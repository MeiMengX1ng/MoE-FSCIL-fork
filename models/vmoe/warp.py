import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.utils import _pair


def set_grad(var):
    def hook(grad):
        var.grad = grad
    return hook


def im2col(input_data, filter_h, filter_w, stride=1, pad=0):
    n, c, h, w = input_data.shape
    out_h = (h + 2 * pad - filter_h) // stride + 1
    out_w = (w + 2 * pad - filter_w) // stride + 1
    img = F.pad(input_data, [pad, pad, pad, pad], 'constant', 0)
    col = torch.zeros((n, c, filter_h, filter_w, out_h, out_w), device=input_data.device, dtype=input_data.dtype)
    for y in range(filter_h):
        y_max = y + stride * out_h
        for x in range(filter_w):
            x_max = x + stride * out_w
            col[:, :, y, x, :, :] = img[:, :, y:y_max:stride, x:x_max:stride]
    return torch.permute(col, (0, 4, 5, 1, 2, 3)).reshape(n * out_h * out_w, -1)


def im2col_from_conv(input_data, conv):
    return im2col(input_data, conv.kernel_size[0], conv.kernel_size[1], conv.stride[0], conv.padding[0])


class WaRPModule(nn.Module):
    def __init__(self, layer):
        super().__init__()
        self.weight = layer.weight
        self.bias = layer.bias
        self.weight.requires_grad = False
        if self.bias is not None:
            self.bias.requires_grad = False
        if self.weight.ndim != 2:
            co, ci, k1, k2 = self.weight.shape
            self.basis_coeff = nn.Parameter(torch.Tensor(co, ci * k1 * k2, 1, 1), requires_grad=True)
            self.register_buffer("UT_forward_conv", torch.Tensor(ci * k1 * k2, ci, k1, k2))
            self.register_buffer("UT_backward_conv", torch.Tensor(co, co, 1, 1))
        else:
            self.basis_coeff = nn.Parameter(torch.Tensor(self.weight.shape), requires_grad=True)
        self.register_buffer("forward_covariance", None)
        self.register_buffer("basis_coefficients", torch.Tensor(self.weight.shape).reshape(self.weight.shape[0], -1))
        self.register_buffer("coeff_mask", torch.zeros(self.basis_coeff.shape))
        self.register_buffer("UT_forward", torch.eye(self.basis_coeff.shape[1]))
        self.register_buffer("UT_backward", torch.eye(self.basis_coeff.shape[0]))
        self.flag = True


class Conv2dWaRP(WaRPModule):
    def __init__(self, conv_layer):
        super().__init__(conv_layer)
        for attr in ['in_channels', 'out_channels', 'kernel_size', 'dilation', 'stride', 'padding', 'padding_mode', 'groups']:
            setattr(self, attr, getattr(conv_layer, attr))
        self.batch_count = 0

    def pre_forward(self, input):
        with torch.no_grad():
            input_col = im2col_from_conv(input.clone(), self)
            return input_col.t() @ input_col

    def post_backward(self):
        with torch.no_grad():
            if self.forward_covariance is not None:
                self.forward_covariance = self.forward_cov + (self.batch_count / (self.batch_count + 1)) * (self.forward_covariance - self.forward_cov)
            else:
                self.forward_covariance = self.forward_cov
            self.batch_count += 1

    def forward(self, input):
        if not self.flag:
            self.forward_cov = self.pre_forward(input)
            if self.padding_mode == 'circular':
                expanded_padding = ((self.padding[1] + 1) // 2, self.padding[1] // 2, (self.padding[0] + 1) // 2, self.padding[0] // 2)
                return F.conv2d(F.pad(input, expanded_padding, mode='circular'), self.weight, self.bias, self.stride, _pair(0), self.dilation, self.groups)
            return F.conv2d(input, self.weight, self.bias, self.stride, self.padding, self.dilation, self.groups)

        utx = F.conv2d(input, self.UT_forward_conv, None, self.stride, self.padding, self.dilation, self.groups)
        autx = F.conv2d(utx, (self.basis_coeff * self.coeff_mask).clone().detach() + self.basis_coeff * (1 - self.coeff_mask), None, 1, 0)
        return F.conv2d(autx, self.UT_backward_conv, self.bias, 1, 0)


warped_modules = {
    nn.Conv2d: Conv2dWaRP,
}


def switch_warp_modules(module):
    new_children = {}
    for name, submodule in module.named_children():
        if isinstance(submodule, nn.Conv2d):
            switched = warped_modules[type(submodule)](submodule)
            new_children[name] = switched
        switch_warp_modules(submodule)

    for name, switched in new_children.items():
        setattr(module, name, switched)
    return module


def _same_device(x_mask, x):
    if x_mask.device != x.device:
        return x_mask.to(x.device)
    return x_mask


@torch.no_grad()
def compute_warp_orthonormal_basis(model, dataloader):
    warped = [module for module in model.modules() if isinstance(module, WaRPModule)]
    for module in warped:
        module.flag = False
        module.forward_covariance = None
        module.batch_count = 0

    for batch in dataloader:
        images = batch[0].cuda()
        _ = model(images)
        for module in warped:
            if hasattr(module, "post_backward"):
                module.post_backward()

    for module in warped:
        module.flag = True
        cov = module.forward_covariance
        if cov is None:
            continue
        _, _, vt = torch.linalg.svd(cov, full_matrices=True)
        weight = module.weight
        if weight.ndim != 2:
            weight = weight.reshape(weight.shape[0], -1)
        ut_backward = torch.eye(weight.shape[0], device=weight.device, dtype=weight.dtype)
        vt = _same_device(vt, weight)
        coeff = ut_backward @ weight @ vt.t()
        module.UT_forward = vt
        module.UT_backward = ut_backward
        module.basis_coefficients.data = coeff.data
        if module.weight.ndim != 2:
            module.UT_forward_conv = vt.reshape(
                vt.shape[0], module.weight.shape[1], module.weight.shape[2], module.weight.shape[3]
            )
            module.UT_backward_conv = ut_backward.t().reshape(module.weight.shape[0], module.weight.shape[0], 1, 1)
            module.basis_coeff.data = coeff.reshape(module.weight.shape[0], -1, 1, 1).data
        else:
            module.basis_coeff.data = coeff.data


def restore_warp_weights(model):
    for module in model.modules():
        if not isinstance(module, WaRPModule):
            continue
        weight = module.weight
        coeff = module.basis_coeff.data.reshape(weight.shape[0], -1)
        restored = module.UT_backward.t() @ coeff @ module.UT_forward
        module.weight.data = restored.reshape(weight.shape).data
