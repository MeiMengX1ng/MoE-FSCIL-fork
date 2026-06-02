echo "Train VMOE on CUB200"
HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
export HF_ENDPOINT
echo "Using HF_ENDPOINT=$HF_ENDPOINT"
seed_num=1
gpu_num=0
model_dir=""

router_disc_type=msd
router_feat_mode=frozen
router_neighbor_k=5
router_sample_n=5
router_epochs_base=40
router_epochs_new=10
num_workers=16

if [ "$router_disc_type" = "dsd" ]; then
    router_neighbor_k=10
fi

python train.py -project vmoe \
        -dataset cub200 \
        -base_mode ft_cos \
        -new_mode ft_cos \
        -backbone_type clip_vit_b16 \
        -model_image_size 224 \
        ${model_dir:+-model_dir $model_dir} \
        -router_disc_type $router_disc_type \
        -router_feat_mode $router_feat_mode \
        -router_neighbor_k $router_neighbor_k \
        -router_sample_n $router_sample_n \
        -lora_rank 8 \
        -lora_alpha 16 \
        -lambda_cross 0.8 \
        -aug_lambda_min 0.45 \
        -aug_lambda_max 0.75 \
        -gamma 0.1 \
        -lr_base 0.01 \
        -lr_new 0.01 \
        -decay 0.0005 \
        -epochs_base 120 \
        -epochs_new 20 \
        -schedule Milestone \
        -milestones 30 60 90 \
        -milestones_new 10 15 \
        -router_epochs_base $router_epochs_base \
        -router_epochs_new $router_epochs_new \
        -batch_size_base 128 \
        -batch_size_new 0 \
        -test_batch_size 100 \
        -temperature 16 \
        -num_workers $num_workers \
        -start_session 0 \
        -gpu $gpu_num \
        -seed $seed_num
