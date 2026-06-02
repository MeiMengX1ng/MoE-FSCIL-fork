echo "Train VMOE on MiniImageNet"
seed_num=1
gpu_num=0

router_disc_type=msd
router_feat_mode=frozen
router_neighbor_k=5
router_sample_n=5
backbone_model_dir=""
router_model_dir=""

if [ "$router_disc_type" = "dsd" ]; then
    router_neighbor_k=10
fi

python train.py -project vmoe \
        -dataset mini_imagenet \
        -base_mode ft_cos \
        -new_mode ft_cos \
        -backbone_type clip_vit_b16 \
        -model_image_size 224 \
        ${backbone_model_dir:+-backbone_model_dir $backbone_model_dir} \
        -router_disc_type $router_disc_type \
        -router_feat_mode $router_feat_mode \
        ${router_model_dir:+-router_model_dir $router_model_dir} \
        -router_neighbor_k $router_neighbor_k \
        -router_sample_n $router_sample_n \
        -lora_rank 8 \
        -lora_alpha 16 \
        -lambda_cross 0.8 \
        -aug_lambda_min 0.45 \
        -aug_lambda_max 0.75 \
        -gamma 0.1 \
        -lr_base 0.1 \
        -lr_new 0.1 \
        -decay 0.0005 \
        -epochs_base 100 \
        -epochs_new 20 \
        -schedule Milestone \
        -milestones 40 80 \
        -milestones_new 10 15 \
        -batch_size_base 128 \
        -batch_size_new 0 \
        -test_batch_size 100 \
        -temperature 16 \
        -start_session 0 \
        -gpu $gpu_num \
        -seed $seed_num
