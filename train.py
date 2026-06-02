import argparse
import importlib
from utils import *

MODEL_DIR=None
DATA_DIR = 'data/'
PROJECT='base'

def get_command_line_parser():
    parser = argparse.ArgumentParser()

    # about dataset and network
    parser.add_argument('-project', type=str, default=PROJECT)
    parser.add_argument('-dataset', type=str, default='cub200',
                        choices=['mini_imagenet', 'cub200', 'cifar100'])
    parser.add_argument('-dataroot', type=str, default=DATA_DIR)

    # about pre-training
    parser.add_argument('-epochs_base', type=int, default=100)
    parser.add_argument('-epochs_new', type=int, default=100)
    parser.add_argument('-lr_base', type=float, default=0.1)
    parser.add_argument('-lr_new', type=float, default=0.1)
    parser.add_argument('-schedule', type=str, default='Step',
                        choices=['Step', 'Milestone'])
    parser.add_argument('-milestones', nargs='+', type=int, default=[60, 70])
    parser.add_argument('-step', type=int, default=40)
    parser.add_argument('-decay', type=float, default=0.0005)
    parser.add_argument('-momentum', type=float, default=0.9)
    parser.add_argument('-gamma', type=float, default=0.1)
    parser.add_argument('-temperature', type=int, default=16)
    parser.add_argument('-not_data_init', action='store_true', help='using average data embedding to init or not')

    parser.add_argument('-batch_size_base', type=int, default=128)
    parser.add_argument('-batch_size_new', type=int, default=0, help='set 0 will use all the availiable training image for new')
    parser.add_argument('-test_batch_size', type=int, default=100)
    parser.add_argument('-base_mode', type=str, default='ft_cos',
                        choices=['ft_dot', 'ft_cos'])
    parser.add_argument('-new_mode', type=str, default='avg_cos',
                        choices=['ft_dot', 'ft_cos', 'avg_cos'])

    # for episode learning
    parser.add_argument('-train_episode', type=int, default=50)
    parser.add_argument('-episode_shot', type=int, default=1)
    parser.add_argument('-episode_way', type=int, default=15)
    parser.add_argument('-episode_query', type=int, default=15)

    # for cec
    parser.add_argument('-lrg', type=float, default=0.1) #lr for graph attention network
    parser.add_argument('-low_shot', type=int, default=1)
    parser.add_argument('-low_way', type=int, default=15)

    parser.add_argument('-start_session', type=int, default=0)
    parser.add_argument('-model_dir', type=str, default=MODEL_DIR, help='loading model parameter from a specific dir')
    parser.add_argument('-set_no_val', action='store_true', help='set validation using test set or no validation')

    parser.add_argument('-backbone_type', type=str, default='clip_vit_b16',
                        choices=['clip_vit_b16', 'resnet18'])
    parser.add_argument('-backbone_feat_dim', type=int, default=768)
    parser.add_argument('-model_image_size', type=int, default=224)
    parser.add_argument('-backbone_model_dir', type=str, default=None)
    parser.add_argument('-router_disc_type', type=str, default='msd',
                        choices=['msd', 'dsd'])
    parser.add_argument('-router_feat_mode', type=str, default='frozen',
                        choices=['frozen', 'warp'])
    parser.add_argument('-router_bottleneck_dim', type=int, default=256)
    parser.add_argument('-router_loss_weight', type=float, default=1.0)
    parser.add_argument('-router_model_dir', type=str, default=None)
    parser.add_argument('-lora_rank', type=int, default=8)
    parser.add_argument('-lora_alpha', type=float, default=16.0)
    parser.add_argument('-lambda_cross', type=float, default=0.8)
    parser.add_argument('-lambda_ortho', type=float, default=1.0)
    parser.add_argument('-aug_lambda_min', type=float, default=0.45)
    parser.add_argument('-aug_lambda_max', type=float, default=0.75)
    parser.add_argument('-router_neighbor_k', type=int, default=5)
    parser.add_argument('-router_sample_n', type=int, default=5)
    parser.add_argument('-milestones_new', nargs='+', type=int, default=[10, 15])

    # about training
    parser.add_argument('-gpu', default='0,1,2,3')
    parser.add_argument('-num_workers', type=int, default=8)
    parser.add_argument('-seed', type=int, default=1)
    parser.add_argument('-debug', action='store_true')

    return parser


if __name__ == '__main__':
    parser = get_command_line_parser()
    args = parser.parse_args()
    set_seed(args.seed)
    pprint(vars(args))
    args.num_gpu = set_gpu(args)

    trainer = importlib.import_module('models.%s.fscil_trainer' % (args.project)).FSCILTrainer(args)
    trainer.train()
