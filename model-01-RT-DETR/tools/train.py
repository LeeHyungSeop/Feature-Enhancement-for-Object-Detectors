"""by lyuwenyu
"""

import os 
import sys 
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
import argparse

import src.misc.dist as dist 
from src.core import YAMLConfig 
from src.solver import TASKS


def main(args, ) -> None:
    '''main
    '''
    dist.init_distributed()
    if args.seed is not None:
        dist.set_seed(args.seed)

    assert not all([args.tuning, args.resume]), \
        'Only support from_scrach or resume or tuning at one time'

    cfg = YAMLConfig(
        args.config,
        resume=args.resume, 
        use_amp=args.amp,
        tuning=args.tuning
    )

    solver = TASKS[cfg.yaml_cfg['task']](cfg)
    
    if args.test_only:
        solver.val()
    else:
        solver.fit()


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--config', '-c', type=str, )
    parser.add_argument('--resume', '-r', type=str, )
    parser.add_argument('--tuning', '-t', type=str, )
    parser.add_argument('--test-only', action='store_true', default=False,)
    parser.add_argument('--amp', action='store_true', default=False,)
    parser.add_argument('--seed', type=int, help='seed',)
    args = parser.parse_args()

    main(args)


'''
pip install -r requirements.txt

# train on multi-gpu
export CUDA_VISIBLE_DEVICES=0,1
torchrun --nproc_per_node=2 tools/train.py -c configs/rtdetr/rtdetr_r50vd_6x_coco.yml \
    2>&1 | tee ./output/s3c_s4o_s5g/train_log.txt
    
export CUDA_VISIBLE_DEVICES=0
torchrun --nproc_per_node=1 tools/train.py -c configs/rtdetr/rtdetr_r50vd_6x_coco.yml \
    --test-only \
    2>&1 | tee ./output/original_swinT/flops.txt
    
export CUDA_VISIBLE_DEVICES=0,1
torchrun --nproc_per_node=2 tools/train.py -c configs/rtdetr/rtdetr_r50vd_6x_coco.yml \
    2>&1 | tee ./outputs/S2_S4_Reg/train_log.txt

# train
export CUDA_VISIBLE_DEVICES=0
python tools/train.py -c configs/rtdetr/rtdetr_r50vd_6x_coco.yml \
    2>&1 | tee ./output/test/log.txt


# test on single-gpu
export CUDA_VISIBLE_DEVICES=0,1
torchrun --nproc_per_node=2 tools/train.py -c configs/rtdetr/rtdetr_r50vd_6x_coco.yml --test-only \
    --resume "/home/hslee/Desktop/Backbone-Neck_Self-Distillation/RT-DETR_FH/output/s3c_s4o_s5g/checkpoint0000.pth" \
    2>&1 | tee ./output/test/test.txt


torchrun --nproc_per_node=2 tools/train.py -c configs/rtdetr/rtdetr_r50vd_6x_coco.yml \
    --resume "/home/hslee/Desktop/Backbone-Neck_Self-Distillation/RT-DETR/output/rtdetr_r50vd_6x_coco_B-N_S-D_KLDiv/checkpoint.pth" \
    2>&1 | tee -a ./output/rtdetr_r50vd_6x_coco_B-N_S-D_KLDiv/train_log4.txt

'''