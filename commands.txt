
conda activate mimic
export MUJOCO_GL=egl
cd /home/tonyzhao/Research/act-plus-plus
python3 imitate_episodes.py \
--task_name sim_transfer_cube_human \
--ckpt_dir /scr/tonyzhao/train_logs/vq_test \
--policy_class ACT --kl_weight 10 --chunk_size 100 \
--hidden_dim 512 --batch_size 8 --dim_feedforward 3200 \
--num_epochs 10000 --lr 1e-5 --seed 0 --vq

conda activate mobile
cd /home/tonyzhao/Research/act-plus-plus
CUDA_VISIBLE_DEVICES=0 python3 imitate_episodes.py \
--task_name all \
--ckpt_dir /scr/tonyzhao/train_logs/pretrain_all \
--policy_class ACT --kl_weight 10 --chunk_size 50 \
--hidden_dim 512 --batch_size 24 --dim_feedforward 3200 --num_epochs 5000 --lr 1e-4 --seed 0


#### NOTE to reproduce this experiment, uncomment the sim data filtering in utils.py
conda activate mobile
cd /home/tonyzhao/Research/act-plus-plus
CUDA_VISIBLE_DEVICES=0 python3 imitate_episodes.py \
--task_name all \
--ckpt_dir /scr/tonyzhao/train_logs/pretrain_all \
--policy_class ACT --kl_weight 10 --chunk_size 50 \
--hidden_dim 512 --batch_size 24 --dim_feedforward 3200 --lr 1e-4 --seed 0 \
--num_steps 1000000 --eval_every 10000000000 --validate_every 2000 --save_every 5000

# generate mirrored data
conda activate mobile
cd /home/tonyzhao/Research/act-plus-plus
python3 record_sim_episodes.py --task_name sim_transfer_cube_scripted_mirror --dataset_dir /scr/tonyzhao/datasets/sim_transfer_cube_scripted_mirror --num_episodes 50
python3 postprocess_episodes.py --dataset_dir /scr/tonyzhao/datasets/sim_transfer_cube_scripted_mirror --num_episodes 50
# the sim_transfer_cube_scripted_mirror will have 100 episodes
# I then copy the whole dir to sim_transfer_cube_scripted then removed all mirrored episodes
# this gives sim_transfer_cube_scripted_mirror (100 episodes) and sim_transfer_cube_scripted (50 episodes)

# visualize the original data
python3 visualize_episodes.py --dataset_dir /scr/tonyzhao/datasets/sim_transfer_cube_scripted_mirror --episode_idx 0
# visualize the artificially mirrored data
python3 visualize_episodes.py --dataset_dir /scr/tonyzhao/datasets/sim_transfer_cube_scripted_mirror --episode_idx 0 --ismirror

# sanity check
# replay the mirrored data action in the original env
python3 replay_episodes.py  --dataset_path /scr/tonyzhao/datasets/sim_transfer_cube_scripted_mirror/mirror_episode_0.hdf5
# replay the original data action in the original env
python3 replay_episodes.py  --dataset_path /scr/tonyzhao/datasets/sim_transfer_cube_scripted_mirror/episode_0.hdf5


# launch experiment on original data
conda activate mobile
export MUJOCO_GL=egl
cd /home/tonyzhao/Research/act-plus-plus
CUDA_VISIBLE_DEVICES=0 python3 imitate_episodes.py \
--task_name sim_transfer_cube_scripted \
--ckpt_dir /scr/tonyzhao/train_logs/cube_scripted \
--policy_class ACT --kl_weight 10 --chunk_size 50 \
--hidden_dim 512 --batch_size 12 --dim_feedforward 3200 --lr 1e-5 --seed 0 \
--num_steps 100000 --eval_every 2000 --validate_every 2000 --save_every 2000 --no_encoder


# launch experiment on all data
conda activate mobile
export MUJOCO_GL=egl
cd /home/tonyzhao/Research/act-plus-plus
CUDA_VISIBLE_DEVICES=0 python3 imitate_episodes.py \
--task_name sim_transfer_cube_scripted_mirror \
--ckpt_dir /scr/tonyzhao/train_logs/cube_scripted_mirror \
--policy_class ACT --kl_weight 10 --chunk_size 50 \
--hidden_dim 512 --batch_size 12 --dim_feedforward 3200 --lr 1e-5 --seed 0 \
--num_steps 100000 --eval_every 2000 --validate_every 2000 --save_every 2000 --no_encoder


####### DIFFUSION POLICY

- first install https://github.com/ARISE-Initiative/robomimic/tree/r2d2 (note the r2d2 branch)
- on top of it pip install the current repo requirements


conda activate mobile
export MUJOCO_GL=egl
cd /home/tonyzhao/Research/act-plus-plus
CUDA_VISIBLE_DEVICES=0 python3 imitate_episodes.py \
--task_name sim_transfer_cube_scripted \
--ckpt_dir /scr/tonyzhao/train_logs/cube_scripted_diffusion_sweep_0 \
--policy_class Diffusion --chunk_size 32 \
--batch_size 32 --lr 1e-5 --seed 0 \
--num_steps 100000 --eval_every 2000 --validate_every 2000 --save_every 2000


conda activate mobile
export MUJOCO_GL=egl
cd /home/tonyzhao/Research/act-plus-plus
CUDA_VISIBLE_DEVICES=1 python3 imitate_episodes.py \
--task_name sim_transfer_cube_scripted \
--ckpt_dir /scr/tonyzhao/train_logs/cube_scripted_diffusion_sweep_1 \
--policy_class Diffusion --chunk_size 16 \
--batch_size 32 --lr 1e-5 --seed 0 \
--num_steps 100000 --eval_every 2000 --validate_every 2000 --save_every 2000


# above are all 100 train diffusion steps, 1e-5

conda activate mobile
export MUJOCO_GL=egl
cd /home/tonyzhao/Research/act-plus-plus
CUDA_VISIBLE_DEVICES=1 python3 imitate_episodes.py \
--task_name sim_transfer_cube_scripted \
--ckpt_dir /scr/tonyzhao/train_logs/cube_scripted_diffusion_sweep_2_50step_1e-4 \
--policy_class Diffusion --chunk_size 32 \
--batch_size 32 --lr 1e-4 --seed 0 \
--num_steps 100000 --eval_every 2000 --validate_every 2000 --save_every 2000

# Dec 10

######################## more diffusion ########################
conda activate mobile
export MUJOCO_GL=egl
cd /home/tonyzhao/Research/act-plus-plus
CUDA_VISIBLE_DEVICES=0 python3 imitate_episodes.py \
--task_name sim_transfer_cube_scripted \
--ckpt_dir /scr/tonyzhao/train_logs/cube_scripted_diffusion_sweep_3_chunk64 \
--policy_class Diffusion --chunk_size 64 \
--batch_size 32 --lr 1e-4 --seed 0 \
--num_steps 200000 --eval_every 4000 --validate_every 4000 --save_every 4000

conda activate mobile
export MUJOCO_GL=egl
cd /home/tonyzhao/Research/act-plus-plus
CUDA_VISIBLE_DEVICES=0 python3 imitate_episodes.py \
--task_name sim_transfer_cube_scripted \
--ckpt_dir /scr/tonyzhao/train_logs/cube_scripted_diffusion_sweep_4_regressionTest \
--policy_class Diffusion --chunk_size 32 \
--batch_size 32 --lr 1e-4 --seed 0 \
--num_steps 200000 --eval_every 6000 --validate_every 6000 --save_every 6000


conda activate mobile
export MUJOCO_GL=egl
cd /home/tonyzhao/Research/act-plus-plus
CUDA_VISIBLE_DEVICES=0 python3 imitate_episodes.py \
--task_name sim_transfer_cube_scripted \
--ckpt_dir /scr/tonyzhao/train_logs/cube_scripted_diffusion_sweep_5_noEMA \
--policy_class Diffusion --chunk_size 32 \
--batch_size 32 --lr 1e-4 --seed 0 \
--num_steps 200000 --eval_every 6000 --validate_every 6000 --save_every 6000

conda activate mobile
export MUJOCO_GL=egl
cd /home/tonyzhao/Research/act-plus-plus
CUDA_VISIBLE_DEVICES=1 python3 imitate_episodes.py \
--task_name sim_transfer_cube_scripted \
--ckpt_dir /scr/tonyzhao/train_logs/cube_scripted_diffusion_sweep_6_noEMA_seed1 \
--policy_class Diffusion --chunk_size 32 \
--batch_size 32 --lr 1e-4 --seed 1 \
--num_steps 200000 --eval_every 6000 --validate_every 6000 --save_every 6000

###### Diffusion Real ######

## deploy
python3 imitate_episodes.py --task_name aloha_mobile_wipe_wine --ckpt_dir /home/mobile-aloha/interbotix_ws/src/act/ckpts/wipe_wine_diffusion_augmentation_seed0/ --policy_class Diffusion --chunk_size 32 --batch_size 32 --lr 1e-4 --seed 0 --num_steps 1000000 --eval_every 1000000 --validate_every 5000 --save_every 5000 --eval

conda activate mobile
export MUJOCO_GL=egl
cd /home/tonyzhao/Research/act-plus-plus
CUDA_VISIBLE_DEVICES=1 python3 imitate_episodes.py \
--task_name aloha_mobile_wipe_wine \
--ckpt_dir /scr/tonyzhao/train_logs/wipe_wine_diffusion_seed0 \
--policy_class Diffusion --chunk_size 32 \
--batch_size 32 --lr 1e-4 --seed 0 \
--num_steps 1000000 --eval_every 1000000 --validate_every 5000 --save_every 5000

## Cotrain
conda activate mobile
export MUJOCO_GL=egl
cd /home/tonyzhao/Research/act-plus-plus
CUDA_VISIBLE_DEVICES=1 python3 imitate_episodes.py \
--task_name aloha_mobile_wipe_wine_cotrain \
--ckpt_dir /scr/tonyzhao/train_logs/wipe_wine_cotrain_diffusion_seed0 \
--policy_class Diffusion --chunk_size 32 \
--batch_size 32 --lr 1e-4 --seed 0 \
--num_steps 1000000 --eval_every 1000000 --validate_every 5000 --save_every 5000

# train no cotrain again with augmentations
conda activate mobile
export MUJOCO_GL=egl
cd /home/tonyzhao/Research/act-plus-plus
CUDA_VISIBLE_DEVICES=0 python3 imitate_episodes.py \
--task_name aloha_mobile_wipe_wine \
--ckpt_dir /scr/tonyzhao/train_logs/wipe_wine_diffusion_augmentation_seed0 \
--policy_class Diffusion --chunk_size 32 \
--batch_size 32 --lr 1e-4 --seed 0 \
--num_steps 1000000 --eval_every 1000000 --validate_every 5000 --save_every 5000

## Cotrain with augmentations
conda activate mobile
export MUJOCO_GL=egl
cd /home/tonyzhao/Research/act-plus-plus
CUDA_VISIBLE_DEVICES=1 python3 imitate_episodes.py \
--task_name aloha_mobile_wipe_wine_cotrain \
--ckpt_dir /scr/tonyzhao/train_logs/wipe_wine_cotrain_diffusion_augmentation_seed0 \
--policy_class Diffusion --chunk_size 32 \
--batch_size 32 --lr 1e-4 --seed 0 \
--num_steps 1000000 --eval_every 1000000 --validate_every 5000 --save_every 5000

# try chunk size 64, no cotrain

conda activate mobile
export MUJOCO_GL=egl
cd /home/tonyzhao/Research/act-plus-plus
CUDA_VISIBLE_DEVICES=0 python3 imitate_episodes.py \
--task_name aloha_mobile_wipe_wine \
--ckpt_dir /scr/tonyzhao/train_logs/wipe_wine_diffusion_augmentation_chunk64_seed0 \
--policy_class Diffusion --chunk_size 64 \
--batch_size 32 --lr 1e-4 --seed 0 \
--num_steps 1000000 --eval_every 1000000 --validate_every 5000 --save_every 5000

# chunk 64 with cotrain

conda activate mobile
export MUJOCO_GL=egl
cd /home/tonyzhao/Research/act-plus-plus
CUDA_VISIBLE_DEVICES=1 python3 imitate_episodes.py \
--task_name aloha_mobile_wipe_wine_cotrain \
--ckpt_dir /scr/tonyzhao/train_logs/wipe_wine_cotrain_diffusion_augmentation_chunk64_seed0 \
--policy_class Diffusion --chunk_size 64 \
--batch_size 32 --lr 1e-4 --seed 0 \
--num_steps 1000000 --eval_every 1000000 --validate_every 5000 --save_every 5000



# chunk 64 with cotrain + EMA

conda activate mobile
export MUJOCO_GL=egl
cd /home/tonyzhao/Research/act-plus-plus
CUDA_VISIBLE_DEVICES=0 python3 imitate_episodes.py \
--task_name aloha_mobile_wipe_wine_2_cotrain \
--ckpt_dir /scr/tonyzhao/train_logs/wipe_wine_cotrain_diffusion_augmentation_chunk64_ema_seed0 \
--policy_class Diffusion --chunk_size 64 \
--batch_size 32 --lr 1e-4 --seed 0 \
--num_steps 1000000 --eval_every 1000000 --validate_every 5000 --save_every 5000


# chunk 64 with cotrain + EMA + 3e-4

conda activate mobile
export MUJOCO_GL=egl
cd /home/tonyzhao/Research/act-plus-plus
CUDA_VISIBLE_DEVICES=1 python3 imitate_episodes.py \
--task_name aloha_mobile_wipe_wine_2_cotrain \
--ckpt_dir /scr/tonyzhao/train_logs/wipe_wine_cotrain_diffusion_augmentation_chunk64_ema_3e-4_seed0 \
--policy_class Diffusion --chunk_size 64 \
--batch_size 32 --lr 3e-4 --seed 0 \
--num_steps 1000000 --eval_every 1000000 --validate_every 5000 --save_every 5000


######################## VINN ########################


conda activate mobile
cd /home/tonyzhao/Research/act-plus-plus/byol_pytorch/examples/lightning
CUDA_VISIBLE_DEVICES=1 python3 train.py --dataset_dir /scr/tonyzhao/datasets/sim_transfer_cube_scripted --cam_name top --seed 0

conda activate mobile
cd /home/tonyzhao/Research/act-plus-plus/byol_pytorch/examples/lightning
CUDA_VISIBLE_DEVICES=0 python3 train.py --dataset_dir /scr/tonyzhao/datasets/sim_transfer_cube_scripted --cam_name left_wrist --seed 0

conda activate mobile
cd /home/tonyzhao/Research/act-plus-plus/byol_pytorch/examples/lightning
CUDA_VISIBLE_DEVICES=1 python3 train.py --dataset_dir /scr/tonyzhao/datasets/sim_transfer_cube_scripted --cam_name right_wrist --seed 0

conda activate mobile
cd /home/tonyzhao/Research/act-plus-plus
TASK_NAME=sim_transfer_cube_scripted
python3 vinn_cache_feature.py --ckpt_path /home/tonyzhao/Research/act-plus-plus/byol_pytorch/examples/lightning/byol-${TASK_NAME}-DUMMY-seed-0.pt

TASK_NAME=sim_transfer_cube_scripted
python3 vinn_select_k.py \
--dataset_dir /scr/tonyzhao/datasets/sim_transfer_cube_scripted \
--ckpt_dir /scr/tonyzhao/train_logs/VINN-eval-seed-0-test

python3 vinn_eval.py \
--dataset_dir /scr/tonyzhao/datasets/sim_transfer_cube_scripted \
--model_dir /home/tonyzhao/Research/act-plus-plus/byol_pytorch/examples/lightning/byol-${TASK_NAME}-DUMMY-seed-0.pt \
--ckpt_dir /scr/tonyzhao/train_logs/VINN-eval-seed-0-test \
--task_name $TASK_NAME 

## TODO
make sure env is consistent
tune a bit more


######################## VINN Real ########################

### test backward compatibility

conda activate mobile
cd /home/tonyzhao/Research/act-plus-plus/byol_pytorch/examples/lightning
CUDA_VISIBLE_DEVICES=1 python3 train.py --task sim_transfer_cube_scripted --cam_name top --seed 0
CUDA_VISIBLE_DEVICES=1 python3 train.py --task sim_transfer_cube_scripted --cam_name left_wrist --seed 0

conda activate mobile
cd /home/tonyzhao/Research/act-plus-plus/byol_pytorch/examples/lightning
CUDA_VISIBLE_DEVICES=1 python3 train.py --task sim_transfer_cube_scripted --cam_name right_wrist --seed 0


conda activate mobile
cd /home/tonyzhao/Research/act-plus-plus
TASK_NAME=sim_transfer_cube_scripted
python3 vinn_cache_feature.py --ckpt_path /home/tonyzhao/Research/act-plus-plus/byol_pytorch/examples/lightning/byol-${TASK_NAME}-DUMMY-seed-0.pt

TASK_NAME=sim_transfer_cube_scripted
python3 vinn_select_k.py \
--dataset_dir /scr/tonyzhao/datasets/sim_transfer_cube_scripted \
--ckpt_dir /scr/tonyzhao/train_logs/VINN-eval-seed-0-test

python3 vinn_eval.py \
--dataset_dir /scr/tonyzhao/datasets/sim_transfer_cube_scripted \
--model_dir /home/tonyzhao/Research/act-plus-plus/byol_pytorch/examples/lightning/byol-${TASK_NAME}-DUMMY-seed-0.pt \
--ckpt_dir /scr/tonyzhao/train_logs/VINN-eval-seed-0-test \
--task_name $TASK_NAME 

### new data loader passed backward compatibility


conda activate mobile
cd /home/tonyzhao/Research/act-plus-plus/byol_pytorch/examples/lightning
#CUDA_VISIBLE_DEVICES=1 python3 train.py --task aloha_mobile_wipe_wine --cam_name cam_high --seed 0
#CUDA_VISIBLE_DEVICES=1 python3 train.py --task aloha_mobile_wipe_wine --cam_name cam_left_wrist --seed 0
#CUDA_VISIBLE_DEVICES=1 python3 train.py --task aloha_mobile_wipe_wine --cam_name cam_right_wrist --seed 0

#CUDA_VISIBLE_DEVICES=1 python3 train.py --task aloha_mobile_wipe_wine_cotrain --cam_name cam_high --seed 0
#CUDA_VISIBLE_DEVICES=1 python3 train.py --task aloha_mobile_wipe_wine_cotrain --cam_name cam_left_wrist --seed 0
CUDA_VISIBLE_DEVICES=1 python3 train.py --task aloha_mobile_wipe_wine_cotrain --cam_name cam_right_wrist --seed 0


conda activate mobile
cd /home/tonyzhao/Research/act-plus-plus/byol_pytorch/examples/lightning
#CUDA_VISIBLE_DEVICES=1 python3 train.py --task aloha_mobile_wash_pan --cam_name cam_high --seed 0
#CUDA_VISIBLE_DEVICES=1 python3 train.py --task aloha_mobile_wash_pan --cam_name cam_left_wrist --seed 0
#CUDA_VISIBLE_DEVICES=1 python3 train.py --task aloha_mobile_wash_pan --cam_name cam_right_wrist --seed 0

#CUDA_VISIBLE_DEVICES=1 python3 train.py --task aloha_mobile_wash_pan_cotrain --cam_name cam_high --seed 0
#CUDA_VISIBLE_DEVICES=1 python3 train.py --task aloha_mobile_wash_pan_cotrain --cam_name cam_left_wrist --seed 0
CUDA_VISIBLE_DEVICES=1 python3 train.py --task aloha_mobile_wash_pan_cotrain --cam_name cam_right_wrist --seed 0


conda activate mobile
cd /home/tonyzhao/Research/act-plus-plus/byol_pytorch/examples/lightning
CUDA_VISIBLE_DEVICES=1 python3 train.py --task aloha_mobile_wipe_wine_cotrain --cam_name cam_right_wrist --seed 0
CUDA_VISIBLE_DEVICES=1 python3 train.py --task aloha_mobile_elevator_truncated --cam_name cam_high --seed 0
CUDA_VISIBLE_DEVICES=1 python3 train.py --task aloha_mobile_elevator_truncated --cam_name cam_left_wrist --seed 0
CUDA_VISIBLE_DEVICES=1 python3 train.py --task aloha_mobile_elevator_truncated --cam_name cam_right_wrist --seed 0


conda activate mobile
cd /home/tonyzhao/Research/act-plus-plus/byol_pytorch/examples/lightning
CUDA_VISIBLE_DEVICES=1 python3 train.py --task aloha_mobile_wash_pan_cotrain --cam_name cam_right_wrist --seed 0
CUDA_VISIBLE_DEVICES=1 python3 train.py --task aloha_mobile_elevator_truncated_cotrain --cam_name cam_high --seed 0
CUDA_VISIBLE_DEVICES=1 python3 train.py --task aloha_mobile_elevator_truncated_cotrain --cam_name cam_left_wrist --seed 0
CUDA_VISIBLE_DEVICES=1 python3 train.py --task aloha_mobile_elevator_truncated_cotrain --cam_name cam_right_wrist --seed 0


conda activate mobile
export CUDA_VISIBLE_DEVICES=1
cd /home/tonyzhao/Research/act-plus-plus
TASK_NAME=aloha_mobile_wipe_wine
DATA_NAME=aloha_mobile_wipe_wine
python3 vinn_cache_feature.py --ckpt_path /home/tonyzhao/Research/act-plus-plus/byol_pytorch/examples/lightning/byol-${TASK_NAME}-DUMMY-seed-0.pt \
--dataset_dir /scr/tonyzhao/mobile_aloha_datasets/${DATA_NAME}


conda activate mobile
export CUDA_VISIBLE_DEVICES=1
cd /home/tonyzhao/Research/act-plus-plus
TASK_NAME=aloha_mobile_wipe_wine_cotrain
DATA_NAME=aloha_mobile_wipe_wine
python3 vinn_cache_feature.py --ckpt_path /home/tonyzhao/Research/act-plus-plus/byol_pytorch/examples/lightning/byol-${TASK_NAME}-DUMMY-seed-0.pt \
--dataset_dir /scr/tonyzhao/mobile_aloha_datasets/${DATA_NAME}



conda activate mobile
export CUDA_VISIBLE_DEVICES=1
cd /home/tonyzhao/Research/act-plus-plus
TASK_NAME=aloha_mobile_wash_pan
DATA_NAME=aloha_mobile_wash_pan
python3 vinn_cache_feature.py --ckpt_path /home/tonyzhao/Research/act-plus-plus/byol_pytorch/examples/lightning/byol-${TASK_NAME}-DUMMY-seed-0.pt \
--dataset_dir /scr/tonyzhao/mobile_aloha_datasets/${DATA_NAME}

conda activate mobile
export CUDA_VISIBLE_DEVICES=1
cd /home/tonyzhao/Research/act-plus-plus
TASK_NAME=aloha_mobile_wash_pan_cotrain
DATA_NAME=aloha_mobile_wash_pan
python3 vinn_cache_feature.py --ckpt_path /home/tonyzhao/Research/act-plus-plus/byol_pytorch/examples/lightning/byol-${TASK_NAME}-DUMMY-seed-0.pt \
--dataset_dir /scr/tonyzhao/mobile_aloha_datasets/${DATA_NAME}


conda activate mobile
export CUDA_VISIBLE_DEVICES=1
cd /home/tonyzhao/Research/act-plus-plus
TASK_NAME=aloha_mobile_elevator_truncated
DATA_NAME=aloha_mobile_elevator_truncated
python3 vinn_cache_feature.py --ckpt_path /home/tonyzhao/Research/act-plus-plus/byol_pytorch/examples/lightning/byol-${TASK_NAME}-DUMMY-seed-0.pt \
--dataset_dir /scr/tonyzhao/mobile_aloha_datasets/${DATA_NAME}

conda activate mobile
export CUDA_VISIBLE_DEVICES=1
cd /home/tonyzhao/Research/act-plus-plus
TASK_NAME=aloha_mobile_elevator_truncated_cotrain
DATA_NAME=aloha_mobile_elevator_truncated
python3 vinn_cache_feature.py --ckpt_path /home/tonyzhao/Research/act-plus-plus/byol_pytorch/examples/lightning/byol-${TASK_NAME}-DUMMY-seed-0.pt \
--dataset_dir /scr/tonyzhao/mobile_aloha_datasets/${DATA_NAME}



# push chair task

conda activate mobile
export CUDA_VISIBLE_DEVICES=0 
cd /home/tonyzhao/Research/act-plus-plus/byol_pytorch/examples/lightning
python3 train.py --task aloha_mobile_chair_truncated --cam_name cam_high --seed 0
python3 train.py --task aloha_mobile_chair_truncated --cam_name cam_left_wrist --seed 0
python3 train.py --task aloha_mobile_chair_truncated --cam_name cam_right_wrist --seed 0

cd /home/tonyzhao/Research/act-plus-plus
TASK_NAME=aloha_mobile_chair_truncated
DATA_NAME=aloha_mobile_chair_truncated
python3 vinn_cache_feature.py --ckpt_path /home/tonyzhao/Research/act-plus-plus/byol_pytorch/examples/lightning/byol-${TASK_NAME}-DUMMY-seed-0.pt \
--dataset_dir /scr/tonyzhao/mobile_aloha_datasets/${DATA_NAME}



conda activate mobile
export CUDA_VISIBLE_DEVICES=1
cd /home/tonyzhao/Research/act-plus-plus/byol_pytorch/examples/lightning
python3 train.py --task aloha_mobile_chair_truncated_cotrain --cam_name cam_high --seed 0
python3 train.py --task aloha_mobile_chair_truncated_cotrain --cam_name cam_left_wrist --seed 0
python3 train.py --task aloha_mobile_chair_truncated_cotrain --cam_name cam_right_wrist --seed 0

cd /home/tonyzhao/Research/act-plus-plus
TASK_NAME=aloha_mobile_chair_truncated_cotrain
DATA_NAME=aloha_mobile_chair_truncated
python3 vinn_cache_feature.py --ckpt_path /home/tonyzhao/Research/act-plus-plus/byol_pytorch/examples/lightning/byol-${TASK_NAME}-DUMMY-seed-0.pt \
--dataset_dir /scr/tonyzhao/mobile_aloha_datasets/${DATA_NAME}


# cache feature again for wipe wine

conda activate mobile
export CUDA_VISIBLE_DEVICES=0
cd /home/tonyzhao/Research/act-plus-plus
TASK_NAME=aloha_mobile_wipe_wine
DATA_NAME=aloha_mobile_wipe_wine
python3 vinn_cache_feature.py --ckpt_path /home/tonyzhao/Research/act-plus-plus/byol_pytorch/examples/lightning/byol-${TASK_NAME}-DUMMY-seed-0.pt \
--dataset_dir /scr/tonyzhao/mobile_aloha_datasets/${DATA_NAME}

cd /home/tonyzhao/Research/act-plus-plus
TASK_NAME=aloha_mobile_wipe_wine_cotrain
DATA_NAME=aloha_mobile_wipe_wine
python3 vinn_cache_feature.py --ckpt_path /home/tonyzhao/Research/act-plus-plus/byol_pytorch/examples/lightning/byol-${TASK_NAME}-DUMMY-seed-0.pt \
--dataset_dir /scr/tonyzhao/mobile_aloha_datasets/${DATA_NAME}



# run on real robot

TASK_NAME=aloha_mobile_chair_truncated
python3 vinn_select_k.py \
--dataset_dir /scr/tonyzhao/mobile_aloha_datasets/${DATA_NAME} \
--ckpt_dir /scr/tonyzhao/train_logs/VINN-eval-${TASK_NAME}-seed-0

python3 vinn_eval.py \
--dataset_dir /scr/tonyzhao/datasets/sim_transfer_cube_scripted \
--model_dir /home/tonyzhao/Research/act-plus-plus/byol_pytorch/examples/lightning/byol-${TASK_NAME}-DUMMY-seed-0.pt \
--ckpt_dir /scr/tonyzhao/train_logs/VINN-eval-${TASK_NAME}-seed-0 \
--task_name $TASK_NAME 



TASK_NAME=aloha_mobile_chair_truncated
python3 vinn_select_k.py \
--dataset_dir /scr/tonyzhao/mobile_aloha_datasets/${DATA_NAME} \
--ckpt_dir /scr/tonyzhao/train_logs/VINN-eval-${TASK_NAME}-seed-0

python3 vinn_eval.py \
--dataset_dir /scr/tonyzhao/datasets/sim_transfer_cube_scripted \
--model_dir /home/tonyzhao/Research/act-plus-plus/byol_pytorch/examples/lightning/byol-${TASK_NAME}-DUMMY-seed-0.pt \
--ckpt_dir /scr/tonyzhao/train_logs/VINN-eval-${TASK_NAME}-seed-0 \
--task_name $TASK_NAME 



# eval on real robot

conda activate aloha
cd /home/mobile-aloha/interbotix_ws/src/act
TASK_NAME=aloha_mobile_wipe_wine
python3 vinn_cache_feature.py --ckpt_path /home/mobile-aloha/interbotix_ws/src/act/ckpts/vinn_ckpts/byol-${TASK_NAME}-DUMMY-seed-0.pt


TASK_NAME=aloha_mobile_wipe_wine
python3 vinn_select_k.py \
--dataset_dir /home/mobile-aloha/data/${TASK_NAME} \
--ckpt_dir /home/mobile-aloha/interbotix_ws/src/act/ckpts/vinn_ckpts/VINN-eval-seed-0-test \


TASK_NAME=aloha_mobile_wipe_wine
python3 vinn_eval.py \
--dataset_dir /home/mobile-aloha/data/${TASK_NAME} \
--model_dir /home/mobile-aloha/interbotix_ws/src/act/ckpts/vinn_ckpts/byol-${TASK_NAME}-DUMMY-seed-0.pt \
--ckpt_dir /home/mobile-aloha/interbotix_ws/src/act/ckpts/vinn_ckpts/VINN-eval-seed-0-test \
--task_name $TASK_NAME 


---------------------------------------------------------------------------------------

NOTE: chunk size cannot be any number, try before launching
TODO: Add history, EMA at test time

conda activate mobile
cd /home/tonyzhao/Research/act-plus-plus
CUDA_VISIBLE_DEVICES=1 python3 train_actuator_network.py



