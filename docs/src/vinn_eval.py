import torch
from torch import nn
import torch.nn.functional as F
import numpy as np
import h5py
import pathlib
import os
import argparse
import matplotlib.pyplot as plt
from PIL import Image
import torchvision
from torchvision import transforms
# from visualize_episodes import visualize_joints
from utils import set_seed, sample_box_pose
# from imitate_episodes import get_image
from sim_env import BOX_POSE
from constants import DT
from imitate_episodes import save_videos
from einops import rearrange
import time

DT = 0.02
import IPython
e = IPython.embed

# modified from https://github.com/jyopari/VINN/blob/main/nearest-neighbor-eval/handle_nn.ipynb

def calculate_nearest_neighbors(curr_feature, support_inputs, support_targets, k, state_weight):
    has_skip = len(support_targets.shape) == 3
    if has_skip: # when there is action skip
        num_targets, skip, a_dim = support_targets.shape
        support_targets = support_targets.view((num_targets, -1))

    curr_vis_feature, curr_s_feature = curr_feature
    support_vis_feature, support_s_feature = support_inputs

    pairwise_dist_vis = torch.norm(curr_vis_feature - support_vis_feature, dim=1).unsqueeze(0)
    pairwise_dist_s = torch.norm(curr_s_feature - support_s_feature, dim=1).unsqueeze(0)
    pairwise_dist = pairwise_dist_vis + pairwise_dist_s * state_weight

    sorted_dist, index = torch.sort(pairwise_dist, dim=1) # sort the support axis
    permuted_support_targets = support_targets[index]
    topk_dist = pairwise_dist[:, :k]
    topk_support_targets = permuted_support_targets[:, :k]
    weights = F.softmax(-topk_dist, dim=1)
    weighted_support_targets = weights.unsqueeze(2) * topk_support_targets
    prediction = torch.sum(weighted_support_targets, dim=1)

    if has_skip:
        num_predictions = prediction.shape[0]
        prediction = prediction.reshape((num_predictions, skip, a_dim))

    return prediction


def main(args):
    # TODO ######################
    k = None # for scripted box transfer
    skip = 100
    real_robot = True
    save_episode = True
    # TODO ######################
    onscreen_cam = 'main'
    state_dim = 14
    dataset_dir = args['dataset_dir']
    onscreen_render = args['onscreen_render']
    ckpt_dir = args['ckpt_dir']
    model_dir = args['model_dir']
    task_name = args['task_name']

    if 'insertion' in task_name:
        sim_episode_len = 400
        env_max_reward = 4
        ks = [None]
    elif 'transfer_cube' in task_name:
        sim_episode_len = 400
        env_max_reward = 4
        ks = [1, 1, 1]
        if 'human' in dataset_dir:
            state_weight = 5
        else:
            state_weight = 10
        print(f'{state_weight=}')
    elif task_name == 'ziploc_slide':
        env_max_reward = 1
        ks = [71]
        state_weight = 0
    elif task_name == 'aloha_mobile_wipe_wine':
        sim_episode_len = 1300
        env_max_reward = 4
        ks = [2, 2, 2]
        state_weight = 5
        print(f'{state_weight=}')
    else:
        raise NotImplementedError

    model_name = pathlib.PurePath(model_dir).name
    seed = int(model_name.split('-')[-1][:-3])

    repr_type = 'byol'
    if 'cotrain' in model_name:
        repr_type += '_cotrain'
    e() # make sure!

    k = ks[seed]

    if real_robot:
        BASE_DELAY = 15
        query_freq = skip - BASE_DELAY

    # load train data
    vis_features = []
    state_features = []
    Y = []
    for episode_id in range(0, 40):
        dataset_path = os.path.join(dataset_dir, f'episode_{episode_id}.hdf5')
        with h5py.File(dataset_path, 'r') as root:
            action = root['/action'][:]
            base_action = root['/base_action'][:]
            action = np.concatenate([action, base_action], axis=1)
            camera_names = list(root[f'/observations/images/'].keys())

        # Visual feature
        all_cam_feature = []
        for cam_name in camera_names:
            feature_dataset_path = os.path.join(dataset_dir, f'{repr_type}_features_seed{seed}_episode_{episode_id}.hdf5')
            with h5py.File(feature_dataset_path, 'r') as root:
                cam_feature = root[f'/features/{cam_name}'][:]
                all_cam_feature.append(cam_feature)
        vis_fea = np.concatenate(all_cam_feature, axis=1)

        ## State feature
        dataset_path = os.path.join(dataset_dir, f'episode_{episode_id}.hdf5')
        with h5py.File(dataset_path, 'r') as root:
            s_fea = root['/observations/qpos'][:]

        # stack actions together
        eps_len = len(action)
        indices = np.tile(np.arange(skip), eps_len).reshape(eps_len, skip) # each row is 0, 1, ... skip
        offset = np.expand_dims(np.arange(eps_len), axis=1)
        indices = indices + offset # row1: 0, 1, ... skip; row2: 1, 2, ... skip+1
        # indices will exceed eps_len, thus clamp to eps_len-1
        indices = np.clip(indices, 0, eps_len-1)
        # stack action
        action = action[indices] # new shape: eps_len, skip, a_dim

        vis_features.append(vis_fea)
        state_features.append(s_fea)
        Y.append(action)

    vis_features = np.concatenate(vis_features)
    state_features  = np.concatenate(state_features)
    Y = np.concatenate(Y)
    train_inputs = [torch.from_numpy(vis_features).cuda(), torch.from_numpy(state_features).cuda()]
    train_targets = torch.from_numpy(Y).cuda()

    set_seed(1000)
    feature_extractors = {}
    for cam_name in camera_names:
        resnet = torchvision.models.resnet18(pretrained=True)
        loading_status = resnet.load_state_dict(torch.load(model_dir.replace('DUMMY', cam_name)))
        print(cam_name, loading_status)
        resnet = nn.Sequential(*list(resnet.children())[:-1])
        resnet = resnet.cuda()
        resnet.eval()
        feature_extractors[cam_name] = resnet



    # load environment
    if real_robot:
        from aloha_scripts.real_env import make_real_env #### TODO TODO
        env = make_real_env(init_node=True, setup_robots=True, setup_base=True)
        max_timesteps = sim_episode_len
        camera_names = ['cam_high', 'cam_left_wrist', 'cam_right_wrist']
    else:
        from sim_env import make_sim_env
        env = make_sim_env(task_name)
        max_timesteps = sim_episode_len


    num_rollouts = 50
    episode_returns = []
    max_rewards = []
    for rollout_id in range(num_rollouts):
        ### set task
        BOX_POSE[0] = sample_box_pose() # used in sim reset
        ts = env.reset()

        ### evaluation loop
        qpos_history = torch.zeros((1, max_timesteps, state_dim)).cuda()
        image_list = [] # for visualization
        qpos_list = []
        target_qpos_list = []
        rewards = []
        with torch.inference_mode():
            for t in range(sim_episode_len):
                start_time = time.time()
                if t % 100 == 0: print(t)
                if t % query_freq == 0:
                    ### process previous timestep to get qpos and image_list
                    obs = ts.observation
                    if 'images' in obs:
                        image_list.append(obs['images'])
                    else:
                        image_list.append({'main': obs['image']})
                    qpos_numpy = np.array(obs['qpos'])
                    # qpos = pre_process(qpos_numpy)
                    qpos = torch.from_numpy(qpos_numpy).float().cuda().unsqueeze(0)
                    qpos_history[:, t] = qpos
                    _, curr_image_raw = get_image(ts, camera_names)

                    image_size = 120
                    transform = transforms.Compose([
                        transforms.Resize(image_size),  # will scale the image
                        transforms.CenterCrop(image_size),
                        transforms.ToTensor(),
                        transforms.Lambda(expand_greyscale),
                        transforms.Normalize(
                            mean=torch.tensor([0.485, 0.456, 0.406]),
                            std=torch.tensor([0.229, 0.224, 0.225])),
                    ])

                    all_cam_features = []
                    for cam_id, curr_image in enumerate(curr_image_raw):
                        curr_image = Image.fromarray(curr_image) # TODO only one camera
                        curr_image = transform(curr_image)
                        curr_image = curr_image.unsqueeze(dim=0).cuda()
                        curr_image_feature = feature_extractors[camera_names[cam_id]](curr_image)
                        curr_image_feature = curr_image_feature.squeeze(3).squeeze(2)
                        all_cam_features.append(curr_image_feature)
                    curr_image_feature = torch.cat(all_cam_features, dim=1)

                    ### Visual feature
                    # curr_feature = curr_image_feature

                    ### State feature
                    # curr_feature = qpos

                    ### Both features
                    curr_feature = [curr_image_feature, qpos]

                    action = calculate_nearest_neighbors(curr_feature, train_inputs, train_targets, k, state_weight) # TODO use this
                    action = action.squeeze(0).cpu().numpy()
                    action = np.concatenate([action[:-BASE_DELAY, :-2], action[BASE_DELAY:, -2:]], axis=1)
                    print(f'Query: {(time.time() - start_time):.3f}s')

                curr_action = action[t % query_freq]
                target_qpos = curr_action[:-2]
                base_action = curr_action[-2:]

                # ### SAFETY
                # max_a = 0.05
                # curr_qpos = qpos.squeeze().cpu().numpy()
                # target_qpos = target_qpos.clip(curr_qpos - max_a, curr_qpos + max_a)
                # ### SAFETY

                ### step the environment
                ts = env.step(target_qpos, base_action=base_action)
                duration = time.time() - start_time
                # print(f'{duration:.3f}')
                time.sleep(max(0, DT - duration))

                ### save things for visualization
                qpos_list.append(qpos_numpy)
                target_qpos_list.append(target_qpos)
                rewards.append(ts.reward)

                # if real_robot and t != 0 and t % 60 == 0:
                #    e()
            plt.close()
        if real_robot:
            env.puppet_bot_left.dxl.robot_set_operating_modes("single", "gripper", "position")
            env.puppet_bot_right.dxl.robot_set_operating_modes("single", "gripper", "position")
            env.puppet_bot_left.dxl.robot_set_operating_modes("single", "gripper", "pwm")
            env.puppet_bot_right.dxl.robot_set_operating_modes("single", "gripper", "pwm")

        rewards = np.array(rewards)
        episode_return = np.sum(rewards[rewards!=None])
        episode_returns.append(episode_return)
        max_reward = np.max(rewards)
        max_rewards.append(max_reward)

        print(f'{episode_return=}, {max_reward=}')
        if save_episode:
            save_videos(image_list, DT, video_path=os.path.join(ckpt_dir, f'video{rollout_id}.mp4'))
            # visualize_joints(qpos_list, target_qpos_list, plot_path=os.path.join(ckpt_dir, f'qpos{rollout_id}.png'))
            # visualize_joints(qpos_list, example_qpos, plot_path=os.path.join(ckpt_dir, f'qpos_reference{rollout_id}.png'), label_overwrite=("policy", "dataset"))

    success_rate = np.mean(np.array(max_rewards) == env_max_reward)
    avg_return = np.mean(episode_returns)
    summary_str = f'\nSuccess rate: {success_rate}\nAverage return: {avg_return}\n\n'
    for r in range(env_max_reward+1):
        more_or_equal_r = (np.array(max_rewards) >= r).sum()
        more_or_equal_r_rate = more_or_equal_r / num_rollouts
        summary_str += f'Reward >= {r}: {more_or_equal_r}/{num_rollouts} = {more_or_equal_r_rate*100}%\n'

    print(summary_str)

    # save success rate to txt
    result_file_name = f'result_{skip}_{k}' + '.txt'
    with open(os.path.join(ckpt_dir, result_file_name), 'w') as f:
        f.write(summary_str)
        f.write(repr(episode_returns))
        f.write('\n\n')
        f.write(repr(max_rewards))

    return success_rate, avg_return



def get_image(ts, camera_names):
    if 'images' in ts.observation:
        curr_images = []
        for cam_name in camera_names:
            curr_image = rearrange(ts.observation['images'][cam_name], 'h w c -> c h w')
            curr_images.append(curr_image)
        curr_image_raw = np.stack(curr_images, axis=0)
    else:
        curr_image_raw = rearrange(ts.observation['image'], 'h w c -> c h w')
    curr_image = torch.from_numpy(curr_image_raw / 255.0).float().cuda().unsqueeze(0)
    curr_image_raw = rearrange(curr_image_raw, 'b c h w -> b h w c')
    return curr_image, curr_image_raw


def expand_greyscale(t):
    return t.expand(3, -1, -1)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--onscreen_render', action='store_true')
    parser.add_argument('--dataset_dir', action='store', type=str, help='The text to parse.', required=True)
    parser.add_argument('--model_dir', action='store', type=str, help='model_dir', required=True)
    parser.add_argument('--task_name', action='store', type=str, help='task_name', required=True)
    parser.add_argument('--ckpt_dir', action='store', type=str, help='The text to parse.', required=True)
    main(vars(parser.parse_args()))
