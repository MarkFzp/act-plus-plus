import os
import numpy as np
import cv2
import h5py
import argparse
import time
from visualize_episodes import visualize_joints, visualize_timestamp, save_videos

import matplotlib.pyplot as plt
from constants import DT

import IPython
e = IPython.embed

JOINT_NAMES = ["waist", "shoulder", "elbow", "forearm_roll", "wrist_angle", "wrist_rotate"]
STATE_NAMES = JOINT_NAMES + ["gripper"]

MIRROR_STATE_MULTIPLY = np.array([-1, 1, 1, -1, 1, -1, 1])[None, ...].astype('float32')


def load_hdf5(dataset_dir, dataset_name):
    dataset_path = os.path.join(dataset_dir, dataset_name + '.hdf5')
    if not os.path.isfile(dataset_path):
        print(f'Dataset does not exist at \n{dataset_path}\n')
        exit()

    with h5py.File(dataset_path, 'r') as root:
        is_sim = root.attrs['sim']
        qpos = root['/observations/qpos'][()]
        qvel = root['/observations/qvel'][()]
        action = root['/action'][()]
        image_dict = dict()
        for cam_name in root[f'/observations/images/'].keys():
            image_dict[cam_name] = root[f'/observations/images/{cam_name}'][()]

    return qpos, qvel, action, image_dict

def main(args):
    dataset_dir = args['dataset_dir']
    num_episodes = args['num_episodes']

    for episode_idx in range(num_episodes):
        dataset_name = f'episode_{episode_idx}'

        qpos, qvel, action, image_dict = load_hdf5(dataset_dir, dataset_name)

        # process proprioception
        qpos = np.concatenate([qpos[:, 7:] * MIRROR_STATE_MULTIPLY, qpos[:, :7] * MIRROR_STATE_MULTIPLY], axis=1)
        qvel = np.concatenate([qvel[:, 7:] * MIRROR_STATE_MULTIPLY, qvel[:, :7] * MIRROR_STATE_MULTIPLY], axis=1)
        action = np.concatenate([action[:, 7:] * MIRROR_STATE_MULTIPLY, action[:, :7] * MIRROR_STATE_MULTIPLY], axis=1)

        # mirror image obs
        image_dict['left_wrist'], image_dict['right_wrist'] = image_dict['right_wrist'][:, :, ::-1], image_dict['left_wrist'][:, :, ::-1]
        image_dict['top'] = image_dict['top'][:, :, ::-1]

        # saving
        data_dict = {
            '/observations/qpos': qpos,
            '/observations/qvel': qvel,
            '/action': action,
        }
        for cam_name in image_dict.keys():
            data_dict[f'/observations/images/{cam_name}'] = image_dict[cam_name]
        max_timesteps = len(qpos)

        # HDF5
        t0 = time.time()
        dataset_path = os.path.join(dataset_dir, f'mirror_episode_{episode_idx}')
        with h5py.File(dataset_path + '.hdf5', 'w', rdcc_nbytes=1024 ** 2 * 2) as root:
            root.attrs['sim'] = True
            obs = root.create_group('observations')
            image = obs.create_group('images')
            for cam_name in image_dict.keys():
                _ = image.create_dataset(cam_name, (max_timesteps, 480, 640, 3), dtype='uint8',
                                            chunks=(1, 480, 640, 3), )
            # compression='gzip',compression_opts=2,)
            # compression=32001, compression_opts=(0, 0, 0, 0, 9, 1, 1), shuffle=False)
            qpos = obs.create_dataset('qpos', (max_timesteps, 14))
            qvel = obs.create_dataset('qvel', (max_timesteps, 14))
            action = root.create_dataset('action', (max_timesteps, 14))

            for name, array in data_dict.items():
                root[name][...] = array
        print(f'Saving {dataset_path}: {time.time() - t0:.1f} secs\n')

        # save_videos(image_dict, DT, video_path=os.path.join(dataset_dir, dataset_name + '_mirror_video.mp4'))
        # visualize_joints(qpos, action, plot_path=os.path.join(dataset_dir, dataset_name + '_mirror_qpos.png'))
        # visualize_timestamp(t_list, dataset_path) # TODO addn timestamp back

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_dir', action='store', type=str, help='Dataset dir.', required=True)
    parser.add_argument('--num_episodes', action='store', type=int, help='Number of episodes.', required=True)
    main(vars(parser.parse_args()))
