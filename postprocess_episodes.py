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

MIRROR_STATE_MULTIPLY = np.array([-1, 1, 1, -1, 1, -1, 1]).astype('float32')
MIRROR_BASE_MULTIPLY = np.array([1, -1]).astype('float32')

def load_hdf5(dataset_dir, dataset_name):
    dataset_path = os.path.join(dataset_dir, dataset_name + '.hdf5')
    if not os.path.isfile(dataset_path):
        print(f'Dataset does not exist at \n{dataset_path}\n')
        exit()

    with h5py.File(dataset_path, 'r') as root:
        is_sim = root.attrs['sim']
        compressed = root.attrs.get('compress', False)
        qpos = root['/observations/qpos'][()]
        qvel = root['/observations/qvel'][()]
        action = root['/action'][()]
        image_dict = dict()
        for cam_name in root[f'/observations/images/'].keys():
            image_dict[cam_name] = root[f'/observations/images/{cam_name}'][()]
        if 'base_action' in root.keys():
            print('base_action exists')
            base_action = root['/base_action'][()]
        else:
            base_action = None
        if compressed:
            compress_len = root['/compress_len'][()]

    if compressed:
        for cam_id, cam_name in enumerate(image_dict.keys()):
            # un-pad and uncompress
            padded_compressed_image_list = image_dict[cam_name]
            image_list = []
            for padded_compressed_image in padded_compressed_image_list: # [:1000] to save memory
                image = cv2.imdecode(padded_compressed_image, 1)
                image_list.append(image)
            image_dict[cam_name] = np.array(image_list)

    return qpos, qvel, action, base_action, image_dict, is_sim

def main(args):
    dataset_dir = args['dataset_dir']
    num_episodes = args['num_episodes']

    start_idx = 0
    for episode_idx in range(start_idx, start_idx + num_episodes):
        dataset_name = f'episode_{episode_idx}'

        qpos, qvel, action, base_action, image_dict, is_sim = load_hdf5(dataset_dir, dataset_name)

        # process proprioception
        qpos = np.concatenate([qpos[:, 7:] * MIRROR_STATE_MULTIPLY, qpos[:, :7] * MIRROR_STATE_MULTIPLY], axis=1)
        qvel = np.concatenate([qvel[:, 7:] * MIRROR_STATE_MULTIPLY, qvel[:, :7] * MIRROR_STATE_MULTIPLY], axis=1)
        action = np.concatenate([action[:, 7:] * MIRROR_STATE_MULTIPLY, action[:, :7] * MIRROR_STATE_MULTIPLY], axis=1)
        if base_action is not None:
            base_action = base_action * MIRROR_BASE_MULTIPLY

        # mirror image obs
        if 'left_wrist' in image_dict.keys():
            image_dict['left_wrist'], image_dict['right_wrist'] = image_dict['right_wrist'][:, :, ::-1], image_dict['left_wrist'][:, :, ::-1]
        elif 'cam_left_wrist' in image_dict.keys():
            image_dict['cam_left_wrist'], image_dict['cam_right_wrist'] = image_dict['cam_right_wrist'][:, :, ::-1], image_dict['cam_left_wrist'][:, :, ::-1]
        else:
            raise Exception('No left_wrist or cam_left_wrist in image_dict')

        if 'top' in image_dict.keys():
            image_dict['top'] = image_dict['top'][:, :, ::-1]
        elif 'cam_high' in image_dict.keys():
            image_dict['cam_high'] = image_dict['cam_high'][:, :, ::-1]
        else:
            raise Exception('No top or cam_high in image_dict')

        # saving
        data_dict = {
            '/observations/qpos': qpos,
            '/observations/qvel': qvel,
            '/action': action,
            '/base_action': base_action,
        } if base_action is not None else {
            '/observations/qpos': qpos,
            '/observations/qvel': qvel,
            '/action': action,
        }
        for cam_name in image_dict.keys():
            data_dict[f'/observations/images/{cam_name}'] = image_dict[cam_name]
        max_timesteps = len(qpos)

        COMPRESS = True

        if COMPRESS:
            # JPEG compression
            t0 = time.time()
            encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 50] # tried as low as 20, seems fine
            compressed_len = []
            for cam_name in image_dict.keys():
                image_list = data_dict[f'/observations/images/{cam_name}']
                compressed_list = []
                compressed_len.append([])
                for image in image_list:
                    result, encoded_image = cv2.imencode('.jpg', image, encode_param) # 0.02 sec # cv2.imdecode(encoded_image, 1)
                    compressed_list.append(encoded_image)
                    compressed_len[-1].append(len(encoded_image))
                data_dict[f'/observations/images/{cam_name}'] = compressed_list
            print(f'compression: {time.time() - t0:.2f}s')

            # pad so it has same length
            t0 = time.time()
            compressed_len = np.array(compressed_len)
            padded_size = compressed_len.max()
            for cam_name in image_dict.keys():
                compressed_image_list = data_dict[f'/observations/images/{cam_name}']
                padded_compressed_image_list = []
                for compressed_image in compressed_image_list:
                    padded_compressed_image = np.zeros(padded_size, dtype='uint8')
                    image_len = len(compressed_image)
                    padded_compressed_image[:image_len] = compressed_image
                    padded_compressed_image_list.append(padded_compressed_image)
                data_dict[f'/observations/images/{cam_name}'] = padded_compressed_image_list
            print(f'padding: {time.time() - t0:.2f}s')

        # HDF5
        t0 = time.time()
        dataset_path = os.path.join(dataset_dir, f'mirror_episode_{episode_idx}')
        with h5py.File(dataset_path + '.hdf5', 'w', rdcc_nbytes=1024 ** 2 * 2) as root:
            root.attrs['sim'] = is_sim
            root.attrs['compress'] = COMPRESS
            obs = root.create_group('observations')
            image = obs.create_group('images')
            for cam_name in image_dict.keys():
                if COMPRESS:
                    _ = image.create_dataset(cam_name, (max_timesteps, padded_size), dtype='uint8',
                                            chunks=(1, padded_size), )
                else:
                    _ = image.create_dataset(cam_name, (max_timesteps, 480, 640, 3), dtype='uint8',
                                            chunks=(1, 480, 640, 3), )
            qpos = obs.create_dataset('qpos', (max_timesteps, 14))
            qvel = obs.create_dataset('qvel', (max_timesteps, 14))
            action = root.create_dataset('action', (max_timesteps, 14))
            if base_action is not None:
                base_action = root.create_dataset('base_action', (max_timesteps, 2))

            for name, array in data_dict.items():
                root[name][...] = array
            
            if COMPRESS:
                _ = root.create_dataset('compress_len', (len(image_dict.keys()), max_timesteps))
                root['/compress_len'][...] = compressed_len

        print(f'Saving {dataset_path}: {time.time() - t0:.1f} secs\n')

        if episode_idx == start_idx:
            save_videos(image_dict, DT, video_path=os.path.join(dataset_dir, dataset_name + '_mirror_video.mp4'))
            # visualize_joints(qpos, action, plot_path=os.path.join(dataset_dir, dataset_name + '_mirror_qpos.png'))
            # visualize_timestamp(t_list, dataset_path) # TODO addn timestamp back

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_dir', action='store', type=str, help='Dataset dir.', required=True)
    parser.add_argument('--num_episodes', action='store', type=int, help='Number of episodes.', required=True)
    main(vars(parser.parse_args()))
