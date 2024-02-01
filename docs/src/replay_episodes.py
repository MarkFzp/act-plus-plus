import os
import h5py
import argparse
from collections import defaultdict 
from sim_env import make_sim_env
from utils import sample_box_pose, sample_insertion_pose
from sim_env import BOX_POSE
from constants import DT
from visualize_episodes import save_videos

import IPython
e = IPython.embed


def main(args):
    dataset_path = args['dataset_path']


    if not os.path.isfile(dataset_path):
        print(f'Dataset does not exist at \n{dataset_path}\n')
        exit()

    with h5py.File(dataset_path, 'r') as root:
        actions = root['/action'][()]

    env = make_sim_env('sim_transfer_cube')
    BOX_POSE[0] = sample_box_pose() # used in sim reset
    ts = env.reset()
    episode_replay = [ts]
    for action in actions:
        ts = env.step(action)
        episode_replay.append(ts)

    # saving
    image_dict = defaultdict(lambda: [])
    while episode_replay:
        ts = episode_replay.pop(0)
        for cam_name, image in ts.observation['images'].items():
            image_dict[cam_name].append(image)

    video_path = dataset_path.replace('episode_', 'replay_episode_').replace('hdf5', 'mp4')
    save_videos(image_dict, DT, video_path=video_path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_path', action='store', type=str, help='Dataset path.', required=True)
    main(vars(parser.parse_args()))
