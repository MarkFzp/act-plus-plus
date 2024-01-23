import torch
import torch.nn.functional as F
import numpy as np
import h5py
import pathlib
import os
import argparse
import matplotlib.pyplot as plt

import IPython
e = IPython.embed

# modified from https://github.com/jyopari/VINN/blob/main/nearest-neighbor-eval/handle_nn.ipynb

def calculate_nearest_neighbors(query_inputs, query_targets, support_inputs, support_targets, max_k):
    with torch.no_grad():
        pairwise_dist = []
        for q_in in query_inputs:
            diff = support_inputs - q_in.unsqueeze(0)
            dist = torch.norm(diff, dim=1)
            pairwise_dist.append(dist)
        pairwise_dist = torch.stack(pairwise_dist)

        sorted_dist, index = torch.sort(pairwise_dist, dim=1) # sort the support axis
        permuted_support_targets = support_targets[index]
        errors = []
        for k in range(1, max_k):
            topk_dist = pairwise_dist[:, :k]
            topk_support_targets = permuted_support_targets[:, :k]
            weights = F.softmax(-topk_dist, dim=1)
            weighted_support_targets = weights.unsqueeze(2) * topk_support_targets
            prediction = torch.sum(weighted_support_targets, dim=1)
            error = F.mse_loss(prediction, query_targets)
            errors.append(error)
        return errors

def chunks(lst, n):
    """Yield successive n-sized chunks from lst."""
    for i in range(0, len(lst), n):
        yield lst[i:i + n]

def main(args):
    # TODO ######################
    dataset_dir = args['dataset_dir']
    ckpt_dir = args['ckpt_dir']
    seed = 0
    max_k = 400
    batch_size = 100
    # TODO ######################

    repr_type = 'byol'
    if 'cotrain' in ckpt_dir:
        repr_type += '_cotrain'
    e() # make sure!

    if not os.path.isdir(ckpt_dir):
        os.makedirs(ckpt_dir)
    
    episode_idxs = [int(name.split('_')[1].split('.')[0]) for name in os.listdir(dataset_dir) if ('.hdf5' in name) and ('features' not in name)]
    episode_idxs.sort()
    assert len(episode_idxs) == episode_idxs[-1] + 1 # no holes
    num_episodes = len(episode_idxs)
    val_split = int(num_episodes * 0.8)

    # load train data
    X = []
    Y = []
    for episode_id in range(0, val_split):
        dataset_path = os.path.join(dataset_dir, f'episode_{episode_id}.hdf5')
        with h5py.File(dataset_path, 'r') as root:
            action = root['/action'][:]
            camera_names = list(root[f'/observations/images/'].keys())
        
        all_cam_feature = []
        feature_dataset_path = os.path.join(dataset_dir, f'{repr_type}_features_seed{seed}_episode_{episode_id}.hdf5')
        with h5py.File(feature_dataset_path, 'r') as root:
            for cam_name in camera_names:
                cam_feature = root[f'/features/{cam_name}'][:]
                all_cam_feature.append(cam_feature)
        cam_feature = np.concatenate(all_cam_feature, axis=1)

        X.append(cam_feature)
        Y.append(action)

    X = np.concatenate(X)
    Y = np.concatenate(Y)
    train_inputs = torch.from_numpy(X).cuda()
    train_targets = torch.from_numpy(Y).cuda()
    print(f'All features: {train_inputs.shape}')

    # load test data
    X = []
    Y = []
    for episode_id in range(val_split, num_episodes):
        dataset_path = os.path.join(dataset_dir, f'episode_{episode_id}.hdf5')
        with h5py.File(dataset_path, 'r') as root:
            action = root['/action'][:]

        all_cam_feature = []
        feature_dataset_path = os.path.join(dataset_dir, f'{repr_type}_features_seed{seed}_episode_{episode_id}.hdf5')
        with h5py.File(feature_dataset_path, 'r') as root:
            for cam_name in camera_names:
                cam_feature = root[f'/features/{cam_name}'][:]
                all_cam_feature.append(cam_feature)
        cam_feature = np.concatenate(all_cam_feature, axis=1)

        X.append(cam_feature)
        Y.append(action)

    X = np.concatenate(X)
    Y = np.concatenate(Y)
    val_inputs = torch.from_numpy(X).cuda()
    val_targets = torch.from_numpy(Y).cuda()

    val_losses = []
    for inputs, targets in zip(chunks(val_inputs, batch_size), chunks(val_targets, batch_size)):
        val_loss = calculate_nearest_neighbors(inputs, targets, train_inputs, train_targets, max_k)
        val_loss = torch.stack(val_loss)
        val_losses.append(val_loss)
    val_losses = torch.mean(torch.stack(val_losses), dim=0)
    val_loss = val_losses

    val_loss = torch.tensor(val_loss).cpu().numpy()
    print(f'min val loss of {np.min(val_loss)} at k={np.argmin(val_loss)}')

    plt.plot(np.arange(1, max_k), val_loss)
    plt.savefig(os.path.join(ckpt_dir, f'k_select-seed{seed}.png'))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_dir', action='store', type=str, help='The text to parse.', required=True)
    parser.add_argument('--ckpt_dir', action='store', type=str, help='The text to parse.', required=True)
    main(vars(parser.parse_args()))