import torch
import argparse
import pathlib
from torch import nn
import torchvision
import os
import time
import h5py
import h5py
from torchvision import models, transforms
from PIL import Image
from tqdm import tqdm
import cv2
import numpy as np

import IPython
e = IPython.embed


def chunks(lst, n):
    """Yield successive n-sized chunks from lst."""
    for i in range(0, len(lst), n):
        yield lst[i:i + n]

def expand_greyscale(t):
    return t.expand(3, -1, -1)


def main(args):
    #################################################
    batch_size = 256
    #################################################

    ckpt_path = args.ckpt_path
    dataset_dir = args.dataset_dir
    ckpt_name = pathlib.PurePath(ckpt_path).name
    dataset_name = ckpt_name.split('-')[1]
    repr_type = ckpt_name.split('-')[0]
    seed = int(ckpt_name.split('-')[-1][:-3])

    if 'cotrain' in ckpt_name:
        repr_type += '_cotrain'

    episode_idxs = [int(name.split('_')[1].split('.')[0]) for name in os.listdir(dataset_dir) if ('.hdf5' in name) and ('features' not in name)]
    episode_idxs.sort()
    assert len(episode_idxs) == episode_idxs[-1] + 1 # no holes
    num_episodes = len(episode_idxs)

    feature_extractors = {}

    for episode_idx in range(num_episodes):

        # load all images
        print(f'loading data')
        dataset_path = os.path.join(dataset_dir, f'episode_{episode_idx}.hdf5')
        with h5py.File(dataset_path, 'r') as root:
            image_dict = {}
            camera_names = list(root[f'/observations/images/'].keys())
            print(f'Camera names: {camera_names}')
            for cam_name in camera_names:
                image = root[f'/observations/images/{cam_name}'][:]
                uncompressed_image = []
                for im in image:
                    im = np.array(cv2.imdecode(im, 1))
                    uncompressed_image.append(im)
                image = np.stack(uncompressed_image, axis=0)

                image_dict[cam_name] = image

        print(f'loading model')
        # load pretrain nets after cam names are known
        if not feature_extractors:
            for cam_name in camera_names:
                resnet = torchvision.models.resnet18(pretrained=True)
                loading_status = resnet.load_state_dict(torch.load(ckpt_path.replace('DUMMY', cam_name)))
                print(cam_name, loading_status)
                resnet = nn.Sequential(*list(resnet.children())[:-1])
                resnet = resnet.cuda()
                resnet.eval()
                feature_extractors[cam_name] = resnet

        # inference with resnet
        feature_dict = {}
        for cam_name, images in image_dict.items():
            # Preprocess images
            image_size = 120 # TODO NOTICE: reduced resolution
            transform = transforms.Compose([
                transforms.Resize(image_size),  # will scale the image
                transforms.CenterCrop(image_size),
                transforms.ToTensor(),
                transforms.Lambda(expand_greyscale),
                transforms.Normalize(
                    mean=torch.tensor([0.485, 0.456, 0.406]),
                    std=torch.tensor([0.229, 0.224, 0.225])),
            ])
            processed_images = []
            for image in tqdm(images):
                image = Image.fromarray(image)
                image = transform(image)
                processed_images.append(image)
            processed_images = torch.stack(processed_images).cuda()

            # query the model
            all_features = []
            with torch.inference_mode():
                for batch in chunks(processed_images, batch_size):
                    print('inference')
                    features = feature_extractors[cam_name](batch)
                    features = features.squeeze(axis=3).squeeze(axis=2)
                    all_features.append(features)
            all_features = torch.cat(all_features, axis=0)
            max_timesteps = all_features.shape[0]
            feature_dict[cam_name] = all_features

            # TODO START diagnostics
            # first_image = images[0]
            # first_processed_image = processed_images[0].cpu().numpy()
            # first_feature = all_features[0].cpu().numpy()
            # import numpy as np
            # np.save('first_image.npy', first_image)
            # np.save('first_processed_image.npy', first_processed_image)
            # np.save('first_feature.npy', first_feature)
            # torch.save(resnet.state_dict(), 'rn.ckpt')
            # e()
            # exit()
            # TODO END diagnostics


        # save
        dataset_path = os.path.join(dataset_dir, f'{repr_type}_features_seed{seed}_episode_{episode_idx}.hdf5')
        print(dataset_path)
        # HDF5
        t0 = time.time()
        with h5py.File(dataset_path, 'w', rdcc_nbytes=1024 ** 2 * 2) as root:
            features = root.create_group('features')
            for cam_name, array in feature_dict.items():
                cam_feature = features.create_dataset(cam_name, (max_timesteps, 512))
                features[cam_name][...] = array.cpu().numpy()
        print(f'Saving: {time.time() - t0:.1f} secs\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='cache features')
    parser.add_argument('--ckpt_path', type=str, required=True, help='ckpt_path')
    parser.add_argument('--dataset_dir', type=str, required=True, help='dataset_dir')
    args = parser.parse_args()

    main(args)