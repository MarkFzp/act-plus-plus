"""
Example usage:
$ python3 script/compress_data.py --dataset_dir /scr/lucyshi/dataset/aloha_test
"""
import os
import h5py
import cv2
import numpy as np
import argparse
from tqdm import tqdm

# Constants
DT = 0.02
JOINT_NAMES = ["waist", "shoulder", "elbow", "forearm_roll", "wrist_angle", "wrist_rotate"]
STATE_NAMES = JOINT_NAMES + ["gripper"]


def compress_dataset(input_dataset_path, output_dataset_path):
    # Check if output path exists
    if os.path.exists(output_dataset_path):
        print(f"The file {output_dataset_path} already exists. Exiting...")
        return

    # Load the uncompressed dataset
    with h5py.File(input_dataset_path, 'r') as infile:
        # Create the compressed dataset
        with h5py.File(output_dataset_path, 'w') as outfile:

            outfile.attrs['sim'] = infile.attrs['sim']
            outfile.attrs['compress'] = True

            # Copy non-image data directly
            for key in infile.keys():
                if key != 'observations':
                    outfile.copy(infile[key], key)

            obs_group = infile['observations']

            # Create observation group in the output
            out_obs_group = outfile.create_group('observations')

            # Copy non-image data in observations directly
            for key in obs_group.keys():
                if key != 'images':
                    out_obs_group.copy(obs_group[key], key)

            image_group = obs_group['images']
            out_image_group = out_obs_group.create_group('images')

            # JPEG compression parameters
            encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 50]

            compressed_lens = []  # List to store compressed lengths for each camera

            for cam_name in image_group.keys():
                if "_depth" in cam_name:  # Depth images are not compressed
                    out_image_group.copy(image_group[cam_name], cam_name)
                else:
                    images = image_group[cam_name]
                    compressed_images = []
                    cam_compressed_lens = []  # List to store compressed lengths for this camera

                    # Compress each image
                    for image in images:
                        result, encoded_image = cv2.imencode('.jpg', image, encode_param)
                        compressed_images.append(encoded_image)
                        cam_compressed_lens.append(len(encoded_image))  # Store the length

                    compressed_lens.append(cam_compressed_lens)

                    # Find the maximum length of the compressed images
                    max_len = max(len(img) for img in compressed_images)

                    # Create dataset to store compressed images
                    compressed_dataset = out_image_group.create_dataset(cam_name, (len(compressed_images), max_len), dtype='uint8')

                    # Store compressed images
                    for i, img in enumerate(compressed_images):
                        compressed_dataset[i, :len(img)] = img

            # Save the compressed lengths to the HDF5 file
            compressed_lens = np.array(compressed_lens)
            _ = outfile.create_dataset('compress_len', compressed_lens.shape)
            outfile['/compress_len'][...] = compressed_lens

    print(f"Compressed dataset saved to {output_dataset_path}")


def save_videos(video, dt, video_path=None):
    if isinstance(video, list):
        cam_names = list(video[0].keys())
        h, w, _ = video[0][cam_names[0]].shape
        w = w * len(cam_names)
        fps = int(1/dt)
        out = cv2.VideoWriter(video_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
        # bitrate = 1000000
        # out.set(cv2.VIDEOWRITER_PROP_BITRATE, bitrate)
        for ts, image_dict in enumerate(video):
            images = []
            for cam_name in cam_names:
                image = image_dict[cam_name]
                image = image[:, :, [2, 1, 0]] # swap B and R channel
                images.append(image)
            images = np.concatenate(images, axis=1)
            out.write(images)
        out.release()
        print(f'Saved video to: {video_path}')
    elif isinstance(video, dict):
        cam_names = list(video.keys())
        # Remove depth images
        cam_names = [cam_name for cam_name in cam_names if '_depth' not in cam_name]
        all_cam_videos = []
        for cam_name in cam_names:
            all_cam_videos.append(video[cam_name])
        all_cam_videos = np.concatenate(all_cam_videos, axis=2) # width dimension

        n_frames, h, w, _ = all_cam_videos.shape
        fps = int(1 / dt)
        out = cv2.VideoWriter(video_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
        for t in range(n_frames):
            image = all_cam_videos[t]
            image = image[:, :, [2, 1, 0]]  # swap B and R channel
            out.write(image)
        out.release()
        print(f'Saved video to: {video_path}')

 
def load_and_save_first_episode_video(dataset_dir, video_path):
    dataset_name = 'episode_0'
    _, _, _, _, image_dict = load_hdf5(dataset_dir, dataset_name)
    save_videos(image_dict, DT, video_path=video_path)


def load_hdf5(dataset_dir, dataset_name):
    dataset_path = os.path.join(dataset_dir, dataset_name + '.hdf5')
    if not os.path.isfile(dataset_path):
        print(f'Dataset does not exist at \n{dataset_path}\n')
        exit()

    with h5py.File(dataset_path, 'r') as root:
        compressed = root.attrs.get('compress', False)
        image_dict = dict()
        for cam_name in root[f'/observations/images/'].keys():
            image_dict[cam_name] = root[f'/observations/images/{cam_name}'][()]
        if compressed:
            compress_len = root['/compress_len'][()]

    if compressed:
        for cam_id, cam_name in enumerate(image_dict.keys()):
            padded_compressed_image_list = image_dict[cam_name]
            image_list = []
            for frame_id, padded_compressed_image in enumerate(padded_compressed_image_list):
                image_len = int(compress_len[cam_id, frame_id])
                compressed_image = padded_compressed_image
                image = cv2.imdecode(compressed_image, 1)
                image_list.append(image)
            image_dict[cam_name] = image_list

    return None, None, None, None, image_dict  # Return only the image dict for this application


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Compress all HDF5 datasets in a directory.")
    parser.add_argument('--dataset_dir', action='store', type=str, required=True, help='Directory containing the uncompressed datasets.')

    args = parser.parse_args()

    output_dataset_dir = args.dataset_dir + '_compressed'
    os.makedirs(output_dataset_dir, exist_ok=True)

    # Iterate over each file in the directory
    for filename in tqdm(os.listdir(args.dataset_dir), desc="Compressing data"):
        if filename.endswith('.hdf5'):
            input_path = os.path.join(args.dataset_dir, filename)
            output_path = os.path.join(output_dataset_dir, filename)
            compress_dataset(input_path, output_path)

    # After processing all datasets, load and save the video for the first episode
    print(f'Saving video for episode 0 in {output_dataset_dir}')
    video_path = os.path.join(output_dataset_dir, 'episode_0_video.mp4')
    load_and_save_first_episode_video(output_dataset_dir, video_path)

