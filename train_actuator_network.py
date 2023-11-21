
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader
import os
import h5py
import math
import wandb
import pickle
import matplotlib.pyplot as plt
from copy import deepcopy
from tqdm import tqdm
from utils import find_all_hdf5
from imitate_episodes import repeater, compute_dict_mean

import IPython
e = IPython.embed

def main():
    ### Idea
    # input : o o o o o o # observed speed 
    # target: a a a a a a # commanded speed
    # at test time, input desired speed profile and convert that to command

    #########################################################
    history_len = 50
    future_len = 50
    prediction_len = 50
    batch_size_train = 16
    batch_size_val  = 16
    lr = 1e-4
    weight_decay = 1e-4

    num_steps = 10000
    validate_every = 2000
    save_every = 2000

    expr_name = f'actuator_network_test_{history_len}_{future_len}_{prediction_len}'
    ckpt_dir = f'/scr/tonyzhao/train_logs/{expr_name}' if os.getlogin() == 'tonyzhao' else f'./ckpts/{expr_name}'
    dataset_dir = '/scr/tonyzhao/compressed_datasets/aloha_mobile_fork/' if os.getlogin() == 'tonyzhao' else '/home/zfu/data/aloha_mobile_fork/'
    #########################################################
    assert(history_len + future_len >= prediction_len)
    assert(future_len % prediction_len == 0)

    wandb.init(project="mobile-aloha2", reinit=True, entity="mobile-aloha2", name=expr_name) # mode='disabled', 

    if not os.path.isdir(ckpt_dir):
        os.makedirs(ckpt_dir)

    dataset_path_list = find_all_hdf5(dataset_dir, skip_mirrored_data=True)
    dataset_path_list = [n for n in dataset_path_list if 'replayed' in n]
    num_episodes = len(dataset_path_list)

    # obtain train test split
    train_ratio = 0.9
    shuffled_episode_ids = np.random.permutation(num_episodes)
    train_episode_ids = shuffled_episode_ids[:int(train_ratio * num_episodes)]
    val_episode_ids = shuffled_episode_ids[int(train_ratio * num_episodes):]
    print(f'\n\nData from: {dataset_dir}\n- Train on {len(train_episode_ids)} episodes\n- Test on {len(val_episode_ids)} episodes\n\n')

    # obtain normalization stats for qpos and action
    # if load_pretrain:
    #     with open(os.path.join('/home/zfu/interbotix_ws/src/act/ckpts/pretrain_all', 'dataset_stats.pkl'), 'rb') as f:
    #         norm_stats = pickle.load(f)
    #     print('Loaded pretrain dataset stats')
    norm_stats, all_episode_len = get_norm_stats(dataset_path_list)
    train_episode_len = [all_episode_len[i] for i in train_episode_ids]
    val_episode_len = [all_episode_len[i] for i in val_episode_ids]
    assert(all_episode_len[0] % prediction_len == 0)

    # save dataset stats
    stats_path = os.path.join(ckpt_dir, f'actuator_net_stats.pkl')
    with open(stats_path, 'wb') as f:
        pickle.dump(norm_stats, f)

    # construct dataset and dataloader
    train_dataset = EpisodicDataset(dataset_path_list, norm_stats, train_episode_ids, train_episode_len, history_len, future_len, prediction_len)
    val_dataset = EpisodicDataset(dataset_path_list, norm_stats, val_episode_ids, val_episode_len, history_len, future_len, prediction_len)
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size_train, shuffle=True, pin_memory=True, num_workers=1, prefetch_factor=1)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size_val, shuffle=True, pin_memory=True, num_workers=1, prefetch_factor=1)
    
    policy = ActuatorNetwork(prediction_len).cuda()
    optimizer = torch.optim.AdamW(policy.parameters(), lr=lr, weight_decay=weight_decay)

    n_parameters = sum(p.numel() for p in policy.parameters() if p.requires_grad)
    print("number of parameters: %.2fM" % (n_parameters/1e6,))

    min_val_loss = np.inf
    best_ckpt_info = None
    train_dataloader = repeater(train_dataloader)
    for step in tqdm(range(num_steps+1)):
        # validation
        if step % validate_every == 0:
            print('validating')

            with torch.inference_mode():
                policy.eval()
                validation_dicts = []
                for batch_idx, data in enumerate(val_dataloader):
                    observed_speed, commanded_speed = data
                    out, forward_dict = policy(observed_speed.cuda(), commanded_speed.cuda())
                    validation_dicts.append(forward_dict)

                validation_summary = compute_dict_mean(validation_dicts)

                epoch_val_loss = validation_summary['loss']
                if epoch_val_loss < min_val_loss:
                    min_val_loss = epoch_val_loss
                    best_ckpt_info = (step, min_val_loss, deepcopy(policy.state_dict()))
            for k in list(validation_summary.keys()):
                validation_summary[f'val_{k}'] = validation_summary.pop(k)            
            wandb.log(validation_summary, step=step)
            print(f'Val loss:   {epoch_val_loss:.5f}')
            summary_string = ''
            for k, v in validation_summary.items():
                summary_string += f'{k}: {v.item():.3f} '
            print(summary_string)

            visualize_prediction(dataset_path_list, val_episode_ids, policy, norm_stats, history_len, future_len, prediction_len, ckpt_dir, step, 'val')
            visualize_prediction(dataset_path_list, train_episode_ids, policy, norm_stats, history_len, future_len, prediction_len, ckpt_dir, step, 'train')


        # training
        policy.train()
        optimizer.zero_grad()
        data = next(train_dataloader)
        observed_speed, commanded_speed = data
        out, forward_dict = policy(observed_speed.cuda(), commanded_speed.cuda())
        # backward
        loss = forward_dict['loss']
        loss.backward()
        optimizer.step()
        wandb.log(forward_dict, step=step) # not great, make training 1-2% slower

        if step % save_every == 0:
            ckpt_path = os.path.join(ckpt_dir, f'actuator_net_step_{step}.ckpt')
            torch.save(policy.state_dict(), ckpt_path)

    ckpt_path = os.path.join(ckpt_dir, f'actuator_net_last.ckpt')
    torch.save(policy.state_dict(), ckpt_path)

    best_step, min_val_loss, best_state_dict = best_ckpt_info
    ckpt_path = os.path.join(ckpt_dir, f'actuator_net_step_{best_step}.ckpt')
    torch.save(best_state_dict, ckpt_path)
    print(f'Training finished:\nval loss {min_val_loss:.6f} at step {best_step}')


def visualize_prediction(dataset_path_list, episode_ids, policy, norm_stats, history_len, future_len, prediction_len, ckpt_dir, step, name):
    num_vis = 2
    episode_ids = episode_ids[:num_vis]
    vis_path = [dataset_path_list[i] for i in episode_ids]

    for i, dataset_path in enumerate(vis_path):
        try:
            with h5py.File(dataset_path, 'r') as root:
                commanded_speed = root['/base_action'][()]
                observed_speed = root['/obs_tracer'][()]
        except Exception as ee:
            print(f'Error loading {dataset_path} in get_norm_stats')
            print(ee)
            quit()
        
        # commanded_speed = (commanded_speed - norm_stats["commanded_speed_mean"]) / norm_stats["commanded_speed_std"]
        norm_observed_speed = (observed_speed - norm_stats["observed_speed_mean"]) / norm_stats["observed_speed_std"]
        out_unnorm_fn = lambda x: (x * norm_stats["commanded_speed_std"]) + norm_stats["commanded_speed_mean"]

        history_pad = np.zeros((history_len, 2))
        future_pad = np.zeros((future_len, 2))
        norm_observed_speed = np.concatenate([history_pad, norm_observed_speed, future_pad], axis=0)

        episode_len = commanded_speed.shape[0]

        all_pred = []
        for t in range(0, episode_len, prediction_len):
            offset_start_ts = t + history_len
            policy_input = norm_observed_speed[offset_start_ts-history_len: offset_start_ts+future_len]
            policy_input = torch.from_numpy(policy_input).float().unsqueeze(dim=0).cuda()
            pred = policy(policy_input)
            pred = pred.detach().cpu().numpy()[0]
            all_pred += out_unnorm_fn(pred).tolist()
        all_pred = np.array(all_pred)

        plot_path = os.path.join(ckpt_dir, f'{name}{i}_step{step}_linear')
        plt.figure()
        plt.plot(commanded_speed[:, 0], label='commanded_speed_linear')
        plt.plot(observed_speed[:, 0], label='observed_speed_linear')
        plt.plot(all_pred[:, 0],  label='pred_commanded_speed_linear')
        # plot vertical grey dotted lines every prediction_len
        for t in range(0, episode_len, prediction_len):
            plt.axvline(t, linestyle='--', color='grey')
        plt.legend()
        plt.savefig(plot_path)
        plt.close()

        plot_path = os.path.join(ckpt_dir, f'{name}{i}_step{step}_angular')
        plt.figure()
        plt.plot(commanded_speed[:, 1], label='commanded_speed_angular')
        plt.plot(observed_speed[:, 1], label='observed_speed_angular')
        plt.plot(all_pred[:, 1], label='pred_commanded_speed_angular')
        # plot vertical dotted lines every prediction_len
        for t in range(0, episode_len, prediction_len):
            plt.axvline(t, linestyle='--', color='grey')
        plt.legend()
        plt.savefig(plot_path)
        plt.close()



class ActuatorNetwork(nn.Module):

    def __init__(self, prediction_len):
        super().__init__()
        d_model = 256
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=8)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=3)
        self.pe = PositionalEncoding(d_model)
        self.in_proj = nn.Linear(2, d_model)
        self.out_proj = nn.Linear(d_model, 2)
        self.prediction_len = prediction_len

    def forward(self, src, tgt=None):
        if tgt is not None: # training time
            # (batch, seq, feature) -> (seq, batch, feature)
            src = self.in_proj(src)
            src = torch.einsum('b s d -> s b d', src)
            src = self.pe(src)
            out = self.transformer(src)
            
            tgt = torch.einsum('b s d -> s b d', tgt)
            assert(self.prediction_len == tgt.shape[0])
            out = out[0: self.prediction_len] # take first few tokens only for prediction
            out = self.out_proj(out)

            l2_loss = loss = F.mse_loss(out, tgt)
            loss_dict = {'loss': l2_loss}
            out = torch.einsum('s b d -> b s d', out)
            return out, loss_dict
        else:
            src = self.in_proj(src)
            src = torch.einsum('b s d -> s b d', src)
            src = self.pe(src)
            out = self.transformer(src)
            out = out[0: self.prediction_len] # take first few tokens only for prediction
            out = self.out_proj(out)
            out = torch.einsum('s b d -> b s d', out)
            return out



class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x):
        """
        Arguments:
            x: Tensor, shape ``[seq_len, batch_size, embedding_dim]``
        """
        x = x + self.pe[:x.size(0)]
        return self.dropout(x)

def get_norm_stats(dataset_path_list):
    all_commanded_speed = []
    all_observed_speed = []
    all_episode_len = []
    for dataset_path in dataset_path_list:
        try:
            with h5py.File(dataset_path, 'r') as root:
                commanded_speed = root['/base_action'][()]
                observed_speed = root['/obs_tracer'][()]
        except Exception as e:
            print(f'Error loading {dataset_path} in get_norm_stats')
            print(e)
            quit()
        all_commanded_speed.append(torch.from_numpy(commanded_speed))
        all_observed_speed.append(torch.from_numpy(observed_speed))
        all_episode_len.append(len(commanded_speed))
    all_commanded_speed = torch.cat(all_commanded_speed, dim=0)
    all_observed_speed = torch.cat(all_observed_speed, dim=0)

    # normalize all_commanded_speed
    commanded_speed_mean = all_commanded_speed.mean(dim=[0]).float()
    commanded_speed_std = all_commanded_speed.std(dim=[0]).float()
    commanded_speed_std = torch.clip(commanded_speed_std, 1e-2, np.inf) # clipping

    # normalize all_observed_speed
    observed_speed_mean = all_observed_speed.mean(dim=[0]).float()
    observed_speed_std = all_observed_speed.std(dim=[0]).float()
    observed_speed_std = torch.clip(observed_speed_std, 1e-2, np.inf) # clipping

    stats = {"commanded_speed_mean": commanded_speed_mean.numpy(), "commanded_speed_std": commanded_speed_std.numpy(),
             "observed_speed_mean": observed_speed_mean.numpy(), "observed_speed_std": observed_speed_std.numpy()}

    return stats, all_episode_len


class EpisodicDataset(torch.utils.data.Dataset):
    def __init__(self, dataset_path_list, norm_stats, episode_ids, episode_len, history_len, future_len, prediction_len):
        super(EpisodicDataset).__init__()
        self.episode_ids = episode_ids
        self.dataset_path_list = dataset_path_list
        self.norm_stats = norm_stats
        self.episode_len = episode_len
        self.cumulative_len = np.cumsum(self.episode_len)
        self.max_episode_len = max(episode_len)
        self.history_len = history_len
        self.future_len = future_len
        self.prediction_len = prediction_len
        self.is_sim = False
        self.history_pad = np.zeros((self.history_len, 2))
        self.future_pad = np.zeros((self.future_len, 2))
        self.prediction_pad = np.zeros((self.prediction_len, 2))
        self.__getitem__(0) # initialize self.is_sim

    def __len__(self):
        return sum(self.episode_len)

    def _locate_transition(self, index):
        assert index < self.cumulative_len[-1]
        episode_index = np.argmax(self.cumulative_len > index) # argmax returns first True index
        start_ts = index - (self.cumulative_len[episode_index] - self.episode_len[episode_index])
        episode_id = self.episode_ids[episode_index]
        return episode_id, start_ts

    def __getitem__(self, index):
        episode_id, start_ts = self._locate_transition(index)
        dataset_path = self.dataset_path_list[episode_id]
        try:
            # print(dataset_path)
            with h5py.File(dataset_path, 'r') as root:
                commanded_speed = root['/base_action'][()]
                observed_speed = root['/obs_tracer'][()]
                observed_speed = np.concatenate([self.history_pad, observed_speed, self.future_pad], axis=0)
                commanded_speed = np.concatenate([commanded_speed, self.prediction_pad], axis=0)

                offset_start_ts = start_ts + self.history_len
                commanded_speed = commanded_speed[start_ts: start_ts+self.prediction_len]
                observed_speed = observed_speed[offset_start_ts-self.history_len: offset_start_ts+self.future_len]

            commanded_speed = torch.from_numpy(commanded_speed).float()
            observed_speed = torch.from_numpy(observed_speed).float()

            # normalize to mean 0 std 1
            commanded_speed = (commanded_speed - self.norm_stats["commanded_speed_mean"]) / self.norm_stats["commanded_speed_std"]
            observed_speed = (observed_speed - self.norm_stats["observed_speed_mean"]) / self.norm_stats["observed_speed_std"]

        except:
            print(f'Error loading {dataset_path} in __getitem__')
            quit()

        # print(image_data.dtype, qpos_data.dtype, action_data.dtype, is_pad.dtype)
        return observed_speed, commanded_speed




if __name__ == '__main__':
    main()
