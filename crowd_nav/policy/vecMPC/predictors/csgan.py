import os
from .sgan import SGAN
import numpy as np
import torch
from .utils.optimized_sgan import OptimizedCSGAN
from .utils.sgan_utils import get_generator, optimized_relative_to_abs
# import rospy

class CSGAN(SGAN):
    def __init__(self):
        super(CSGAN, self).__init__()
        self.model = OptimizedCSGAN

    def create_input(self, trajectory, actions): # actions: N x T''' x 2
        indexes = torch.flip(torch.arange(
            len(trajectory)-1, -1, -self.iskip, device=self.device)[:self.history_length+1], dims=(0,))
        trajectory = trajectory[indexes, :, :2] # (T'+1) x (1+H) x 2
        if len(trajectory) <= 1:
            trajectory = torch.nn.functional.pad(trajectory, (0, 0, 0, 0, 1, 0), mode='constant')
                                                                                                                # rospy.loginfo("Input: {}".format(obs_traj.shape, obs_traj_rel.shape))
        N = actions.shape[0]
        T = actions.shape[1]
        indexes = torch.arange(0, T, self.oskip)
        actions = actions[:, indexes, :2] # N x T' x 2
        ego_traj_full = trajectory[None, :, 0].repeat(actions.shape[0], 1, 1) # N x T' x 2
        ego_traj_full = torch.cat((ego_traj_full, actions), axis=1).permute((1, 0, 2)) # TF' x N x 2
        
        ego_traj = ego_traj_full[1:] # TF' x N x 2
        ego_traj_rel = ego_traj - ego_traj_full[:-1] # TF' x N x 2

        obs_traj_full = trajectory[:, 1:] # T' x (1+H) x 2
        
        T = obs_traj_full.shape[0]
        H = obs_traj_full.shape[1]
        obs_traj_full = obs_traj_full[:, None].repeat(1, N, 1, 1).view(T, N*H, 2)
        
        obs_traj = obs_traj_full[1:] # T' x H x 2
        obs_traj_rel = obs_traj - obs_traj_full[:-1] # T' x H x 2
    
        return obs_traj, obs_traj_rel, ego_traj, ego_traj_rel, H # T' x (1+H) x 2

    def get_predictions(self, trajectory, actions):
        with torch.no_grad():
            trajectory = torch.tensor(trajectory, device=self.device, dtype=torch.float)
            actions = torch.tensor(actions, device=self.device, dtype=torch.float)
            obs_traj, obs_traj_rel, ego_traj, ego_rel_traj, H = self.create_input(trajectory, actions)

            N = actions.shape[0]
            indexes = torch.arange(0, N*H+1, H, device=self.device, dtype=torch.int)
            # rospy.loginfo("{} {}".format(actions.shape, indexes.shape))
            seq_start_end = torch.stack((indexes[:-1], indexes[1:]), axis=1)

            # rospy.loginfo("{} {} {} {} {}".format(obs_traj.shape, obs_traj_rel.shape, seq_start_end.shape, ego_traj.shape, ego_rel_traj.shape))
            pred_traj_fake_rel = self.generator(
                obs_traj, obs_traj_rel, seq_start_end, ego_traj, ego_rel_traj, self.num_samples)
            pred_traj_fake = optimized_relative_to_abs(
                pred_traj_fake_rel, obs_traj[-1]) # S x T'' x N*H x 2
            
            S = pred_traj_fake.shape[0]
            T = pred_traj_fake.shape[1]
            pred_traj_fake = pred_traj_fake.reshape(S, T, N, -1, 2).permute(2, 0, 1, 3, 4) # N x S x T'' x H x 2
            # rospy.loginfo("from model: {}".format(pred_traj_fake.shape))
            pred_traj_fake = self.create_output(
                pred_traj_fake, trajectory[-1, 1:])  # N x S x T' x H x 2
            pred_traj_fake = torch.cat(
                (pred_traj_fake[:, :, :-1], self.predict_velocity(pred_traj_fake)), dim=-1)  # N x S x T' x H x 4
                                                                                                                    # rospy.loginfo("final_pred: {}\n\n".format(pred_traj_fake.shape))
        return pred_traj_fake.cpu().numpy()  # N x S x T' x H x 4
    
    def predictor_cost(self, state, actions, predictions):
        cost = np.array([0.0])
        return self.Q_dev * (cost**2)
    
    # def create_input(self, trajectory, actions): # actions: T''' x 2
    #     indexes = torch.flip(torch.arange(
    #         len(trajectory)-1, -1, -self.iskip, device=self.device)[:self.history_length+1], dims=(0,))
    #     trajectory = trajectory[indexes, :, :2] # (T'+1) x (1+H) x 2
    #     if len(trajectory) <= 1:
    #         trajectory = torch.nn.functional.pad(trajectory, (0, 0, 0, 0, 1, 0), mode='constant')
                                                                                                                # rospy.loginfo("Input: {}".format(obs_traj.shape, obs_traj_rel.shape))
    #     T = actions.shape[1]
    #     indexes = torch.arange(0, T, self.oskip)
    #     actions = actions[:, indexes, :2] # N x T' x 2

    #     ego_traj_full = trajectory[:, 0] # T' x 2

    #     obs_traj_full = trajectory[:, 1:] # T' x H x 2
    #     obs_traj = obs_traj_full[1:] # T' x H x 2
    #     obs_traj_rel = obs_traj - obs_traj_full[:-1] # T' x H x 2
    
    #     return obs_traj, obs_traj_rel, ego_traj_full, actions # T' x (1+H) x 2

    # def get_predictions(self, trajectory, actions):
    #     with torch.no_grad():
    #         trajectory = torch.tensor(trajectory, device=self.device, dtype=torch.float)
    #         actions = torch.tensor(actions, device=self.device, dtype=torch.float)

    #         obs_traj, obs_traj_rel, ego_traj_past, actions = self.create_input(trajectory, actions)
    #         seq_start_end = torch.tensor([[0, obs_traj.shape[1]]], dtype=torch.int, device=self.device)

    #         pred_traj_fake = []
    #         for action in actions:
    #             ego_traj_full  = torch.cat((ego_traj_past, action), axis=0)[:, None]

    #             ego_traj = ego_traj_full[1:]
    #             ego_traj_rel = ego_traj - ego_traj_full[:-1]

    #             pred_traj_fake_rel = self.generator(
    #                 obs_traj, obs_traj_rel, seq_start_end, ego_traj, ego_traj_rel,  self.num_samples)
    #             _pred_traj_fake = optimized_relative_to_abs(
    #                 pred_traj_fake_rel, obs_traj[-1]) # S x T'' x H x 2
                
    #             pred_traj_fake.append(_pred_traj_fake)
                                                                                                    #    rospy.loginfo("from model: {}".format(pred_traj_fake.shape))
    #         pred_traj_fake = torch.stack(pred_traj_fake, axis=0) # N x S x T'' x H x 2
    #         pred_traj_fake = self.create_output(
    #             pred_traj_fake, trajectory[-1, 1:])  # N x S x T' x H x 2
    #         pred_traj_fake = torch.cat(
    #             (pred_traj_fake[:, :, :-1], self.predict_velocity(pred_traj_fake)), dim=-1)  # N x S x T' x H x 4

                                                                                                                    # rospy.loginfo("final_pred: {}\n\n".format(pred_traj_fake.shape))
    #     return pred_traj_fake.cpu().numpy()  # N x S x T' x H x 4
