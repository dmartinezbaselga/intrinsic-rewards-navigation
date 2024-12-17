import torch
import torch.nn as nn
from .sgan_models import TrajectoryGenerator, ConditionalTrajectoryGenerator

class OptimizedCSGAN(ConditionalTrajectoryGenerator):
    def __init__(
        self, obs_len, pred_len, embedding_dim=64, encoder_h_dim=64,
        decoder_h_dim=128, mlp_dim=1024, num_layers=1, noise_dim=(0, ),
        noise_type='gaussian', noise_mix_type='ped', pooling_type=None,
        pool_every_timestep=True, dropout=0.0, bottleneck_dim=1024,
        activation='relu', batch_norm=True, neighborhood_size=2.0, grid_size=8,
        device='cpu'
    ):
        super(OptimizedCSGAN, self).__init__(
            obs_len, pred_len, embedding_dim, encoder_h_dim, decoder_h_dim,
            mlp_dim, num_layers, noise_dim, noise_type, noise_mix_type,
            pooling_type, pool_every_timestep, dropout, bottleneck_dim,
            activation, batch_norm, neighborhood_size, grid_size, device
        )

    def forward(self, obs_traj, obs_traj_rel, seq_start_end, ego_traj, ego_traj_rel, num_samples, user_noise=None):
        """
        Inputs:
        - obs_traj: Tensor of shape (obs_len, batch, 2)
        - obs_traj_rel: Tensor of shape (obs_len, batch, 2)
        - seq_start_end: A list of tuples which delimit sequences within batch.
        - ego_traj: Tensor of shape (obs_len+future_len, batch, 2) for the ego agent
        - ego_traj_rel: Tensor of shape (obs_len+future_len, batch, 2) for the ego agent
        - user_noise: Generally used for inference when you want to see
        relation between different types of noise and outputs.
        Output:
        - pred_traj_rel: Tensor of shape (self.pred_len, batch, 2)
        """
        batch = obs_traj_rel.size(1)
        # Encode seq
        final_encoder_h = self.encoder(obs_traj_rel)
        # Pool States
        if self.pooling_type:
            end_pos = obs_traj[-1, :, :]
            pool_h = self.pool_net(final_encoder_h, seq_start_end, end_pos)
            # Construct input hidden states for decoder
            mlp_decoder_context_input = torch.cat(
                [final_encoder_h.view(-1, self.encoder_h_dim), pool_h], dim=1)
        else:
            mlp_decoder_context_input = final_encoder_h.view(
                -1, self.encoder_h_dim)

        # Ego Hidden State
        ego_encoder_h = self.encoder(ego_traj_rel).view(-1, self.encoder_h_dim)
        mlp_decoder_context_input = self.add_ego_agent(
            mlp_decoder_context_input, seq_start_end, ego_encoder_h)

        decoder_c = torch.zeros(
            self.num_layers, batch, self.decoder_h_dim, device=self.device
        )

        pred_traj_samples = []

        for _ in range(num_samples):
            # Add Noise
            if self.mlp_decoder_needed():
                noise_input = self.mlp_decoder_context(
                    mlp_decoder_context_input)
            else:
                noise_input = mlp_decoder_context_input
            decoder_h = self.add_noise(
                noise_input, seq_start_end, user_noise=user_noise)
            decoder_h = torch.unsqueeze(decoder_h, 0)

            state_tuple = (decoder_h, decoder_c)
            last_pos = obs_traj[-1]
            last_pos_rel = obs_traj_rel[-1]
            # Predict Trajectory

            pred_traj_samples.append(self.decoder(
                last_pos,
                last_pos_rel,
                state_tuple,
                seq_start_end)[0]
            )
        
        return torch.stack(pred_traj_samples, axis=0)

class OptimizedSGAN(TrajectoryGenerator):
    def __init__(
        self, obs_len, pred_len, embedding_dim=64, encoder_h_dim=64,
        decoder_h_dim=128, mlp_dim=1024, num_layers=1, noise_dim=(0, ),
        noise_type='gaussian', noise_mix_type='ped', pooling_type=None,
        pool_every_timestep=True, dropout=0.0, bottleneck_dim=1024,
        activation='relu', batch_norm=True, neighborhood_size=2.0, grid_size=8,
        device='cpu'
    ):
        super(OptimizedSGAN, self).__init__(
            obs_len, pred_len, embedding_dim, encoder_h_dim, decoder_h_dim,
            mlp_dim, num_layers, noise_dim, noise_type, noise_mix_type,
            pooling_type, pool_every_timestep, dropout, bottleneck_dim,
            activation, batch_norm, neighborhood_size, grid_size, device
        )
    
    def forward(self, obs_traj, obs_traj_rel, seq_start_end, num_samples, user_noise=None):
        """
        Inputs:
        - obs_traj: Tensor of shape (obs_len, batch, 2)
        - obs_traj_rel: Tensor of shape (obs_len, batch, 2)
        - seq_start_end: A list of tuples which delimit sequences within batch.
        - user_noise: Generally used for inference when you want to see
        relation between different types of noise and outputs.
        Output:
        - pred_traj_rel: Tensor of shape (self.pred_len, batch, 2)
        """
        batch = obs_traj_rel.size(1)
        # Encode seq
        final_encoder_h = self.encoder(obs_traj_rel)
        # Pool States
        if self.pooling_type:
            end_pos = obs_traj[-1, :, :]
            pool_h = self.pool_net(final_encoder_h, seq_start_end, end_pos)
            # Construct input hidden states for decoder
            mlp_decoder_context_input = torch.cat(
                [final_encoder_h.view(-1, self.encoder_h_dim), pool_h], dim=1)
        else:
            mlp_decoder_context_input = final_encoder_h.view(
                -1, self.encoder_h_dim)

        decoder_c = torch.zeros(
            self.num_layers, batch, self.decoder_h_dim, device=self.device
        )

        pred_traj_samples = []

        for _ in range(num_samples):
            # Add Noise
            if self.mlp_decoder_needed():
                noise_input = self.mlp_decoder_context(
                    mlp_decoder_context_input)
            else:
                noise_input = mlp_decoder_context_input
            decoder_h = self.add_noise(
                noise_input, seq_start_end, user_noise=user_noise)
            decoder_h = torch.unsqueeze(decoder_h, 0)

            state_tuple = (decoder_h, decoder_c)
            last_pos = obs_traj[-1]
            last_pos_rel = obs_traj_rel[-1]
            # Predict Trajectory

            pred_traj_samples.append(self.decoder(
                last_pos,
                last_pos_rel,
                state_tuple,
                seq_start_end)[0]
            )
        
        return torch.stack(pred_traj_samples, axis=0)