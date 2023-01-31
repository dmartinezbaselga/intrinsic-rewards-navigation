from gym.spaces import Box, Discrete, Space
from crowd_nav.policy.graph_model import GAT_RL, RGL
import numpy as np
from typing import List, Optional, Union
import torch
import torch.nn as nn
from crowd_nav.policy.cadrl import mlp
import torch.nn.utils.rnn as rnn_utils



class MovingMeanStd:
    """Track moving mean, std and count."""

    def __init__(self, epsilon: float = 1e-4, shape: Optional[List[int]] = None):
        """Initialize object.

        Args:
            epsilon: Initial count.
            shape: Shape of the trackables mean and std.
        """
        if not shape:
            shape = []
        self.mean = torch.zeros(shape, dtype=torch.float32)
        self.var = torch.ones(shape, dtype=torch.float32)
        self.count = epsilon

    def __call__(self, inputs):
        """Normalize input batch using moving mean and std.

        Args:
            inputs: Input batch to normalize.

        Returns:
            Logarithmic scaled normalized output.
        """
        batch_mean = torch.mean(torch.Tensor.float(inputs), axis=0)
        batch_var = torch.var(torch.Tensor.float(inputs), axis=0)
        batch_count = inputs.shape[0]
        self.update_params(batch_mean, batch_var, batch_count)
        return torch.log(inputs / self.std + 1)

    def update_params(
        self, batch_mean: float, batch_var: float, batch_count: float
    ) -> None:
        """Update moving mean, std and count.

        Args:
            batch_mean: Input batch mean.
            batch_var: Input batch variance.
            batch_count: Number of cases in the batch.
        """
        delta = batch_mean - self.mean
        tot_count = self.count + batch_count

        # This moving mean calculation is from reference implementation.
        self.mean = self.mean + delta + batch_count / tot_count
        m_a = self.var * self.count
        m_b = batch_var * batch_count
        M2 = m_a + m_b + torch.square(delta) * self.count * batch_count / tot_count
        self.var = M2 / tot_count
        self.count = tot_count

    @property
    def std(self) -> float:
        """Get moving standard deviation.

        Returns:
            Returns moving standard deviation.
        """
        return torch.sqrt(self.var)


def update_beta(beta_schedule: str, beta: float, rho: float, step: int) -> float:
    """Update beta based on schedule and training step.

    Args:
        beta_schedule: Schedule for beta update.
        beta: Initial beta.
        rho: Schedule decay parameter.
        step: Current training iteration.

    Returns:
        Updated beta as per input schedule.
    """
    if beta_schedule == "linear_decay":
        return beta * ((1.0 - rho) ** step)
    return beta


def compute_states_entropy(
    obs_embeds, embed_dim: int, k_nn: int
):
    """Compute states entropy using K nearest neighbour method.

    Args:
        obs_embeds: Observation latent representation using
            encoder model.
        embed_dim: Embedding vector dimension.
        k_nn: Number of nearest neighbour for K-NN estimation.

    Returns:
        Computed states entropy.
    """
    # .detach().cpu().numpy()
    obs_embeds_ = torch.reshape(obs_embeds, [-1, embed_dim])
    dist = torch.linalg.norm(obs_embeds_[:, None, :] - obs_embeds_[None, :, :], axis=-1)
    return dist.argsort(axis=-1)[:, :k_nn][:, -1]

class GraphModel(nn.Module):
    def __init__(self, config, robot_state_dim, human_state_dim):
        super().__init__()
        self.model = GAT_RL(config, robot_state_dim, human_state_dim)
    
    def forward(self, state):
        return self.model(self.rotate(state))[:, 0, :]

    def rotate(self, state):
        """
        Transform the coordinate to agent-centric.
        Input tuple include robot state tensor and human state tensor.
        robot state tensor is of size (batch_size, number, state_length)(for example 100*1*9)
        human state tensor is of size (batch_size, number, state_length)(for example 100*5*5)
        """
        # for robot
        # 'px', 'py', 'vx', 'vy', 'radius', 'gx', 'gy', 'v_pref', 'theta'
        #  0     1      2     3      4        5     6      7         8
        # for human
        #  'px', 'py', 'vx', 'vy', 'radius'
        #  0     1      2     3      4
        assert len(state[0].shape) == 3
        if len(state[1].shape) == 3:
            batch = state[0].shape[0]
            robot_state = state[0]
            human_state = state[1]
            human_num = state[1].shape[1]
            dx = robot_state[:, :, 5] - robot_state[:, :, 0]
            dy = robot_state[:, :, 6] - robot_state[:, :, 1]
            dx = dx.unsqueeze(1)
            dy = dy.unsqueeze(1)
            dg = torch.norm(torch.cat([dx, dy], dim=2), 2, dim=2, keepdim=True)
            rot = torch.atan2(dy, dx)
            cos_rot = torch.cos(rot)
            sin_rot = torch.sin(rot)
            transform_matrix = torch.cat((cos_rot, -sin_rot, sin_rot, cos_rot), dim=1).reshape(batch, 2, 2)
            robot_velocities = torch.bmm(robot_state[:, :, 2:4], transform_matrix)
            radius_r = robot_state[:, :, 4].unsqueeze(1)
            v_pref = robot_state[:, :, 7].unsqueeze(1)
            target_heading = torch.zeros_like(radius_r)
            pos_r = torch.zeros_like(robot_velocities)
            cur_heading = (robot_state[:, :, 8].unsqueeze(1) - rot + np.pi) % (2 * np.pi) - np.pi
            new_robot_state = torch.cat((pos_r, robot_velocities, radius_r, dg, target_heading, v_pref, cur_heading), dim=2)
            human_positions = human_state[:, :, 0:2] - robot_state[:, :, 0:2]
            human_positions = torch.bmm(human_positions, transform_matrix)
            human_velocities = human_state[:, :, 2:4]
            human_velocities = torch.bmm(human_velocities, transform_matrix)
            human_radius = human_state[:, :, 4].unsqueeze(2) + 0.3
            new_human_state = torch.cat((human_positions, human_velocities, human_radius), dim=2)
            new_state = (new_robot_state, new_human_state)
            return new_state
        else:
            batch = state[0].shape[0]
            robot_state = state[0]
            dx = robot_state[:, :, 5] - robot_state[:, :, 0]
            dy = robot_state[:, :, 6] - robot_state[:, :, 1]
            dx = dx.unsqueeze(1)
            dy = dy.unsqueeze(1)
            radius_r = robot_state[:, :, 4].unsqueeze(1)
            dg = torch.norm(torch.cat([dx, dy], dim=2), 2, dim=2, keepdim=True)
            rot = torch.atan2(dy, dx)
            cos_rot = torch.cos(rot)
            sin_rot = torch.sin(rot)
            vx = (robot_state[:, :, 2].unsqueeze(1) * cos_rot +
                robot_state[:, :, 3].unsqueeze(1) * sin_rot).reshape((batch, 1, -1))
            vy = (robot_state[:, :, 3].unsqueeze(1) * cos_rot -
                robot_state[:, :, 2].unsqueeze(1) * sin_rot).reshape((batch, 1, -1))
            v_pref = robot_state[:, :, 7].unsqueeze(1)
            theta = robot_state[:, :, 8].unsqueeze(1)
            px_r = torch.zeros_like(v_pref)
            py_r = torch.zeros_like(v_pref)
            new_robot_state = torch.cat((px_r, py_r, vx, vy, radius_r, dg, rot, v_pref, theta), dim=2)
            new_state = (new_robot_state, None)
            return new_state

class CADRLModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(
                        nn.Linear(13, 150),
                        nn.ReLU())
    
    def forward(self, state):
        return self.model(state[0])

class LSTMModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.mlp1 = mlp(13, [150, 100, 50])
        self.lstm = nn.LSTM(50, 50, batch_first=True)
        self.mlp = mlp(6 + 50, [150], last_relu=True)
    
    def forward(self, state):
        # state = torch.stack(state).permute((1,0,2))

        size = state.shape
        self_state = state[:, 0, :6]

        state = torch.reshape(state, (-1, size[2]))
        mlp1_output = self.mlp1(state)
        mlp1_output = torch.reshape(mlp1_output, (size[0], size[1], -1))
        packed_mlp1_output = torch.nn.utils.rnn.pack_padded_sequence(mlp1_output, [10], batch_first=True)

        output, (hn, cn) = self.lstm(packed_mlp1_output)
        hn = hn.squeeze(0)
        joint_state = torch.cat([self_state, hn], dim=1)
        value = self.mlp(joint_state)
        return value

class SARLModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.mlp1 = mlp(61, [150, 100], last_relu=True)
        self.mlp2 = mlp(100, [100, 50])
        self.attention = mlp(100 * 2, [100, 100, 1])
        self.mlp3 = mlp(56, [150], last_relu=True)


        # self.mlp1 = mlp(13, [150, 100, 50])
        # self.lstm = nn.LSTM(50, 50, batch_first=True)
        # self.mlp = mlp(6 + 50, [150], last_relu=True)
    
    def forward(self, state):
        # state = torch.stack(state).permute((1,0,2))

        lengths = torch.IntTensor([state.size()[1]])

        size = state.shape
        self_state = state[:, 0, :6]
        mlp1_output = self.mlp1(state.reshape((-1, size[2])))
        mlp2_output = self.mlp2(mlp1_output)


            # compute attention scores
        global_state = torch.mean(mlp1_output.view(size[0], size[1], -1), 1, keepdim=True)
        global_state = global_state.expand((size[0], size[1], 100)).\
            contiguous().view(-1, 100)
        attention_input = torch.cat([mlp1_output, global_state], dim=1)

        scores = self.attention(attention_input).view(size[0], size[1], 1).squeeze(dim=2)

        # masked softmax
        mask = rnn_utils.pad_sequence([torch.ones(length.item()) for length in lengths], batch_first=True)
        masked_scores = scores * mask.float()
        max_scores = torch.max(masked_scores, dim=1, keepdim=True)[0]
        exps = torch.exp(masked_scores - max_scores)
        masked_exps = exps * mask.float()
        masked_sums = masked_exps.sum(1, keepdim=True)
        weights = (masked_exps / masked_sums).unsqueeze(2)
        self.attention_weights = weights[0, :, 0].data.cpu().numpy()

        # output feature is a linear combination of input features
        features = mlp2_output.view(size[0], size[1], -1)
        weighted_feature = torch.sum(torch.mul(weights, features), dim=1)

        # concatenate agent's state with global weighted humans' state
        joint_state = torch.cat([self_state, weighted_feature], dim=1)
        value = self.mlp3(joint_state)
        return value

class MPRLModel(nn.Module):
    def __init__(self, config, robot_state_dim, human_state_dim):
        super().__init__()
        self.model = RGL(config, robot_state_dim, human_state_dim)


        # self.mlp1 = mlp(13, [150, 100, 50])
        # self.lstm = nn.LSTM(50, 50, batch_first=True)
        # self.mlp = mlp(6 + 50, [150], last_relu=True)
    
    def forward(self, state):
        return self.model(self.trans_no_rotation(state))[:, 0, :]

    def trans_no_rotation(self, state):
        """
        Transform the coordinate to agent-centric.
        Input tuple include robot state tensor and human state tensor.
        robot state tensor is of size (batch_size, number, state_length)(for example 100*1*9)
        human state tensor is of size (batch_size, number, state_length)(for example 100*5*5)
        """
        # for robot
        # 'px', 'py', 'vx', 'vy', 'radius', 'gx', 'gy', 'v_pref', 'theta'
        #  0     1      2     3      4        5     6      7         8
        # for human
        #  'px', 'py', 'vx', 'vy', 'radius'
        #  0     1      2     3      4
        assert len(state[0].shape) == 3
        assert len(state[1].shape) == 3
        batch = state[0].shape[0]
        robot_state = state[0]
        human_state = state[1]
        human_num = state[1].shape[1]
        dx = robot_state[:, :, 5] - robot_state[:, :, 0]
        dy = robot_state[:, :, 6] - robot_state[:, :, 1]
        dx = dx.unsqueeze(1)
        dy = dy.unsqueeze(1)
        radius_r = robot_state[:, :, 4].unsqueeze(1)
        dg = torch.norm(torch.cat([dx, dy], dim=2), 2, dim=2, keepdim=True)
        rot = torch.atan2(dy, dx)
        vx = robot_state[:, :, 2].unsqueeze(1)
        vy = robot_state[:, :, 3].unsqueeze(1)
        v_pref = robot_state[:, :, 7].unsqueeze(1)
        theta = robot_state[:, :, 8].unsqueeze(1)
        new_robot_state = torch.cat((theta, theta, vx, vy, radius_r, dg, rot, v_pref, theta), dim=2)
        new_human_state = None
        for i in range(human_num):
            dx1 = human_state[:, i, 0].unsqueeze(1) - robot_state[:, :, 0]
            dy1 = human_state[:, i, 1].unsqueeze(1) - robot_state[:, :, 1]
            dx1 = dx1.unsqueeze(1).reshape((batch, 1, -1))
            dy1 = dy1.unsqueeze(1).reshape((batch, 1, -1))
            vx1 = (human_state[:, i, 2].unsqueeze(1).unsqueeze(2)).reshape((batch, 1, -1))
            vy1 = (human_state[:, i, 3].unsqueeze(1).unsqueeze(2)).reshape((batch, 1, -1))
            radius_h = human_state[:, i, 4].unsqueeze(1).unsqueeze(2)
            cur_human_state = torch.cat((dx1, dy1, vx1, vy1, radius_h), dim=2)
            if new_human_state is None:
                new_human_state = cur_human_state
            else:
                new_human_state = torch.cat((new_human_state, cur_human_state), dim=1)
        new_state = (new_robot_state, new_human_state)
        return new_state

class RE3():
    """Random Encoder for Efficient Exploration.

    Implementation of:
    [1] State entropy maximization with random encoders for efficient
    exploration. Seo, Chen, Shin, Lee, Abbeel, & Lee, (2021).
    arXiv preprint arXiv:2102.09430.

    Estimates state entropy using a particle-based k-nearest neighbors (k-NN)
    estimator in the latent space. The state's latent representation is
    calculated using an encoder with randomly initialized parameters.

    The entropy of a state is considered as intrinsic reward and added to the
    environment's extrinsic reward for policy optimization.
    Entropy is calculated per batch, it does not take the distribution of
    the entire replay buffer into consideration.
    """

    def __init__(
        self,
        config, robot_state_dim, human_state_dim, device,
        embeds_dim: int = 128,
        beta: float = 0.2,
        beta_schedule: str = "constant",
        rho: float = 0.1,
        k_nn: int = 50,
    ):
        """Initialize RE3.

        Args:
            action_space: The action space in which to explore.
            framework: Supports "tf", this implementation does not
                support torch.
            model: The policy's model.
            embeds_dim: The dimensionality of the observation embedding
                vectors in latent space.
            encoder_net_config: Optional model
                configuration for the encoder network, producing embedding
                vectors from observations. This can be used to configure
                fcnet- or conv_net setups to properly process any
                observation space.
            beta: Hyperparameter to choose between exploration and
                exploitation.
            beta_schedule: Schedule to use for beta decay, one of
                "constant" or "linear_decay".
            rho: Beta decay factor, used for on-policy algorithm.
            k_nn: Number of neighbours to set for K-NN entropy
                estimation.
            random_timesteps: The number of timesteps to act completely
                randomly (see [1]).
            sub_exploration: The config dict for the underlying Exploration
                to use (e.g. epsilon-greedy for DQN). If None, uses the
                FromSpecDict provided in the Policy's default config.

        Raises:
            ValueError: If the input framework is Torch.
        """

        self.beta = beta
        self.rho = rho
        self.k_nn = k_nn
        self.embeds_dim = embeds_dim
        self.beta_schedule = beta_schedule
        self._rms = MovingMeanStd()
        self.name = "RE3"

        if config.name == "cadrl":
            self._encoder_net = nn.Sequential(
                CADRLModel(),
                nn.Linear(150, 256),
                nn.ReLU(),
                nn.Linear(256, 128)
            )
        elif config.name == "lstm_rl":
            self._encoder_net = nn.Sequential(
                LSTMModel(),
                nn.Linear(150, 256),
                nn.ReLU(),
                nn.Linear(256, 128)
            )
        elif config.name == "sarl":
            self._encoder_net = nn.Sequential(
                SARLModel(),
                nn.Linear(150, 256),
                nn.ReLU(),
                nn.Linear(256, 128)
            )
        elif config.name == "model_predictive_rl":
            self._encoder_net = nn.Sequential(
                MPRLModel(config, robot_state_dim, human_state_dim),
                nn.Linear(config.gcn.X_dim, 256),
                nn.ReLU(),
                nn.Linear(256, 128)
            )
        elif config.name == "tree_search_rl":
            self._encoder_net = nn.Sequential(
                GraphModel(config, robot_state_dim, human_state_dim),
                nn.Linear(config.gcn.X_dim, 256),
                nn.ReLU(),
                nn.Linear(256, embeds_dim)
            )
        else:
            raise ValueError("Curiosity not implemented for this method")
        # Creates ModelV2 embedding module / layers.
        # self._encoder_net = nn.Sequential(
        #     self.graph_model(config, robot_state_dim, human_state_dim),
        #     nn.Linear(config.gcn.X_dim, 256),
        #     nn.ReLU(),
        #     nn.Linear(256, embeds_dim)
        # ).to(device)


    def get_embeddings(self, obs):
        """Calculate states' embeddings and add it to SampleBatch."""
        obs_embeds = self._encoder_net(obs)
        return obs_embeds
    
    def optimize(self, state, next_state, action):
        return
    
    def compute_intrinsic_reward(self, state, next_state, action):
        raise Exception("This function should not be called")
        return 0.0

    def compute_intrinsic_reward_batch(self, embeddings, iteration):
        states_entropy = compute_states_entropy(
            embeddings, self.embeds_dim, self.k_nn
        )
        states_entropy = update_beta(
            self.beta_schedule, self.beta, self.rho, iteration
        ) * torch.reshape(
            self._rms(states_entropy),
            embeddings.shape[:-1],
        )
        return states_entropy
