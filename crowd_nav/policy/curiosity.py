from crowd_nav.policy.graph_model import GAT_RL, RGL
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from crowd_nav.policy.cadrl import mlp
import torch.nn.utils.rnn as rnn_utils



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
        state = torch.stack(state).permute((1,0,2))

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
        state = torch.stack(state).permute((1,0,2))

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
    
class ICM():
    """Implementation of:
    [1] Curiosity-driven Exploration by Self-supervised Prediction
    Pathak, Agrawal, Efros, and Darrell - UC Berkeley - ICML 2017.
    https://arxiv.org/pdf/1705.05363.pdf

    Learns a simplified model of the environment based on three networks:
    1) Embedding observations into latent space ("feature" network).
    2) Predicting the action, given two consecutive embedded observations
    ("inverse" network).
    3) Predicting the next embedded obs, given an obs and action
    ("forward" network).

    The less the agent is able to predict the actually observed next feature
    vector, given obs and action (through the forwards network), the larger the
    "intrinsic reward", which will be added to the extrinsic reward.
    Therefore, if a state transition was unexpected, the agent becomes
    "curious" and will further explore this transition leading to better
    exploration in sparse rewards environments.
    """

    def __init__(self, config, robot_state_dim, human_state_dim, device, lr, scaling_factor, policy="tree_search_rl"):
        self.scaling_factor = scaling_factor
        self.action_num = config.action_space.speed_samples * config.action_space.rotation_samples + 1
        self.device = device
        self.lr = lr
        self.cross_entropy_loss = torch.nn.CrossEntropyLoss()
        self.beta = 0.2
        self.name = "ICM"
        if policy == "cadrl":
            self._curiosity_feature_net = nn.Sequential(
                CADRLModel(),
                nn.Linear(150, 256),
                nn.ReLU(),
                nn.Linear(256, 128)
            )
        elif policy == "lstm_rl":
            self._curiosity_feature_net = nn.Sequential(
                LSTMModel(),
                nn.Linear(150, 256),
                nn.ReLU(),
                nn.Linear(256, 128)
            )
        elif policy == "sarl":
            self._curiosity_feature_net = nn.Sequential(
                SARLModel(),
                nn.Linear(150, 256),
                nn.ReLU(),
                nn.Linear(256, 128)
            )
        elif policy == "model_predictive_rl":
            self._curiosity_feature_net = nn.Sequential(
                MPRLModel(config, robot_state_dim, human_state_dim),
                nn.Linear(config.gcn.X_dim, 256),
                nn.ReLU(),
                nn.Linear(256, 128)
            )
        elif policy == "tree_search_rl":
            self._curiosity_feature_net = nn.Sequential(
                GraphModel(config, robot_state_dim, human_state_dim),
                nn.Linear(config.gcn.X_dim, 256),
                nn.ReLU(),
                nn.Linear(256, 128)
            )
        else:
            raise ValueError("Curiosity not implemented for this method")

        self._curiosity_inverse_fcnet = nn.Sequential(
            nn.Linear(2*128, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, self.action_num)
        )

        self._curiosity_forward_fcnet = nn.Sequential(
            nn.Linear(128+self.action_num, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 128)
        )
        self.get_exploration_optimizer()

    def get_exploration_optimizer(self):
        # Create, but don't add Adam for curiosity NN updating to the policy.
        # If we added and returned it here, it would be used in the policy's
        # update loop, which we don't want (curiosity updating happens inside
        # `postprocess_trajectory`).
        feature_params = list(self._curiosity_feature_net.parameters())
        inverse_params = list(self._curiosity_inverse_fcnet.parameters())
        forward_params = list(self._curiosity_forward_fcnet.parameters())

        # Now that the Policy's own optimizer(s) have been created (from
        # the Model parameters (IMPORTANT: w/o(!) the curiosity params),
        # we can add our curiosity sub-modules to the Policy's Model.
        self._curiosity_feature_net = self._curiosity_feature_net.to(
            self.device
        )
        self._curiosity_inverse_fcnet = self._curiosity_inverse_fcnet.to(
            self.device
        )
        self._curiosity_forward_fcnet = self._curiosity_forward_fcnet.to(
            self.device
        )
        self._optimizer = torch.optim.Adam(
            forward_params + inverse_params + feature_params, lr=self.lr
        )

    def compute_intrinsic_reward(self, state, next_state, action):
        # When the reward is stored in memory and when new rewards are created in the tree search
        phi = self._curiosity_feature_net(state)
        next_phi = self._curiosity_feature_net(next_state)
        predicted_next_phi = self._curiosity_forward_fcnet(
            torch.cat((phi, F.one_hot(torch.Tensor([action]).to(torch.int64).to(phi.device), self.action_num).float()), 1)
        )
        intrinsic_reward = self.scaling_factor * F.mse_loss(next_phi, predicted_next_phi, reduction='none').mean(-1)
        return intrinsic_reward.data.cpu().numpy()[0]

    def compute_intrinsic_reward_batch(self, state, next_state, action):
        # When the reward is stored in memory and when new rewards are created in the tree search
        phi = self._curiosity_feature_net(state)
        next_phi = self._curiosity_feature_net(next_state)
        predicted_next_phi = self._curiosity_forward_fcnet(
            torch.cat((phi, F.one_hot(action.to(torch.int64), self.action_num).float()), 1)
        )
        forward_l2_norm_sqared = 0.5 * torch.sum(
            torch.pow(predicted_next_phi - next_phi, 2.0), dim=-1
        )
        intrinsic_reward = self.scaling_factor * forward_l2_norm_sqared
        return intrinsic_reward.unsqueeze(1)
    
    def optimize(self, state, next_state, action):
        phi = self._curiosity_feature_net(state)
        next_phi = self._curiosity_feature_net(next_state)
        real_actions = F.one_hot(action.to(torch.int64), self.action_num).float()
        predicted_next_phi = self._curiosity_forward_fcnet(
            torch.cat((phi, real_actions), 1)
        )
        forward_l2_norm_sqared = 0.5 * torch.sum(
            torch.pow(predicted_next_phi - next_phi, 2.0), dim=-1
        )
        forward_loss = torch.mean(forward_l2_norm_sqared)
        pred_actions = self._curiosity_inverse_fcnet(torch.cat((phi, next_phi), -1))
        inverse_loss = self.cross_entropy_loss(pred_actions, action)

        loss = (1.0 - self.beta) * inverse_loss + self.beta * forward_loss
        self._optimizer.zero_grad()
        loss.backward()
        self._optimizer.step()