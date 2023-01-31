import logging
import torch
import numpy as np
from numpy.linalg import norm
import itertools
from crowd_sim.envs.policy.policy import Policy
from crowd_sim.envs.utils.action import ActionRot, ActionXY
from crowd_sim.envs.utils.state import tensor_to_joint_state
from crowd_sim.envs.utils.info import  *
from crowd_sim.envs.utils.utils import point_to_segment_dist
from crowd_nav.policy.state_predictor import StatePredictor, LinearStatePredictor_batch
from crowd_nav.policy.graph_model import RGL,GAT_RL
from crowd_nav.policy.value_estimator import DQNNetwork, Noisy_DQNNetwork, DQNDropoutNetwork


class TreeSearchRL(Policy):
    def __init__(self):
        super().__init__()
        self.name = 'TreeSearchRL'
        self.trainable = True
        self.multiagent_training = True
        self.kinematics = None
        self.epsilon = None
        self.exploration_alg = "epsilon_greedy"
        self.gamma = None
        self.sampling = None
        self.speed_samples = None
        self.rotation_samples = None
        self.action_space = None
        self.rotation_constraint = None
        self.speeds = None
        self.rotations = None
        self.action_values = None
        self.robot_state_dim = 9
        self.human_state_dim = 5
        self.v_pref = 1
        self.share_graph_model = None
        self.value_estimator = None
        self.linear_state_predictor = None
        self.state_predictor = None
        self.planning_depth = None
        self.planning_width = None
        self.do_action_clip = None
        self.sparse_search = None
        self.sparse_speed_samples = 2
        self.sparse_rotation_samples = 8
        self.action_group_index = []
        self.traj = None
        self.use_noisy_net = False
        self.count=0
        self.time_step = 0.25

    def configure(self, config, device):
        self.set_common_parameters(config)
        self.planning_depth = config.model_predictive_rl.planning_depth
        self.do_action_clip = config.model_predictive_rl.do_action_clip
        if hasattr(config.model_predictive_rl, 'sparse_search'):
            self.sparse_search = config.model_predictive_rl.sparse_search
        self.planning_width = config.model_predictive_rl.planning_width
        self.share_graph_model = config.model_predictive_rl.share_graph_model
        self.linear_state_predictor = config.model_predictive_rl.linear_state_predictor
        # self.set_device(device)
        self.device = device
        if config.use_noisy_net is not None:
            self.use_noisy_net = config.use_noisy_net

        if self.exploration_alg == "dropout":
            if self.linear_state_predictor:
                self.state_predictor = LinearStatePredictor_batch(config, self.time_step)
                graph_model = RGL(config, self.robot_state_dim, self.human_state_dim)
                self.value_estimator = DQNDropoutNetwork(config, graph_model)
                self.model = [graph_model, self.value_estimator.value_network]
            else:
                if self.share_graph_model:
                    graph_model = GAT_RL(config, self.robot_state_dim, self.human_state_dim)
                    self.value_estimator = DQNDropoutNetwork(config, graph_model)
                    self.state_predictor = StatePredictor(config, graph_model, self.time_step)
                    self.model = [graph_model, self.value_estimator.value_network, self.state_predictor.human_motion_predictor]
                else:
                    graph_model1 = GAT_RL(config, self.robot_state_dim, self.human_state_dim)
                    self.value_estimator = DQNDropoutNetwork(config, graph_model1)
                    graph_model2 = GAT_RL(config, self.robot_state_dim, self.human_state_dim)
                    self.state_predictor = StatePredictor(config, graph_model2, self.time_step)
                    self.model = [graph_model1, graph_model2, self.value_estimator.value_network,
                                self.state_predictor.human_motion_predictor]
        elif self.use_noisy_net:
            if self.linear_state_predictor:
                self.state_predictor = LinearStatePredictor_batch(config, self.time_step)
                graph_model = RGL(config, self.robot_state_dim, self.human_state_dim)
                self.value_estimator = Noisy_DQNNetwork(config, graph_model)
                self.model = [graph_model, self.value_estimator.value_network]
            else:
                if self.share_graph_model:
                    graph_model = GAT_RL(config, self.robot_state_dim, self.human_state_dim)
                    self.value_estimator = Noisy_DQNNetwork(config, graph_model)
                    self.state_predictor = StatePredictor(config, graph_model, self.time_step)
                    self.model = [graph_model, self.value_estimator.value_network, self.state_predictor.human_motion_predictor]
                else:
                    graph_model1 = GAT_RL(config, self.robot_state_dim, self.human_state_dim)
                    self.value_estimator = Noisy_DQNNetwork(config, graph_model1)
                    graph_model2 = GAT_RL(config, self.robot_state_dim, self.human_state_dim)
                    self.state_predictor = StatePredictor(config, graph_model2, self.time_step)
                    self.model = [graph_model1, graph_model2, self.value_estimator.value_network,
                                self.state_predictor.human_motion_predictor]
        else:
            if self.linear_state_predictor:
                self.state_predictor = LinearStatePredictor_batch(config, self.time_step)
                graph_model = RGL(config, self.robot_state_dim, self.human_state_dim)
                self.value_estimator = DQNNetwork(config, graph_model)
                self.model = [graph_model, self.value_estimator.value_network]
            else:
                if self.share_graph_model:
                    graph_model = GAT_RL(config, self.robot_state_dim, self.human_state_dim)
                    self.value_estimator = DQNNetwork(config, graph_model)
                    self.state_predictor = StatePredictor(config, graph_model, self.time_step)
                    self.model = [graph_model, self.value_estimator.value_network, self.state_predictor.human_motion_predictor]
                else:
                    graph_model1 = GAT_RL(config, self.robot_state_dim, self.human_state_dim)
                    self.value_estimator = DQNNetwork(config, graph_model1)
                    graph_model2 = GAT_RL(config, self.robot_state_dim, self.human_state_dim)
                    self.state_predictor = StatePredictor(config, graph_model2, self.time_step)
                    self.model = [graph_model1, graph_model2, self.value_estimator.value_network,
                                self.state_predictor.human_motion_predictor]
        logging.info('Planning depth: {}'.format(self.planning_depth))
        logging.info('Planning width: {}'.format(self.planning_width))
        logging.info('Sparse search: {}'.format(self.sparse_search))

        if self.planning_depth > 1 and not self.do_action_clip:
            logging.warning('Performing d-step planning without action space clipping!')

    def set_common_parameters(self, config):
        self.gamma = config.rl.gamma
        self.kinematics = config.action_space.kinematics
        self.sampling = config.action_space.sampling
        self.speed_samples = config.action_space.speed_samples
        self.rotation_samples = config.action_space.rotation_samples
        self.v_pref = config.action_space.v_pref
        self.rotation_constraint = config.action_space.rotation_constraint

    def set_device(self, device):
        self.device = device
        for model in self.model:
            model.to(device)

    def set_epsilon(self, epsilon):
        if self.exploration_alg == "dropout":
            self.epsilon = 0.0
            self.value_estimator.set_dropout(epsilon)
        else:
            self.epsilon = epsilon
    
    def set_exploration_alg(self, alg):
        self.exploration_alg = alg

    def set_noisy_net(self, use_noisy_net):
        self.use_noisy_net = use_noisy_net

    def set_time_step(self, time_step):
        self.time_step = time_step
        self.state_predictor.time_step = time_step

    def get_normalized_gamma(self):
        return pow(self.gamma, self.time_step * self.v_pref)

    def get_model(self):
        return self.value_estimator

    def get_state_dict(self):
        if self.state_predictor.trainable:
            if self.share_graph_model:
                return {
                    'graph_model': self.value_estimator.graph_model.state_dict(),
                    'value_network': self.value_estimator.value_network.state_dict(),
                    'motion_predictor': self.state_predictor.human_motion_predictor.state_dict()
                }
            else:
                return {
                    'graph_model1': self.value_estimator.graph_model.state_dict(),
                    'graph_model2': self.state_predictor.graph_model.state_dict(),
                    'value_network': self.value_estimator.value_network.state_dict(),
                    'motion_predictor': self.state_predictor.human_motion_predictor.state_dict()
                }
        else:
            return {
                    'graph_model': self.value_estimator.graph_model.state_dict(),
                    'value_network': self.value_estimator.value_network.state_dict()
                }

    def get_traj(self):
        return self.traj

    def load_state_dict(self, state_dict):
        if self.state_predictor.trainable:
            if self.share_graph_model:
                self.value_estimator.graph_model.load_state_dict(state_dict['graph_model'])
            else:
                self.value_estimator.graph_model.load_state_dict(state_dict['graph_model1'])
                self.state_predictor.graph_model.load_state_dict(state_dict['graph_model2'])

            self.value_estimator.value_network.load_state_dict(state_dict['value_network'])
            self.state_predictor.human_motion_predictor.load_state_dict(state_dict['motion_predictor'])
        else:
            self.value_estimator.graph_model.load_state_dict(state_dict['graph_model'])
            self.value_estimator.value_network.load_state_dict(state_dict['value_network'])

    def save_model(self, file):
        torch.save(self.get_state_dict(), file)

    def load_model(self, file):
        checkpoint = torch.load(file)
        self.load_state_dict(checkpoint)

    def build_action_space(self, v_pref):
        """
        Action space consists of 25 uniformly sampled actions in permitted range and 25 randomly sampled actions.
        """
        holonomic = True if self.kinematics == 'holonomic' else False
        # speeds = [(np.exp((i + 1) / self.speed_samples) - 1) / (np.e - 1) * v_pref for i in range(self.speed_samples)]
        speeds = [(i+1)/self.speed_samples * v_pref for i in range(self.speed_samples)]
        if holonomic:
            rotations = np.linspace(0, 2 * np.pi, self.rotation_samples, endpoint=False)
        else:
            if self.rotation_constraint == np.pi:
                rotations = np.linspace(-self.rotation_constraint, self.rotation_constraint, self.rotation_samples, endpoint=False)
            else:
                rotations = np.linspace(-self.rotation_constraint, self.rotation_constraint, self.rotation_samples)

        action_space = [ActionXY(0, 0) if holonomic else ActionRot(0, 0)]
        self.action_group_index.append(0)

        for i, rotation in enumerate(rotations):
            for j, speed in enumerate(speeds):
                action_index = i * self.speed_samples + j + 1
                self.action_group_index.append(action_index)
                if holonomic:
                    action_space.append(ActionXY(speed * np.cos(rotation), speed * np.sin(rotation)))
                else:
                    action_space.append(ActionRot(speed, rotation))
        self.speeds = speeds
        self.rotations = rotations
        self.action_space = action_space

    def predict(self, state):
        """
        A base class for all methods that takes pairwise joint state as input to value network.
        The input to the value network is always of shape (batch_size, # humans, rotated joint state length)

        """
        self.count=self.count+1
        # if self.count == 34:
        #     print('debug')
        if self.phase is None or self.device is None:
            raise AttributeError('Phase, device attributes have to be set!')
        if self.phase == 'train' and self.epsilon is None:
            raise AttributeError('Epsilon attribute has to be set in training phase')
        # self.v_pref = state.robot_state.v_pref
        if self.reach_destination(state):
            return ActionXY(0, 0) if self.kinematics == 'holonomic' else ActionRot(0, 0)
        if self.action_space is None:
            self.build_action_space(self.v_pref)
        max_action = None
        origin_max_value = float('-inf')
        state_tensor = state.to_tensor(add_batch_size=True, device=self.device)
        if self.exploration_alg == "epsilon_greedy":
            probability = np.random.random()
            if self.phase == 'train' and probability < self.epsilon and self.use_noisy_net is False:
                max_action_index = np.random.choice(len(self.action_space))
                max_action = self.action_space[max_action_index]
                self.last_state = self.transform(state)
                return max_action, max_action_index
            else:
                max_value, max_action_index, max_traj = self.V_planning(state_tensor, self.planning_depth, self.planning_width)
                if max_value[0] > origin_max_value:
                    max_action = self.action_space[max_action_index[0]]
                if max_action is None:
                    raise ValueError('Value network is not well trained.')
        elif (self.exploration_alg == "boltzmann" or self.exploration_alg == "dropout" or self.exploration_alg == "curiosity" or 
                self.exploration_alg == "random_encoder" or self.exploration_alg == "noisy_net"):
            max_value, max_action_index, max_traj = self.V_planning(state_tensor, self.planning_depth, self.planning_width)
            if max_value[0] > origin_max_value:
                max_action = self.action_space[max_action_index[0]]
            if max_action is None:
                raise ValueError('Value network is not well trained.')
        else:
            raise ValueError('Exploration algorithm unknown')

        if self.phase == 'train':
            self.last_state = self.transform(state)
        else:
            self.last_state = self.transform(state)
            self.traj = max_traj[0]
        return max_action, int(max_action_index[0])

    def V_planning(self, state, depth, width):
        """ Plans n steps into future based on state action value function. Computes the value for the current state as well as the trajectories
        defined as a list of (state, action, reward) triples
        """
        # current_state_value = self.value_estimator(state)
        robot_state_batch = state[0]
        human_state_batch = state[1]
        if state[1] is None:
            if depth == 0:
                q_value = self.value_estimator(state)
                if self.exploration_alg == "boltzmann":
                    q_dist = torch.softmax(q_value/self.epsilon,-1,float)
                    max_action_indexes = torch.stack([torch.multinomial(q_dist_aux, 1) for q_dist_aux in q_dist])
                    max_action_value = torch.gather(q_value, 1, max_action_indexes)
                else:
                    max_action_value, max_action_indexes = torch.max(q_value, dim=1)
                trajs = []
                for i in range(robot_state_batch.shape[0]):
                    cur_state = (robot_state_batch[i, :, :].unsqueeze(0), None)
                    trajs.append([(cur_state, None, None)])
                return max_action_value, max_action_indexes, trajs
            else:
                q_value = self.value_estimator(state)
                if self.exploration_alg == "boltzmann":
                    q_dist = torch.softmax(q_value/self.epsilon,-1,float)
                    max_action_indexes = torch.stack([torch.multinomial(q_dist_aux, width) for q_dist_aux in q_dist])
                    max_action_value = torch.gather(q_value, 1, max_action_indexes)
                else:
                    max_action_value, max_action_indexes = torch.topk(q_value, width, dim=1)
            action_stay = []
            for i in range(robot_state_batch.shape[0]):
                if self.kinematics == "holonomic":
                    action_stay.append(ActionXY(0, 0))
                else:
                    action_stay.append(ActionRot(0, 0))
            pre_next_state = None
            next_robot_state_batch = None
            next_human_state_batch = None
            reward_est = torch.zeros(state[0].shape[0], width) * float('inf')

            for i in range(robot_state_batch.shape[0]):
                cur_state = (robot_state_batch[i, :, :].unsqueeze(0), None)
                next_human_state = None
                for j in range(width):
                    cur_action = self.action_space[max_action_indexes[i][j]]
                    next_robot_state = self.compute_next_robot_state(cur_state[0], cur_action)
                    if next_robot_state_batch is None:
                        next_robot_state_batch = next_robot_state
                    else:
                        next_robot_state_batch = torch.cat((next_robot_state_batch, next_robot_state), dim=0)
                    reward_est[i][j], _ = self.reward_estimator.estimate_reward_on_predictor(
                        tensor_to_joint_state(cur_state), tensor_to_joint_state((next_robot_state, next_human_state)))

            next_state_batch = (next_robot_state_batch, next_human_state_batch)
            if self.planning_depth - depth >= 2 and self.planning_depth > 2:
                cur_width = 1
            else:
                cur_width = int(self.planning_width / 2)
            next_values, next_action_indexes, next_trajs = self.V_planning(next_state_batch, depth - 1, cur_width)
            next_values = next_values.view(state[0].shape[0], width)
            returns = (reward_est + self.get_normalized_gamma() * next_values + max_action_value) / 2
            if self.exploration_alg == "boltzmann":
                q_dist = torch.softmax(returns/self.epsilon,-1,float)
                max_action_index = torch.stack([torch.multinomial(q_dist_aux, 1) for q_dist_aux in q_dist])
                max_action_return = torch.gather(returns, 1, max_action_index)
            else:
                max_action_return, max_action_index = torch.max(returns, dim=1)
            trajs = []
            max_returns = []
            max_actions = []
            for i in range(robot_state_batch.shape[0]):
                cur_state = (robot_state_batch[i, :, :].unsqueeze(0), None)
                action_id = max_action_index[i]
                trajs_id = i * width + action_id
                action = max_action_indexes[i][action_id]
                next_traj = next_trajs[trajs_id]
                trajs.append([(cur_state, action, reward_est)] + next_traj)
                max_returns.append(max_action_return[i].data)
                max_actions.append(action)
            max_returns = torch.tensor(max_returns)
            return max_returns, max_actions, trajs
        else:
            if depth == 0:
                q_value = self.value_estimator(state)
                if self.exploration_alg == "boltzmann":
                    q_dist = torch.softmax(q_value/self.epsilon,-1,float)
                    max_action_indexes = torch.stack([torch.multinomial(q_dist_aux, 1) for q_dist_aux in q_dist])
                    max_action_value = torch.gather(q_value, 1, max_action_indexes)
                else:
                    max_action_value, max_action_indexes = torch.max(q_value, dim=1)
                trajs = []
                for i in range(robot_state_batch.shape[0]):
                    cur_state = (robot_state_batch[i, :, :].unsqueeze(0), human_state_batch[i, :, :].unsqueeze(0))
                    trajs.append([(cur_state, None, None)])
                return max_action_value, max_action_indexes, trajs
            else:
                # q_value = self.value_estimator(state)
                q_value = self.value_estimator(state)
                if self.exploration_alg == "boltzmann":
                    q_dist = torch.softmax(q_value/self.epsilon,-1,float)
                    max_action_indexes = torch.stack([torch.multinomial(q_dist_aux, width) for q_dist_aux in q_dist])
                    max_action_value = torch.gather(q_value, 1, max_action_indexes)
                else:
                    max_action_value, max_action_indexes = torch.topk(q_value, width, dim=1)
            action_stay = []
            for i in range(robot_state_batch.shape[0]):
                if self.kinematics == "holonomic":
                    action_stay.append(ActionXY(0, 0))
                else:
                    action_stay.append(ActionRot(0, 0))
            _, pre_next_state = self.state_predictor(state, action_stay)
            next_robot_state_batch = None
            next_human_state_batch = None
            reward_est = (torch.zeros(state[0].shape[0], width) * float('inf')).to(self.device)

            for i in range(robot_state_batch.shape[0]):
                cur_state = (robot_state_batch[i, :, :].unsqueeze(0), human_state_batch[i, :, :].unsqueeze(0))
                next_human_state = pre_next_state[i, :, :].unsqueeze(0)
                for j in range(width):
                    cur_action = self.action_space[max_action_indexes[i][j]]
                    next_robot_state = self.compute_next_robot_state(cur_state[0], cur_action)
                    if next_robot_state_batch is None:
                        next_robot_state_batch = next_robot_state
                        next_human_state_batch = next_human_state
                    else:
                        next_robot_state_batch = torch.cat((next_robot_state_batch, next_robot_state), dim=0)
                        next_human_state_batch = torch.cat((next_human_state_batch, next_human_state), dim=0)
                    reward_est[i][j], _ = self.reward_estimator.estimate_reward_on_predictor(
                        tensor_to_joint_state(cur_state), tensor_to_joint_state((next_robot_state, next_human_state)))
            next_state_batch = (next_robot_state_batch, next_human_state_batch)
            if self.planning_depth - depth >= 2 and self.planning_depth > 2:
                cur_width = 1
            else:
                cur_width = int(self.planning_width/2)
            next_values, next_action_indexes, next_trajs = self.V_planning(next_state_batch, depth-1, cur_width)
            next_values = next_values.view(state[0].shape[0], width)
            returns = (reward_est + self.get_normalized_gamma()*next_values + max_action_value) / 2
            if self.exploration_alg == "boltzmann":
                q_dist = torch.softmax(returns/self.epsilon,-1,float)
                max_action_index = torch.stack([torch.multinomial(q_dist_aux, 1) for q_dist_aux in q_dist])
                max_action_return = torch.gather(returns, 1, max_action_index)
            else:
                max_action_return, max_action_index = torch.max(returns, dim=1)
            trajs = []
            max_returns = []
            max_actions = []
            for i in range(robot_state_batch.shape[0]):
                cur_state = (robot_state_batch[i, :, :].unsqueeze(0), human_state_batch[i, :, :].unsqueeze(0))
                action_id = max_action_index[i]
                trajs_id = i * width + action_id
                action = max_action_indexes[i][action_id]
                next_traj = next_trajs[trajs_id]
                trajs.append([(cur_state, action, reward_est)] + next_traj)
                max_returns.append(max_action_return[i].data)
                max_actions.append(action)
            max_returns = torch.tensor(max_returns)
            return max_returns, max_actions, trajs

    def compute_next_robot_state(self, robot_state, action):
        if robot_state.shape[0] != 1:
            raise NotImplementedError
        next_state = robot_state.clone().squeeze()
        if self.kinematics == 'holonomic':
            next_state[0] = next_state[0] + action.vx * self.time_step
            next_state[1] = next_state[1] + action.vy * self.time_step
            next_state[2] = action.vx
            next_state[3] = action.vy
        else:
            next_state[8] = (next_state[8] + action.r) % (2 * np.pi)
            next_state[0] = next_state[0] + torch.cos(next_state[8]) * action.v * self.time_step
            next_state[1] = next_state[1] + torch.sin(next_state[8]) * action.v * self.time_step
            next_state[2] = torch.cos(next_state[8]) * action.v
            next_state[3] = torch.sin(next_state[8]) * action.v
        return next_state.unsqueeze(0).unsqueeze(0)

    def generate_simulated_trajectory(self, robot_state_batch, human_state_batch, action_batch, next_human_state_batch):
        # next_state = robot_state.clone()
        # action_list = []
        # if self.kinematics == 'holonomic':
        #     for i in range(next_state.shape[0]):
        #         action = self.action_space[action_index[i]]
        #         action_list.append([action.vx, action.vy])
        #     action_tensor = torch.tensor(action_list)
        #     next_state[:, :, 0:2] = next_state[:, :, 0:2] + action_tensor * self.time_step
        #     next_state[:, :, 2:4] = action_tensor
        # else:
        #     for i in range(next_state.shape[0]):
        #         action = self.action_space[action_index[i]]
        #         action_list.append([action.v, action.r])
        #     action_tensor = torch.tensor(action_list)
        #     next_state[:, :, 8] = (next_state[:, :, 8] + action_tensor[:, 1]) % (2 * np.pi)
        #     next_state[:, :, 2] = np.cos(next_state[:, :, 8]) * action_tensor[:, 0]
        #     next_state[:, :, 3] = np.sin(next_state[:, :, 8]) * action_tensor[:, 0]
        #     next_state[:, :, 0:2] = next_state[:, :, 0:2] + next_state[:, :, 2:4] * self.time_step
        # return next_state
        expand_next_robot_state = None
        expand_reward = []
        expand_done = []
        for i in range(robot_state_batch.shape[0]):
            action = self.action_space[action_batch[i]]
            cur_robot_state = robot_state_batch[i, :, :]
            cur_human_state = human_state_batch[i, :, :]
            cur_state = tensor_to_joint_state((cur_robot_state, cur_human_state))
            next_robot_state = self.compute_next_robot_state(cur_robot_state, action)
            next_human_state = next_human_state_batch[i, :, :]
            next_state = tensor_to_joint_state((next_robot_state, next_human_state))
            reward, info = self.reward_estimator.estimate_reward_on_predictor(cur_state, next_state)
            expand_reward.append(reward)
            done = False
            if info is ReachGoal() or info is Collision():
                done = True
            expand_done.append(done)
            if expand_next_robot_state is None:
                expand_next_robot_state = next_robot_state
            else:
                expand_next_robot_state = torch.cat((expand_next_robot_state, next_robot_state), dim=0)
            # expand_next_robot_state.append(next_robot_state)
        # expand_next_robot_state = torch.Tensor(expand_next_robot_state)
        expand_reward = torch.Tensor(expand_reward).unsqueeze(dim=1)
        expand_done = torch.Tensor(expand_done).unsqueeze(dim=1)
        return expand_next_robot_state, expand_reward, expand_done

    def get_attention_weights(self):
        return self.value_estimator.graph_model.attention_weights