import numpy as np                                                                                                                                                                         # import rospy
# import rospy
class CV(object):
    def __init__(self):
        super(CV, self).__init__()
        self.dt = None
        self.prediction_horizon = None
        self.wrap = np.vectorize(self._wrap)
    
    def set_params(self, params):
        self.dt = params['dt']
        self.prediction_horizon = params['prediction_horizon']
        self.rollout_steps = int(np.ceil(self.prediction_horizon / self.dt))
        self.prediction_length = int(np.ceil(self.prediction_horizon / self.dt)) + 1
        self.history_length = params['history_length']

        self.q_obs = params['cost']['q']['obs']
        self.q_goal = params['cost']['q']['goal']
        self.q_wind = params['cost']['q']['wind']

        self.sigma_h = params['cost']['sigma']['h']
        self.sigma_s = params['cost']['sigma']['s']
        self.sigma_r = params['cost']['sigma']['r']

        # Normalization factors for Q weights
        self.q_goal_norm = np.square(2 / float(self.prediction_horizon))

        # # Empirically found
        q_wind_norm = 0.1 * np.deg2rad(350)
        self.q_wind_norm = np.square(q_wind_norm)
        self.q_obs_norm = np.square(0.5)

        # Normalized weights
        self.Q_obs = self.q_obs / self.q_obs_norm
        self.Q_goal = self.q_goal / self.q_goal_norm
        self.Q_discrete = self.q_wind / self.q_wind_norm
        self.Q_dev = 0

        self.log_cost = params['log_cost']
        self.discrete_cost_type = params['cost']['discrete_cost_type']
    
    def reset(self):
        pass

    def get_predictions(self, trajectory, actions):
        velocity = trajectory[None, -1, 1:, 2:4]  # 1 x H x 2
        init_pos = trajectory[None, -1, 1:, 0:2]  # 1 x H x 2
        steps = 1 + np.arange(self.prediction_length, dtype=np.float32)[:, None, None]  # T' x 1 x 1
 #       rospy.loginfo("1: {}".format(steps.shape))
        steps = np.multiply(velocity, steps) * self.dt  # T' x H x 2
        steps = (init_pos + steps)[None, None] # N x S x T' x H x 2
        steps = np.concatenate((steps[:, :, :-1], self.predict_velocity(steps)), axis=-1) # N x S x T' x H x 4
#        rospy.loginfo("{} {}\n\n".format(steps.shape, self.prediction_length))
        return steps
    
    def predict_velocity(self, steps):
        return (steps[:, :, 1:]-steps[:, :, :-1])/self.dt

    def predict(self, trajectory, state, actions, goal): # (T x (1+H) x 5), ((1+H) x 5), (N x T' x 2), (2, )
        predictions = self.get_predictions(trajectory, actions) # N x S x T' x H x 4

        c_goal = self.goal_cost(state, actions, goal)
        c_obs = self.obstacle_cost(state, actions, predictions)
        c_discrete = self.discrete_cost(state, actions, predictions)
        c_predictor = self.predictor_cost(state, actions, predictions)
        c_total = c_goal + c_obs + c_discrete + c_predictor
        
        best_action_idx = np.argmin(np.mean(c_total, axis=-1))
        best_action = actions[best_action_idx][0]

        # if self.log_cost:
        #     rospy.loginfo("type: \t\t goal \t\t obs \t\t discrete \t\t dev")
        #     rospy.loginfo("type: \t {} \t {} \t {} \t {}".format(self.Q_goal, self.Q_obs, self.Q_discrete, self.Q_dev))
        #     rospy.loginfo("type: \t {} \t {} \t {} \t {}\n\n".format(c_goal.mean(), c_obs.mean(), c_discrete.mean(), c_predictor.mean()))
    
        return predictions, c_total, actions, best_action 

    def predictor_cost(self, state, actions, predictions):
        return np.array([0.0])

    def goal_cost(self, state, actions, goal):
        """Ratio of extra distance that needs to be travelled towards the goal"""
        init_dist = np.linalg.norm(goal-state[0, :2])
        dist = np.linalg.norm(goal[None, None]-actions[:, :, :2], axis=-1) # (N x T')
        st_dist = np.clip(self.vpref * self.dt * np.arange(1, self.prediction_length), 0, init_dist)
        opt_dist = init_dist - st_dist
        cost = np.sum((dist - opt_dist) / (2 * st_dist), axis=-1)[:, None]
        return self.Q_goal * (cost ** 2) # (N, 1)

    def obstacle_cost(self, state, actions, predictions):
        """
        Cost using 2D Gaussian around obstacles
        """
        # Distance to other agents
  #      rospy.loginfo("act: {} pred: {} state: {}\n\n".format(actions.shape, predictions.shape, state.shape))
        dx = actions[:, None, :, None, 0] - predictions[:, :, :, :, 0] - (state[None, None, None, 1:, 4] + state[0, 4]) # N x S x T' x H
        dy = actions[:, None, :, None, 1] - predictions[:, :, :, :, 1] - (state[None, None, None, 1:, 4] + state[0, 4]) # N x S x T' x H
                                                                                                                                                                                                            # rospy.loginfo(" dx:{} dy:{}".format(dx.shape, dy.shape))
        # Heading of "other agent"
        obs_theta = np.arctan2(predictions[:, :, :, :, 3], predictions[:, :, :, :, 2]) # N x S x T' x H
        # Checking for static obstacles
        static_obs = (np.linalg.norm(predictions[:, :, :, :, 2:4], axis=-1) < 0.01) # N x S x T' x H
        # Alpha calculates whether ego agent is in front or behind "other agent"
        alpha = self.wrap(np.arctan2(dy, dx) - obs_theta + np.pi/2.0) <= 0 # N x S x T' x H
                                                                                                                                                                                                            # rospy.loginfo(" obs_theta:{} static_obs:{} alpha:{}".format(obs_theta.shape, static_obs.shape, alpha.shape))

        # Sigma values used to create 2D gaussian around obstacles for cost penalty
        sigma = np.where(alpha, self.sigma_r, self.sigma_h)
        sigma = static_obs + np.multiply(1-static_obs, sigma) # N x S x T' x H
        sigma_s = 1.0 * static_obs + self.sigma_s * (1 - static_obs) # N x S x T' x H
                                                                                                                                                                                                            # rospy.loginfo("s:{} ss:{}".format(sigma.shape, sigma_s.shape))

        # Variables used in cost_obs function based on sigma and obs_theta
        a = np.cos(obs_theta) ** 2 / (2 * sigma ** 2) + np.sin(obs_theta) ** 2 / (2 * sigma_s ** 2)
        b = np.sin(2 * obs_theta) / (4 * sigma ** 2) - np.sin(2 * obs_theta) / (4 * sigma_s ** 2)
        c = np.sin(obs_theta) ** 2 / (2 * sigma ** 2) + np.cos(obs_theta) ** 2 / (2 * sigma_s ** 2)

        cost = np.exp(-((a * dx ** 2) + (2 * b * dx * dy) +  (c * dy ** 2))) # N x S x T' x H
        cost = np.mean(cost, axis=3)
        cost = np.sum(cost, axis=-1)
                                                                                                                                                                                                            # rospy.loginfo("c: {}\n\n".format(cost.shape))
        return self.Q_obs * (cost ** 2) # (N, S)
    
    def discrete_cost(self, state, actions, predictions): # (1+H) x 5, N x T' x 4, N x S x T' x H x 4
        N = actions.shape[0]
        S = predictions.shape[1]
        state_ = np.tile(state[None, None, None, None, 0, :2]-state[None, None, None, 1:, :2], (N, S, 1, 1, 1))
        dxdy = np.concatenate((state_, actions[:, None, :, None, :2] - predictions[:, :, :, :, :2]), axis=2)
        winding_nums = np.arctan2(dxdy[:, :, :, :, 1], dxdy[:, :, :, :, 0]) # N x S x T' x H
        winding_nums = winding_nums[:, :, 1:]-winding_nums[:, :, :-1]

        if self.discrete_cost_type == 'entropy':
            winding_nums = np.mean(winding_nums, axis=2) < 0 # N x S x H
            p = np.mean(winding_nums, axis=1) # N x H
            # Using mean entropy
            entropy = - (p * np.log(p+1e-8) + (1-p) * np.log(1-p+1e-8))
            entropy = np.mean(entropy, axis=1)[:, None]

            return self.Q_discrete * (entropy ** 2)
        else:
            winding_nums = np.abs(np.mean(winding_nums, axis=2)) # N x S x H

            # considering all agents we are in front of
            dxdy = state[None, 0, :2] - state[1:, :2]
            obs_theta = np.arctan2(state[1:, 3], state[1:, 2])
            alpha = self.wrap(np.arctan2(dxdy[:, 1], dxdy[:, 0]) - obs_theta + np.pi/2.0) >= 0 # N x S x H
            winding_nums = np.multiply(winding_nums, alpha)
            
            winding_nums = np.multiply(winding_nums, alpha)
            winding_nums = np.mean(winding_nums, axis=-1) # N x S

            return - self.Q_discrete * (winding_nums ** 2)
        

    @staticmethod
    def _wrap(angle):  # keep angle between [-pi, pi]
        while angle >= np.pi:
            angle -= 2 * np.pi
        while angle < -np.pi:
            angle += 2 * np.pi
        return angle
