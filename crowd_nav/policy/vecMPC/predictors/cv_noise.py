import numpy as np                                                                                                                                                                         # import rospy
# import rospy
from .cv import CV

class CVN(CV):
    def __init__(self):
        super(CVN, self).__init__()
    
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

        self.num_samples = params['predictor']['num_samples']
        self.sample_angle_std = params['predictor']['sample_angle_std']

    def get_predictions(self, trajectory, actions):
        velocity = trajectory[None, -1, 1:, 2:4]  # 1 x H x 2
        init_pos = trajectory[None, -1, 1:, 0:2]  # 1 x H x 2
        steps = 1 + np.arange(self.prediction_length, dtype=np.float)[:, None, None]  # T' x 1 x 1
        steps = np.multiply(velocity, steps) * self.dt  # T' x H x 2
        steps = (steps).transpose((1, 2, 0)) # H x 2 x T'
        samples = []
        for _ in range(self.num_samples):
            sampled_angle = np.random.normal(0, self.sample_angle_std, 1)[0]
            theta = (sampled_angle * np.pi)/ 180.
            c, s = np.cos(theta), np.sin(theta)
            rotation_mat = np.array([[c, s],[-s, c]])
            samples.append(init_pos + (np.matmul(rotation_mat, steps)).transpose((2, 0, 1)))
        samples = np.stack(samples, axis=0)[None]# N x S x T' x H x 2
        steps = np.concatenate((samples[:, :, :-1], self.predict_velocity(samples)), axis=-1) # N x S x T' x H x 4
        return steps