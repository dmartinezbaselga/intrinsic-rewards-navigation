# import rospy
import numpy as np
import copy
# import rospy
from .cv import CV
from filterpy.kalman import KalmanFilter


class CVKF(CV):
    def __init__(self):
        super(CVKF, self).__init__()
        self.dt = None
        self.prediction_horizon = None
        self.filters = None
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

        self.num_samples = params['predictor']['num_samples']

    def get_kf(self):
        kf = KalmanFilter(dim_x=4, dim_z=4)
        # kf.P *= 100.0
        kf.H = np.eye(4)
        kf.F = np.eye(4)
        temp = np.zeros((4, 4))
        temp[0, 2] = self.dt
        temp[1, 3] = self.dt
        kf.B = temp
        return kf

    def reset(self):
        self.filters = None

    def unroll_kf(self, kf, s):
        kf_ = copy.deepcopy(kf)
        trajectory = []
        for _ in range(self.rollout_steps):
            kf_.predict()
            # rospy.loginfo("{} {} {}\n\n\n\n\n".format(kf_.x.shape, np.linalg.cholesky(kf_.P).shape, (self.num_samples, 4)))
            trajectory.append(np.random.multivariate_normal(kf_.x.squeeze(), kf_.P, size=self.num_samples))
        return np.stack(trajectory, axis=1)

    def get_predictions(self, trajectory, actions):
        if self.filters is None:
            self.filters = [self.get_kf()
                            for _ in range(trajectory.shape[1]-1)]
        predictions = []
        for kf, s in zip(self.filters, trajectory[-1, 1:, :4]):
            kf.predict()
            kf.update(s)
            predictions.append(self.unroll_kf(kf, s)) # S x T' x 4
        
        predictions = np.stack(predictions, axis=2)[None] # N x S x T' x H x 4
        return predictions

    # # (T x (1+H) x 5), ((1+H) x 5), (N x T' x 2), (2, )
    # def predict(self, trajectory, state, actions, goal):
    #     predictions = self.get_predictions(
    #         trajectory, actions)  # N x S x T' x H x 4

    #     c_goal = self.goal_cost(state, actions, goal)
    #     c_obs = self.obstacle_cost(state, actions, predictions)
    #     c_total = c_goal + c_obs

    #     best_action = actions[np.argmin(np.mean(c_total, axis=-1))][0]
    #     return predictions, c_total, actions, best_action
