from crowd_nav.configs.icra_benchmark.config import BaseEnvConfig, BasePolicyConfig, BaseTrainConfig, Config


class EnvConfig(BaseEnvConfig):
    def __init__(self, debug=False):
        super(EnvConfig, self).__init__(debug)


class PolicyConfig(BasePolicyConfig):
    def __init__(self, debug=False):
        super(PolicyConfig, self).__init__(debug)
        self.name = 'tree_search_rl'

        # gcn
        self.gcn.num_layer = 1
        self.gcn.X_dim = 32
        self.gcn.similarity_function = 'concatenation'
        self.gcn.layerwise_graph = False
        self.gcn.skip_connection = True

        self.model_predictive_rl = Config()
        self.model_predictive_rl.linear_state_predictor = False
        self.model_predictive_rl.planning_depth = 1
        self.model_predictive_rl.planning_width = 10
        self.model_predictive_rl.do_action_clip = False
        self.model_predictive_rl.motion_predictor_dims = [64, 5]
        self.model_predictive_rl.value_network_dims = [32, 100, 100, 1]
        self.model_predictive_rl.share_graph_model = False
        self.use_noisy_net = False


class TrainConfig(BaseTrainConfig):
    def __init__(self, debug=False):
        super(TrainConfig, self).__init__(debug)

        self.train.freeze_state_predictor = False
        self.train.detach_state_predictor = False
        self.train.reduce_sp_update_frequency = False
        # We reuse the same variable for epsilon and for boltzmann temperature
        self.train.epsilon_start = 1.0
        self.train.epsilon_end = 0.01
        self.train.epsilon_decay = 4000
        self.train.exploration_alg = "boltzmann"
