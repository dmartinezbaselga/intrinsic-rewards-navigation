from genericpath import exists
import os
import yaml
import numpy as np

class BaseConfig(object):
    def __init__(self):
        pass


class Config(object):
    def __init__(self, model) -> None:

        # environment settings
        self.env = BaseConfig()
        self.env.env_name = 'CrowdSimDict-v0'  # name of the environment
        self.env.time_limit = 50 # time limit of each episode (second)
        self.env.time_step = 0.1 # length of each timestep/control frequency (second)
        self.env.val_size = 100
        self.env.test_size = 20 # number of episodes for test.py
        self.env.randomize_attributes = True # randomize the preferred velocity and radius of humans or not
        self.env.seed = 0  # random seed for environment

        # reward function
        self.reward = BaseConfig()
        self.reward.success_reward = 10
        self.reward.collision_penalty = -20
        # discomfort distance for the front half of the robot
        self.reward.discomfort_dist_front = 0.25
        # discomfort distance for the back half of the robot
        self.reward.discomfort_dist_back = 0.25
        self.reward.discomfort_penalty_factor = 10
        self.reward.gamma = 0.99  # discount factor for rewards

        # environment settings
        self.sim = BaseConfig()
        self.sim.render = True # show GUI for visualization
        self.sim.circle_radius = 8 # radius of the circle where all humans start on
        self.sim.human_num = 1 # total number of humans
        # Group environment: set to true; FoV environment: false
        self.sim.group_human = False

        # human settings
        self.humans = BaseConfig()
        self.humans.visible = True # a human is visible to other humans and the robot
        # policy to control the humans: orca or social_force
        self.humans.policy = "orca"
        self.humans.radius = 0.2 # radius of each human
        self.humans.v_pref = 0.8 # max velocity of each human
        # FOV = this values * PI
        self.humans.FOV = 2.

        # a human may change its goal before it reaches its old goal
        self.humans.random_goal_changing = False
        self.humans.goal_change_chance = 0.25

        # a human may change its goal after it reaches its old goal
        self.humans.end_goal_changing = False
        self.humans.end_goal_change_chance = 1.0

        # a human may change its radius and/or v_pref after it reaches its current goal
        self.humans.random_radii = False
        self.humans.random_v_pref = False

        # one human may have a random chance to be blind to other agents at every time step
        self.humans.random_unobservability = False
        self.humans.unobservable_chance = 0.3

        self.humans.random_policy_changing = False

        # config for ORCA
        self.orca = BaseConfig()
        self.orca.neighbor_dist = 10
        self.orca.safety_space = 0.15
        self.orca.time_horizon = 5
        self.orca.time_horizon_obst = 5

        # config for social force
        self.sf = BaseConfig()
        self.sf.A = 2.
        self.sf.B = 1
        self.sf.KI = 1

        # robot settings
        self.robot = BaseConfig()
        self.robot.visible = False  # the robot is visible to humans
        # robot policy: srnn for now
        self.robot.policy = 'vecmpc'
        self.robot.radius = 0.2  # radius of the robot
        self.robot.v_pref = 1.0  # max velocity of the robot
        # robot FOV = this values * PI
        self.robot.FOV = 2.

        # add noise to observation or not
        self.noise = BaseConfig()
        self.noise.add_noise = False
        # uniform, gaussian
        self.noise.type = "uniform"
        self.noise.magnitude = 0.1

        # robot action type
        self.action_space = BaseConfig()
        # holonomic or unicycle
        self.action_space.kinematics = "holonomic"


        self.model = model
        self.path = os.path.join(os.path.dirname(os.path.realpath(__file__)), "params", f"{model}.yaml")
        if not exists(self.path):
            raise FileNotFoundError
        with open(self.path, "r") as fin:
            self.MPC = yaml.safe_load(fin)
        self.MPC['params']['dt'] = 0.1
        self.MPC['params']['prediction_length'] = 1.2

        self.save_path = "REAL_WORLD"
        self.exp_name = "REAL_WORLD"

        self.test_setting = "circle"

        self.real_human_path = "cv_trajs"
