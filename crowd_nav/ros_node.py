#!/usr/bin/env python
import logging
import argparse
import importlib.util
import os
import torch
import numpy as np
import matplotlib.pyplot as plt
import gym
from math import atan2, sqrt, pi, cos, sin
from crowd_nav.utils.explorer import Explorer
from crowd_nav.policy.policy_factory import policy_factory
from crowd_sim.envs.utils.robot import Robot
from crowd_sim.envs.policy.orca import ORCA
from crowd_sim.envs.policy.socialforce import SocialForce
from crowd_nav.policy.reward_estimate import Reward_Estimator
import rospy
from crowd_sim.envs.utils.state import ObservableState, FullState, JointState
from geometry_msgs.msg import Twist, PoseStamped
from lmpcc_msgs.msg import lmpcc_obstacle_array, lmpcc_obstacle
from nav_msgs.msg import Odometry
from crowd_sim.envs.utils.action import ActionRot, ActionXY
from tf.transformations import euler_from_quaternion


class baseline_planner:
    def __init__(self, policy_name):
        self.robot_policy = None
        self.cur_state = None
        rospy.init_node('baseline_planner_node', anonymous=True)
        self.robot_policy = policy_factory[policy_name]()
        self.peds_full_state = []
        self.obstacle_radius = 0.2
        self.time_step = 0.25
        self.robot_full_state = FullState(0,0,0,0,0,0,0,0,0)
        self.robot_full_state.v_pref = 1.0
        self.robot_full_state.radius = 0.2
        self.current_goal = False

    def start(self):
        self.sub_obs = rospy.Subscriber("obstacles", lmpcc_obstacle_array, self.obstacles_callback)
        self.sub_goal = rospy.Subscriber("/roadmap/goal", PoseStamped, self.goal_callback)
        self.sub_state = rospy.Subscriber("/Robot_1/pose", PoseStamped, self.state_callback)
        # self.human_vel_pub = rospy.Publisher('human_vel_cmd', VelInfo, queue_size=10)
        self.robot_action_pub = rospy.Publisher('/Robot_1/cmd_vel', Twist, queue_size=10)
        print("Ros node started")
        rospy.spin()

    def load_policy_model(self, args):
        if not isinstance(self.robot_policy, SocialForce) and not isinstance(self.robot_policy, ORCA):
            device = torch.device("cuda:0" if torch.cuda.is_available() and args.gpu else "cpu")

            if args.model_dir is not None:
                if args.config is not None:
                    config_file = args.config
                else:
                    config_file = os.path.join(args.model_dir, 'config.py')
                if args.il:
                    model_weights = os.path.join(args.model_dir, 'il_model.pth')
                elif args.rl:
                    if os.path.exists(os.path.join(args.model_dir, 'resumed_rl_model.pth')):
                        model_weights = os.path.join(args.model_dir, 'resumed_rl_model.pth')
                    else:
                        print(os.listdir(args.model_dir))
                        model_weights = os.path.join(args.model_dir, sorted(os.listdir(args.model_dir))[-1])
                else:
                    model_weights = os.path.join(args.model_dir, 'best_val.pth')

            else:
                config_file = args.config

            spec = importlib.util.spec_from_file_location('config', config_file)
            if spec is None:
                parser.error('Config file not found.')
            config = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(config)

            # configure policy
            policy_config = config.PolicyConfig(False)
            policy = policy_factory[policy_config.name]()
            reward_estimator = Reward_Estimator()
            env_config = config.EnvConfig(False)
            reward_estimator.configure(env_config)
            policy.reward_estimator = reward_estimator
            if args.planning_depth is not None:
                policy_config.model_predictive_rl.do_action_clip = True
                policy_config.model_predictive_rl.planning_depth = args.planning_depth
            if args.planning_width is not None:
                policy_config.model_predictive_rl.do_action_clip = True
                policy_config.model_predictive_rl.planning_width = args.planning_width
            if args.sparse_search:
                policy_config.model_predictive_rl.sparse_search = True

            policy.configure(policy_config, device)
            if policy.trainable:
                if args.model_dir is None:
                    parser.error('Trainable policy must be specified with a model weights directory')
                policy.load_model(model_weights)

            # for continous action
            action_dim = 2
            max_action = [1., 0.5235988]
            min_action = [0., -0.5235988]
            if policy.name == 'TD3RL':
                policy.set_action(action_dim, max_action, min_action)
            self.robot_policy = policy
            policy.set_v_pref(1.0)
            self.robot_policy.set_time_step(self.time_step)
            if not isinstance(self.robot_policy, ORCA) and not isinstance(self.robot_policy, SocialForce):
                self.robot_policy.set_epsilon(0.01)
            policy.set_phase("test")
            policy.set_device(device)

            # set safety space for ORCA in non-cooperative simulation
            if isinstance(self.robot_policy, ORCA):
                self.robot_policy.safety_space = args.safety_space
        else:
            self.robot_policy.time_step = self.time_step
        print("policy loaded")

    def obstacles_callback(self, obstacles: lmpcc_obstacle_array):
        self.peds_full_state.clear()
        for o in obstacles.lmpcc_obstacles:
            self.peds_full_state.append(ObservableState(px=o.pose.position.x, py=o.pose.position.y,
                                                        vx=o.velocity.linear.x, vy=o.velocity.linear.y,
                                                        radius=0.2))
    
    def goal_callback(self, goal: PoseStamped):
        self.robot_full_state.gx = goal.pose.position.x
        self.robot_full_state.gy = goal.pose.position.y
        print(self.robot_full_state.gx , self.robot_full_state.gy)
        self.current_goal = True

    def state_callback(self, robot_state: PoseStamped):
        if self.current_goal:
            self.robot_full_state.px = robot_state.pose.position.x
            self.robot_full_state.py = robot_state.pose.position.y
            print(self.robot_full_state.px, self.robot_full_state.py)
            _, _, theta = euler_from_quaternion([robot_state.pose.orientation.x, robot_state.pose.orientation.y,
                                                 robot_state.pose.orientation.z, robot_state.pose.orientation.w])
            self.robot_full_state.theta = theta

            self.cur_state = JointState(self.robot_full_state, self.peds_full_state)
            action_cmd = Twist()

            dis = np.sqrt((self.robot_full_state.px - self.robot_full_state.gx)**2 + (self.robot_full_state.py - self.robot_full_state.gy)**2)
            if dis < 0.3:
                action_cmd.linear.x = 0.0
                action_cmd.linear.y = 0.0
                action_cmd.angular.z = 0.0
                self.current_goal = None
            else:
                robot_action = self.robot_policy.predict(self.cur_state)
                print('robot_action', robot_action)

                if isinstance(robot_action, ActionXY):
                    action_cmd.angular.z = min(pi/1.5, max(-pi/1.5, (atan2(robot_action.vy, robot_action.vx)%(2*pi) - self.robot_full_state.theta%(2*pi))/self.time_step))
                    if abs(action_cmd.angular.z) > pi/2:
                        action_cmd.linear.x = min(0.2, sqrt(robot_action.vy**2 + robot_action.vx**2))
                    else:    
                        action_cmd.linear.x = sqrt(robot_action.vy**2 + robot_action.vx**2)
                else:
                    robot_action = robot_action[0]
                    action_cmd.angular.z = min(pi/1.5, max(-pi/1.5, robot_action.r/self.time_step))
                    if abs(action_cmd.angular.z) > pi/2:
                        action_cmd.linear.x = min(0.2, robot_action.r/self.time_step)
                    else:    
                        action_cmd.linear.x = robot_action.v
                angle = action_cmd.angular.z * self.time_step + self.robot_full_state.theta
                self.robot_full_state.vx = action_cmd.linear.x * cos(angle)
                self.robot_full_state.vy = action_cmd.linear.x * sin(angle)
            self.robot_action_pub.publish(action_cmd)


if __name__ == '__main__':
    parser = argparse.ArgumentParser('Parse configuration file')
    parser.add_argument('--config', type=str, default=None)
    parser.add_argument('--policy', type=str, default='tree_search_rl')
    parser.add_argument('-m', '--model_dir', type=str, default='data/tsrl5rot/tsrl/1')#None
    parser.add_argument('--il', default=False, action='store_true')
    parser.add_argument('--rl', default=False, action='store_true')
    parser.add_argument('--gpu', default=False, action='store_true')
    parser.add_argument('--safety_space', type=float, default=0.2)
    parser.add_argument('-d', '--planning_depth', type=int, default=None)
    parser.add_argument('-w', '--planning_width', type=int, default=None)
    parser.add_argument('--sparse_search', default=False, action='store_true')
    sys_args = parser.parse_args()
    try:
        planner = baseline_planner(sys_args.policy)
        planner.load_policy_model(sys_args)
        planner.start()
    except rospy.ROSException:
        pass
