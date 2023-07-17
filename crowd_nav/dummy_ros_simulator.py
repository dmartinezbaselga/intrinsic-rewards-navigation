#!/usr/bin/env python
import logging
import argparse
import importlib.util
import os
import torch
import numpy as np
import matplotlib.pyplot as plt
import gym
import math
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
from crowd_sim.envs.utils.action import ActionRot
from visualization_msgs.msg import Marker, MarkerArray
from tf.transformations import quaternion_from_euler

class Robot:
    def __init__(self, x, y, theta) -> None:
        self.theta = theta
        self.x = x
        self.y = y
        self.v = 0.
        self.w = 0.
    
class Obstacle:
    def __init__(self, x, y, max_x, max_y, vx, vy, radius) -> None:
        self.x = x
        self.y = y
        self.vx = vx
        self.vy = vy    
        self.max_x = max_x
        self.max_y = max_y
        self.radius = radius

class Simulator:
    def __init__(self, obstacles, robot: Robot):
        rospy.init_node('simulator', anonymous=True)
        self.robot = robot
        self.obstacles = obstacles
        self.dt = 0.1
        self.obstacle_marker = MarkerArray()
        for i in range(len(obstacles)):
            marker = Marker()
            marker.header.frame_id = "map"
            marker.header.stamp = rospy.Time.now()

            # set shape, Arrow: 0; Cube: 1 ; Sphere: 2 ; Cylinder: 3
            marker.type = 3
            marker.id = i

            # Set the scale of the marker
            marker.scale.x = 0.4
            marker.scale.y = 0.4
            marker.scale.z = 1.0

            # Set the color
            marker.color.r = 0.0
            marker.color.g = 1.0
            marker.color.b = 0.0
            marker.color.a = 1.0

            self.obstacle_marker.markers.append(marker)
        self.robot_marker = Marker()
        self.robot_marker.header.frame_id = "map"
        self.robot_marker.header.stamp = rospy.Time.now()

        # set shape, Arrow: 0; Cube: 1 ; Sphere: 2 ; Cylinder: 3
        self.robot_marker.type = 3
        self.robot_marker.id = 0

        # Set the scale of the self.robot_marker
        self.robot_marker.scale.x = 0.4
        self.robot_marker.scale.y = 0.4
        self.robot_marker.scale.z = 1.0

        # Set the color
        self.robot_marker.color.r = 1.0
        self.robot_marker.color.g = 0.0
        self.robot_marker.color.b = 0.0
        self.robot_marker.color.a = 1.0        

    def start(self):
        self.pub_obs = rospy.Publisher("obstacles", lmpcc_obstacle_array, queue_size=10)
        # self.sub_goal = rospy.Subscriber("goal", Pose, self.goal_callback)
        self.pub_state = rospy.Publisher("/Robot_1/pose", PoseStamped, queue_size=10)
        self.pub_robot_marker = rospy.Publisher("robot_marker", Marker, queue_size=10)
        self.pub_obs_marker = rospy.Publisher("obs_marker", MarkerArray, queue_size=10)
        # self.human_vel_pub = rospy.Publisher('human_vel_cmd', VelInfo, queue_size=10)
        self.vel_sub = rospy.Subscriber('/Robot_1/cmd_vel', Twist, self.vel_callback)
        rospy.Timer(rospy.Duration(self.dt/2.0), self.simulate)
        rospy.spin()
    
    def vel_callback(self, vel: Twist):
        self.robot.v = vel.linear.x
        self.robot.w = vel.angular.z

    def simulate(self, event):
        stamp = rospy.Time.now()
        print("v: ", self.robot.v, ". w: ", self.robot.w, ". theta: ", self.robot.theta)
        self.robot.x = self.robot.x + self.robot.v * math.cos(self.robot.theta) * self.dt
        self.robot.y = self.robot.y + self.robot.v * math.sin(self.robot.theta) * self.dt
        self.robot.theta = self.robot.theta + self.robot.w * self.dt
        for o in self.obstacles:
            if abs(o.x) >= abs(o.max_x):
                o.vx = -o.vx
            if abs(o.y) >= abs(o.max_y):
                o.vy = -o.vy
            o.x = o.x + o.vx * self.dt
            o.y = o.y + o.vy * self.dt
        obs_msg = lmpcc_obstacle_array()
        for o in self.obstacles:
            obs = lmpcc_obstacle()
            obs.pose.position.x = o.x
            obs.pose.position.y = o.y
            obs.velocity.linear.x = o.vx
            obs.velocity.linear.y = o.vy
            obs_msg.lmpcc_obstacles.append(obs)
        obs_msg.header.stamp = stamp
        self.pub_obs.publish(obs_msg)
        state_msg = PoseStamped()
        state_msg.pose.position.x = self.robot.x
        state_msg.pose.position.y = self.robot.y
        q = quaternion_from_euler(0,0,self.robot.theta)
        state_msg.pose.orientation.x = q[0]
        state_msg.pose.orientation.y = q[1]
        state_msg.pose.orientation.z = q[2]
        state_msg.pose.orientation.w = q[3]   
        state_msg.header.stamp = stamp    
        self.pub_state.publish(state_msg)
        self.robot_marker.pose.position.x = self.robot.x
        self.robot_marker.pose.position.y = self.robot.y
        self.pub_robot_marker.publish(self.robot_marker)
        for i in range(len(self.obstacles)):
            self.obstacle_marker.markers[i].pose.position.x = self.obstacles[i].x
            self.obstacle_marker.markers[i].pose.position.y = self.obstacles[i].y
        self.pub_obs_marker.publish(self.obstacle_marker)

if __name__ == '__main__':
    robot = Robot(x=0., y=0., theta=0.)
    obs1 = Obstacle(1.0, 1.0, 2.0, 2.0, 0.25, 0.25, 0.2)
    obs2 = Obstacle(-1.0, 1.0, 4.0, 4.0, -0.5, 0.5, 0.2)
    simulator = Simulator(obstacles=[obs1, obs2], robot=robot)
    simulator.start()
