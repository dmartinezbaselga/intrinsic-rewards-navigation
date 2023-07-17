import struct
import random
import rospy
from geometry_msgs.msg import PoseStamped, Twist
from nav_msgs.msg import Path
from tf.transformations import euler_from_quaternion
import math
import argparse
import numpy as np

MIN_Y = -1.0
MAX_Y = 8.0
MAX_X = 5.0
MIN_X = -5.0

class Goal:
    def __init__(self, x=0., y=0.) -> None:
        self.x = x
        self.y = y

def generate_random_goals(n_goals):
    with open('random_goals.bin', 'wb') as f:
        for _ in range(n_goals):

            is_x = random.randint(0, 1)
            is_negative = random.randint(0, 1)
            print(is_x)

            if is_x:
                x = random.random() * (MAX_X - MIN_X) + MIN_X
                if is_negative:
                    y = MIN_Y + 0.5
                else:
                    y = MAX_Y - 0.5
            else:
                y = random.random() * (MAX_Y - MIN_Y) + MIN_Y
                if is_negative:
                    x = MIN_X + 0.5
                else:
                    x = MAX_X - 0.5
            # x = random.random()*(MAX_X-MIN_X) + MIN_X
            # y = random.random()*(MAX_Y-MIN_Y) + MIN_Y
            data = struct.pack('d', x)
            f.write(data)
            data = struct.pack('d', y)
            f.write(data)
            print(x, y)

class GoalsPublisher:
    def __init__(self, random_goals, max_goals) -> None:
        rospy.init_node('goal_publisher')
        self.max_goals = max_goals
        self.num_goals_reached = 0
        self.random_file_name = 'random_goals.bin'
        self.random_goals = random_goals

        self.robot_pose_sub_ = rospy.Subscriber('/Robot_1/pose', PoseStamped, self.robot_pose_callback)
        self.goal_pub_ = rospy.Publisher('/roadmap/goal', PoseStamped, queue_size=10)
        self.ref_path_pub_ = rospy.Publisher('/roadmap/reference', Path, queue_size=10)
        self.vel_pub_ = rospy.Publisher('/Robot_1/cmd_vel', Twist, queue_size=10)

        self.last_pub_stamp_ = rospy.Time.now()

        self.min_dist_goal = 0.3
        self.aligning_to_goal = False
        if random_goals:
            self.goals_file = open('random_goals.bin', 'rb')
            self.current_goal = None
        rospy.spin()
    
    def read_goal(self):
        data = self.goals_file.read(8)
        goal = Goal()
        goal.x = struct.unpack('d', data)[0]
        data = self.goals_file.read(8)
        goal.y = struct.unpack('d', data)[0]
        print(goal.x, goal.y)
        return goal

    def publish_current_goal(self):
        msg = PoseStamped()
        msg.pose.position.x = self.current_goal.x
        msg.pose.position.y = self.current_goal.y
        msg.header.stamp = rospy.Time.now()
        msg.header.frame_id = 'map'
        self.goal_pub_.publish(msg)

    def construct_reference_path(self, msg):

        robot_x = msg.pose.position.x
        robot_y = msg.pose.position.y

        path = Path()

        # Add the start
        new_g = PoseStamped()
        new_g.pose.position.x = robot_x
        new_g.pose.position.y = robot_y
        new_g.header.stamp = msg.header.stamp
        new_g.header.frame_id = 'map'
        path.poses.append(new_g)
        theta = math.atan2(self.current_goal.y - robot_y, self.current_goal.x - robot_x)

        # Add the rest
        while math.sqrt((path.poses[-1].pose.position.x - self.current_goal.x)**2 + (path.poses[-1].pose.position.y - self.current_goal.y)**2) > 2.0:
            new_g = PoseStamped()
            new_g.pose.position.x = path.poses[-1].pose.position.x + 2. * math.cos(theta)
            new_g.pose.position.y = path.poses[-1].pose.position.y + 2. * math.sin(theta)
            new_g.header.stamp = msg.header.stamp
            new_g.header.frame_id = 'map'
            path.poses.append(new_g)

        # Add the goal
        new_g = PoseStamped()
        new_g.pose.position.x = self.current_goal.x
        new_g.pose.position.y = self.current_goal.y
        new_g.header.stamp = msg.header.stamp
        new_g.header.frame_id = 'map'
        path.poses.append(new_g)

        path.header.frame_id = 'map'
        path.header.stamp = msg.header.stamp

        self.reference_path = path

    def publish_path(self):
        # print("GoalPublisher: Publishing reference path")
        # print("-----------")
        # for p in self.reference_path.poses:
        #     print(f"Point (x = {p.pose.position.x}, y = {p.pose.position.y})")
        self.ref_path_pub_.publish(self.reference_path)
    
    def robot_pose_callback(self, msg: PoseStamped):
        # TODO: Filter poses
        if self.current_goal:
            # @Note: Robot radius subtracted
            goal_dist = -0.325 + math.sqrt((msg.pose.position.x - self.current_goal.x)**2 + (msg.pose.position.y - self.current_goal.y)**2)
            # print(f"Goal distance: {goal_dist}")
        if self.current_goal is None:
            self.current_goal = self.read_goal()
            self.publish_current_goal()
            self.construct_reference_path(msg)
            self.publish_path()
            self.last_pub_stamp_ = msg.header.stamp
            goal_dist = -0.325 + math.sqrt((msg.pose.position.x - self.current_goal.x)**2 + (msg.pose.position.y - self.current_goal.y)**2)
        if goal_dist < self.min_dist_goal and not self.aligning_to_goal:
            self.num_goals_reached = self.num_goals_reached + 1
            print("Number of goals reched: ", self.num_goals_reached)
            if self.num_goals_reached == self.max_goals:
                print("Configured number of goals reached! Exiting.")
                rospy.signal_shutdown("Finished successfully")
            else:
                self.aligning_to_goal = True
        if self.aligning_to_goal:
            _, _, theta = euler_from_quaternion([msg.pose.orientation.x, msg.pose.orientation.y, msg.pose.orientation.z, msg.pose.orientation.w])
            angle_to_goal = math.atan2(msg.pose.position.y - self.current_goal.y, msg.pose.position.y - self.current_goal.y)%(2*math.pi) - theta%(2*math.pi)
            if abs(angle_to_goal) < 0.4:
                self.aligning_to_goal = False
                self.current_goal = self.read_goal()
                self.publish_current_goal()
                self.construct_reference_path(msg)
                self.publish_path()
                self.last_pub_stamp_ = msg.header.stamp
            else:
                vel_msg = Twist()
                vel_msg.angular.z = np.sign(angle_to_goal)
                self.vel_pub_.publish(vel_msg)
        elif self.last_pub_stamp_ + rospy.Duration(1. / 20.) <= msg.header.stamp:
            self.publish_path()
            self.publish_current_goal()
            self.last_pub_stamp_ = msg.header.stamp


if __name__ == '__main__':
    parser = argparse.ArgumentParser('Parse configuration file')
    parser.add_argument('--n_goals', type=int, required=True)
    parser.add_argument('--publish', default=False, action='store_true')
    args = parser.parse_args()
    if args.publish:
        goal_publisher = GoalsPublisher(True, args.n_goals)
    else:
        generate_random_goals(args.n_goals)