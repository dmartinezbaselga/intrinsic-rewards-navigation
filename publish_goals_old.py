import struct
import random
import rospy
from geometry_msgs.msg import PoseStamped
from nav_msgs.msg import Path
import math
import argparse
import time

MIN_Y = 0.0
MAX_Y = 7.0
MAX_X = 4.0
MIN_X = -4.0

class Goal:
    def __init__(self, x=0., y=0.) -> None:
        self.x = x
        self.y = y

def generate_random_goals(n_goals):
    with open('random_goals.bin', 'wb') as f:
        for _ in range(n_goals):
            x = random.random()*(MAX_X-MIN_X) + MIN_X
            y = random.random()*(MAX_Y-MIN_Y) + MIN_Y
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
        self.robot_pose_sub_ = rospy.Subscriber('robot_pose', PoseStamped, self.robot_pose_callback)
        self.goal_pub_ = rospy.Publisher('goal', PoseStamped, queue_size=10)
        self.ref_path_pub_ = rospy.Publisher('ref_path', Path, queue_size=10)
        self.min_dist_goal = 0.1
        if random_goals:
            self.goals_file = open('random_goals.bin', 'rb')
            self.current_goal = None
        rospy.spin()
    
    def read_goal(self, robot_x, robot_y):
        data = self.goals_file.read(8)
        goal = Goal()
        goal.x = struct.unpack('d', data)[0]
        data = self.goals_file.read(8)
        goal.y = struct.unpack('d', data)[0]
        print(goal.x, goal.y)
        self.current_goal = goal
        msg = PoseStamped()
        msg.pose.position.x = goal.x
        msg.pose.position.y = goal.y
        msg.header.stamp = rospy.Time.now()
        msg.header.frame_id = 'map'
        self.goal_pub_.publish(msg)
        theta = math.atan2(goal.y - robot_y, goal.x - robot_x)
        path = Path()
        new_g = PoseStamped()
        new_g.pose.position.x = robot_x
        new_g.pose.position.y = robot_y
        new_g.header.stamp = msg.header.stamp
        new_g.header.frame_id = 'map'
        path.poses.append(new_g)
        while math.sqrt((path.poses[-1].pose.position.x - goal.x)**2 + (path.poses[-1].pose.position.y - goal.y)**2) > 2.0:
            new_g = PoseStamped()
            new_g.pose.position.x = path.poses[-1].pose.position.x + 2. * math.cos(theta)
            new_g.pose.position.y = path.poses[-1].pose.position.y + 2. * math.sin(theta)
            new_g.header.stamp = msg.header.stamp
            new_g.header.frame_id = 'map'
            path.poses.append(new_g)
        new_g = PoseStamped()
        new_g.pose.position.x = goal.x
        new_g.pose.position.y = goal.y
        new_g.header.stamp = msg.header.stamp
        new_g.header.frame_id = 'map'
        path.poses.append(new_g)
        path.header.frame_id = 'map'
        path.header.stamp = msg.header.stamp
        self.ref_path_pub_.publish(path)
    
    def robot_pose_callback(self, msg: PoseStamped):
        # TODO: Filter poses
        if self.current_goal is None or math.sqrt((msg.pose.position.x - self.current_goal.x)**2 + 
            (msg.pose.position.y - self.current_goal.y)**2) < self.min_dist_goal:
            self.num_goals_reached = self.num_goals_reached + 1
            if self.num_goals_reached == self.max_goals:
                exit(0)
            else:
                self.read_goal(msg.pose.position.x, msg.pose.position.y)


if __name__ == '__main__':
    parser = argparse.ArgumentParser('Parse configuration file')
    parser.add_argument('--n_goals', type=int, required=True)
    parser.add_argument('--publish', default=False, action='store_true')
    args = parser.parse_args()
    if args.publish:
        goal_publisher = GoalsPublisher(True, args.n_goals)
    else:
        generate_random_goals(args.n_goals)