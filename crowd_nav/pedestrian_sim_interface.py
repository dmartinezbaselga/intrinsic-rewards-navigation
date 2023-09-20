import rospy
from sensor_msgs.msg import Joy
from nav_msgs.msg import Odometry
from geometry_msgs.msg import PoseStamped, Pose
from std_msgs.msg import Float64MultiArray, ColorRGBA
import numpy as np
from visualization_msgs.msg import Marker
from tf.transformations import quaternion_from_euler
import ros_visuals
import math
from derived_object_msgs.msg import ObjectArray
from std_msgs.msg import Empty
from std_srvs.srv import Empty as empty_srv

class State:
    def __init__(self) -> None:
        self.x = 0.
        self.y = 0.
        self.psi = 0.

class Obstacle:
    def __init__(self) -> None:
        self.x = 0.
        self.y = 0.
        self.psi = 0.
        self.vx = 0.
        self.vy = 0.
        self.id = -2
        self.last_time = rospy.Time.now()

class Goal:
    def __init__(self) -> None:
        self.x = 0.
        self.y = 0.

class PedestrianSimInterface:
    def __init__(self) -> None:
        self.enable_output_ = True
        self.first_time_step = None
        self.state_received = False
        # self.state_sub_ = rospy.Subscriber("Robot_1/pose", PoseStamped, self.state_optitrack_callback, queue_size=1)
        # self.goal_sub_ = rospy.Subscriber("/roadmap/goal", PoseStamped, self.goal_callback, queue_size=1)
        self.obs_subs_ = []
        self.dynamic_obstacles = []
        self.obs_subs_.append(rospy.Subscriber("/pedestrian_simulator/pedestrians", ObjectArray, self.obstacle_callback, queue_size=1))
        self.robot_action_pub = rospy.Publisher('/lmpcc/reset_environment', Empty, queue_size=10)
        self.reset_world_srv = rospy.ServiceProxy('/gazebo/reset_world', empty_srv)
        # Need to fix dt and the path prediction topic
    
    def obstacle_callback(self, msg: ObjectArray):
        self.dynamic_obstacles.clear()
        for obs in msg.objects:
            dyn_obs = Obstacle()
            dyn_obs.id = obs.id
            dyn_obs.x = obs.pose.position.x
            dyn_obs.y = obs.pose.position.y
            dyn_obs.vx = obs.twist.linear.x
            dyn_obs.vy = obs.twist.linear.y
            dyn_obs.psi = 0.
            self.dynamic_obstacles.append(dyn_obs)

    def update_robot(self, robot_x):
        if self.first_time_step == None:
            self.first_time_step = rospy.Time.now()
        elif robot_x >= 25. or rospy.Time.now() - self.first_time_step >= rospy.Duration(secs=40):
            msg = Empty()
            resp = self.reset_world_srv()
            self.robot_action_pub.publish(msg)
            self.first_time_step = rospy.Time.now()
