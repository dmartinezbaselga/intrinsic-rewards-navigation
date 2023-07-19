import rospy
from sensor_msgs.msg import Joy
from geometry_msgs.msg import PoseStamped, Point
from std_msgs.msg import Float64MultiArray
import numpy as np
from visualization_msgs.msg import Marker

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

class JackalInterface:
    def __init__(self) -> None:
        self.bluetooth_sub_ = rospy.Subscriber("/bluetooth_teleop/joy", Joy, self.bluetooth_callback, queue_size=1)
        self.last_state_time_ = rospy.Time.now()
        self.state_received = False
        self.robot_state = State()
        self.goal = Goal()
        # self.state_sub_ = rospy.Subscriber("Robot_1/pose", PoseStamped, self.state_optitrack_callback, queue_size=1)
        # self.goal_sub_ = rospy.Subscriber("/roadmap/goal", PoseStamped, self.goal_callback, queue_size=1)
        self.max_obstacles = 12
        self.obs_received_ = np.zeros(self.max_obstacles)
        self.obs_subs_ = []
        self.obs_pub_ = []
        for i in range(self.max_obstacles):
            self.obs_subs_.append(rospy.Subscriber("/obstacle" + str(i+1) + "/path_prediction", Float64MultiArray, self.optitrack_obstacle_callback, queue_size=1))
            self.obs_pub_.append(rospy.Publisher("/obstacle" + str(i+1) + "/visual", Marker, queue_size=10))

        self.first_obstacle_id = -1
        self.dynamic_obstacles = []
        self.dt = 0.2
        # Need to fix dt and the path prediction topic

    def bluetooth_callback(self, msg: Joy):
        self.enable_output_ = msg.axes[2] < -0.9
    
    def state_optitrack_callback(self, msg: PoseStamped):
        if self.last_state_time_ + rospy.Duration(1. / 50.) > msg.header.stamp:
            return
        self.state_received = True
        self.robot_state.x = msg.pose.position.x
        self.robot_state.y = msg.pose.position.y
        # Not quaternion
        self.robot_state.psi = msg.pose.orientation.z

    def optitrack_obstacle_callback(self, msg: Float64MultiArray):
        obstacle_id = msg.data[2]
        self.obs_received_[obstacle_id] = True
        obstacle_found = False
        stamp = rospy.Time.now()
        for obs in self.dynamic_obstacles:
            if obs.id == obstacle_id:
                obs.x = msg.data[0]
                obs.y = msg.data[1]
                obs.psi = msg.data[3]
                obs.vx = (msg.data[4] - msg.data[0])/self.dt
                obs.vy = (msg.data[5] - msg.data[1])/self.dt
                obs.stamp = stamp
                obstacle_found = True
        if not obstacle_found:
            new_obs = Obstacle()
            new_obs.id = obstacle_id
            new_obs.x = msg.data[0]
            new_obs.y = msg.data[1]
            new_obs.psi = msg.data[3]
            new_obs.vx = (msg.data[4] - msg.data[0])/self.dt
            new_obs.vy = (msg.data[5] - msg.data[1])/self.dt
            obs.stamp = stamp
            self.dynamic_obstacles.append(new_obs)
        if obstacle_id == self.first_obstacle_id:
            self.post_process_obstacles()
        elif self.first_obstacle_id == -1:
            self.first_obstacle_id = obstacle_id
        marker = Marker()
        marker.header.frame_id = "map"
        marker.header.stamp = rospy.Time.now()

        # set shape, Arrow: 0; Cube: 1 ; Sphere: 2 ; Cylinder: 3
        marker.type = 0
        marker.id = obstacle_id
        start = Point()
        start.x = obs.x
        start.y = obs.y
        marker.points.append(start)
        end = Point()
        end.x = msg.data[4]
        end.y = msg.data[5]
        marker.points.append(end)
        # Set the scale of the marker
        marker.scale.x = 0.4
        marker.scale.y = 0.4
        marker.scale.z = 1.0

        # Set the color
        marker.color.r = 0.0
        marker.color.g = 1.0
        marker.color.b = 0.0
        marker.color.a = 1.0
        self.obs_pub_[obstacle_id-1].publish(marker)
    
    def goal_callback(self, goal: PoseStamped):
        self.goal.x = goal.pose.position.x
        self.goal.y = goal.pose.position.y
        
