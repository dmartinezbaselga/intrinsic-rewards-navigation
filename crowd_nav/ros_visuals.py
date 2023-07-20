import rospy
from visualization_msgs.msg import MarkerArray, Marker
from geometry_msgs.msg import Pose, Point
from copy import deepcopy

COLORS = [[153, 51, 102], [0, 255, 0], [254, 254, 0], [254, 0, 254], [0, 255, 255], [255, 153, 0]]

class ROSMarkerPublisher:

    """
    Collects ROS Markers
    """

    def __init__(self, topic, max_size):
        self.pub_ = rospy.Publisher(topic, MarkerArray, queue_size=1)
        self.marker_list_ = MarkerArray()

        self.clear_all()

        self.ros_markers_ = []

        self.topic_ = topic
        self.max_size_ = max_size
        self.id_ = 0
        self.prev_id_ = 0

    def clear_all(self):
        remove_marker = Marker()
        remove_marker.action = remove_marker.DELETEALL
        self.marker_list_.markers = []
        self.marker_list_.markers.append(remove_marker)
        self.pub_.publish(self.marker_list_)

    def __del__(self):
        self.clear_all()

    def publish(self):
        # print('DEBUG: Publishing markers')

        remove_marker = Marker()
        remove_marker.action = remove_marker.DELETE

        if self.prev_id_ > self.id_:
            for i in range(self.id_, self.prev_id_):
                remove_marker.id = i
                self.marker_list_.markers.append(deepcopy(remove_marker))

        # Define stamp properties
        for marker in self.marker_list_.markers:
            marker.header.stamp = rospy.Time.now()

        # Publish
        # print(self.marker_list_.markers)
        #print('Publishing {} markers'.format(len(self.marker_list_.markers)))
        self.pub_.publish(self.marker_list_)

        # Reset
        self.marker_list_.markers = []

        self.prev_id_ = deepcopy(self.id_)
        self.id_ = 0

    def get_cube(self, frame_id="map"):
        new_marker = ROSMarker(1, frame_id, self)
        return self.get_point_marker(new_marker)

    def get_sphere(self, frame_id="map"):
        new_marker = ROSMarker(2, frame_id, self)
        return self.get_point_marker(new_marker)

    def get_cylinder(self, frame_id="map"):
        new_marker = ROSMarker(3, frame_id, self)
        return self.get_point_marker(new_marker)

    def get_circle(self, frame_id="map"):
        new_marker = ROSMarker(3, frame_id, self)
        new_marker.set_scale(1., 1., 0.01)
        return self.get_point_marker(new_marker)

    def get_line(self, frame_id="map"):
        new_marker = ROSLineMarker(frame_id, self)
        return self.get_point_marker(new_marker)
    
    def get_arrow(self, frame_id="map"):
        new_marker = ROSMarker(0, frame_id, self)
        return self.get_point_marker(new_marker)

    def get_point_marker(self, new_marker):
        self.ros_markers_.append(new_marker)
        return new_marker  # Gives the option to modify it

    def add_marker_(self, new_marker):
        self.marker_list_.markers.append(new_marker)

    def get_id_(self):
        cur_id = deepcopy(self.id_)
        self.id_ += 1
        return cur_id

class ROSMarker:

    def __init__(self, type, frame_id, ros_marker_publisher):
        self.marker_ = Marker()
        self.marker_.header.frame_id = frame_id
        self.type_ = type
        self.marker_.type = self.type_
        self.ros_marker_publisher_ = ros_marker_publisher

        self.set_color(0)
        self.set_scale(1, 1, 1)

    def set_scale(self, x, y, z):
        self.marker_.scale.x = x
        self.marker_.scale.y = y
        self.marker_.scale.z = z

    def set_scale_all(self, all=1.0):
        self.marker_.scale.x = all
        self.marker_.scale.y = all
        self.marker_.scale.z = all

    def set_color(self, int_val, alpha=1.0):
        red, green, blue = get_viridis_color(int_val)
        self.marker_.color.r = red
        self.marker_.color.g = green
        self.marker_.color.b = blue
        self.marker_.color.a = alpha

    def set_lifetime(self, lifetime):
        self.marker_.lifetime = rospy.Time(lifetime)

    def add_marker(self, pose, add_orientation=False):
        self.marker_.id = self.ros_marker_publisher_.get_id_()
        self.marker_.pose = pose
        if not add_orientation:
            self.marker_.pose.orientation.x = 0
            self.marker_.pose.orientation.y = 0
            self.marker_.pose.orientation.z = 0
            self.marker_.pose.orientation.w = 1
            
        self.ros_marker_publisher_.add_marker_(deepcopy(self.marker_)) # Note the deepcopy

class ROSLineMarker:

    def __init__(self, frame_id, ros_marker_publisher):
        self.marker_ = Marker()
        self.marker_.header.frame_id = frame_id
        self.marker_.type = 5
        self.ros_marker_publisher_ = ros_marker_publisher

        self.marker_.scale.x = 0.25
        self.marker_.pose.orientation.x = 0
        self.marker_.pose.orientation.y = 0
        self.marker_.pose.orientation.z = 0
        self.marker_.pose.orientation.w = 1
        self.set_color(1)

    # Ax <= b (visualized)
    def add_constraint_line(self, A, b, length=10.0):
        a1 = A[0]
        a2 = A[1]
        b = float(b)
        p1 = Point()
        p2 = Point()
        if abs(a1) < 0.01 and abs(a2) >= 0.01:
            p1.x = -length
            p1.y = (b + a1 * length) / float(a2)
            p1.z = 0.

            p2.x = length
            p2.y = (b - a1 * length) / float(a2)
            p2.z = 0.
        elif(abs(a1) >= 0.01):
            p1.y = -length
            p1.x = (b + a2 * length) / float(a1)
            p1.z = 0.

            p2.y = length
            p2.x = (b - a2 * length) / float(a1)
            p2.z = 0.
        else:
            raise Exception("visualized constrained is zero")

        p1.x = min(max(p1.x, -1e5), 1e5)
        p2.x = min(max(p2.x, -1e5), 1e5)
        p1.y = min(max(p1.y, -1e5), 1e5)
        p2.y = min(max(p2.y, -1e5), 1e5)

        self.add_line(p1, p2)

    def add_line_from_poses(self, pose_one, pose_two):
        p1 = Point()
        p1.x = pose_one.position.x
        p1.y = pose_one.position.y
        p1.z = pose_one.position.z

        p2 = Point()
        p2.x = pose_two.position.x
        p2.y = pose_two.position.y
        p2.z = pose_two.position.z

        self.add_line(p1, p2)

    def add_line(self, point_one, point_two):
        self.marker_.id = self.ros_marker_publisher_.get_id_()

        self.marker_.points.append(point_one)
        self.marker_.points.append(point_two)

        self.ros_marker_publisher_.add_marker_(deepcopy(self.marker_))

        self.marker_.points = []

    def set_color(self, int_val, alpha=1.0):
        red, green, blue = get_viridis_color(int_val)
        self.marker_.color.r = red
        self.marker_.color.g = green
        self.marker_.color.b = blue
        self.marker_.color.a = alpha

    def set_scale(self, thickness):
        self.marker_.scale.x = thickness


def get_viridis_color(select):
    # Obtained from https://waldyrious.net/viridis-palette-generator/
    viridis_vals = [253, 231, 37, 234, 229, 26, 210, 226, 27, 186, 222, 40, 162, 218, 55, 139, 214, 70, 119, 209, 83, 99,
    203, 95, 80, 196, 106, 63, 188, 115, 49, 181, 123, 38, 173, 129, 33, 165, 133, 30, 157, 137, 31, 148, 140, 34, 140,
    141, 37, 131, 142, 41, 123, 142, 44, 115, 142, 47, 107, 142, 51, 98, 141, 56, 89, 140]

    VIRIDIS_COLORS = 20
    select %= VIRIDIS_COLORS  # only 20 values specified

    # Invert the color range
    select = VIRIDIS_COLORS - 1 - select
    red = viridis_vals[select * 3 + 0]
    green = viridis_vals[select * 3 + 1]
    blue = viridis_vals[select * 3 + 2]

    red /= 256.0
    green /= 256.0
    blue /= 256.0

    return red, green, blue