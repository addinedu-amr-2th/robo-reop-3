import rclpy
from rclpy.qos import QoSProfile
from geometry_msgs.msg import Twist
import rclpy
from std_msgs.msg import Int32
import time

right_threshold = 50
left_threshold = 50

lane_distance = 0
right_distance_difference = 0
left_distance_difference = 0
intersection_flag = 0

def lane_distance_callback(msg):
    global lane_distance
    lane_distance = msg.data

def right_distance_callback(msg):
    global right_distance_difference
    right_distance_difference = msg.data

def left_distance_callback(msg):
    global left_distance_difference
    left_distance_difference = msg.data

def intersection_flag_callback(msg):
    global intersection_flag
    intersection_flag = msg.data

def main(args=None):
    rclpy.init(args=args)
    qos = QoSProfile(depth=1)
    node = rclpy.create_node('teleop_twist_publisher')
    # Creating Subscribers for Lane Distance Information
    node.create_subscription(Int32, '/lane_distance', lane_distance_callback, qos)
    node.create_subscription(Int32, '/right_distance_difference', right_distance_callback, qos)
    node.create_subscription(Int32, '/left_distance_difference', left_distance_callback, qos)
    node.create_subscription(Int32, '/intersection_flag', intersection_flag_callback, qos)
    cmd_vel_pub = node.create_publisher(Twist, 'cmd_vel', qos)
    twist = Twist()

    while rclpy.ok():
        # Set the angular velocity value proportional to the lane_distance 

        if intersection_flag == 0:
            if right_distance_difference > right_threshold or left_distance_difference > left_threshold:
                twist.angular.z = -lane_distance * 0.025
                twist.linear.x = 0.1
            else:
                twist.linear.x = 0.1
                twist.angular.z = -lane_distance * 0.012
        elif intersection_flag == 1:
            for i in range(8):
                twist.linear.x = 0.1
                twist.angular.z = 0.04
                cmd_vel_pub.publish(twist)
                time.sleep(0.2)

        cmd_vel_pub.publish(twist)

        print("Publishing cmd_vel: linear.x=%.2f angular.z=%.2f" % (twist.linear.x, twist.angular.z))
        print("Lane distance: %d, Right distance difference: %d, Left distance difference: %d" % (lane_distance, right_distance_difference, left_distance_difference))
        print('intersection_flag:', intersection_flag)
        rclpy.spin_once(node)

    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()