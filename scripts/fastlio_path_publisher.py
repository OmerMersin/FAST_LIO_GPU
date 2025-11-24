#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from nav_msgs.msg import Odometry, Path
from geometry_msgs.msg import PoseStamped

class FastLIOPath(Node):
    def __init__(self):
        super().__init__('fastlio_path_node')

        self.declare_parameter('odom_topic', '/lio_sam/mapping/odometry')
        odom_topic = self.get_parameter('odom_topic').value

        self.path_pub = self.create_publisher(Path, '/fastlio/path', 10)
        self.odom_sub = self.create_subscription(Odometry, odom_topic, self.odom_callback, 10)

        self.path = Path()
        self.path.header.frame_id = 'map'

        self.get_logger().info(f"[FAST-LIO PATH] Listening: {odom_topic}")

    def odom_callback(self, msg: Odometry):
        pose = PoseStamped()
        pose.header = msg.header
        pose.pose = msg.pose.pose

        self.path.header.stamp = msg.header.stamp
        self.path.poses.append(pose)

        self.path_pub.publish(self.path)


def main():
    rclpy.init()
    node = FastLIOPath()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
