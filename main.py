#!/usr/bin/env python3
# main.py
import rclpy
from rclpy.node import Node
import tf2_ros
import sys
sys.path.append('./channels/')
from objectdetection import ObjectDetector
from localization import Localization
from semanticslam import SemanticSLAM
from navigation import Navigation
from armcontroller import ArmController

class MainNode(Node):
    def __init__(self):
        super().__init__('NARTECH_node')
        # Create a single TF2 buffer and listener to be shared across modules.
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)
        # Instantiate the helper modules.
        self.pick = lambda x: None #overridden
        self.drop = lambda x: None #overridden
        self.object_detector = ObjectDetector(self, self.tf_buffer)
        self.localization = Localization(self, self.tf_buffer)
        self.semantic_slam = SemanticSLAM(self, self.tf_buffer, self.localization, self.object_detector)
        self.navigation = Navigation(self, self.semantic_slam, self.localization)
        self.start_navigation_to_coordinate = self.navigation.start_navigation_to_coordinate
        self.arm_controller = ArmController(self, self.semantic_slam)
        self.pick = self.arm_controller.pick
        self.drop = self.arm_controller.drop

def main(args=None):
    rclpy.init(args=args)
    node = MainNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
