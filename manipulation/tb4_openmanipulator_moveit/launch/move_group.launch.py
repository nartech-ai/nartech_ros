from launch import LaunchDescription
from launch_ros.actions import Node
from moveit_configs_utils import MoveItConfigsBuilder
import yaml

def generate_launch_description():
    moveit_config = (
        MoveItConfigsBuilder("turtlebot4", package_name="tb4_openmanipulator_moveit")
        .robot_description()
        .trajectory_execution()
        .planning_pipelines()
        .joint_limits()
        .to_moveit_configs()
    )

    # TEMP FIX: Hardcoded path to moveit_controllers.yaml
    controller_yaml_path = "/home/nartech/nartech_ws/src/tb4_openmanipulator_moveit/config/moveit_controllers.yaml"

    with open(controller_yaml_path, "r") as f:
        controller_yaml_dict = yaml.safe_load(f)
        
    ros__params = controller_yaml_dict["moveit_simple_controller_manager"]["ros__parameters"]
    mgr_params  = controller_yaml_dict["moveit_controller_manager"]["ros__parameters"]

    return LaunchDescription([
        Node(
            package="moveit_ros_move_group",
            executable="move_group",
            output="screen",
            parameters=[
                moveit_config.to_dict(),
                {"use_sim_time": True},
                {"moveit_simple_controller_manager": ros__params},
                {"moveit_controller_manager": mgr_params},
            ]
        )
    ])
