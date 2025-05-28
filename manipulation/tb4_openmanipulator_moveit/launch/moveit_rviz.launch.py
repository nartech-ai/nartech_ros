from launch import LaunchDescription
from launch_ros.actions import Node
from moveit_configs_utils import MoveItConfigsBuilder

def generate_launch_description():
    moveit_config = (
        MoveItConfigsBuilder("turtlebot4", package_name="tb4_openmanipulator_moveit")
        .planning_pipelines()
        .joint_limits()
        .trajectory_execution()
        .to_moveit_configs()
    )

    return LaunchDescription([
        Node(
            package="rviz2",
            executable="rviz2",
            name="rviz2",
            output="screen",
            arguments=["-d", str(moveit_config.package_path / "config" / "moveit.rviz")],
            parameters=[
                moveit_config.to_dict(),
                {"use_sim_time": True}
            ]
        )
    ])
