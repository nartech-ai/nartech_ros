import rclpy
import time
from rclpy.node import Node
from geometry_msgs.msg import PoseStamped
from moveit_msgs.srv import GetPositionIK, GetMotionPlan
from moveit_msgs.msg import RobotState, JointConstraint, Constraints, MotionPlanRequest, RobotTrajectory
from trajectory_msgs.msg import JointTrajectoryPoint, JointTrajectory
from sensor_msgs.msg import JointState
from rclpy.action import ActionClient
from std_srvs.srv import Empty
from control_msgs.action import FollowJointTrajectory

class ArmController(Node):
    def __init__(self):
        super().__init__('arm_controller')
        self.ik_client = self.create_client(GetPositionIK, 'compute_ik')
        self.plan_client = self.create_client(GetMotionPlan, 'plan_kinematic_path')
        self.arm_client = ActionClient(self, FollowJointTrajectory, '/arm_controller/follow_joint_trajectory')
        self.gripper_client = ActionClient(self, FollowJointTrajectory, '/gripper_controller/follow_joint_trajectory')
        for client, name in [(self.ik_client, 'IK'), (self.plan_client, 'Planner')]:
            if not client.wait_for_service(timeout_sec=5.0):
                self.get_logger().error(f"{name} service not available")
                rclpy.shutdown()
                return
        if not self.arm_client.wait_for_server(timeout_sec=5.0):
            self.get_logger().error("arm_controller action not available")
            rclpy.shutdown()
            return

    def wait_for_joint_states(self, joint_names):
        received = False
        joint_msg = JointState()
        def callback(msg):
            nonlocal joint_msg, received
            if all(j in msg.name for j in joint_names):
                joint_msg = msg
                received = True
        sub = self.create_subscription(JointState, '/joint_states', callback, 10)
        while rclpy.ok() and not received:
            rclpy.spin_once(self)
        self.destroy_subscription(sub)
        return joint_msg

    def move_end_effector_to(self, x=0.15, y=0.0, z=0.2):
        joint_names = ['joint1', 'joint2', 'joint3', 'joint4']
        joint_state_cur = self.wait_for_joint_states(joint_names)
        # Step 1: Target pose:
        pose = PoseStamped()
        pose.header.frame_id = 'base_link'
        pose.pose.position.x = x
        pose.pose.position.y = y
        pose.pose.position.z = z
        pose.pose.orientation.w = 1.0
        # Step 2: IK of it:
        ik_req = GetPositionIK.Request()
        ik_req.ik_request.group_name = 'arm'
        ik_req.ik_request.ik_link_name = 'link5'
        ik_req.ik_request.pose_stamped = pose
        ik_req.ik_request.robot_state = RobotState(joint_state=joint_state_cur)
        ik_req.ik_request.timeout.sec = 2
        ik_future = self.ik_client.call_async(ik_req)
        rclpy.spin_until_future_complete(self, ik_future)
        ik_res = ik_future.result()
        if ik_res.error_code.val != 1:
            self.get_logger().error(f"IK failed with code {ik_res.error_code.val}")
            rclpy.shutdown()
            return
        # Step 3: Set as joint goal:
        goal_positions = [ik_res.solution.joint_state.position[ik_res.solution.joint_state.name.index(j)] for j in joint_names]
        # Step 4: Plan to joint goal
        req = GetMotionPlan.Request()
        plan_req = MotionPlanRequest()
        plan_req.group_name = 'arm'
        # Step 5: Start state as current joint states
        start_state = RobotState()
        start_state.joint_state.name = joint_names
        start_state.joint_state.position = [joint_state_cur.position[joint_state_cur.name.index(j)] for j in joint_names]
        plan_req.start_state = start_state
        # Step 6: Goal state as joint constraints from IK
        goal_constraints = Constraints()
        for j, pos in zip(joint_names, goal_positions):
            jc = JointConstraint()
            jc.joint_name = j
            jc.position = pos
            jc.tolerance_above = 0.01
            jc.tolerance_below = 0.01
            jc.weight = 1.0
            goal_constraints.joint_constraints.append(jc)
        plan_req.goal_constraints = [goal_constraints]
        # Step 7: Carry out planning to obtain trajectory:
        plan_req.num_planning_attempts = 5
        plan_req.allowed_planning_time = 5.0
        req.motion_plan_request = plan_req
        plan_future = self.plan_client.call_async(req)
        rclpy.spin_until_future_complete(self, plan_future)
        plan_res = plan_future.result()
        if plan_res.motion_plan_response.error_code.val != 1:
            self.get_logger().error(f"Planning failed with code {plan_res.motion_plan_response.error_code.val}")
            rclpy.shutdown()
            return
        trajectory: JointTrajectory = plan_res.motion_plan_response.trajectory.joint_trajectory
        self.get_logger().info(f"Planned trajectory with {len(trajectory.points)} points.")
        # Step 8: Execute planned trajectory
        goal_msg = FollowJointTrajectory.Goal()
        goal_msg.trajectory = trajectory
        self.get_logger().info("Sending trajectory to arm_controller...")
        send_future = self.arm_client.send_goal_async(goal_msg)
        rclpy.spin_until_future_complete(self, send_future)
        goal_handle = send_future.result()
        if not goal_handle.accepted:
            self.get_logger().error("Trajectory rejected by arm_controller")
            rclpy.shutdown()
            return
        result_future = goal_handle.get_result_async()
        rclpy.spin_until_future_complete(self, result_future)
        self.get_logger().info("Trajectory execution complete.")
    
    def control_gripper(self, command): #command="open"/"close"
        open_gripper = command == "open"
        position = 0.019 if open_gripper else -0.01  # adjust values as needed
        goal_msg = FollowJointTrajectory.Goal()
        goal_msg.trajectory.joint_names = ['gripper_left_joint', 'gripper_right_joint']
        point = JointTrajectoryPoint()
        point.positions = [position, position]
        point.time_from_start.sec = 1
        goal_msg.trajectory.points = [point]
        future = self.gripper_client.send_goal_async(goal_msg)
        rclpy.spin_until_future_complete(self, future)
        goal_handle = future.result()
        if not goal_handle.accepted:
            self.get_logger().error('Gripper goal rejected')
            return
        result_future = goal_handle.get_result_async()
        rclpy.spin_until_future_complete(self, result_future)
        self.get_logger().info('Gripper moved')
        time.sleep(2.0)

def main():
    rclpy.init()
    poseplannerexecutor = ArmController()
    poseplannerexecutor.control_gripper("open")
    poseplannerexecutor.move_end_effector_to(x=0.15, y=0.0, z=0.3)
    poseplannerexecutor.move_end_effector_to(x=0.15, y=0.0, z=0.2)
    poseplannerexecutor.control_gripper("close")
    poseplannerexecutor.move_end_effector_to(x=0.15, y=0.0, z=0.3)
    poseplannerexecutor.move_end_effector_to(x=0.15, y=0.0, z=0.2)
    poseplannerexecutor.control_gripper("open")
    poseplannerexecutor.move_end_effector_to(x=0.15, y=0.0, z=0.3)
    rclpy.shutdown()

if __name__ == '__main__':
    main()
