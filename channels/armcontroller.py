import rclpy
import time
import math
from rclpy.node import Node
from rclpy.duration import Duration
from geometry_msgs.msg import PoseStamped
from moveit_msgs.srv import GetPositionIK, GetMotionPlan
from moveit_msgs.msg import RobotState, JointConstraint, Constraints, MotionPlanRequest, RobotTrajectory
from trajectory_msgs.msg import JointTrajectoryPoint, JointTrajectory
from sensor_msgs.msg import JointState
from rclpy.action import ActionClient
from std_srvs.srv import Empty
from control_msgs.action import FollowJointTrajectory
from geometry_msgs.msg import Twist

# Assume mettabridge defines these constants and functions:
if __name__ == '__main__': #arm test does not need that
    NAV_STATE_BUSY = NAV_STATE_SUCCESS = NAV_STATE_FAIL = 42
    NAV_STATE_SET = lambda x: 42
    NAV_STATE_GET = lambda: 42
else:
    from mettabridge import NAV_STATE_SET, NAV_STATE_GET, NAV_STATE_BUSY, NAV_STATE_SUCCESS, NAV_STATE_FAIL

class ArmController:
    def __init__(self, node=None, semantic_slam=None):
        self.holding = False
        self.semantic_slam = semantic_slam
        if node is None:
            self.own_node = Node('arm_controller')
            self.node = self.own_node
        else:
            self.own_node = None
            self.node = node
        self.cmd_pub = self.node.create_publisher(Twist, '/cmd_vel', 10)
        self.ik_client = self.node.create_client(GetPositionIK, 'compute_ik')
        self.plan_client = self.node.create_client(GetMotionPlan, 'plan_kinematic_path')
        self.arm_client = ActionClient(self.node, FollowJointTrajectory, '/arm_controller/follow_joint_trajectory')
        self.gripper_client = ActionClient(self.node, FollowJointTrajectory, '/gripper_controller/follow_joint_trajectory')
        for client, name in [(self.ik_client, 'IK'), (self.plan_client, 'Planner')]:
            if not client.wait_for_service(timeout_sec=5.0):
                self.node.get_logger().error(f"{name} service not available")
                rclpy.shutdown()
                return
        if not self.arm_client.wait_for_server(timeout_sec=5.0):
            self.node.get_logger().error("arm_controller action not available")
            rclpy.shutdown()
            return
        self.control_gripper("open")
        self.move_end_effector_to(x=0.15, y=0.0, z=0.3)

    def pick_at(self, x, y, z):
        self.node.get_logger().info("arm_controller: PICK AT COORDINATE " + f"{x} {y} {z}")
        #sanitize end location
        #x=0.2..0.3 z=0.1..0.2 works for pick!
        #unused: for x=0.2 it can reach up to z=0.4 if necessary (to grap from a platform)
        if y != 0 : y = 0.0
        if x < 0.2: x = 0.2
        if x > 0.3: x = 0.3
        if z < 0.1: z = 0.1
        if z > 0.2: z = 0.2
        self.node.get_logger().info("arm_controller: PICK CORRECTED " + f"{x} {y} {z}")
        #proceed in the following order: (async programming style)
        def after_open(_):
            self.move_end_effector_to(x=0.15, y=0.0, z=0.3).add_done_callback(after_lifted)
        def after_lifted(_):
            self.move_end_effector_to(x=x, y=y, z=z).add_done_callback(after_descended)
        def after_descended(_):
            self.control_gripper("close").add_done_callback(after_closed)
        def after_closed(_):
            self.move_end_effector_to(x=0.15, y=0.0, z=0.3).add_done_callback(after_final_lift)
        def after_final_lift(_):
            self.node.get_logger().info("arm_controller: DONE")
        self.control_gripper("open").add_done_callback(after_open)

    def pick(self, objectlabel):
        """
        Drive the mobile base until the target object is centred in the camera
        and 20 cm away, then trigger pick_at().

        Parameters
        ----------
        objectlabel : str
            The semantic-SLAM label of the object to pick.
        """
        # ---------- Preconditions ------------------------------------------------
        if NAV_STATE_GET() == NAV_STATE_BUSY or getattr(self, "holding", False):
            return
        NAV_STATE_SET(NAV_STATE_BUSY)

        if not (objectlabel and objectlabel in self.semantic_slam.previous_detections):
            self.node.get_logger().info("arm_controller: Pick failed, object location not observed or remembered")
            NAV_STATE_SET(NAV_STATE_FAIL)
            return

        # Remember where the object was last seen (for pick_at).
        (_, _, _, _, _, _, spoint_base_link, imagecoords_depth) = self.semantic_slam.previous_detections[objectlabel]
        target_point = spoint_base_link.point
        # ---------- Controller tuning constants ----------------------------------
        yaw_tol   = 0.04       # ± ~2.3 degrees (image plane error)
        depth_tol = 0.1       # ± 3 cm
        k_yaw     = 0.1 #0.6        # rad / s per unit x_rel
        k_fwd     = 0.1 #25       # m / s per metre depth error
        MIN_ANG = 0.08      # rad / s  → tweak until the robot just starts turning
        MIN_LIN = 0.04      # m / s
        # ---------- Internal state -----------------------------------------------
        self.correction_attempts = 0
        self.max_corrections     = 10
        # Publish a zero-velocity command so the base really stops.
        def _stop_motion():
            self.cmd_pub.publish(Twist())
        # Store on the instance so the callback can call it.
        self._stop_motion = _stop_motion
        # ---------- Timer callback -----------------------------------------------
        def correction_step():
            # ❶ Abort if the target is gone.
            if objectlabel not in self.semantic_slam.previous_detections:
                self._stop_motion()
                self.correction_timer.cancel()
                self.node.get_logger().info("arm_controller: Object absent in semantic map")
                NAV_STATE_SET(NAV_STATE_FAIL)
                return
            (t, _, _, _, _, _, spoint_base_link, imagecoords_depth) = self.semantic_slam.previous_detections[objectlabel]
            if time.time() - t > 1.0:
                self._stop_motion()
                self.correction_timer.cancel()
                self.node.get_logger().info("arm_controller: Object lost from observation")
                NAV_STATE_SET(NAV_STATE_FAIL)
                return
            x_rel, _, depth = imagecoords_depth
            x_err = (x_rel - 0.5) * 2.0
            target_point = spoint_base_link.point
            twist = Twist()
            # ❷ Yaw correction (same sign as x_rel → turn toward the object).
            if abs(x_err) > yaw_tol:
                twist.angular.z = -k_yaw * x_err
                if 0 < abs(twist.angular.z) < MIN_ANG:
                    twist.angular.z = math.copysign(MIN_ANG, twist.angular.z)
            # ❸ Forward correction (stop ~40 cm in front of object).
            depth_error = depth - 0.40
            if depth_error > depth_tol:
                twist.linear.x = k_fwd * depth_error
                if 0 < twist.linear.x < MIN_LIN:
                    twist.linear.x = MIN_LIN
            # Publish base command.
            self.cmd_pub.publish(twist)
            self.node.get_logger().info(
                f"Correction {self.correction_attempts + 1} | "
                f"x_err={x_err:+.2f}, depth={depth:.2f}, "
                f"cmd=({twist.linear.x:.2f}, {twist.angular.z:.2f})")
            # ❹ Convergence check.
            if abs(x_err) <= yaw_tol and abs(depth_error) <= depth_tol:
                self._stop_motion()
                self.correction_timer.cancel()
                self.node.get_logger().info("arm_controller: Aligned and close. Executing pick.")
                self.pick_at(target_point.x, 0.0, target_point.z)
                NAV_STATE_SET(NAV_STATE_SUCCESS)        # optional, if you have it
                return
            # ❺ Bail out if we’re stuck.
            self.correction_attempts += 1
            if self.correction_attempts >= self.max_corrections:
                self._stop_motion()
                self.correction_timer.cancel()
                self.node.get_logger().info(
                    "arm_controller: Too many corrections.")
                NAV_STATE_SET(NAV_STATE_FAIL)
        # ---------- Kick off the periodic controller -----------------------------
        self.correction_timer = self.node.create_timer(0.5, correction_step)

    def drop(self):
        if NAV_STATE_GET() == NAV_STATE_BUSY or not self.holding:
            return
        NAV_STATE_SET(NAV_STATE_BUSY)
        def after_up(_):
            self.move_end_effector_to(x=0.15, y=0.0, z=0.2).add_done_callback(after_down)
        def after_down(_):
            self.control_gripper("open").add_done_callback(after_open)
        def after_open(_):
            self.move_end_effector_to(x=0.15, y=0.0, z=0.3).add_done_callback(after_drop)
        def after_drop(_):
            self.node.get_logger().info("arm_controller: DROP complete")
            self.holding = False
            NAV_STATE_SET(NAV_STATE_SUCCESS)
        self.move_end_effector_to(x=0.15, y=0.0, z=0.3).add_done_callback(after_up)

    def wait_for_joint_states(self, joint_names):
        future = rclpy.task.Future()
        def callback(msg):
            if all(j in msg.name for j in joint_names):
                future.set_result(msg)
        self._joint_state_sub = self.node.create_subscription(JointState, '/joint_states', callback, 10)
        def cleanup_on_done(_):
            self.node.destroy_subscription(self._joint_state_sub)
            self._joint_state_sub = None
        future.add_done_callback(cleanup_on_done)
        return future

    def move_end_effector_to(self, x=0.15, y=0.0, z=0.2):
        future = rclpy.task.Future()
        joint_names = ['joint1', 'joint2', 'joint3', 'joint4']
        def after_joint_state(fut):
            joint_state_cur = fut.result()
            # Step 1: Target pose:
            self.node.get_logger().info('arm_controller: STEP 1')
            pose = PoseStamped()
            pose.header.frame_id = 'base_link'
            pose.pose.position.x = x
            pose.pose.position.y = y
            pose.pose.position.z = z
            pose.pose.orientation.w = 1.0
            # Step 2: IK of it:
            self.node.get_logger().info('arm_controller: STEP 2')
            ik_req = GetPositionIK.Request()
            ik_req.ik_request.group_name = 'arm'
            ik_req.ik_request.ik_link_name = 'link5'
            ik_req.ik_request.pose_stamped = pose
            ik_req.ik_request.robot_state = RobotState(joint_state=joint_state_cur)
            ik_req.ik_request.timeout.sec = 2
            def after_ik(ik_fut):
                ik_res = ik_fut.result()
                if ik_res.error_code.val != 1:
                    self.node.get_logger().error(f"arm_controller: IK failed with code {ik_res.error_code.val}")
                    future.set_result(False)
                    return
                # Step 3: Set as joint goal:
                self.node.get_logger().info('arm_controller: STEP 3')
                goal_positions = [ik_res.solution.joint_state.position[
                    ik_res.solution.joint_state.name.index(j)] for j in joint_names]
                # Step 4: Plan to joint goal
                self.node.get_logger().info('arm_controller: STEP 4')
                req = GetMotionPlan.Request()
                plan_req = MotionPlanRequest()
                plan_req.group_name = 'arm'
                # Step 5: Start state as current joint states
                self.node.get_logger().info('arm_controller: STEP 5')
                start_state = RobotState()
                start_state.joint_state.name = joint_names
                start_state.joint_state.position = [joint_state_cur.position[
                    joint_state_cur.name.index(j)] for j in joint_names]
                plan_req.start_state = start_state
                # Step 6: Goal state as joint constraints from IK
                self.node.get_logger().info('arm_controller: STEP 6')
                goal_constraints = Constraints()
                for j, pos in zip(joint_names, goal_positions):
                    jc = JointConstraint()
                    jc.joint_name = j
                    jc.position = pos
                    #if j == "joint1":
                    #    jc.tolerance_above = 0.1  # allow full ±180° yaw
                    #    jc.tolerance_below = 0.1
                    #else:
                    jc.tolerance_above = 0.01
                    jc.tolerance_below = 0.01
                    jc.weight = 1.0
                    goal_constraints.joint_constraints.append(jc)
                plan_req.goal_constraints = [goal_constraints]
                # Step 7: Carry out planning to obtain trajectory:
                self.node.get_logger().info('arm_controller: STEP 7')
                plan_req.num_planning_attempts = 5
                plan_req.allowed_planning_time = 5.0
                req.motion_plan_request = plan_req
                def after_plan(plan_fut):
                    plan_res = plan_fut.result()
                    if plan_res.motion_plan_response.error_code.val != 1:
                        self.node.get_logger().error(f"arm_controller: Planning failed with code {plan_res.motion_plan_response.error_code.val}")
                        future.set_result(False)
                        return
                    trajectory: JointTrajectory = plan_res.motion_plan_response.trajectory.joint_trajectory
                    self.node.get_logger().info(f"arm_controller: Planned trajectory with {len(trajectory.points)} points.")
                    # Step 8: Execute planned trajectory
                    self.node.get_logger().info('arm_controller: STEP 8')
                    goal_msg = FollowJointTrajectory.Goal()
                    goal_msg.trajectory = trajectory
                    self.node.get_logger().info("arm_controller: Sending trajectory to arm_controller...")
                    def after_traj(goal_fut):
                        goal_handle = goal_fut.result()
                        if not goal_handle.accepted:
                            self.node.get_logger().error("arm_controller: Trajectory rejected by arm_controller")
                            future.set_result(False)
                            return
                        goal_handle.get_result_async().add_done_callback(
                            lambda res_fut: future.set_result(True)
                        )
                    self.arm_client.send_goal_async(goal_msg).add_done_callback(after_traj)
                self.plan_client.call_async(req).add_done_callback(after_plan)
            self.ik_client.call_async(ik_req).add_done_callback(after_ik)
        self.wait_for_joint_states(joint_names).add_done_callback(after_joint_state)
        return future
    
    def control_gripper(self, command):
        future = rclpy.task.Future()
        open_gripper = command == "open"
        self.node.get_logger().info("arm_controller: Gripper open=" + str(open_gripper))
        position = 0.019 if open_gripper else -0.01
        goal_msg = FollowJointTrajectory.Goal()
        goal_msg.trajectory.joint_names = ['gripper_left_joint', 'gripper_right_joint']
        point = JointTrajectoryPoint()
        point.positions = [position, position]
        point.time_from_start.sec = 1
        goal_msg.trajectory.points = [point]
        def goal_sent_cb(fut):
            goal_handle = fut.result()
            if not goal_handle.accepted:
                self.node.get_logger().error('arm_controller: Gripper goal rejected')
                future.set_result(False)
                return
            goal_handle.get_result_async().add_done_callback(
                lambda res_fut: future.set_result(True)
            )
        self.gripper_client.send_goal_async(goal_msg).add_done_callback(goal_sent_cb)
        return future

def main():
    rclpy.init()
    armcontroller = ArmController()
    #self.control_gripper("open")
    armcontroller.pick_at(0.2, 0.0, 0.1) 
    end_time = time.time() + 10.0
    while time.time() < end_time and rclpy.ok():
        rclpy.spin_once(armcontroller.node, timeout_sec=0.1)
    #armcontroller.control_gripper("open")
    #armcontroller.move_end_effector_to(x=0.15, y=0.0, z=0.3)
    #armcontroller.move_end_effector_to(x=0.15, y=0.0, z=0.2)
    #armcontroller.control_gripper("close")
    #armcontroller.move_end_effector_to(x=0.15, y=0.0, z=0.3)
    #armcontroller.move_end_effector_to(x=0.15, y=0.0, z=0.2)
    #armcontroller.control_gripper("open")
    #armcontroller.move_end_effector_to(x=0.15, y=0.0, z=0.3)
    rclpy.shutdown()

if __name__ == '__main__':
    main()
