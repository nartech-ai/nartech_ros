import rclpy
import time
import math
import functools
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
from rclpy.task import Future

# Approach and grab params
GOAL_TARGET_DISTANCE = 0.38 #0.37
yaw_tol, depth_tol = 0.05, 0.02
k_yaw,  k_fwd      = 0.3,  0.3
CORRECTION_DT      = 0.5
GRIPPER_GRIP_HEIGHT = 0.2
# Gripper Open and close angle
GRIPPER_OPEN_TARGET = 0.019
GRIPPER_CLOSE_TARGET = -0.01
# Slip detection params
GRIPPER_SLIP_TOL    = 0.002   # +- tolerance around closed angle
SLIP_CHECK_PERIOD   = 0.2     # Seconds between checks

# Assume mettabridge defines these constants and functions:
if __name__ == '__main__': #arm test does not need that
    NAV_STATE_BUSY = NAV_STATE_SUCCESS = NAV_STATE_FAIL = 42
    NAV_STATE_SET = lambda x: 42
    NAV_STATE_GET = lambda: 42
    ARM_STATE_SET = lambda x: 42
    ARM_STATE_GET = lambda: 42
else:
    from mettabridge import NAV_STATE_SET, NAV_STATE_GET, NAV_STATE_BUSY, NAV_STATE_SUCCESS, NAV_STATE_FAIL, ARM_STATE_SET, ARM_STATE_GET

class ArmController:
    def __init__(self, node=None, semantic_slam=None, navigation=None):
        ARM_STATE_SET("FREE")
        self.semantic_slam = semantic_slam
        self.navigation = navigation
        self.objectlabel = None
        if node is None:
            self.own_node = Node('arm_controller')
            self.node = self.own_node
        else:
            self.own_node = None
            self.node = node
        self.picking = False
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
        # Cache latest /joint_states for slip detection
        self._latest_joint_state = None
        def _joint_state_cb(msg):
            self._latest_joint_state = msg
        self._joint_state_sub = self.node.create_subscription(JointState, '/joint_states', _joint_state_cb, 10)
        # Start periodic slip-check timer
        self._slip_timer = self.node.create_timer(SLIP_CHECK_PERIOD, self._check_gripper_slip)
        # Open the gripper
        self.control_gripper("open")
        # Move arm to pre-grap position
        self.move_end_effector_to(x=0.15, y=0.0, z=0.3)

    # ─────────────────────────────────────────────────────────────────────────
    # Slip detection: opens gripper if it ever reaches fully-closed angle
    # while we think we are holding something.
    # ─────────────────────────────────────────────────────────────────────────
    def _check_gripper_slip(self):
        if ARM_STATE_GET() == "FREE" or self._latest_joint_state is None:
            return
        #update semantic map inventory (object locations move with robot)
        self.semantic_slam.inventory = [ARM_STATE_GET()]
        try:
            li = self._latest_joint_state.name.index('gripper_left_joint')
            ri = self._latest_joint_state.name.index('gripper_right_joint')
        except ValueError:
            return  # joint names not published yet
        l_pos = self._latest_joint_state.position[li]
        r_pos = self._latest_joint_state.position[ri]
        fully_closed = (l_pos <= GRIPPER_CLOSE_TARGET + GRIPPER_SLIP_TOL and
                        r_pos <= GRIPPER_CLOSE_TARGET + GRIPPER_SLIP_TOL)
        if not fully_closed:
            return
        self.node.get_logger().warn(f'arm_controller: Slip detected at {l_pos:+.3f}, {r_pos:+.3f}')
        ARM_STATE_SET("FREE")
        self.navigation.cancel_goals()
        NAV_STATE_SET(NAV_STATE_FAIL)                               # block nav while recovering
        self.control_gripper('open')

    def pick_at(self, x: float, y: float, z: float, *, done_cb=None) -> Future:
        # ─── contract object we’ll fulfil at the end ────────────────────────────
        result_future: Future = Future()
        # ─── sanitize end-effector target ───────────────────────────────────────
        if y != 0: y = 0.0
        x = min(max(x, 0.2), 0.23)   # clamp 0.2 ≤ x ≤ 0.23 (z=0.05) # clamp 0.2 ≤ x ≤ 0.3 (z=0.1)
        z = min(max(z, 0.05), 0.2)   # clamp 0.1 ≤ z ≤ 0.2 (z=0.05)
        self.node.get_logger().info(f"arm_controller: PICK CORRECTED {x:.2f} {y:.2f} {z:.2f}")
        # ─── common success / failure paths ─────────────────────────────────────
        def _finish(success: bool) -> None:
            if done_cb:
                try:
                    done_cb(success)
                except Exception as e:  # noqa: BLE001
                    self.node.get_logger().warn(f"arm_controller: done_cb raised {e!r}")
            if not result_future.done():
                result_future.set_result(success)
        def _abort(reason: str, exc: BaseException | None = None) -> None:
            msg = f"arm_controller: ABORT – {reason}"
            if exc: msg += f" ({exc!r})"
            self.node.get_logger().error(msg)
            _finish(False)
        # ─── async chain helpers (each checks the previous step) ────────────────
        def after_open(fut):
            if fut.exception():
                return _abort("gripper open failed", fut.exception())
            self.move_end_effector_to(x=0.15, y=0.0, z=0.3).add_done_callback(after_lifted)
        def after_lifted(fut):
            if fut.exception():
                return _abort("pre-lift failed", fut.exception())
            self.move_end_effector_to(x=x, y=y, z=z).add_done_callback(after_descended)
        def after_descended(fut):
            if fut.exception():
                return _abort("descend failed", fut.exception())
            self.control_gripper("close").add_done_callback(after_closed)
        def after_closed(fut):
            if fut.exception():
                return _abort("gripper close failed", fut.exception())
            self.move_end_effector_to(x=0.15, y=0.0, z=0.3).add_done_callback(after_final_lift)
        def after_final_lift(fut):
            if fut.exception():
                return _abort("final lift failed", fut.exception())
            self.node.get_logger().info("arm_controller: DONE")
            _finish(True)
        # ─── kick things off ────────────────────────────────────────────────────
        self.control_gripper("open").add_done_callback(after_open)
        return result_future

    def pick(self, objectlabel: str, recover=False) -> None:
        # ─── Guards ─────────────────────────────────────────────────────────────
        self.objectlabel = objectlabel
        if not recover:
            if NAV_STATE_GET() == NAV_STATE_BUSY or self.picking or ARM_STATE_GET() != "FREE":
                NAV_STATE_SET(NAV_STATE_FAIL)
                return
        else:
            if self.picking:
                self.node.get_logger().warn("arm_controller: Recovering, but pick already in progress")
                NAV_STATE_SET(NAV_STATE_FAIL)
                return
        NAV_STATE_SET(NAV_STATE_BUSY)
        self.picking = True
        if not objectlabel or objectlabel not in self.semantic_slam.previous_detections:
            self.node.get_logger().info("arm_controller: Pick failed, object location not observed or remembered")
            ARM_STATE_SET("FREE"); NAV_STATE_SET(NAV_STATE_FAIL)
            self.picking = False
            return
        self.correction_attempts = 0
        self.max_corrections     = 50
        # Stop all motion
        def _stop_motion():
            self.cmd_pub.publish(Twist())
        # ─── Timer callback ─────────────────────────────────────────────────────
        def correction_step() -> None:
            # ❶ Latest observation (needed *before* any read)
            (t, _, _, _, _, _, spoint_base_link, imagecoords_depth) = self.semantic_slam.previous_detections[objectlabel]
            # ❷ Abort if object missing for >5 s
            if time.time() - t > 5.0 or spoint_base_link is None:
                back_twist = Twist()
                back_twist.linear.x = -0.15
                self.cmd_pub.publish(back_twist)
                def _after_reverse():
                    nonlocal reverse_timer
                    _stop_motion()
                    self.node.get_logger().info("arm_controller: Object absent → backed away 15 cm")
                    ARM_STATE_SET("FREE"); NAV_STATE_SET(NAV_STATE_FAIL)
                    self.picking = False
                    reverse_timer.cancel()
                reverse_timer = self.node.create_timer(1.0, _after_reverse)
                self.correction_timer.cancel()
                return
            target_point = spoint_base_link.point
            # ❸ Control law
            x_rel, _, depth = imagecoords_depth
            x_err           = (x_rel - 0.5) * 2.0
            depth_err       = depth - GOAL_TARGET_DISTANCE
            twist = Twist()
            if abs(x_err) > yaw_tol:
                twist.angular.z = -k_yaw * x_err
            if abs(depth_err) > depth_tol:
                twist.linear.x = k_fwd * depth_err
            self.cmd_pub.publish(twist)
            self.node.get_logger().info(f"Correction {self.correction_attempts + 1} | "
                                        f"x_err={x_err:+.2f}, depth={depth:.2f}, "
                                        f"cmd=({twist.linear.x:.2f}, {twist.angular.z:.2f})")
            # ❹ Converged?
            if abs(x_err) <= yaw_tol and abs(depth_err) <= depth_tol:
                _stop_motion()
                self.correction_timer.cancel()
                self.node.get_logger().info("arm_controller: Aligned and close. Executing pick.")
                def _on_grasp_done(success: bool):
                    if success:
                        ARM_STATE_SET(objectlabel)
                    else:
                        ARM_STATE_SET("FREE")
                    NAV_STATE_SET(NAV_STATE_SUCCESS if success else NAV_STATE_FAIL)
                    self.picking = False
                self.pick_at(target_point.x, 0.0, target_point.z - GRIPPER_GRIP_HEIGHT, done_cb=_on_grasp_done)
                return
            # ❺ Bail-out if stuck
            self.correction_attempts += 1
            if self.correction_attempts >= self.max_corrections:
                _stop_motion()
                self.correction_timer.cancel()
                self.node.get_logger().info("arm_controller: Too many corrections.")
                ARM_STATE_SET("FREE"); NAV_STATE_SET(NAV_STATE_FAIL)
                self.picking = False
                return
        # ─────────── Start periodic controller ──────────────────────────────────────
        self.correction_timer = self.node.create_timer(CORRECTION_DT, correction_step)

    def drop(self):
        if NAV_STATE_GET() == NAV_STATE_BUSY or ARM_STATE_GET() == "FREE":
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
            ARM_STATE_SET("FREE")
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
                        goal_handle.get_result_async().add_done_callback(lambda res_fut: future.set_result(True))
                    self.arm_client.send_goal_async(goal_msg).add_done_callback(after_traj)
                self.plan_client.call_async(req).add_done_callback(after_plan)
            self.ik_client.call_async(ik_req).add_done_callback(after_ik)
        self.wait_for_joint_states(joint_names).add_done_callback(after_joint_state)
        return future
    
    def control_gripper(self, command):
        future = rclpy.task.Future()
        open_gripper = command == "open"
        self.node.get_logger().info("arm_controller: Gripper open=" + str(open_gripper))
        position = GRIPPER_OPEN_TARGET if open_gripper else GRIPPER_CLOSE_TARGET
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
            goal_handle.get_result_async().add_done_callback(lambda res_fut: future.set_result(True))
        self.gripper_client.send_goal_async(goal_msg).add_done_callback(goal_sent_cb)
        return future

def main():
    rclpy.init()
    armcontroller = ArmController()
    armcontroller.pick_at(0.23, 0.0, 0.05)
    end_time = time.time() + 10.0
    while time.time() < end_time and rclpy.ok():
        rclpy.spin_once(armcontroller.node, timeout_sec=0.1)
    rclpy.shutdown()

if __name__ == '__main__':
    main()
