import rclpy
from rclpy.task import Future
from rclpy.node import Node
from sensor_msgs.msg import JointState, Image
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
from geometry_msgs.msg import PoseStamped
from moveit_msgs.srv import GetPositionFK, GetPositionIK
from scipy.spatial.transform import Rotation as R
import numpy as np
from cv_bridge import CvBridge
from control_msgs.action import GripperCommand
from rclpy.action import ActionClient
from lerobot.policies.diffusion.modeling_diffusion import DiffusionPolicy
from lerobot.policies.act.modeling_act import ACTPolicy
from lerobot.policies.pi0.modeling_pi0 import PI0Policy
from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy
from message_filters import ApproximateTimeSynchronizer, Subscriber # ROS2 message filters
import torch
import argparse
import time
import matplotlib.pyplot as plt
import os
import cv2

from dualprocess_vla import DualProcess_VLA

parser = argparse.ArgumentParser()
parser.add_argument('--model_type', type = str, default = 'diffusion', help = 'diffusion_act_etc')
parser.add_argument('--ckpt_path', type = str, required = True, help = 'path of ckpt')
parser.add_argument('--action_type', type = str, required = True, help = 'model_prediction_type : rel_cart, rel_joint, abs_joint')
parser.add_argument('--prompt', type = str, required = True, help = 'pick, place, open, close')
option_args = parser.parse_args()

def ensure_safe_goal_position(
    goal_present_pos: dict[str, tuple[float, float]],
    max_relative_target: float | dict[str, float]
) -> dict[str, float]:
    """Caps relative action target magnitude for safety."""
    if isinstance(max_relative_target, float):
        diff_cap = dict.fromkeys(goal_present_pos, max_relative_target)
    elif isinstance(max_relative_target, dict):
        if set(goal_present_pos) != set(max_relative_target):
            raise ValueError("max_relative_target keys must match those of goal_present_pos.")
        diff_cap = max_relative_target
    else:
        raise TypeError(f"max_relative_target must be float or dict, got {type(max_relative_target)}")

    warnings_dict = {}
    safe_goal_positions = {}
    for key, (goal_pos, present_pos) in goal_present_pos.items():
        diff = goal_pos - present_pos
        max_diff = diff_cap[key]
        # clamp diff
        safe_diff = min(diff, max_diff)
        safe_diff = max(safe_diff, -max_diff)
        safe_goal_pos = present_pos + safe_diff
        safe_goal_positions[key] = safe_goal_pos
        if abs(safe_goal_pos - goal_pos) > 1e-4:
            warnings_dict[key] = {
                "original goal_pos": goal_pos,
                "safe goal_pos": safe_goal_pos,
            }
    return safe_goal_positions

class PandaController(Node):
    def __init__(self, model_type, ckpt_path, action_type, prompt):
        super().__init__('panda_policy_controller')
        self.joint_state = None
        self.left_image = None
        self.right_image = None
        self.mid_image = None
        self.prev_gripper_state = None
        self.i = 1


        self.bridge = CvBridge()
        self.gripper_flag = 0.0

        self.declare_parameter("control_frequency", 15.0)
        self.control_frequency = self.get_parameter("control_frequency").value
        self.control_period = 1.0 / self.control_frequency
        self.declare_parameter("excution_time_from_start", 1/(self.control_frequency))
        self.excution_time_from_start = self.get_parameter("excution_time_from_start").value

        self.declare_parameter("gripper_interval", 1.5)
        self.gripper_interval = self.get_parameter("gripper_interval").value

        self.joint_state_sub = Subscriber(self, JointState, '/joint_states')
        self.gripper_state_sub = Subscriber(self, JointState, '/robotiq/joint_states')
        self.img_l_sub = Subscriber(self, Image, '/zed_left/zed_node_0/left/image_rect_color')
        self.img_r_sub = Subscriber(self, Image, '/zed_right/zed_node_1/right/image_rect_color')
        self.img_m_sub = Subscriber(self, Image, '/webcam/image_raw')

        self.ts = ApproximateTimeSynchronizer([self.joint_state_sub,
                                               self.gripper_state_sub,
                                               self.img_l_sub,
                                               self.img_r_sub,
                                               self.img_m_sub,
                                               ],
                                               queue_size=1, slop=1/15)
        

        self.ts.registerCallback(self.get_sync_callback)
        self.get_logger().info("Syncing data...")

        # publishers and action clients
        self.trajectory_pub = self.create_publisher(
            JointTrajectory, '/panda/joint_trajectory_controller/joint_trajectory', 1)

        # FK/IK service clients
        self.fk_client = self.create_client(GetPositionFK, '/compute_fk')
        self.ik_client = self.create_client(GetPositionIK, '/compute_ik')
        while not self.fk_client.wait_for_service(timeout_sec=1.0):
            self.get_logger().info('Waiting for FK service...')
        while not self.ik_client.wait_for_service(timeout_sec=1.0):
            self.get_logger().info('Waiting for IK service...')

        self.gripper_client = ActionClient(self, GripperCommand, '/robotiq/robotiq_gripper_controller/gripper_cmd')
        self.gripper_client.wait_for_server()

        # load pretrained policy
        if model_type.lower() == 'diffusion':
            self.policy = DiffusionPolicy.from_pretrained(ckpt_path)
            self.get_logger().info(f'Loaded DiffusionPolicy from {ckpt_path}')
        elif model_type.lower() == 'act':
            self.policy = ACTPolicy.from_pretrained(ckpt_path)
            self.get_logger().info(f'Loaded ActPolicy from {ckpt_path}')
        elif model_type.lower() == 'pi0':
            self.policy = PI0Policy.from_pretrained(ckpt_path)
            self.get_logger().info(f'Loaded PI0Policy from {ckpt_path}')
            self.policy.eval()
        elif model_type.lower() == 'smolvla':
            self.policy = SmolVLAPolicy.from_pretrained(ckpt_path)
            self.get_logger().info(f'Loaded PI0Policy from {ckpt_path}')
        elif model_type.lower() == 'dp_vla':
            self.policy = DualProcess_VLA(
                system1_cfg_path=ckpt_path,
                system2_model_path="Qwen/Qwen2.5-VL-7B-Instruct",
                mode='real'
            )
            self.get_logger().info(f'Loaded DP_VLA Policy from {ckpt_path}')
        
        self.action_type = action_type
        self.model_type = model_type
        self.prompt = prompt
    
    def get_sync_callback(self, msg1, msg2, img1, img2, img3): # (5) abs_tgt_joint, msg5
        self.get_logger().info('>>> get_sync_callback fired')
        # 1. get zed left image data
        img_l = self.bridge.imgmsg_to_cv2(img1, desired_encoding="rgb8")
        self.left_image = img_l[:,180:-100,:] / 255.

        # 2. get zed right image data
        img_r = self.bridge.imgmsg_to_cv2(img2, desired_encoding="rgb8")
        self.right_image = img_r[:,60:-220,:] / 255.

        # 3. get web cam image data
        img_m = self.bridge.imgmsg_to_cv2(img3, desired_encoding="rgb8")
        self.mid_image = img_m[120:,150:-130,:] / 255.
        
        # 4. get robot absolute joint angle data
        self.joint_state = list(msg1.position)[:7]
        self.joint_state.append(msg2.position[0]) # append gripper state
        self.joint_state = np.array(self.joint_state)

        #self.check_time_stamp(msg1, msg2, img1, img2, img3)

    def call_fk_service(self, joint_positions: np.ndarray):
        request = GetPositionFK.Request()
        request.header.frame_id = ''
        request.fk_link_names = ['panda_link8']
        request.robot_state.joint_state.name = [
            'panda_joint1', 'panda_joint2', 'panda_joint3',
            'panda_joint4', 'panda_joint5', 'panda_joint6', 'panda_joint7'
        ]
        request.robot_state.joint_state.position = joint_positions.tolist()
        future = self.fk_client.call_async(request)
        rclpy.spin_until_future_complete(self, future)
        if future.result() is not None:
            fk_result = future.result()
            
            fk_robot_pose = fk_result.pose_stamped[0].pose
            
            rot_base_to_8 = R.from_quat([
                fk_robot_pose.orientation.x,
                fk_robot_pose.orientation.y,
                fk_robot_pose.orientation.z,
                fk_robot_pose.orientation.w,
            ])

            p_offset = np.array([0.0, 0.0, 0.15])
            p_8 = np.array([
                fk_robot_pose.position.x,
                fk_robot_pose.position.y,
                fk_robot_pose.position.z
            ])
            p_tool = rot_base_to_8.apply(p_offset) + p_8
            fk_result.pose_stamped[0].pose.position.x = float(p_tool[0])
            fk_result.pose_stamped[0].pose.position.y = float(p_tool[1])
            fk_result.pose_stamped[0].pose.position.z = float(p_tool[2])
            return fk_result
        else:
            self.get_logger().error('Failed to call FK service')
            return None
        
    def call_ik_service(self, pose: PoseStamped):
        # pose_tool to pose_8
        rot_base_to_tool = R.from_quat([
            pose.pose.orientation.x,
            pose.pose.orientation.y,
            pose.pose.orientation.z,
            pose.pose.orientation.w
        ])
        p_tool = np.array([
            pose.pose.position.x,
            pose.pose.position.y,
            pose.pose.position.z])
        p_offset = np.array([0.0, 0.0, 0.15])
        p_8 = p_tool - rot_base_to_tool.apply(p_offset)

        pose.pose.position.x = p_8[0]
        pose.pose.position.y = p_8[1]
        pose.pose.position.z = p_8[2]
        
        request = GetPositionIK.Request()
        request.ik_request.group_name = 'panda_arm'
        request.ik_request.pose_stamped.header.frame_id = ''
        request.ik_request.pose_stamped.pose.position = pose.pose.position
        request.ik_request.pose_stamped.pose.orientation = pose.pose.orientation
        request.ik_request.robot_state.joint_state.name = [
            'panda_joint1', 'panda_joint2', 'panda_joint3',
            'panda_joint4', 'panda_joint5', 'panda_joint6', 'panda_joint7'
        ]
        request.ik_request.robot_state.joint_state.position = self.joint_state[:7].tolist()
        future = self.ik_client.call_async(request)
        rclpy.spin_until_future_complete(self, future)
        if future.result() is not None:
            return future.result()
        else:
            self.get_logger().error('Failed to call IK service')
            return None
    
    def compute_new_pose(self, current_robot_pose, delta):
        converted_delta_translation = delta[:3]
        current_position = np.array([
            current_robot_pose.position.x,
            current_robot_pose.position.y,
            current_robot_pose.position.z
        ])
        new_robot_position = current_position + converted_delta_translation

        scaled_rotvec = delta[3:6]
        converted_delta_rot = R.from_rotvec(scaled_rotvec)

        current_robot_rot = R.from_quat([
            current_robot_pose.orientation.x,
            current_robot_pose.orientation.y,
            current_robot_pose.orientation.z,
            current_robot_pose.orientation.w
        ])

        new_robot_rot = converted_delta_rot * current_robot_rot
        new_robot_quat = new_robot_rot.as_quat()  # [x, y, z, w]

        new_pose = PoseStamped()
        new_pose.header.stamp = self.get_clock().now().to_msg()
        new_pose.header.frame_id = ''
        new_pose.pose.position.x = float(new_robot_position[0])
        new_pose.pose.position.y = float(new_robot_position[1])
        new_pose.pose.position.z = float(new_robot_position[2])
        new_pose.pose.orientation.x = float(new_robot_quat[0])
        new_pose.pose.orientation.y = float(new_robot_quat[1])
        new_pose.pose.orientation.z = float(new_robot_quat[2])
        new_pose.pose.orientation.w = float(new_robot_quat[3])
        
        return new_pose
    
    def publish_trajectory(self, joint_positions):
        # 로봇 궤적
        traj_msg = JointTrajectory()
        traj_msg.joint_names = [
            'panda_joint1', 'panda_joint2', 'panda_joint3',
            'panda_joint4', 'panda_joint5', 'panda_joint6', 'panda_joint7'
        ]
        point = JointTrajectoryPoint() # 목표관절 상태
        point.positions = joint_positions
        point.time_from_start.nanosec = int(1000000000 * self.excution_time_from_start) # 로봇이 포인트에 도달하는데 걸리는 시간을 지정
        traj_msg.points = [point] # 생성한 포인트를 리스트형태로 저장
        self.trajectory_pub.publish(traj_msg)
    
    def gripper_send_goal(self, position: float, max_effort: float):
        # 그리퍼 제어
        goal_msg = GripperCommand.Goal()
        goal_msg.command.position = position
        goal_msg.command.max_effort = max_effort
        
        #self.get_logger().info(f'Sending goal: position={position}, max_effort={max_effort}')
        self.gripper_client.wait_for_server()
        future = self.gripper_client.send_goal_async(goal_msg, feedback_callback=self.gripper_feedback_callback)
        future.add_done_callback(self.gripper_goal_response_callback)

    def gripper_goal_response_callback(self, future: Future):
        # 그리퍼 제어 목표 수락 여부 확인 후 결과 처리
        goal_handle = future.result()
        if not goal_handle.accepted:
            self.get_logger().info('Goal rejected')
            return
        
        #self.get_logger().info('Goal accepted, waiting for result...')
        result_future = goal_handle.get_result_async()
        result_future.add_done_callback(self.gripper_result_callback)

    def gripper_feedback_callback(self, feedback_msg):
        # 액션 실행 중에 발생하는 피드백 메세지를 처리
        self.get_logger().info(f'Feedback: {feedback_msg.feedback}')

    def gripper_result_callback(self, future: Future):
        # 액션 실행이 완료된 후 최종 결과를 받아 로그에 출력
        result = future.result().result
        self.get_logger().info(f'Result: {result}')


    def check_time_stamp(self, msg1, msg2, img1, img2, img3): # (11) abs_tgt_joint

        ts1 = msg1.header.stamp.sec + msg1.header.stamp.nanosec * 1e-9
        ts2 = msg2.header.stamp.sec + msg2.header.stamp.nanosec * 1e-9
        ts3 = img1.header.stamp.sec + img1.header.stamp.nanosec * 1e-9
        ts4 = img2.header.stamp.sec + img2.header.stamp.nanosec * 1e-9
        ts5 = img3.header.stamp.sec + img3.header.stamp.nanosec * 1e-9
        

        self.data_idx += 1
        log_messages = [
            f"data index: {self.data_idx}_timestamp",
            "===============================================================",
            f"joint state:                  {ts1} ",
            f"gripper state:                    {ts2}",
            f"left image:                   {ts3} ",
            f"right image:                  {ts4} ",
            f"mid image:                    {ts5} ",
            "==============================================================="
        ]
        self.get_logger().info("\n".join(log_messages))

        self.prev_ts1 = ts1
        self.prev_ts2 = ts2
        self.prev_ts3 = ts3
        self.prev_ts4 = ts4
        self.prev_ts5 = ts5

    def control_loop(self):
        if self.joint_state is None or self.left_image is None or self.right_image is None or self.mid_image is None:
            return
        
        # [1] Model predicts trajectory
        p_start = time.time()
        device = next(self.policy.parameters()).device
        if self.model_type.lower() == 'dp_vla':
            observation = {
                'task': self.prompt,
                'observation.state': self.joint_state,
                'observation.left_image': self.left_image,
                'observation.right_image': self.right_images,
                'observation.mid_image' : self.mid_image
            }
        elif self.model_type.lower() == 'pi0' or self.model_type.lower() == 'smolvla':
            observation = {
                'task' : [self.prompt],
                'observation.state': torch.from_numpy(self.joint_state).type(torch.float32).unsqueeze(0).to(device),
                'observation.left_image': torch.from_numpy(self.left_image).permute(2,0,1).unsqueeze(0).type(torch.float32).to(device),
                'observation.right_image': torch.from_numpy(self.right_image).permute(2,0,1).unsqueeze(0).type(torch.float32).to(device),
                'observation.mid_image' : torch.from_numpy(self.mid_image).permute(2,0,1).unsqueeze(0).type(torch.float32).to(device)
                }
        else:
            observation = {
                'observation.state': torch.from_numpy(self.joint_state).type(torch.float32).unsqueeze(0).to(device),
                'observation.left_image': torch.from_numpy(self.left_image).permute(2,0,1).unsqueeze(0).type(torch.float32).to(device),
                'observation.right_image': torch.from_numpy(self.right_image).permute(2,0,1).unsqueeze(0).type(torch.float32).to(device),
                'observation.mid_image' : torch.from_numpy(self.mid_image).permute(2,0,1).unsqueeze(0).type(torch.float32).to(device)
                }
            

        if self.model_type.lowel() == 'dp_vla':
            if self.is_first:
                self.policy.start_system2_thread(obs_dict = observation)
                self.is_first = False
            preds = self.policy.forward_system1(obs_dict = observation)
        elif self.model_type.lower() == 'pi0':
            with torch.inference_mode():
                with torch.no_grad():
                    preds = self.policy.select_action(observation)
                    #print(preds.shape)
                    #imgs, masks = self.policy.prepare_images(observation)    # batch에 이미지/상태/태스크 포함
                    #emb = self.policy.model.paligemma_with_expert.embed_image(imgs[0])
                    #print("emb mean/std:", emb.mean().item(), emb.std().item())
        else:
            with torch.inference_mode():
                preds = self.policy.select_action(observation)

        if not isinstance(preds, np.ndarray):
            preds = preds.cpu().numpy()

        print(f'action_num : {self.i}')
        self.i += 1

        cycle_start = time.time()

        if self.action_type == 'rel_cart':
            delta = preds[0]
            
            # [2] Solve FK : current_cartesian_pose
            fk_result = self.call_fk_service(self.joint_state[:7])
            if fk_result is None:
                return
            
            # [3] Compute target cartesian pose
            current_robot_pose = fk_result.pose_stamped[0].pose
            new_pose = self.compute_new_pose(current_robot_pose, delta)
            
            # [4] Solve IK : target joint angle
            ik_result = self.call_ik_service(new_pose)
            if ik_result is None:
                return
            
            # [5] Publish robot target joint angle
            target_joints = ik_result.solution.joint_state.position[:7]
            self.publish_trajectory(target_joints)
            
            # [6] Send gripper goal action
            curr_gripper_state = float(delta[-1])
            if self.prev_gripper_state is None or curr_gripper_state != self.prev_gripper_state:
                self.gripper_send_goal(position=float(curr_gripper_state), max_effort=5.0)
                self.prev_gripper_state = curr_gripper_state

        elif self.action_type == 'rel_joint':
            delta = preds[0]
            current_joint_angle = self.joint_state[:7]
            target_joints = current_joint_angle + delta[:7]
            target_joints = [float(x) for x in target_joints]
            self.publish_trajectory(target_joints)
            curr_gripper_state = delta[7]
            if self.prev_gripper_state is None or curr_gripper_state != self.prev_gripper_state:
                self.gripper_send_goal(position=float(curr_gripper_state), max_effort=5.0)
                self.prev_gripper_state = curr_gripper_state

        elif self.action_type == 'abs_joint':
            target_joints = preds[0]
            
            target_joints = [float(x) for x in target_joints]
            #print(target_joints[:7])

            ###########################################################################
            goal = preds[0][:7]
            present = self.joint_state[:7].tolist()

            #raw_vel_limits = [2.1750]*4 + [2.6100]*3
            raw_vel_limits = [2.617]*4 + [3.1415]*3 
            scale          = 0.1
            dt             = self.control_period
            vel_limits     = [v * scale for v in raw_vel_limits]
            max_deltas     = [v * dt    for v in vel_limits]
            gp      = { f"joint{i}": (goal[i], present[i]) for i in range(7) }
            max_rel = { f"joint{i}": max_deltas[i]           for i in range(7) }
            
            safe     = ensure_safe_goal_position(gp, max_rel)
            safe_goals = [ safe[f"joint{i}"] for i in range(7) ]
            print("clamped abs_joint:", safe_goals)
            self.publish_trajectory(safe_goals)
            ###########################################################################
            #self.publish_trajectory(target_joints[:7])
            curr_gripper_state = target_joints[7]
            if self.prev_gripper_state is None or curr_gripper_state != self.prev_gripper_state:
                self.gripper_send_goal(position=float(curr_gripper_state), max_effort=5.0)
                self.prev_gripper_state = curr_gripper_state
        
        # Wait for cycle time to match
        cycle_elapsed = time.time() - cycle_start
        #print("{:.10f}".format(cycle_elapsed))
        remaining = self.control_period - cycle_elapsed
        print(remaining)
        if remaining > 0:
            time.sleep(remaining)
        #self.get_logger().info(f"{1.0/(time.time()-cycle_start)} Hz")

    def destroy(self):
        super().destroy_node()

def main(option_args = option_args):
    rclpy.init()
    node = PandaController(option_args.model_type, option_args.ckpt_path, option_args.action_type, option_args.prompt)
    try:
        # spin_once 로 콜백을 처리하면서 직접 control_loop 호출
        while rclpy.ok():
            node.joint_state = None
            #cycle_start = time.time()

            while node.joint_state is None:
                rclpy.spin_once(node)
            node.control_loop()

            #cycle_elapsed = time.time() - cycle_start
            #remaining = 0.01 - cycle_elapsed
            #if remaining > 0:
            #    time.sleep(remaining)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__=='__main__':
    main()
