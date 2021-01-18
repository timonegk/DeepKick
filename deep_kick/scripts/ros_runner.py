#!/usr/bin/env python3
import math
import os.path
import pickle
from collections import defaultdict

import numpy as np
import rospy
import actionlib
import stable_baselines3
from bitbots_moveit_bindings import get_position_fk, get_position_ik
from bitbots_msgs.msg import KickAction, JointCommand, FootPressure, KickGoal, KickFeedback
from geometry_msgs.msg import PoseStamped, Point, Quaternion
from moveit_msgs.srv import GetPositionFKRequest, GetPositionFKResponse, GetPositionIKRequest, GetPositionIKResponse
from sensor_msgs.msg import Imu, JointState
from transforms3d.affines import compose, decompose
from transforms3d.euler import quat2euler, mat2euler, euler2mat, euler2quat
from transforms3d.quaternions import quat2mat
from urdf_parser_py.urdf import URDF
from stable_baselines3.common.vec_env.vec_normalize import VecNormalize

from deep_kick.state import PhaseState, OrientationState


class DummyRobot:
    imu_rpy = (0, 0)
    velocity = ((0, 0, 0), (0, 0, 0))


class DummyEnv:
    robot = DummyRobot()
    progress = 0


class DeepKick:
    def __init__(self):
        path = os.path.join(os.path.dirname(__file__), '../../rl-baselines3-zoo/logs/cl04/ppo/WolfgangBulletEnv-v1_67/')
        # needed in python >= 3.8
        custom_objects = {
            "learning_rate": 0.0,
            "lr_schedule": lambda _: 0.0,
            "clip_range": lambda _: 0.0,
        }
        self.policy = stable_baselines3.PPO.load(os.path.join(path, 'best_model.zip'), custom_objects=custom_objects)
        self.vecnormalize = pickle.load(open(os.path.join(path, 'WolfgangBulletEnv-v1', 'vecnormalize.pkl'), 'rb'))  # type: VecNormalize
        self.env = DummyEnv()
        self.joint_names = ["LAnklePitch", "LAnkleRoll", "LHipPitch", "LHipRoll", "LHipYaw", "LKnee",
                            "RAnklePitch", "RAnkleRoll", "RHipPitch", "RHipRoll", "RHipYaw", "RKnee"]
        self.state = OrientationState(self.env)
        self.joint_publisher = rospy.Publisher('kick_motor_goals', JointCommand, queue_size=1)
        self.imu_sub = rospy.Subscriber('imu/data', Imu, self.imu_cb, queue_size=1)
        self.joint_state_sub = rospy.Subscriber('joint_states', JointState, self.joint_state_cb, queue_size=1)
        self.left_pressure_sub = rospy.Subscriber('foot_pressure_filtered/left', FootPressure,
                                                  lambda msg: self.foot_pressure_cb(msg, True),
                                                  queue_size=1)
        self.right_pressure_sub = rospy.Subscriber('foot_pressure_filtered/right', FootPressure,
                                                   lambda msg: self.foot_pressure_cb(msg, False),
                                                   queue_size=1)

        self.pose_scaling_factors_left = [0.2, 0.1, -0.08, math.tau / 12, math.tau / 12,
                                          math.tau / 12]  # todo maybe these limits are to hard
        self.pose_scaling_factors_right = self.pose_scaling_factors_left
        # shift center by following m
        leg_height_offset = -0.32
        leg_sidewards_offset = 0.1
        self.pose_scaling_addends_left = [0,
                                          leg_sidewards_offset / self.pose_scaling_factors_left[1],
                                          leg_height_offset / self.pose_scaling_factors_left[2],
                                          0, 0, 0]
        self.pose_scaling_addends_right = self.pose_scaling_addends_left.copy()
        self.pose_scaling_addends_right[1] *= -1



        self.last_joint_state_time = 0
        urdf = URDF.from_xml_string(rospy.get_param('robot_description'))
        self.joint_limits = {
            joint.name: (joint.limit.lower, joint.limit.upper) for joint in urdf.joints if joint.limit
        }
        self.action_server = actionlib.SimpleActionServer('dynamic_kick', KickAction,
                                                          execute_cb=self.execute_cb, auto_start=False)
        self.action_server.start()

    def execute_cb(self, goal: KickGoal):
        r = rospy.Rate(30)
        self.env.progress = 0
        while self.env.progress <= 2:
            feedback = KickFeedback()
            feedback.percent_done = int(self.env.progress * 100)
            self.action_server.publish_feedback(feedback)
            observation = self.state.get_state_array(scaled=True)
            observation = self.vecnormalize.normalize_obs(observation)
            self.state.publish_debug()
            action, state = self.policy.predict(observation, deterministic=True)
            #msg = self.get_joint_action(action)
            msg = self.get_cartesian_action(action)
            self.joint_publisher.publish(msg)
            self.env.progress += 0.02
            r.sleep()
        self.action_server.set_succeeded()

    def get_joint_action(self, action):
        msg = JointCommand()
        msg.header.stamp = rospy.Time.now()
        msg.joint_names = self.joint_names
        msg.positions = self.scale_action_to_motor_goal(action)
        msg.velocities = [-1] * len(self.joint_names)
        return msg

    def scale_action_to_pose(self, pos_left, rpy_left, pos_right, rpy_right):
        scaled_pos_left = []
        scaled_rpy_left = []
        for i in range(3):
            scaled_pos_left.append(
                self.pose_scaling_factors_left[i] * (pos_left[i] + self.pose_scaling_addends_left[i]))
        for i in range(3, 6):
            scaled_rpy_left.append(
                self.pose_scaling_factors_left[i] * (rpy_left[i - 3] + self.pose_scaling_addends_left[i]))
        scaled_pos_right = []
        scaled_rpy_right = []
        for i in range(3):
            scaled_pos_right.append(
                self.pose_scaling_factors_right[i] * (pos_right[i] + self.pose_scaling_addends_right[i]))
        for i in range(3, 6):
            scaled_rpy_right.append(
                self.pose_scaling_factors_right[i] * (rpy_right[i - 3] + self.pose_scaling_addends_right[i]))
        return scaled_pos_left, scaled_rpy_left, scaled_pos_right, scaled_rpy_right

    def get_cartesian_action(self, action):
        left_pos, left_rpy, right_pos, right_rpy = self.scale_action_to_pose(
            action[0:3],
            action[3:6],
            action[6:9],
            action[9:12])

        request = GetPositionIKRequest()
        request.ik_request.avoid_collisions = True
        request.ik_request.group_name = "LeftLeg"
        request.ik_request.ik_link_name = "l_sole"
        request.ik_request.pose_stamped.pose.position = Point(*left_pos)
        quat = euler2quat(*left_rpy)
        request.ik_request.pose_stamped.pose.orientation = Quaternion(quat[1], quat[2], quat[3], quat[0])
        ik_result = get_position_ik(request, approximate=True)
        first_error_code = ik_result.error_code.val
        request.ik_request.group_name = "RightLeg"
        request.ik_request.ik_link_name = "r_sole"
        request.ik_request.pose_stamped.pose.position = Point(*right_pos)
        quat = euler2quat(*right_rpy)
        request.ik_request.pose_stamped.pose.orientation = Quaternion(quat[1], quat[2], quat[3], quat[0])
        ik_result = get_position_ik(request, approximate=True)
        # check if no solution or collision
        msg = JointCommand()
        msg.header.stamp = rospy.Time.now()
        for name, position in zip(ik_result.solution.joint_state.name, ik_result.solution.joint_state.position):
            msg.joint_names.append(name)
            msg.positions.append(position)
            msg.velocities.append(-1)
        return msg

    def scale_action_to_motor_goal(self, action):
        res = []
        for name, scaled_position in zip(self.joint_names, action):
            lower, upper = self.joint_limits[name]
            res.append((scaled_position * (upper - lower) + upper + lower) / 2)
        return res

    def imu_cb(self, msg: Imu):
        orientation = quat2euler((msg.orientation.w, msg.orientation.x, msg.orientation.y, msg.orientation.z))
        self.env.robot.imu_rpy = orientation
        self.env.robot.velocity = ((0, 0, 0), (msg.angular_velocity.x, msg.angular_velocity.y, msg.angular_velocity.z))

    def foot_pressure_cb(self, msg: FootPressure, is_left: bool):
        return
        # LLB, LLF, LRF, LRB, RLB, RLF, RRF, RRB
        pressures = [msg.left_back, msg.left_front, msg.right_front, msg.right_back]
        prefix = 'L' if is_left else 'R'
        self.env.robot.pressure_sensors[prefix + 'LB'].pressure = pressures[0]
        self.env.robot.pressure_sensors[prefix + 'LF'].pressure = pressures[1]
        self.env.robot.pressure_sensors[prefix + 'RF'].pressure = pressures[2]
        self.env.robot.pressure_sensors[prefix + 'RB'].pressure = pressures[3]

    def joint_state_cb(self, msg: JointState):
        request = GetPositionFKRequest()
        request.robot_state.joint_state = msg
        request.fk_link_names = ['l_sole', 'r_sole']
        response: GetPositionFKResponse = get_position_fk(request)
        new_left_pose = self.msg2array(response.pose_stamped[0])
        new_right_pose = self.msg2array(response.pose_stamped[1])
        if self.last_joint_state_time != 0:
            # calc velocities
            time_diff = (msg.header.stamp - self.last_joint_state_time).to_sec()
            left_lin, left_ang = self.get_velocities(self.env.robot.left_foot_pose, new_left_pose, time_diff)
            right_lin, right_ang = self.get_velocities(self.env.robot.right_foot_pose, new_right_pose, time_diff)
            self.env.robot.left_foot_vel = (left_lin, left_ang)
            self.env.robot.right_foot_vel = (right_lin, right_ang)

        left_rpy = quat2euler(new_left_pose[1])
        right_rpy = quat2euler(new_right_pose[1])
        self.env.robot.left_foot_pose = new_left_pose[0], left_rpy
        self.env.robot.right_foot_pose = new_right_pose[0], right_rpy

        self.last_joint_state_time = msg.header.stamp

    def msg2array(self, msg: PoseStamped):
        pose_array = ((msg.pose.position.x,
                       msg.pose.position.y,
                       msg.pose.position.z),
                      (msg.pose.orientation.w,
                       msg.pose.orientation.x,
                       msg.pose.orientation.y,
                       msg.pose.orientation.z))
        return pose_array

    def get_velocities(self, old_pose, new_pose, time_diff):
        new_mat = quat2mat(new_pose[1])
        # old_pose is in rpy
        old_mat = euler2mat(*old_pose[1])
        new_transform = compose(np.array(new_pose[0]), new_mat, np.ones(3))
        old_transform = compose(np.array(old_pose[0]), old_mat, np.ones(3))
        lin_diff, ang_diff, _, _ = decompose(np.matmul(np.linalg.inv(new_transform), old_transform))
        linear_velocity = lin_diff / time_diff
        angular_velocity = np.array(mat2euler(ang_diff)) / time_diff
        return linear_velocity, angular_velocity


if __name__ == '__main__':
    rospy.init_node('deep_kick')
    DeepKick()
