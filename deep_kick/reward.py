from abc import ABC
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from deep_kick.env import DeepKickEnv

import math
import numpy as np
import rospy
from std_msgs.msg import Float32
import pybullet as p


class AbstractReward(ABC):
    def __init__(self, lower_bound, upper_bound):
        self.name = self.__class__.__name__
        self.lower_bound = lower_bound
        self.upper_bound = upper_bound
        self.episode_reward = 0
        self.current_reward = 0
        self.publisher = rospy.Publisher(self.name, Float32, queue_size=1)

    def get_lower_bound(self):
        return self.lower_bound

    def get_upper_bound(self):
        return self.upper_bound

    def reset_episode_reward(self):
        self.episode_reward = 0

    def get_episode_reward(self):
        return self.episode_reward

    def get_name(self):
        return self.name

    def publish_reward(self, env: "DeepKickEnv"):
        if self.publisher.get_num_connections() > 0:
            self.publisher.publish(Float32(self.current_reward))

    def compute_reward(self, env: "DeepKickEnv"):
        raise NotImplementedError

    def compute_current_reward(self, env: "DeepKickEnv"):
        self.current_reward = self.compute_reward(env)
        self.episode_reward += self.current_reward

        return self.current_reward

    def get_info_dict(self):
        return {}


class CombinedReward(AbstractReward):
    def __init__(self, list_reward_classes):
        self.rewards = []
        lower_bound = 0
        upper_bound = 0
        for reward in list_reward_classes:
            self.rewards.append(reward)
            lower_bound += reward.get_lower_bound()
            upper_bound += reward.get_upper_bound()

        super().__init__(lower_bound, upper_bound)

    def compute_current_reward(self, env):
        self.current_reward = 0
        for reward_type in self.rewards:
            self.current_reward += reward_type.compute_current_reward(env)
        self.episode_reward += self.current_reward
        return self.current_reward

    def reset_episode_reward(self):
        self.episode_reward = 0
        for reward in self.rewards:
            reward.reset_episode_reward()

    def get_info_dict(self):
        info = {
            reward.get_name(): reward.episode_reward for reward in self.rewards
        }
        return info

    def publish_reward(self, env):
        self.publisher.publish(Float32(self.current_reward))
        for reward in self.rewards:
            reward.publish_reward(env)


class WeightedCombinedReward(CombinedReward):
    def __init__(self, reward_classes, weights):
        self.weights = weights
        self.rewards = reward_classes
        super().__init__(reward_classes)

    def compute_current_reward(self, env):
        self.current_reward = 0
        # weight the rewards
        for i in range(0, len(self.rewards)):
            self.current_reward += self.weights[i] * self.rewards[i].compute_current_reward(env)
        self.episode_reward += self.current_reward
        return self.current_reward


class DeepMimicReward(WeightedCombinedReward):
    def __init__(self):
        self.list_reward_classes = [JointPositionReward(factor=2),
                                    JointVelocityReward(factor=0.1),
                                    EndEffectorReward(factor=40),
                                    RootPositionReward(factor=10),
                                    BallVelocityReward()]
        reference_weights = np.array([0.65, 0.1, 0.15, 0.1])
        task_weights = np.array([1])
        self.weights = np.concatenate([reference_weights * 0.7, task_weights * 0.3])
        super().__init__(self.list_reward_classes, self.weights)


class JointPositionReward(AbstractReward):
    """This reward function rewards joint positions similar to those in a reference action"""
    def __init__(self, factor=5):
        self.factor = factor
        super().__init__(0, 1)

    def compute_reward(self, env):
        diff_sum = 0
        for joint_name in env.robot.used_joint_names:
            # take difference between joint in simulation and joint in reference trajectory
            current_joint_position = env.robot.joints[joint_name].get_position()
            reference_position = env.current_frame.joint_positions[joint_name]
            position_diff = reference_position - current_joint_position
            # add squared diff to sum
            diff_sum += position_diff ** 2

        error = math.e ** (-self.factor * diff_sum)
        return error


class JointVelocityReward(AbstractReward):
    """This reward function rewards joint velocities similar to the joint velocities of a reference action"""

    def __init__(self, factor=0.1):
        self.factor = factor
        super().__init__(0, 1)

    def compute_reward(self, env):
        if not env.previous_frame:
            # we don't have enough data yet
            return 0

        diff_sum = 0
        for joint_name in env.robot.used_joint_names:
            # take difference between joint in simulation and joint in reference trajectory
            joint_velocity = env.robot.joints[joint_name].get_velocity()
            # target velocity with difference quotient
            ref_joint_diff = (env.current_frame.joint_positions[joint_name] -
                              env.previous_frame.joint_positions[joint_name])
            ref_joint_velocity = ref_joint_diff / env.policy_timestep

            # add  squared diff to sum
            diff_sum += (joint_velocity - ref_joint_velocity) ** 2

        error = math.exp(-self.factor * diff_sum)
        return error


class EndEffectorReward(AbstractReward):
    """This reward function rewards a correct position of end effectors in the world compared to a reference state"""
    def __init__(self, factor):
        self.factor = factor
        super().__init__(0, 1)

    def compute_reward(self, env):
        left_leg, right_leg = env.robot.get_legs_in_world()
        ref_left_leg, ref_right_leg = env.refbot.get_legs_in_world()
        left_distance = np.linalg.norm(left_leg - ref_left_leg)
        right_distance = np.linalg.norm(right_leg - ref_right_leg)
        return math.exp(-self.factor * (left_distance**2 + right_distance**2))


class RootPositionReward(AbstractReward):
    """This reward function rewards a center of mass close to that of a reference action."""
    def __init__(self, factor):
        self.factor = factor
        super().__init__(0, 1)

    def compute_reward(self, env):
        # we assume center of mass = base link
        position, _ = env.robot.get_pose_world_frame()
        ref_position = env.current_frame.robot_position
        # distance between position and ref_position
        distance = np.linalg.norm(np.array(position) - np.array(ref_position))
        return math.exp(-self.factor * distance**2)


class EndPositionReachedReward(AbstractReward):
    def __init__(self):
        super().__init__(0, 1)

    def compute_reward(self, env: "DeepKickEnv"):
        # Whether the ball moved in the last time step
        ball_moved = np.linalg.norm(env.last_ball_position - env.get_ball_position()) > 0.001
        left_x, left_y, _ = env.robot.left_foot_pose[0]
        right_x, right_y, _ = env.robot.right_foot_pose[0]
        left_z = env.robot.left_world_pos[2]
        right_z = env.robot.right_world_pos[2]
        # correct position would be [0, +-0.1, -0.3], but we give sole tolerance
        left_at_end = (-0.05 - env.foot_goal_tolerance <= left_x <= -0.05 + env.foot_goal_tolerance and
                       0 <= left_z <= 0.05 + env.foot_goal_tolerance)
        right_at_end = (-0.05 - env.foot_goal_tolerance <= right_x <= -0.05 + env.foot_goal_tolerance and
                        0 <= right_z <= 0.05 + env.foot_goal_tolerance)
        foot_distance_good = 0.15 <= abs(left_y - right_y) <= 0.25
        # whether angular velocities are almost zero
        angular_vels = env.robot.velocity[1]
        ang_vel_zero = all(abs(vel) < env.angular_velocity_tolerance for vel in angular_vels)

        # regular termination is when the demonstration is finished, the ball is no longer moving,
        # the feet are at their terminal position and the angular velocities are zero
        regular_termination = env.demonstration_finished and not ball_moved and left_at_end and right_at_end and ang_vel_zero and foot_distance_good

        return 1 if regular_termination else 0


class ActionPossibleReward(AbstractReward):
    def __init__(self):
        super().__init__(-1, 0)

    def compute_reward(self, env):
        if env.action_possible:
            return 0
        else:
            return -1


class DeepMimicStrikeReward(AbstractReward):
    """This rewards the distance of the foot from the ball, and gives one if the ball has been hit"""
    def __init__(self, factor=4):
        super().__init__(0, 1)
        self.factor = factor

    def compute_reward(self, env: "DeepKickEnv"):
        if env.ball_moved:
            return 1
        else:
            if env.is_left_kick:
                kick_foot_position = np.array(env.robot.left_world_pos)
            else:
                kick_foot_position = np.array(env.robot.right_foot_pose)
            ball_position = np.append(env.get_ball_position(), env.ball_radius)
            distance = np.linalg.norm(kick_foot_position - ball_position)
            return math.exp(-self.factor * distance**2)


class BallVelocityReward(AbstractReward):
    """Reward for the ball velocity, scaled exponentially"""
    def __init__(self, factor=2):
        super().__init__(0, 1)
        self.factor = factor

    def compute_reward(self, env: "DeepKickEnv"):
        if env.use_demonstration and not env.kick_time <= env.demonstration_time <= env.kick_time + 0.5:
            return 0

        ball_change = np.linalg.norm(env.last_ball_position - env.get_ball_position())
        ball_velocity = ball_change / env.policy_timestep

        return 1 - math.exp(-self.factor * ball_velocity)


class ExponentialBallDistanceReward(AbstractReward):
    """This rewards the distance of the ball from its start position, but exponentially"""
    def __init__(self, factor=4):
        super().__init__(0, 10)
        self.factor = factor

    def compute_reward(self, env: "DeepKickEnv"):
        if not env.done:
            # We only want to give this reward once, when we are done
            return 0
        ball_start_position = np.array((env.ball_x, env.ball_y))
        current_ball_position = env.get_ball_position()
        distance = np.linalg.norm(ball_start_position - current_ball_position)
        reward = 1 - math.exp(-self.factor * distance)
        if env.robot.is_alive():
            return reward
        else:
            # If the robot fell, don't reward it as much
            return 0.5 * reward


class BallDistanceReward(AbstractReward):
    """This rewards the distance of the ball from its start position"""
    def __init__(self, factor=1):
        super().__init__(0, 10)
        self.factor = factor

    def compute_reward(self, env: "DeepKickEnv"):
        ball_start_position = np.array((env.ball_x, env.ball_y))
        current_ball_position = env.get_ball_position()
        distance = np.linalg.norm(ball_start_position - current_ball_position)
        return self.factor * distance


class DemonstrationFinishedReward(AbstractReward):
    """This gives negative reward when the demonstration is finished but we are not"""
    def __init__(self):
        super().__init__(-1, 0)

    def compute_reward(self, env: "DeepKickEnv"):
        if env.demonstration_finished:
            return -1
        else:
            return 0
