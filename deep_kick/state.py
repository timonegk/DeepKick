from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from deep_kick.env import DeepKickEnv

import math
import numpy as np
from rospy import Publisher
from std_msgs.msg import Float32MultiArray


class State:
    def __init__(self, env: "DeepKickEnv"):
        self.env = env
        self.debug_names = None
        self.debug_publishers = {}

    def publish_debug(self):
        entries = self.get_state_entries(True)
        if len(self.debug_publishers.keys()) == 0:
            # initialize publishers
            for name in entries.keys():
                self.debug_publishers[name] = Publisher("state_" + name, Float32MultiArray, queue_size=1)
        for entry_name in entries.keys():
            publisher = self.debug_publishers[entry_name]
            if publisher.get_num_connections() > 0:
                publisher.publish(Float32MultiArray(data=entries[entry_name]))

    def get_state_entries(self, scaled):
        raise NotImplementedError

    def get_state_array(self, scaled):
        return np.concatenate(list(self.get_state_entries(scaled).values()))

    def get_num_observations(self):
        return len(self.get_state_array(True))


class PhaseState(State):
    def get_state_entries(self, scaled):
        output = dict()
        output["time"] = [self.env.progress]
        return output


class PhaseCommandState(State):
    def get_state_entries(self, scaled):
        output = dict()
        output["time"] = [self.env.progress]
        output["command"] = self.env.current_command
        return output


class OrientationState(PhaseState):
    def get_state_entries(self, scaled):
        output = super().get_state_entries(scaled)
        output["roll"] = [self.env.robot.imu_rpy[0] / (math.tau / 2)] if scaled else [self.env.robot.imu_rpy[0]]
        output["pitch"] = [self.env.robot.imu_rpy[1] / (math.tau / 2)] if scaled else [self.env.robot.imu_rpy[1]]
        return output


class GyroState(OrientationState):
    def get_state_entries(self, scaled):
        output = super().get_state_entries(scaled)
        output["ang_vel"] = self.env.robot.velocity[1] / np.array(10) if scaled else self.env.robot.velocity[1]
        return output


class FootState(PhaseState):
    def get_state_entries(self, scaled):
        output = super().get_state_entries(scaled)
        output["left_pos"] = self.env.robot.left_foot_pose[0]
        output["left_rot"] = self.env.robot.left_foot_pose[1] / np.array(math.tau / 2) if scaled else \
            self.env.robot.left_foot_pose[1]
        output["right_pos"] = self.env.robot.right_foot_pose[0]
        output["right_rot"] = self.env.robot.right_foot_pose[1] / np.array(math.tau / 2) if scaled else \
            self.env.robot.right_foot_pose[1]
        return output


class FootVelocityState(FootState):
    def get_state_entries(self, scaled):
        output = super().get_state_entries(scaled)
        output["left_lin"] = self.env.robot.left_foot_vel[0] / np.array(5) if scaled else \
            self.env.robot.left_foot_vel[0]
        output["left_ang"] = self.env.robot.left_foot_vel[1] / np.array(4 * math.tau) if scaled else \
            self.env.robot.right_foot_vel[1]
        output["right_lin"] = self.env.robot.right_foot_vel[0] / np.array(5) if scaled else \
            self.env.robot.right_foot_vel[0]
        output["right_ang"] = self.env.robot.right_foot_vel[1] / np.array(4 * math.tau) if scaled else \
            self.env.robot.right_foot_vel[1]
        return output


class OrientationFootState(OrientationState, FootState):
    def get_state_entries(self, scaled):
        output = dict()
        output.update(OrientationState.get_state_entries(self, scaled))
        output.update(FootState.get_state_entries(self, scaled))
        return output


class PressureSensorState(PhaseState):
    def get_state_entries(self, scaled):
        output = super().get_state_entries(scaled)
        foot_pressures = [self.env.robot.pressure_sensors["LLB"].get_force()[1],
                          self.env.robot.pressure_sensors["LLF"].get_force()[1],
                          self.env.robot.pressure_sensors["LRF"].get_force()[1],
                          self.env.robot.pressure_sensors["LRB"].get_force()[1],
                          self.env.robot.pressure_sensors["RLB"].get_force()[1],
                          self.env.robot.pressure_sensors["RLF"].get_force()[1],
                          self.env.robot.pressure_sensors["RRF"].get_force()[1],
                          self.env.robot.pressure_sensors["RRB"].get_force()[1]]
        output["foot_pressure"] = foot_pressures / np.array(100) if scaled else foot_pressures
        return output


class PressureSensorFootState(PressureSensorState, FootState):
    def get_state_entries(self, scaled):
        output = dict()
        output.update(PressureSensorState.get_state_entries(self, scaled))
        output.update(FootState.get_state_entries(self, scaled))
        return output


class ComprehensiveState(GyroState, FootVelocityState, PressureSensorFootState):
    def get_state_entries(self, scaled):
        output = dict()
        output.update(GyroState.get_state_entries(self, scaled))
        output.update(FootVelocityState.get_state_entries(self, scaled))
        output.update(PressureSensorFootState.get_state_entries(self, scaled))
        return output


class JointSpaceState(PhaseState):
    def __init__(self, env: "DeepKickEnv"):
        super().__init__(env)

    def get_state_entries(self, scaled):
        output = super().get_state_entries(scaled)
        joint_positions = []
        joint_velocities = []
        for joint_name in self.env.robot.joints.keys():
            joint = self.env.robot.joints[joint_name]
            if scaled:
                joint_positions.append(joint.get_scaled_position())
                joint_velocities.append(joint.get_scaled_velocity())
            else:
                joint_positions.append(joint.get_position())
                joint_velocities.append(joint.get_velocity())
        output["joint_positions"] = joint_positions
        output["joint_velocities"] = joint_velocities
        return output
