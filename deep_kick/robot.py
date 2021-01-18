import math
import pybullet as p
from geometry_msgs.msg import Point, Quaternion
from scipy import signal
import os
import numpy as np
import rospkg
from sensor_msgs.msg import JointState
from bitbots_moveit_bindings import get_position_fk, get_position_ik
from moveit_msgs.srv import GetPositionIKRequest, GetPositionFKRequest, GetPositionFKResponse
from transforms3d.affines import compose, decompose
from transforms3d.euler import quat2euler, euler2quat, mat2euler
from transforms3d.quaternions import quat2mat, rotate_vector, qinverse, qmult

from deep_kick.demonstration import Frame


class Robot:
    def __init__(self, pybullet_client, observation_type, used_joints="Legs", physics=True):
        self.pybullet_client = pybullet_client
        # config values
        self.pose_on_episode_start = ([0, 0, 0.43], p.getQuaternionFromEuler((0, 0.25, 0)))
        self.pose_world_frame = self.pose_on_episode_start
        self.imu_rpy = None
        self.velocity = (0, 0, 0), (0, 0, 0)
        self.last_linear_vel = (0, 0, 0)
        self.linear_acc = (0, 0, 0)
        self.left_foot_pose = None
        self.right_foot_pose = None
        self.left_world_pos = None
        self.right_world_pos = None
        self.left_foot_vel = None
        self.right_foot_vel = None
        self.alpha = 1
        self.observation_type = observation_type

        self.initial_joints_positions = {"LAnklePitch": -30, "LAnkleRoll": 0, "LHipPitch": 30, "LHipRoll": 0,
                                         "LHipYaw": 0, "LKnee": 60, "RAnklePitch": 30, "RAnkleRoll": 0,
                                         "RHipPitch": -30, "RHipRoll": 0, "RHipYaw": 0, "RKnee": -60,
                                         "LShoulderPitch": 0, "LShoulderRoll": 0, "LElbow": 45, "RShoulderPitch": 0,
                                         "RShoulderRoll": 0, "RElbow": -45, "HeadPan": 0, "HeadTilt": 0}

        # how the foot pos and rpy are scaled. from [-1:1] action to meaningful m or rad
        # foot goal is relative to the start of the leg, so that 0 would be the center of possible poses
        # x, y, z, roll, pitch, yaw
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

        self.leg_joints = ["LAnklePitch", "LAnkleRoll", "LHipPitch", "LHipRoll", "LHipYaw", "LKnee",
                           "RAnklePitch", "RAnkleRoll", "RHipPitch", "RHipRoll", "RHipYaw", "RKnee"]

        if used_joints == "Legs":
            self.used_joint_names = self.leg_joints
        elif used_joints == "LegsAndArms":
            self.used_joint_names = self.leg_joints + ["LShoulderPitch", "LShoulderRoll", "LElbow", "RShoulderPitch",
                                                       "RShoulderRoll", "RElbow"]
        elif used_joints == "All":
            self.used_joint_names = ["LAnklePitch", "LAnkleRoll", "LHipPitch", "LHipRoll", "LHipYaw", "LKnee",
                                     "RAnklePitch", "RAnkleRoll", "RHipPitch", "RHipRoll", "RHipYaw", "RKnee",
                                     "LShoulderPitch", "LShoulderRoll", "LElbow", "RShoulderPitch",
                                     "RShoulderRoll", "RElbow", "HeadPan", "HeadTilt"]
        else:
            print("Used joint group \"{}\" not known".format(used_joints))
        self.num_used_joints = len(self.used_joint_names)

        # Loading robot
        if physics:
            flags = self.pybullet_client.URDF_USE_SELF_COLLISION + \
                    self.pybullet_client.URDF_USE_INERTIA_FROM_FILE + \
                    self.pybullet_client.URDF_USE_SELF_COLLISION_EXCLUDE_ALL_PARENTS
        else:
            flags = self.pybullet_client.URDF_USE_INERTIA_FROM_FILE

        rospack = rospkg.rospack.RosPack()
        urdf_path = os.path.join(rospack.get_path('wolfgang_description'), 'urdf/robot.urdf')
        self.robot_index = self.pybullet_client.loadURDF(urdf_path,
                                                         self.pose_on_episode_start[0], self.pose_on_episode_start[1],
                                                         flags=flags, useFixedBase=not physics)

        if not physics:
            self._disable_physics()

        # Retrieving joints and foot pressure sensors
        self.joints = {}
        self.pressure_sensors = {}
        self.links = {}

        self.log_file_name = 'logs.txt'
        with open(self.log_file_name, 'w') as f:
            f.write(','.join(
                ['roll', 'pitch', 'ang_x', 'ang_y', 'ang_z', 'left_lin_x', 'left_lin_y', 'left_lin_z', 'left_ang_x',
                 'left_ang_y', 'left_ang_z', 'LLB', 'LLF', 'LRF', 'LRB', 'RLB', 'RLF', 'RRF', 'RRB']) + '\n')

        # Collecting the available joints
        for i in range(self.pybullet_client.getNumJoints(self.robot_index)):
            joint_info = self.pybullet_client.getJointInfo(self.robot_index, i)
            name = joint_info[1].decode('utf-8')
            # we can get the links by seeing where the joint is attached
            self.links[joint_info[12].decode('utf-8')] = joint_info[16]
            if name in self.initial_joints_positions.keys():
                # remember joint
                self.joints[name] = Joint(i, self.robot_index, self.pybullet_client)
            elif name in ["LLB", "LLF", "LRF", "LRB", "RLB", "RLF", "RRF", "RRB"]:
                self.pybullet_client.enableJointForceTorqueSensor(self.robot_index, i)
                self.pressure_sensors[name] = PressureSensor(name, i, self.robot_index, 1, 1, self.pybullet_client)

        # set friction for feet
        for link_name in self.links.keys():
            if link_name in ["l_foot", "r_foot", "llb", "llf", "lrf", "lrb", "rlb", "rlf", "rrf", "rrb"]:
                # print(self.parts[part].body_name)
                self.pybullet_client.changeDynamics(self.robot_index, self.links[link_name])

        # reset robot to initial position
        self.reset()

    def transform_world_to_robot(self, foot_in_world_pos, foot_in_world_quat, foot_in_world_lin_vel,
                                 foot_in_world_ang_vel):
        # transform to relative to base link
        # quat of library is wxyz
        robot_in_world_quat = (self.pose_world_frame[1][3], self.pose_world_frame[1][0],
                               self.pose_world_frame[1][1], self.pose_world_frame[1][2])
        robot_in_world_mat = quat2mat(robot_in_world_quat)
        robot_in_world_trans = compose(self.pose_world_frame[0], robot_in_world_mat, [1, 1, 1])
        world_in_robot_trans = np.linalg.inv(robot_in_world_trans)
        foot_in_world_mat = quat2mat(foot_in_world_quat)
        foot_in_world_trans = compose(foot_in_world_pos, foot_in_world_mat, [1, 1, 1])
        foot_in_robot_trans = np.matmul(world_in_robot_trans, foot_in_world_trans)
        pos_in_robot, mat_in_robot, _, _ = decompose(foot_in_robot_trans)
        rpy_in_robot = mat2euler(mat_in_robot)
        # rotate velocities so that they are in robot frame
        lin_vel_in_robot = rotate_vector(foot_in_world_lin_vel, qinverse(robot_in_world_quat))
        # subtract linear velocity of robot, since it should be relative to it
        rel_lin_vel = lin_vel_in_robot - self.velocity[0]

        # same for angular vels
        ang_vel_in_robot = rotate_vector(foot_in_world_ang_vel, qinverse(robot_in_world_quat))
        rel_ang_vel = ang_vel_in_robot - self.velocity[1]

        return pos_in_robot, rpy_in_robot, rel_lin_vel, rel_ang_vel

    def compute_imu_orientation_from_world(self, robot_in_world_quat):
        # imu orientation has roll and pitch relative to gravity vector. yaw in world frame
        # get global yaw
        yrp_world_frame = quat2euler(robot_in_world_quat, axes='szxy')
        # remove global yaw rotation from roll and pitch
        yaw_quat = euler2quat(yrp_world_frame[0], 0, 0, axes='szxy')
        rp = rotate_vector((yrp_world_frame[1], yrp_world_frame[2], 0), qinverse(yaw_quat))
        # save in correct order
        return (rp[0], rp[1], yrp_world_frame[0])

    def update(self, log=False):
        """
        Updates the state of the robot from pybullet. This is only done once per step to improve performance.
        @return:
        """
        (x, y, z), (qx, qy, qz, qw) = self.pybullet_client.getBasePositionAndOrientation(self.robot_index)
        self.pose_world_frame = (x, y, z), (qx, qy, qz, qw)
        # create quat with wxyz order for transforms3d
        robot_in_world_quat = (qw, qx, qy, qz)

        # imu orientation has roll and pitch relative to gravity vector. yaw in world frame
        self.imu_rpy = self.compute_imu_orientation_from_world(robot_in_world_quat)

        # rotate velocities from world to robot frame
        lin_vel_world_frame, ang_vel_world_frame = self.pybullet_client.getBaseVelocity(self.robot_index)

        lin_vel_robot_frame = rotate_vector(lin_vel_world_frame, qinverse(robot_in_world_quat))
        angular_vel_robot_frame = rotate_vector(ang_vel_world_frame, qinverse(robot_in_world_quat))
        self.last_linear_vel, _ = self.velocity
        self.velocity = lin_vel_robot_frame, angular_vel_robot_frame
        # simple acceleration computation by using diff of velocities
        linear_acc = np.array(list(map(lambda i, j: i - j, self.last_linear_vel, lin_vel_robot_frame)))
        # adding gravity to the acceleration
        gravity_world_frame = np.array([0, 0, 9.81])
        gravity_robot_frame = rotate_vector(gravity_world_frame, qinverse(robot_in_world_quat))
        self.linear_acc = linear_acc + gravity_robot_frame
        for joint in self.joints.values():
            joint.update()

        # foot positions
        _, _, _, _, world_position, world_orientation, world_linear_vel, world_angular_vel = p.getLinkState(
            self.robot_index, self.links["l_sole"], 1, 0)
        self.left_world_pos = world_position
        # pybullet does weird stuff and rotates our feet
        world_orientation = qmult(
            (world_orientation[3], world_orientation[0], world_orientation[1], world_orientation[2]),
            euler2quat(0, 0, math.tau / 4))
        pos_in_robot, rpy_in_robot, lin_vel_in_robot, ang_vel_in_robot = self.transform_world_to_robot(
            world_position, world_orientation, world_linear_vel, world_angular_vel)
        self.left_foot_pose = pos_in_robot, rpy_in_robot
        self.left_foot_vel = lin_vel_in_robot, ang_vel_in_robot
        # right foot
        _, _, _, _, world_position, world_orientation, world_linear_vel, world_angular_vel = p.getLinkState(
            self.robot_index, self.links["r_sole"], 1, 0)
        self.right_world_pos = world_position
        world_orientation = qmult(
            (world_orientation[3], world_orientation[0], world_orientation[1], world_orientation[2]),
            euler2quat(0, 0, math.tau / 4))
        pos_in_robot, rpy_in_robot, lin_vel_in_robot, ang_vel_in_robot = self.transform_world_to_robot(
            world_position, world_orientation, world_linear_vel, world_angular_vel)
        self.right_foot_pose = pos_in_robot, rpy_in_robot
        self.right_foot_vel = lin_vel_in_robot, ang_vel_in_robot

        if log:
            with open(self.log_file_name, 'a') as f:
                f.write(','.join((str(x) for x in [
                    self.imu_rpy[0] / (math.tau / 2),
                    self.imu_rpy[1] / (math.tau / 2),
                    self.velocity[1][0] / 10,
                    self.velocity[1][1] / 10,
                    self.velocity[1][2] / 10,
                    self.left_foot_vel[0][0] / 5,
                    self.left_foot_vel[0][1] / 5,
                    self.left_foot_vel[0][2] / 5,
                    self.left_foot_vel[1][0] / (4 * math.tau),
                    self.left_foot_vel[1][1] / (4 * math.tau),
                    self.left_foot_vel[1][2] / (4 * math.tau),
                    self.pressure_sensors["LLB"].get_force()[1] / 100,
                    self.pressure_sensors["LLF"].get_force()[1] / 100,
                    self.pressure_sensors["LRF"].get_force()[1] / 100,
                    self.pressure_sensors["LRB"].get_force()[1] / 100,
                    self.pressure_sensors["RLB"].get_force()[1] / 100,
                    self.pressure_sensors["RLF"].get_force()[1] / 100,
                    self.pressure_sensors["RRF"].get_force()[1] / 100,
                    self.pressure_sensors["RRB"].get_force()[1] / 100,
                ])) + '\n')

    def calc_state_cartesian(self):
        request = GetPositionFKRequest()
        for joint_name in self.used_joint_names:
            # from radiant to range of -1 +1
            joint = self.joints[joint_name]
            request.robot_state.joint_state.name.append(joint_name)
            request.robot_state.joint_state.position.append(joint.get_position())
        request.fk_link_names = ['l_sole', 'r_sole']
        result = get_position_fk(request)  # type: GetPositionFKResponse
        l_sole = result.pose_stamped[result.fk_link_names.index('l_sole')].pose
        l_sole_quat = l_sole.orientation
        l_sole_rpy = quat2euler((l_sole_quat.w, l_sole_quat.x, l_sole_quat.y, l_sole_quat.z))
        r_sole = result.pose_stamped[result.fk_link_names.index('r_sole')].pose
        r_sole_quat = r_sole.orientation
        r_sole_rpy = quat2euler((r_sole_quat.w, r_sole_quat.x, r_sole_quat.y, r_sole_quat.z))
        # everything has to be between -1 and 1. For xyz that is always true, but rpy have to be scaled with pi.
        return np.concatenate([
            [l_sole.position.x, l_sole.position.y, l_sole.position.z],
            np.divide(l_sole_rpy, math.pi),
            [r_sole.position.x, r_sole.position.y, r_sole.position.z],
            np.divide(r_sole_rpy, math.pi),
        ])

    def get_legs_in_world(self):
        end_effector_positions = self.calc_state_cartesian()
        robot_pose = self.get_pose_world_frame()
        robot_rotation = quat2mat((robot_pose[1][3], robot_pose[1][0], robot_pose[1][1], robot_pose[1][2]))
        world_to_base_link = compose(robot_pose[0], robot_rotation, [1, 1, 1])
        # We have to append 1 for the matrix multiplication, we remove it afterwards
        left_leg_vector = [*end_effector_positions[:3], 1]
        left_leg_in_world = np.matmul(world_to_base_link, left_leg_vector)[:-1]
        right_leg_vector = [*end_effector_positions[6:9], 1]
        right_leg_in_world = np.matmul(world_to_base_link, right_leg_vector)[:-1]
        return left_leg_in_world, right_leg_in_world

    def get_joint_state(self):
        msg = JointState()
        for joint_name in self.used_joint_names:
            msg.name.append(joint_name)
            joint = self.joints[joint_name]
            msg.position.append(joint.get_position())
            msg.velocity.append(joint.get_velocity())
        return msg

    def apply_action(self, action, cartesian, relative, current_frame):
        assert (np.isfinite(action).all())
        if not cartesian:
            i = 0
            # iterate through all joints, always in same order to keep the same matching between NN and simulation
            for joint_name in self.used_joint_names:
                # scaling needed since action space is -1 to 1, but joints have lower and upper limits
                joint = self.joints[joint_name]
                goal_position = action[i]
                if relative:
                    goal_position += current_frame.joint_positions[joint_name]  # todo we can get out of bounds by this
                joint.set_scaled_position(goal_position)
                i += 1
            return True
        else:
            # split action values into corresponding position and rpy
            if relative:
                left_foot_pos, left_foot_rpy, right_foot_pos, right_foot_rpy = self.scale_action_to_relative_pose(
                    action[0:3],
                    action[3:6],
                    action[6:9],
                    action[9:12], current_frame)
            else:
                left_foot_pos, left_foot_rpy, right_foot_pos, right_foot_rpy = self.scale_action_to_pose(action[0:3],
                                                                                                         action[3:6],
                                                                                                         action[6:9],
                                                                                                         action[9:12])
            ik_result, success = self.compute_ik(left_foot_pos, left_foot_rpy, right_foot_pos, right_foot_rpy,
                                                 collision=False, approximate=True)
            if success:
                for name, position in ik_result.items():
                    if name in self.used_joint_names:
                        self.joints[name].set_position(position)
                return True
            else:
                return False

    def compute_ik(self, left_foot_pos, left_foot_rpy, right_foot_pos, right_foot_rpy, collision=True, approximate=False):
        request = GetPositionIKRequest()
        # request.ik_request.timeout = rospy.Time.from_seconds(0.01)
        # request.ik_request.attempts = 1
        request.ik_request.avoid_collisions = collision

        request.ik_request.group_name = "LeftLeg"
        request.ik_request.ik_link_name = "l_sole"
        request.ik_request.pose_stamped.pose.position = Point(*left_foot_pos)
        quat = euler2quat(*left_foot_rpy)
        request.ik_request.pose_stamped.pose.orientation = Quaternion(quat[1], quat[2], quat[3], quat[0])
        ik_result = get_position_ik(request, approximate=approximate)
        first_error_code = ik_result.error_code.val
        request.ik_request.group_name = "RightLeg"
        request.ik_request.ik_link_name = "r_sole"
        request.ik_request.pose_stamped.pose.position = Point(*right_foot_pos)
        quat = euler2quat(*right_foot_rpy)
        request.ik_request.pose_stamped.pose.orientation = Quaternion(quat[1], quat[2], quat[3], quat[0])
        ik_result = get_position_ik(request, approximate=approximate)
        # check if no solution or collision
        success = first_error_code == ik_result.error_code.val == 1
        joint_result = {}
        for name, position in zip(ik_result.solution.joint_state.name, ik_result.solution.joint_state.position):
            joint_result[name] = position
        return joint_result, success

    def is_alive(self):
        alive = True
        (x, y, z), (a, b, c, d), _, _, _, _ = self.get_head_pose()
        robot_position, robot_orientation = self.get_pose_world_frame()
        robot_rpy = p.getEulerFromQuaternion(robot_orientation)

        # head higher than starting position of body
        alive = alive and z > self.get_start_height()
        # angle of the robot in roll and pitch not to far from zero
        alive = alive and abs(robot_rpy[0] < math.pi / 2) and abs(robot_rpy[1] < math.pi / 2)
        return alive

    def reset(self):
        self.pose_on_episode_start = ([0, 0, 0.43], p.getQuaternionFromEuler((0, 0.25, 0)))
        # set joints to initial position
        for name in self.joints:
            joint = self.joints[name]
            pos_in_rad = math.radians(self.initial_joints_positions[name])
            joint.reset_position(pos_in_rad, 0)
            joint.set_position(pos_in_rad)

        # reset body pose and velocity
        self.pybullet_client.resetBasePositionAndOrientation(self.robot_index, self.pose_on_episode_start[0],
                                                             self.pose_on_episode_start[1])
        self.pybullet_client.resetBaseVelocity(self.robot_index, [0, 0, 0], [0, 0, 0])
        self.update()

    def reset_to_frame(self, frame):
        self.pose_on_episode_start = (frame.robot_position, frame.robot_orientation)
        self.pybullet_client.resetBasePositionAndOrientation(self.robot_index, frame.robot_position,
                                                             frame.robot_orientation)
        self.pybullet_client.resetBaseVelocity(self.robot_index, [0, 0, 0], [0, 0, 0])

        # set all joints to initial position since they can be modified from last fall
        for name in self.joints:
            pos_in_rad = math.radians(self.initial_joints_positions[name])
            self.joints[name].reset_position(pos_in_rad, 0)
        # set all joints in the frame to their position
        for joint_name in frame.joint_positions.keys():
            self.joints[joint_name].reset_position(frame.joint_positions[joint_name], 0)

        self.update()

    def get_frame(self):
        position, orientation = self.get_pose_world_frame()
        joint_names = self.used_joint_names
        joint_positions = [self.joints[name].get_position() for name in joint_names]
        joint_velocities = [self.joints[name].get_velocity() for name in joint_names]
        return Frame(0, position, orientation, joint_names, joint_positions, joint_velocities)

    def get_init_bias(self, cartesian):
        if cartesian:
            return np.zeros(12)
        else:
            joint_positions = []
            for joint_name in self.used_joint_names:
                # from radiant to range of -1 +1
                joint = self.joints[joint_name]
                joint_positions.append(joint.convert_radiant_to_scaled(math.radians(self.initial_joints_positions[joint_name])))

            return joint_positions

    def get_pose_world_frame(self):
        return self.pose_world_frame

    def get_velocity(self):
        return self.velocity

    def get_foot_pose(self, is_left):
        if is_left:
            return self.left_foot_pose
        else:
            return self.right_foot_pose

    def get_foot_vel(self, is_left):
        if is_left:
            return self.left_foot_vel
        else:
            return self.right_foot_vel

    def get_head_pose(self):
        if "head" in self.links.keys():
            return self.pybullet_client.getLinkState(self.robot_index, self.links["head"])
        elif "r_knee" in self.links.keys():
            return self.pybullet_client.getLinkState(self.robot_index, self.links["r_knee"])
        else:
            print("Head link not existing")
            return None

    def get_start_height(self):
        if "head" in self.links.keys():
            return self.pose_on_episode_start[0][2]
        elif "r_knee" in self.links.keys():
            return 0.2
        else:
            print("Head link not existing")
            return None

    def get_scaled_positions_for_frame(self, frame):
        frame_positions = frame.joint_positions
        scaled = []
        for joint_name in self.used_joint_names:
            joint = self.joints[joint_name]
            scaled.append(joint.convert_radiant_to_scaled(frame_positions[joint_name]))
        return scaled

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

    def scale_pose_to_action(self, left_pos, left_rpy, right_pos, right_rpy):
        action = []
        for i in range(3):
            action.append((left_pos[i] / self.pose_scaling_factors_left[i]) - self.pose_scaling_addends_left[i])
        for i in range(3, 6):
            action.append((left_rpy[i - 3] / self.pose_scaling_factors_left[i]) - self.pose_scaling_addends_left[i])
        for i in range(3):
            action.append((right_pos[i] / self.pose_scaling_factors_right[i]) - self.pose_scaling_addends_right[i])
        for i in range(3, 6):
            action.append((right_rpy[i - 3] / self.pose_scaling_factors_right[i]) - self.pose_scaling_addends_right[i])
        return action

    def scale_action_to_relative_pose(self, pos_left, rpy_left, pos_right, rpy_right, current_frame):
        # since we just shift pos and rpy relativly we can scale easily
        pos_left = current_frame.left_foot_pos + np.array(pos_left) * 0.1
        rpy_left = current_frame.left_foot_rpy + np.array(rpy_left) * (math.tau / 12)
        pos_right = current_frame.right_foot_pos + np.array(pos_right) * 0.1
        rpy_right = current_frame.right_foot_rpy + np.array(rpy_right) * (math.tau / 12)
        return pos_left, rpy_left, pos_right, rpy_right

    def set_alpha(self, alpha):
        if self.alpha != alpha:
            # only change if the value changed, for better performance
            ref_col = [1, 1, 1, alpha]
            self.pybullet_client.changeVisualShape(self.robot_index, -1, rgbaColor=ref_col)
            for l in range(self.pybullet_client.getNumJoints(self.robot_index)):
                self.pybullet_client.changeVisualShape(self.robot_index, l, rgbaColor=ref_col)
            self.alpha = alpha

    def _disable_physics(self):
        self.pybullet_client.changeDynamics(self.robot_index, -1, linearDamping=0, angularDamping=0)
        self.pybullet_client.setCollisionFilterGroupMask(self.robot_index, -1, collisionFilterGroup=0,
                                                         collisionFilterMask=0)
        self.pybullet_client.changeDynamics(self.robot_index, -1,
                                            activationState=self.pybullet_client.ACTIVATION_STATE_SLEEP +
                                                            self.pybullet_client.ACTIVATION_STATE_ENABLE_SLEEPING +
                                                            self.pybullet_client.ACTIVATION_STATE_DISABLE_WAKEUP)
        num_joints = self.pybullet_client.getNumJoints(self.robot_index)
        for j in range(num_joints):
            self.pybullet_client.setCollisionFilterGroupMask(self.robot_index, j, collisionFilterGroup=0,
                                                             collisionFilterMask=0)
            self.pybullet_client.changeDynamics(self.robot_index, j,
                                                activationState=self.pybullet_client.ACTIVATION_STATE_SLEEP +
                                                                self.pybullet_client.ACTIVATION_STATE_ENABLE_SLEEPING +
                                                                self.pybullet_client.ACTIVATION_STATE_DISABLE_WAKEUP)


class Joint:
    def __init__(self, joint_index, body_index, pybullet_client):
        self.pybullet_client = pybullet_client
        self.joint_index = joint_index
        self.body_index = body_index
        joint_info = self.pybullet_client.getJointInfo(self.body_index, self.joint_index)
        self.name = joint_info[1].decode('utf-8')
        self.type = joint_info[2]
        self.max_force = joint_info[10]
        self.max_velocity = joint_info[11]
        self.lowerLimit = joint_info[8]
        self.upperLimit = joint_info[9]
        position, velocity, forces, applied_torque = self.pybullet_client.getJointState(self.body_index,
                                                                                        self.joint_index)
        self.state = position, velocity, forces, applied_torque

    def update(self):
        """
        Called just once per step to update state from simulation. Improves performance.
        @return:
        """
        position, velocity, forces, applied_torque = self.pybullet_client.getJointState(self.body_index,
                                                                                        self.joint_index)
        self.state = position, velocity, forces, applied_torque

    def reset_position(self, position, velocity):
        self.pybullet_client.resetJointState(self.body_index, self.joint_index, targetValue=position,
                                             targetVelocity=velocity)
        # self.disable_motor()

    def disable_motor(self):
        self.pybullet_client.setJointMotorControl2(self.body_index, self.joint_index,
                                                   controlMode=self.pybullet_client.POSITION_CONTROL, targetPosition=0,
                                                   targetVelocity=0,
                                                   positionGain=0.1, velocityGain=0.1, force=0)

    def set_position(self, position):
        self.pybullet_client.setJointMotorControl2(self.body_index, self.joint_index,
                                                   self.pybullet_client.POSITION_CONTROL,
                                                   targetPosition=position, force=self.max_force,
                                                   maxVelocity=self.max_velocity)

    def set_scaled_position(self, position):
        # sets position inside limits with a given position values in [-1, 1]
        pos_mid = 0.5 * (self.lowerLimit + self.upperLimit)
        radiant_position = position * (self.upperLimit - self.lowerLimit) / 2 + pos_mid
        self.set_position(radiant_position)

    def reset_scaled_position(self, position):
        # sets position inside limits with a given position values in [-1, 1]
        pos_mid = 0.5 * (self.lowerLimit + self.upperLimit)
        radiant_position = position * (self.upperLimit - self.lowerLimit) / 2 + pos_mid
        self.reset_position(radiant_position, 0)

    def get_state(self):
        return self.state

    def get_position(self):
        position, velocity, forces, applied_torque = self.state
        return position

    def get_scaled_position(self):
        # gives position inside limits scaled to [-1, 1]
        pos, _, _, _ = self.state
        pos_mid = 0.5 * (self.lowerLimit + self.upperLimit)
        # do clipping for rounding errors
        return np.clip(2 * (pos - pos_mid) / (self.upperLimit - self.lowerLimit), -1, 1)

    def get_velocity(self):
        position, velocity, forces, applied_torque = self.state
        return velocity

    def get_scaled_velocity(self):
        return self.get_velocity() * 0.1

    def get_torque(self):
        position, velocity, forces, applied_torque = self.state
        return applied_torque

    def convert_radiant_to_scaled(self, position):
        # helper method to convert to scaled position for this joint using min max scaling
        scaled_position = -1 + ((position - self.lowerLimit) * 2) / (self.upperLimit - self.lowerLimit)
        return scaled_position


class PressureSensor:
    def __init__(self, name, joint_index, body_index, cutoff, order, pybullet_client):
        self.pybullet_client = pybullet_client
        self.joint_index = joint_index
        self.name = name
        self.body_index = body_index
        nyq = 240 * 0.5  # nyquist frequency from simulation frequency
        normalized_cutoff = cutoff / nyq  # cutoff freq in hz
        self.filter_b, self.filter_a = signal.butter(order, normalized_cutoff, btype='low')
        self.filter_state = signal.lfilter_zi(self.filter_b, 1)
        self.unfiltered = 0
        self.filtered = [0]

    def filter_step(self):
        self.unfiltered = self.pybullet_client.getJointState(self.body_index, self.joint_index)[2][2] * -1
        self.filtered, self.filter_state = signal.lfilter(self.filter_b, self.filter_a, [self.unfiltered],
                                                          zi=self.filter_state)

    def get_force(self):
        return max(self.unfiltered, 0), max(self.filtered[0], 0)

    def get_scaled_force(self):
        return self.get_force()[1] * 0.01
