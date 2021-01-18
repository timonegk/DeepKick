import copy
import math
import os
import sys
import time
import gym
import pybullet_data
from std_msgs.msg import Float32MultiArray
from sensor_msgs.msg import JointState
import random

from deep_kick.ros_interface import ROSInterface
from deep_kick.robot import Robot
from parallel_parameter_search.utils import load_robot_param, load_yaml_to_param

from deep_kick import reward
import pybullet as p
import numpy as np
import rospy
import rospkg

from stable_baselines3.common.env_checker import check_env

from bitbots_dynamic_kick.py_kick_wrapper import PyKick
from bitbots_msgs.msg import KickGoal

from deep_kick.state import JointSpaceState, PhaseState, OrientationState, GyroState, FootState, FootVelocityState, \
    OrientationFootState, PressureSensorState, PressureSensorFootState, ComprehensiveState
from deep_kick.utils import BulletClient
from deep_kick.demonstration import Frame


class DeepKickEnv(gym.Env):
    """This is an OpenAi environment for RL. It extends the simulation to provide the necessary methods for
    compatibility with openai RL algorithms, e.g. PPO.
    """
    metadata = {
        'render.modes': ['human', 'rgb_array'],
        'video.frames_per_second': 240
    }

    def __init__(self, reward_function: reward.AbstractReward, used_joints="Legs", step_freq=30, ros_debug=False, gui=False,
                 use_demonstration=True, early_termination=True, time_horizon=3, gravity=True, random_initialization=True,
                 observation_type='comprehensive', cartesian_action=False, randomize_commands=False, force=None) -> None:
        """
        @param reward_function: a reward object that specifies the reward function
        @param used_joints: which joints should be enabled
        @param step_freq: the frequency of policy steps in hertz
        @param ros_debug: enables ROS debug messages (needs roscore)
        @param gui: enables pybullet debug GUI
        @param use_demonstration: whether demonstration should be used for training
        @param early_termination: if episode should be terminated early when robot falls
        @param time_horizon: maximum time in seconds (time horizon)
        """
        self.gui = gui
        self.paused = False
        self.realtime = False
        self.gravity = gravity
        self.ros_debug = ros_debug
        self.reward_function = reward_function
        self.early_termination = early_termination
        self.time_horizon = time_horizon
        self.observation_type = observation_type
        self.cartesian_action = cartesian_action
        self.random_initialization = random_initialization
        self.force = force

        # The last action that has been performed
        self.last_action = None
        # The number of calls to step() since the last reset
        self.step_count = 0
        # The wall time of the last call to step()
        self.last_step_time = 0
        # The current progress of the demonstration
        self.progress = 0
        # The current time in the demonstration, in seconds
        self.demonstration_time = 0
        # The tolerance of foot goal position
        self.foot_goal_tolerance = 0.05
        # The tolerance for the final angular velocity
        self.angular_velocity_tolerance = 0.1
        # Whether the episode is done
        self.done = False
        # The time from which on the robot was stable
        self.stable_time = 0
        # time of random force impact
        self.force_time = 0
        # direction of random force
        self.force_direction = 0

        self.camera_distance = 1.0
        self.camera_yaw = 0
        self.camera_pitch = -30
        self.render_width = 800
        self.render_height = 600

        # Instantiating Bullet
        if self.gui:
            self.pybullet_client = BulletClient(p.GUI)
            self.pybullet_client.configureDebugVisualizer(self.pybullet_client.COV_ENABLE_GUI, True)
            self.pybullet_client.configureDebugVisualizer(self.pybullet_client.COV_ENABLE_SEGMENTATION_MARK_PREVIEW, 0)
            self.pybullet_client.configureDebugVisualizer(self.pybullet_client.COV_ENABLE_DEPTH_BUFFER_PREVIEW, 0)
            self.pybullet_client.configureDebugVisualizer(self.pybullet_client.COV_ENABLE_RGB_BUFFER_PREVIEW, 0)
            self.debug_alpha_index = self.pybullet_client.addUserDebugParameter("display reference", 0, 1, 0.5)
        else:
            self.pybullet_client = BulletClient(p.DIRECT)

        if self.gravity:
            # Loading floor
            self.pybullet_client.setAdditionalSearchPath(pybullet_data.getDataPath())  # optionally
            self.plane_index = self.pybullet_client.loadURDF('plane.urdf')
            self.pybullet_client.changeDynamics(self.plane_index, -1, lateralFriction=1,
                                                spinningFriction=0.1, rollingFriction=0.1, restitution=0.9)
            self.pybullet_client.setGravity(0, 0, -9.81)
        else:
            self.pybullet_client.setGravity(0, 0, 0)

        # Add ball
        self.ball_radius = 0.07
        self.pybullet_client.setAdditionalSearchPath(os.path.join(os.path.dirname(__file__), 'bullet_data'))
        self.ball_start_position = (0.2, 0, self.ball_radius)
        self.ball_index = self.pybullet_client.loadURDF('soccerball.urdf', (0.2, 0, self.ball_radius), (0, 0, 0, 1),
                                                        flags=self.pybullet_client.URDF_USE_INERTIA_FROM_FILE,
                                                        useFixedBase=False)
        self.last_ball_position = self.ball_start_position

        # Load robot model
        self.namespace = ""
        rospack = rospkg.RosPack()
        load_robot_param(self.namespace, rospack, "wolfgang")

        # Engine parameters
        # time step should be at 240Hz (due to pyBullet documentation)
        self.timestep = 1 / 240
        # standard parameters seem to be best. leave them like they are
        # self.pybullet_client.setPhysicsEngineParameter(fixedTimeStep=self.timestep, numSubSteps=1)
        # no real time, as we will publish own clock
        self.pybullet_client.setRealTimeSimulation(0)

        # How many simulation steps have to be done per policy step
        self.sim_steps = int((1 / self.timestep) / step_freq)
        self.policy_timestep = self.timestep * self.sim_steps

        # create real robot + reference robot which is only to display ref trajectory
        self.robot = Robot(self.pybullet_client, self.observation_type, used_joints)
        self.refbot = Robot(self.pybullet_client, 'none', used_joints, physics=False)
        self.refbot.set_alpha(0.5)

        self.randomize_commands = randomize_commands
        if self.randomize_commands:
            self.current_command = self.get_random_command()
        else:
            self.current_command = (0.2, 0.08)  # ball position
        # simulation origin is support foot, this is the offset to the base footprint, i.e. half of the foot distance
        self.y_offset = 0.1
        # will be set later, in reset()
        self.ball_x = 0
        self.ball_y = 0
        # Which foot is kicking
        self.is_left_kick = True
        # Whether the ball has been moved
        self.ball_moved = False

        if self.cartesian_action:
            num_actions = 12
        else:
            num_actions = self.robot.num_used_joints
        self.action_space = gym.spaces.Box(low=-1.0, high=1.0, shape=(num_actions,), dtype=np.float32)
        self.action_possible = True  # this is set to False if the current action cannot be applied

        if self.observation_type == 'phase':
            self.state = PhaseState(self)
        elif self.observation_type == 'orientation':
            self.state = OrientationState(self)
        elif self.observation_type == 'gyro':
            self.state = GyroState(self)
        elif self.observation_type == 'foot':
            self.state = FootState(self)
        elif self.observation_type == 'foot_velocity':
            self.state = FootState(self)
        elif self.observation_type == 'orientation_foot':
            self.state = OrientationFootState(self)
        elif self.observation_type == 'pressure_sensor':
            self.state = PressureSensorState(self)
        elif self.observation_type == 'pressure_sensor_foot':
            self.state = PressureSensorFootState(self)
        elif self.observation_type == 'comprehensive':
            self.state = ComprehensiveState(self)
        else:
            sys.exit(f'Unknown observation type {self.observation_type}')
        num_observations = self.state.get_num_observations()
        self.observation_space = gym.spaces.Box(low=-1.0, high=1.0, shape=(num_observations,), dtype=np.float32)


        # add publisher if ROS debug is active
        if self.ros_debug:
            self.state_publisher = rospy.Publisher("state_debug", Float32MultiArray, queue_size=1)
            self.action_publisher = rospy.Publisher("action_debug", Float32MultiArray, queue_size=1)
            # todo value publisher
            self.ros_interface = ROSInterface(self.robot)

        self.use_demonstration = use_demonstration
        # Set to True when the demonstration is finished but the robot still needs to stand still
        self.demonstration_finished = False

        if self.use_demonstration:
            load_yaml_to_param(self.namespace + '/dynamic_kick', "bitbots_dynamic_kick", "/config/kick_config.yaml", rospack)
            params = rospy.get_param(self.namespace + '/dynamic_kick')
            self.overall_time = sum(params[name] for name in params if name.endswith('_time'))
            self.kick_time = (params['move_trunk_time'] + params['raise_foot_time'] +
                              params['move_to_ball_time'] + 0.5 * params['kick_time'])
            self.engine = PyKick()
            self.engine.set_params(rospy.get_param(self.namespace + '/dynamic_kick'))

        # The current output of the demonstration
        self.current_frame = None  # type: Frame
        # The previous output of the demonstration
        self.previous_frame = None  # type: Frame

        self.walkready_joint_state = JointState()
        self.walkready_joint_state.name = ["HeadPan", "HeadTilt", "LHipYaw", "LHipRoll",
                                           "LHipPitch", "LKnee", "LAnklePitch",
                                           "LAnkleRoll", "LShoulderPitch", "LShoulderRoll", "LElbow",
                                           "RHipYaw", "RHipRoll", "RHipPitch",
                                           "RKnee", "RAnklePitch", "RAnkleRoll",
                                           "RShoulderPitch", "RShoulderRoll", "RElbow"]
        self.walkready_joint_state.position = [0.0, 0.0, -3.349828585645047e-06, 0.10200620312963775,
                                               0.4801425653470249, 1.159038255771891, -0.4171025496819984,
                                               0.10199380318793154, 0.0012236716819459838, 0.0, 1.0472,
                                               4.346170893296005e-06, -0.09709665347552558, -0.4787056376651516,
                                               -1.1594305199861608, 0.4553927723524701, -0.11957603480745559,
                                               -0.0012445634097944063, 0.0, -1.0472]
        self.walkready_joint_state.velocity = [-1] * len(self.walkready_joint_state.name)

        # run check after everything is initialized
        #check_env(self)

    def get_kick_goal(self, command):
        kick_goal = KickGoal()
        kick_goal.header.frame_id = 'base_footprint'
        kick_goal.kick_speed = 1
        kick_goal.ball_position.x = command[0]
        kick_goal.ball_position.y = command[1]
        kick_goal.kick_direction.w = 1
        return kick_goal

    def get_random_command(self):
        return [np.random.uniform(0.15, 0.25), np.random.uniform(-0.15, 0.15)]

    def reset(self):
        if self.randomize_commands:
            command = self.get_random_command()
        else:
            command = self.current_command

        self.force_time = random.uniform(0, self.overall_time)
        self.force_direction = random.uniform(-math.pi, math.pi)

        kick_goal = self.get_kick_goal(command)
        # if we have a demonstration we set the simulation to a random start in it
        start_time = random.uniform(0, self.overall_time - 0.01) if self.random_initialization else 0
        self.demonstration_time = 0
        if self.use_demonstration:
            self.engine.set_goal(kick_goal, joint_state=self.walkready_joint_state)
            # We have to step twice because the first position is garbage due to bio_ik
            self.step_demonstration(start_time)
            self.current_frame = self.step_demonstration(0)
            self.previous_frame = None
            self.robot.reset_to_frame(self.current_frame)
            self.refbot.reset_to_frame(self.current_frame)
        else:
            # without demonstration we just go to init
            self.robot.reset()

        # Set ball position and velocity
        self.ball_x = command[0]
        self.ball_y = command[1]
        if not self.use_demonstration or self.engine.is_left_kick():
            self.is_left_kick = True
            self.ball_y += self.y_offset
        else:
            self.is_left_kick = False
            self.ball_y -= self.y_offset
        ball_x = self.ball_x
        if self.random_initialization and start_time > self.kick_time:
            # When the kick has already happened, move the ball to the ball distance of the reference kick (~25cm)
            ball_x += 0.25
        self.pybullet_client.resetBasePositionAndOrientation(
            self.ball_index,
            (ball_x, self.ball_y, self.ball_radius),
            (0, 0, 0, 1))
        self.pybullet_client.resetBaseVelocity(self.ball_index, [0, 0, 0], [0, 0, 0])
        self.ball_moved = False

        self.done = False
        self.step_count = 0
        self.reward_function.reset_episode_reward()
        self.demonstration_finished = False
        # return robot state and current time value
        return self.state.get_state_array(scaled=True)

    def step_simulation(self):
        # get keyboard events if gui is active
        single_step = False
        if self.gui:
            # rest if R-key was pressed
            rKey = ord('r')
            nKey = ord('n')
            sKey = ord('s')
            tKey = ord('t')
            spaceKey = self.pybullet_client.B3G_SPACE
            keys = self.pybullet_client.getKeyboardEvents()
            if rKey in keys and keys[rKey] & self.pybullet_client.KEY_WAS_TRIGGERED:
                self.reset()
            if spaceKey in keys and keys[spaceKey] & self.pybullet_client.KEY_WAS_TRIGGERED:
                self.paused = not self.paused
            if sKey in keys and keys[sKey] & self.pybullet_client.KEY_IS_DOWN:
                single_step = True
            if nKey in keys and keys[nKey] & self.pybullet_client.KEY_WAS_TRIGGERED:
                if self.gravity:
                    self.pybullet_client.setGravity(0, 0, 0)
                else:
                    self.pybullet_client.setGravity(0, 0, -9.81)
                self.gravity = not self.gravity
            if tKey in keys and keys[tKey] & self.pybullet_client.KEY_WAS_TRIGGERED:
                self.realtime = not self.realtime
                print("Realtime is " + str(self.realtime))

        # check if simulation should continue currently
        if not self.paused or single_step:
            self.pybullet_client.stepSimulation()
            for name, ps in self.robot.pressure_sensors.items():
                ps.filter_step()

    def step_demonstration(self, dt):
        joint_command = self.engine.step(dt, self.robot.get_joint_state())
        self.progress = self.engine.get_progress()
        if len(joint_command.joint_names) == 0:
            frame = copy.deepcopy(self.previous_frame)
            frame.time = self.demonstration_time
            return frame
        else:
            trunk_pose = self.engine.get_trunk_pose()
            position = (trunk_pose.position.x, trunk_pose.position.y, trunk_pose.position.z)
            orientation = (trunk_pose.orientation.x, trunk_pose.orientation.y,
                           trunk_pose.orientation.z, trunk_pose.orientation.w)
            return Frame(time=self.demonstration_time, position=position, orientation=orientation, joint_names=joint_command.joint_names,
                         joint_positions=joint_command.positions, joint_velocities=joint_command.velocities)

    def step(self, action):
        # save action as class variable since we may need it to compute reward
        self.last_action = action

        # step the current frame further, based on the time
        self.demonstration_time += self.policy_timestep
        if self.use_demonstration:
            self.previous_frame = self.current_frame
            self.current_frame = self.step_demonstration(self.policy_timestep)

        # apply action and let environment perform a step (which are maybe multiple simulation steps)
        self.robot.apply_action(action,
                                cartesian=self.cartesian_action,
                                current_frame=self.current_frame,
                                relative=False)

        self.last_ball_position = self.get_ball_position()

        if self.force and self.demonstration_time < self.force_time < self.demonstration_time + 0.333:
            magnitude = self.force
            force = [magnitude * math.sin(self.force_direction), magnitude * math.cos(self.force_direction)]
            self.pybullet_client.applyExternalForce(self.robot.robot_index, -1,
                                                    [force[0], force[1], 0],
                                                    [0, 0, 0], p.LINK_FRAME)

        for i in range(self.sim_steps):
            self.step_simulation()

        ball_distance = np.linalg.norm(np.array((self.ball_x, self.ball_y)) - self.get_ball_position())
        self.ball_moved = ball_distance > 0.01

        if np.any(np.array(self.robot.velocity[1]) > self.angular_velocity_tolerance):
            # We are still unstable, update the stable_time
            self.stable_time = self.demonstration_time + self.policy_timestep

        # update the robot model just once after simulation to save performance
        self.robot.update(log=True)
        self.refbot.update()
        self.step_count += 1

        dead = self.early_termination and not self.robot.is_alive()

        # Terminate when we fell or the time horizon has been reached
        self.done = (dead or self.demonstration_time >= self.time_horizon)
        # compute reward
        reward = self.reward_function.compute_current_reward(self)

        info = dict()
        if self.done:
            info["rewards"] = self.reward_function.get_info_dict()
            info["evaluation"] = {
                "fallen": 0 if self.robot.is_alive() else 1,
                "ball_distance": ball_distance,
                "stable_time": self.stable_time,
            }
        else:
            # render ref trajectory
            if self.gui:
                alpha = self.pybullet_client.readUserDebugParameter(self.debug_alpha_index)
                self.refbot.set_alpha(alpha)
            if self.use_demonstration:
                self.refbot.reset_to_frame(self.current_frame)
            else:
                self.refbot.reset()

            if self.realtime:
                # sleep long enough to run the simulation in real time and not in accelerated speed
                step_computation_time = time.time() - self.last_step_time
                self.last_step_time = time.time()
                time_to_sleep = self.timestep * self.sim_steps - step_computation_time
                if time_to_sleep > 0:
                    time.sleep(time_to_sleep)

        if self.ros_debug:
            self.publish(self.state, action)

        return self.state.get_state_array(scaled=True), reward, bool(self.done), info

    def get_ball_position(self):
        return np.array(self.pybullet_client.getBasePositionAndOrientation(self.ball_index)[0][:2])

    def get_action_biases(self):
        return self.robot.get_init_bias(self.cartesian_action)

    def _seed(self, seed=None):
        self.np_random, seed = gym.utils.seeding.np_random(seed)
        self.robot.np_random = self.np_random  # use the same np_randomizer for robot as for env
        return [seed]

    def publish(self, state, action):
        state.publish_debug()

        if self.action_publisher.get_num_connections() > 0:
            action_msg = Float32MultiArray()
            action_msg.data = action
            self.action_publisher.publish(action_msg)

        self.reward_function.publish_reward(self)

        self.ros_interface.publish_true_odom()
        self.ros_interface.publish_joints()
        self.ros_interface.publish_foot_pressure()
        self.ros_interface.publish_imu()

    def render(self, mode='rgb_array'):
        if mode not in ('rgb_array', 'human'):
            return
            # raise ValueError('Unsupported render mode:{}'.format(mode))
        base_pos = self.robot.get_pose_world_frame()[0]
        view_matrix = self.pybullet_client.computeViewMatrixFromYawPitchRoll(cameraTargetPosition=base_pos,
                                                                             distance=self.camera_distance,
                                                                             yaw=self.camera_yaw,
                                                                             pitch=self.camera_pitch, roll=0,
                                                                             upAxisIndex=2)
        proj_matrix = self.pybullet_client.computeProjectionMatrixFOV(fov=60, aspect=float(
            self.render_width) / self.render_height, nearVal=0.1, farVal=100.0)
        (_, _, px, _, _) = self.pybullet_client.getCameraImage(width=self.render_width, height=self.render_height,
                                                               renderer=self.pybullet_client.ER_BULLET_HARDWARE_OPENGL,
                                                               viewMatrix=view_matrix, projectionMatrix=proj_matrix)
        rgb_array = np.array(px)
        rgb_array = rgb_array[:, :, :3]
        if mode == 'human':
            import cv2
            bgr_array = cv2.cvtColor(rgb_array, cv2.COLOR_RGB2BGR)
            cv2.imshow('render', bgr_array)
            cv2.waitKey(1)
        return rgb_array

    def close(self):
        self.pybullet_client.disconnect()


class WolfgangEnv(DeepKickEnv):
    def __init__(self, gui=False, debug=False, use_demonstration=True, early_termination=True, gravity=True,
                 step_freq=30, reward_function_name=None, used_joints="Legs", force=None,
                 random_initialization=True, cartesian_action=True, observation_type=None):
        reward_function = reward.__getattribute__(reward_function_name)

        super().__init__(reward_function=reward_function(), used_joints=used_joints, step_freq=step_freq,
                         ros_debug=debug, gui=gui, use_demonstration=use_demonstration,
                         early_termination=early_termination, gravity=gravity, force=force,
                         random_initialization=random_initialization, cartesian_action=cartesian_action,
                         observation_type=observation_type)
