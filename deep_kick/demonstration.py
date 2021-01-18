class Frame:
    def __init__(self, time, position, orientation, joint_names, joint_positions, joint_velocities):
        self.time = time
        self.robot_position = position
        self.robot_orientation = orientation
        self.joint_positions = {}
        self.joint_velocities = {}
        i = 0
        for name in joint_names:
            self.joint_positions[name] = joint_positions[i]
            self.joint_velocities[name] = joint_velocities[i]
            i += 1

    def get_as_msg(self):
        msg = ReferenceFrame()
        msg.header.stamp = rospy.Time.now()
        msg.time = self.time
        msg.phase = self.phase
        msg.pose.position.x = self.robot_position[0]
        msg.pose.position.y = self.robot_position[1]
        msg.pose.position.z = self.robot_position[2]
        msg.pose.orientation.x = self.robot_orientation[0]
        msg.pose.orientation.y = self.robot_orientation[1]
        msg.pose.orientation.z = self.robot_orientation[2]
        msg.pose.orientation.w = self.robot_orientation[3]
        msg.joint_states.name = self.joint_positions.keys()
        msg.joint_states.position = self.joint_positions.values()
        return msg
