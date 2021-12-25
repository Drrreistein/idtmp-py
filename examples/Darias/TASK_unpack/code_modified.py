
class sdg_sample_grasp_dir(object):
    def __init__(self, robot, dic_body_info):
        self.robot = robot
        self.dic_body_info = dic_body_info

    def __call__(self, input_tuple, seed=None):
        body, pose = input_tuple
        list_available = [0, 1, 2, 3, 4]
        if seed is None:
            idx = random.sample(list_available, 1)[0]
        else:
            idx = np.array([seed]).flatten()[0]

        direction = list_available[int(idx)]

        return (GraspDirection(body, pose, direction, self.robot, self.dic_body_info),)

class sdg_sample_grasp0(object):
    def __init__(self, robot):
        self.robot = robot
        self.end_effector_link = link_from_name(robot, TOOL_FRAMES[get_body_name(robot)])

    def __call__(self, input_tuple, seed=None):
        body, grasp_dir = input_tuple

        assert body == grasp_dir.body

        list_available = [0, 1, 2, 3, 4]

        grasp_poses = get_sucker_grasps2(body, direction=grasp_dir.direction, under=True, tool_pose=Pose(),
                                         grasp_length=0)  # 0,3,4
        grasp_pose = random.sample(grasp_poses, 1)[0]

        approach_pose = Pose(0.1 * Point(z=-1))  # pose bias wrt end-effector frame
        body_grasp = BodyGrasp(body, grasp_pose, approach_pose, self.robot, self.end_effector_link)
        return (body_grasp,)  # return a tuple

class sdg_sample_grasp(object):
    def __init__(self, robot):
        self.robot = robot
        self.end_effector_link = link_from_name(robot, TOOL_FRAMES[get_body_name(robot)])

    def search(self, input_tuple, seed=None):
        """return the ee_frame wrt the object_frame of the object"""
        body, pose, grasp_dir = input_tuple  # grasp_dir defined in ellipsoid_frame of the body

        assert body == grasp_dir.body
        ellipsoid_frame, obj_extent, direction = grasp_dir.ellipsoid_frame, grasp_dir.obj_extent, grasp_dir.direction

        ex, ey, ez = obj_extent

        translate_z = Pose(point=[0, 0, -0.001])
        list_grasp = []

        if direction == 0:
            """ee at +X of the ellipsoid_frame"""
            swap_z = Pose(euler=[0, -np.pi / 2, 0])
            # translate_point: choose from the grasping surface with 2 dof
            d1, d2 = 0., 0.  # [-0.5, 0.5]
            translate_point = Pose(point=[ex / 2, 0 + d1 * ey, ez / 2 + d2 * ez])
            for j in range(2):
                rotate_z = Pose(euler=[0, 0, j * np.pi])  # gripper open with +Y direction
                grasp = multiply(translate_point, swap_z, rotate_z, translate_z)
                list_grasp.append(grasp)

        elif direction == 1:
            """ee at +Y"""
            swap_z = Pose(euler=[np.pi / 2, 0, 0])
            d1, d2 = 0., 0.  # [-0.5, 0.5]
            translate_point = Pose(point=[0 - d1 * ex, ey / 2, ez / 2 + d2 * ez])
            for j in range(2):
                rotate_z = Pose(euler=[0, 0, j * np.pi + np.pi / 2])
                grasp = multiply(translate_point, swap_z, rotate_z, translate_z)
                list_grasp.append(grasp)

        elif direction == 2:
            """ee at +Z"""
            swap_z = Pose(euler=[0, np.pi, 0])
            d1, d2 = 0., 0.  # [-0.5, 0.5]
            translate_point = Pose(point=[0 - d2 * ex, 0 + d1 * ey, ez])
            for j in range(4):
                rotate_z = Pose(euler=[0, 0, j * np.pi / 2])
                grasp = multiply(translate_point, swap_z, rotate_z, translate_z)
                list_grasp.append(grasp)

        elif direction == 3:
            """ee at -X"""
            swap_z = Pose(euler=[0, np.pi / 2, 0])
            d1, d2 = 0., 0.  # [-0.5, 0.5]
            translate_point = Pose(point=[-ex / 2, 0 - d1 * ey, ez / 2 + d2 * ez])
            for j in range(2):
                rotate_z = Pose(euler=[0, 0, j * np.pi + np.pi])
                grasp = multiply(translate_point, swap_z, rotate_z, translate_z)
                list_grasp.append(grasp)

        elif direction == 4:
            """ee at -Y"""
            swap_z = Pose(euler=[-np.pi / 2, 0, 0])
            d1, d2 = 0., 0.  # [-0.5, 0.5]
            translate_point = Pose(point=[0 + d1 * ex, -ey / 2, ez / 2 + d2 * ez])
            for j in range(2):
                rotate_z = Pose(euler=[0, 0, j * np.pi - np.pi / 2])
                grasp = multiply(translate_point, swap_z, rotate_z, translate_z)
                list_grasp.append(grasp)
        """ee_frame wrt ellipsoid_frame"""
        grasp_pose = random.sample(list_grasp, 1)[0]
        """ee_frame wrt object_frame: get_pose()"""
        grasp_pose = multiply(invert(get_pose(body)), pose_from_tform(ellipsoid_frame), grasp_pose)

        approach_pose = Pose(0.1 * Point(z=-1))  # pose bias wrt end-effector frame
        body_grasp = BodyGrasp(body, grasp_pose, approach_pose, self.robot, self.end_effector_link)
        return (body_grasp,)  # return a tuple

    def __call__(self, input_tuple, seed=None):
        return self.search(input_tuple, seed=None)

class sdg_ik_grasp(object):
    def __init__(self, robot, all_bodies=[], teleport=False, num_attempts=50):
        self.all_bodies = all_bodies
        self.teleport = teleport
        self.num_attempts = num_attempts
        self.movable_joints = get_movable_joints(robot)
        self.sample_fn = get_sample_fn(robot, self.movable_joints)
        self.robot = robot
        self.visualization_collision = False
        self.max_distance = MAX_DISTANCE

    def search(self, input_tuple, seed=None):
        body, pose, grasp = input_tuple  # pose is measured by get_pose()

        self.msg_yg = 0  # Collision distance of IK error
        self.msg_generator = False
        self.msg_obstacle = None
        self.msg_bi = False

        set_pose(body, pose.pose)

        obstacles = self.all_bodies

        grasp_pose_ee = multiply(pose.pose, grasp.grasp_pose)  # in world frame
        approach_pose_ee = multiply(grasp_pose_ee, grasp.approach_pose)  # 右乘,以当前ee坐标系为基准进行变换

        list_q_approach = []
        list_q_grasp = []
        list_command_approach = []
        list_collision_obstacle = []

        for _ in range(self.num_attempts):
            sampled_conf = self.sample_fn()
            set_joint_positions(self.robot, self.movable_joints, sampled_conf)  # Random seed

            q_approach = inverse_kinematics(self.robot, grasp.link, approach_pose_ee)
            if not q_approach:
                point, quat = get_link_pose(self.robot, grasp.link)
                target_point, target_quat = approach_pose_ee
                error_dist = np.linalg.norm(np.array(point) - np.array(target_point)) + np.linalg.norm(
                    np.array(quat) - np.array(target_quat))
                if error_dist > self.msg_yg:
                    self.msg_yg = error_dist

            q_grasp = inverse_kinematics(self.robot, grasp.link, grasp_pose_ee)
            if not q_grasp:
                point, quat = get_link_pose(self.robot, grasp.link)
                target_point, target_quat = grasp_pose_ee
                error_dist = np.linalg.norm(np.array(point) - np.array(target_point)) + np.linalg.norm(
                    np.array(quat) - np.array(target_quat))
                if error_dist > self.msg_yg:
                    self.msg_yg = error_dist

            if q_approach and q_grasp:
                """If it is reachable"""
                self.msg_generator = True
                list_q_approach.append(q_approach)
                list_q_grasp.append(q_grasp)
                set_joint_positions(self.robot, self.movable_joints, q_approach)
                no_collision = True
                for b in obstacles:
                    if pairwise_collision(self.robot, b,
                                          visualization=self.visualization_collision,
                                          max_distance=self.max_distance):
                        no_collision = False
                        list_collision_obstacle.append(b)
                        c_dist = collision_dist(self.robot, b,
                                                visualization=self.visualization_collision,
                                                max_distance=self.max_distance)
                        if c_dist and c_dist > self.msg_yg:
                            self.msg_yg = c_dist
                set_joint_positions(self.robot, self.movable_joints, q_grasp)
                for b in obstacles:
                    if pairwise_collision(self.robot, b,
                                          visualization=self.visualization_collision,
                                          max_distance=self.max_distance):
                        no_collision = False
                        list_collision_obstacle.append(b)
                        c_dist = collision_dist(self.robot, b,
                                                visualization=self.visualization_collision,
                                                max_distance=self.max_distance)
                        if c_dist and c_dist > self.msg_yg:
                            self.msg_yg = c_dist
                command = None
                approach_conf = None
                if no_collision:
                    """If there is no collision"""
                    approach_conf = BodyConf(self.robot, q_approach)
                    if self.teleport:
                        path = [q_approach, q_grasp]
                    else:
                        approach_conf.assign()
                        # The path from q_approach to q_grasp.
                        path = plan_direct_joint_motion(self.robot, approach_conf.joints, q_grasp,
                                                        obstacles=obstacles,
                                                        disabled_collisions=DISABLED_COLLISION_PAIR,
                                                        max_distance=self.max_distance)
                        if path:
                            command = Command([BodyPath(self.robot, path),
                                               Attach(body, self.robot, grasp.link),
                                               BodyPath(self.robot, path[::-1], attachments=[grasp])])
                list_command_approach.append(command)
                if command:
                    set_joint_positions(self.robot, self.movable_joints, list_q_grasp[0])
                    return approach_conf, command, q_approach, q_grasp
                """Find the obstacle with the maximum occurrence"""
                if list_collision_obstacle:
                    self.msg_obstacle = max(list_collision_obstacle, key=list_collision_obstacle.count)
        # jp = get_joint_positions(self.robot, self.movable_joints)
        # ee_pose = get_link_pose(self.robot, grasp.link)
        #
        # err1 = np.array(ee_pose[0]) - np.array(grasp_pose_ee[0])
        # err2 = np.array(ee_pose[0]) - np.array(approach_pose_ee[0])

        if list_q_approach and list_q_grasp:
            set_joint_positions(self.robot, self.movable_joints, list_q_grasp[0])
            return None, None, list_q_approach[0], list_q_grasp[0]
        return None, None, None, None

    def __call__(self, input_tuple, seed=None):
        approach_conf, command, q_approach, q_grasp = self.search(input_tuple, seed=None)

        if command is None:
            return None
        else:
            return approach_conf, command

    def get_error_message(self):
        return SDG_MSG(self.msg_generator, self.msg_obstacle, self.msg_yg, self.msg_bi)

class sdg_plan_free_motion(object):
    def __init__(self, robot, all_bodies=[], rrt_iteration=None, teleport=False, self_collisions=True):
        self.all_bodies = all_bodies
        self.teleport = teleport
        self.self_collisions = self_collisions
        self.robot = robot
        self.max_distance = MAX_DISTANCE
        if rrt_iteration is None:
            self.rrt_iteration = 20
        else:
            self.rrt_iteration = rrt_iteration

    def __call__(self, input_tuple, seed=None):
        conf1, conf2 = input_tuple

        self.msg_yg = 0  # Collision distance of IK error
        self.msg_generator = False
        self.msg_obstacle = None
        self.msg_bi = True

        assert ((conf1.body == conf2.body) and (conf1.joints == conf2.joints))
        if self.teleport:
            path = [conf1.configuration, conf2.configuration]
        else:
            conf1.assign()
            # obstacles = fixed + assign_fluent_state(fluents)
            obstacles = self.all_bodies
            path = plan_joint_motion(self.robot, conf2.joints, conf2.configuration, obstacles=obstacles,
                                     self_collisions=self.self_collisions, disabled_collisions=DISABLED_COLLISION_PAIR,
                                     max_distance=self.max_distance, iterations=self.rrt_iteration)
            if path is None:
                if DEBUG_FAILURE:
                    user_input('Free motion failed')
                self.msg_yg = 1
                return None
        command = Command([BodyPath(self.robot, path, joints=conf2.joints)])
        return (command,)  # return a tuple

    def get_error_message(self):
        return SDG_MSG(self.msg_generator, self.msg_obstacle, self.msg_yg, self.msg_bi)

class sdg_plan_holding_motion(object):
    def __init__(self, robot, all_bodies=[], rrt_iteration=None, teleport=False, self_collisions=True):
        self.all_bodies = all_bodies
        self.teleport = teleport
        self.self_collisions = self_collisions
        self.robot = robot
        self.max_distance = MAX_DISTANCE
        if rrt_iteration is None:
            self.rrt_iteration = 20
        else:
            self.rrt_iteration = rrt_iteration

    def __call__(self, input_tuple, seed=None):
        conf1, conf2, body, grasp = input_tuple

        self.msg_yg = 0  # Collision distance of IK error
        self.msg_generator = False
        self.msg_obstacle = None
        self.msg_bi = True
        assert ((conf1.body == conf2.body) and (conf1.joints == conf2.joints))
        if self.teleport:
            path = [conf1.configuration, conf2.configuration]
        else:
            conf1.assign()
            # obstacles = all_bodies + assign_fluent_state(fluents)
            obstacles = list(set(self.all_bodies) - {grasp.body})
            path = plan_joint_motion(self.robot, conf2.joints, conf2.configuration,
                                     obstacles=obstacles, attachments=[grasp.attachment()],
                                     self_collisions=self.self_collisions, disabled_collisions=DISABLED_COLLISION_PAIR,
                                     max_distance=self.max_distance, iterations=self.rrt_iteration)
            if path is None:
                if DEBUG_FAILURE:
                    user_input('Holding motion failed')
                self.msg_yg = 1
                return None
        command = Command([BodyPath(self.robot, path, joints=conf2.joints, attachments=[grasp])])
        return (command,)  # return a tuple

    def get_error_message(self):
        return SDG_MSG(self.msg_generator, self.msg_obstacle, self.msg_yg, self.msg_bi)
