from itertools import islice
from IPython import embed
import os, sys, time, re

from prompt_toolkit.filters import app

import tmsmt as tm
import numpy as np
from codetiming import Timer

from pddl_parse.PDDL import PDDL_Parser
import pybullet as p
import pybullet_tools.utils as pu
import pybullet_tools.kuka_primitives3 as pk
import pybullet_tools.pr2_primitives as pp
import pybullet_tools.pr2_utils as ppu
from build_scenario import PlanningScenario
from task_planner import TaskPlanner

from logging_utils import *
logging.setLoggerClass(ColoredLogger)
logger = logging.getLogger('MAIN')

EPSILON = 0.01
RESOLUTION = 0.2
MOTION_TIMEOUT = 200

class DomainSemantics(object):
    def __init__(self, scene):
        self.scn = scene
        self.robot = self.scn.robots[0]
        self.arm = 'left'

        self.base_joints = [0,1,2]
        self.arm_joints = list(pp.get_arm_joints(self.robot, self.arm))
        self.movable_joints = self.base_joints + self.arm_joints

        self.all_bodies = self.scn.all_bodies
        self.max_distance = pp.MAX_DISTANCE
        self.self_collision = True
        self.approach_pose = (( 0. ,  0. , -0.02), (0.0, 0.0, 0.0, 1.0))
        self.ik_fn = pp.get_ik_fn(self.scn)
        self.ir_sampler = pp.get_ir_sampler(self.scn)
        self.end_effector_link = pp.get_gripper_link(self.robot, self.arm)
        self.base_motion_obstacles = list(set(self.scn.env_bodies) | set(self.scn.regions))
        self.arm_motion_obstacles = list(set(self.scn.env_bodies) | set(self.scn.regions))
        self.custom_limits = []
        self.base_lower_limits, self.base_upper_limits = pp.get_custom_limits(self.robot, self.base_joints, self.custom_limits)
        self.disabled_collision_pairs={}
        self.sample_fn = pp.get_sample_fn(self.robot, self.arm_joints)

    def get_global_ik(self, input_tuple, attachment=[]):

        arm, body, pose, grasp = input_tuple
        ir_generator = self.ir_sampler(*input_tuple)
        self.max_attempts = 25
        
        for i in range(self.max_attempts):
            try:
                ir_outputs = next(ir_generator)
            except StopIteration:
                return None
            if ir_outputs is None:
                continue
            ik_outputs = self.ik_fn(*(input_tuple + ir_outputs))
            if ik_outputs is None:
                continue
            joints_path = []
            for p in ik_outputs[0].commands[0].path:
                joints_path.append(p.values)
            path = pk.BodyPath(self.robot, joints_path, self.arm_joints, attachments=attachment)
            return ir_outputs[0].values, path

    def pr2_inverse_kinematics_random(self, robot, arm, target_poses, obstacles=[], custom_limits={}, **kwargs):
        arm_link = pp.get_gripper_link(robot, arm)
        arm_joints = pp.get_arm_joints(robot, arm)
        
        # arm_conf = pp.sub_inverse_kinematics(robot, arm_joints[0], arm_link, target_pose, custom_limits=custom_limits)
        solutions = pp.plan_cartesian_motion(robot, arm_joints[0], arm_link, target_poses, **kwargs)

        if solutions is None:
            print(f"ik failed: no candidate IK")
            return None
        if not any(pp.pairwise_collision(robot, b) for b in obstacles):
            print(f"ik failed: collided ik")
            return None            
        return pp.get_joint_positions(robot, arm_joints)

    def get_ik_custom(self, input_tuple, attachment=[]):
        arm, body, pose, grasp = input_tuple
        init_base = pp.get_joint_positions(self.robot, self.base_joints)
        init_arm = pp.get_joint_positions(self.robot, self.arm_joints)

        if attachment==[]:
            arm_obstacles = self.scn.all_bodies
        else:
            arm_obstacles = list(set(self.scn.all_bodies) - {body})

        gripper_pose = pp.multiply(pose.value, pp.invert(grasp.value))  # w_f_g = w_f_o * (g_f_o)^-1

        pick_pose = pp.multiply(pose.value, ((0,0,-0.02),pp.quat_from_euler((0,-np.pi/2,0))))
        approach_pose = pp.multiply(pose.value, ((0,0,-0.1),pp.quat_from_euler((0,-np.pi/2,0))))
        pp.draw_pose(pick_pose)

        base_generator = pp.learned_pose_generator(self.robot, pick_pose, arm=arm, grasp_type=grasp.grasp_type)
        for base_conf in islice(base_generator, 50):
            if not pp.all_between(self.base_lower_limits, base_conf, self.base_upper_limits):
                logger.warning(f"generate base conf succeed")
                continue
            pp.set_joint_positions(self.robot, self.base_joints, base_conf)
        # ik of arm
        # approach_pose = pp.multiply(pose.value, ((0,0,-0.1),(0,0,0,1)))
        # grasp_conf = pp.pr2_inverse_kinematics(self.robot, arm, gripper_pose,
        #                                     custom_limits=self.custom_limits)
            pick_conf = self.pr2_inverse_kinematics_random(self.robot, arm, [pick_pose],
                                                    custom_limits=self.custom_limits)
            approach_conf = self.pr2_inverse_kinematics_random(self.robot, arm, [approach_pose],
                                                    custom_limits=self.custom_limits)
            if pick_conf is None or approach_conf is None:
                logger.warning(f"arm ik failed")
                continue
            break
        pp.set_joint_positions(self.robot, self.arm_joints, init_arm)
        return base_conf, [approach_conf, pick_conf]

    def plan_base_motion(self, goal_conf, attachment=[]):
        raw_path = pp.plan_joint_motion(self.robot, [0,1,2], goal_conf, attachments=[],
                                obstacles=self.base_motion_obstacles, custom_limits=self.custom_limits,
                                self_collisions=pp.SELF_COLLISIONS,
                                restarts=4, iterations=50, smooth=50)
        path = pk.BodyPath(self.robot, raw_path, self.base_joints, attachments=attachment)
        return path

    def plan_joints_motion(self):
        pass

    def motion_plan(self, body, goal_pose, attaching=False):
        # pu.draw_pose(goal_pose)
        init_conf = pp.get_joint_positions(self.robot, self.movable_joints)
        # attachment
        if attaching:
            body_pose = pu.get_pose(body)
            tcp_pose = pu.get_link_pose(self.robot, self.end_effector_link)
            grasp_pose = pu.multiply(pu.invert(tcp_pose), body_pose)
            grasp = pp.BodyGrasp(body, grasp_pose, self.approach_pose, self.robot, attach_link=self.end_effector_link)
            obstacles = list(set(self.scn.all_bodies) - {body})
            attachment = [grasp.attachment()]
        else:
            obstacles = self.scn.all_bodies
            attachment = []

        goal_pose = pp.BodyPose(body, goal_pose)
        approach_vector = pp.APPROACH_DISTANCE * pp.get_unit_vector([1, 0, 0])
        grasps = []
        for g in pp.get_top_grasps(body, grasp_length=pp.GRASP_LENGTH):
            grasps.append(pp.Grasp('top', body, g, pp.multiply((approach_vector, pp.unit_quat()), g), pp.TOP_HOLDING_LEFT_ARM))

        import random
        filtered_grasps = []
        for grasp in grasps:
            grasp_width = 0.0
            if grasp_width is not None:
                grasp.grasp_width = grasp_width
                filtered_grasps.append(grasp)
        random.shuffle(filtered_grasps)

        input_tuple = ('left', body, goal_pose, filtered_grasps[0])

        iteration = 0
        while iteration<10:        
            base_conf, arm_path = self.get_ik_custom(input_tuple, attachment)

            print(f"ik generated, plan motion")
            path = pp.plan_joint_motion(self.robot, self.arm_joints, arm_path[0], obstacles=obstacles,self_collisions=self.self_collision, 
            disabled_collisions=self.disabled_collision_pairs, attachments=attachment,
            max_distance=self.max_distance, iterations=MOTION_TIMEOUT)
            if path is None:
                iteration += 1
                pp.set_joint_positions(self.robot, self.movable_joints, init_conf)
            else:
                break

        pp.set_joint_positions(self.robot, self.movable_joints, init_conf)
        
        base_path = self.plan_base_motion(base_conf, attachment)

        embed()

        pp.set_joint_positions(self.robot, self.base_joints, base_path.path[-1])
        pp.set_joint_positions(self.robot, self.arm_joints, arm_path.path[-1])

        base_path
        return True, [base_path, arm_path]
        if path is None:
            # logger.error(f"free motion planning failed")
            return False, path
        cmd = pp.Command([pp.BodyPath(self.robot, [path[0],path[-1]], joints=self.movable_joints, attachments=attachment)])
        cmd.execute()
        return True, path

class UnpackDomainSemantics(DomainSemantics):
    def activate(self):
        tm.bind_refine_operator(self.op_pick_up, "pick-up")
        tm.bind_refine_operator(self.op_put_down, "put-down")
        tm.bind_refine_operator(self.op_cook_or_clean, 'cook')
        tm.bind_refine_operator(self.op_cook_or_clean, 'clean')

    def op_pick_up(self, args):
        [a, obj, region, i, j] = args
        nop = tm.op_nop(self.scn)
        body = self.scn.bd_body[obj]
        (point, rotation) = pu.get_pose(body)
        center = pu.get_aabb_center(pu.get_aabb(body))
        extend = pu.get_aabb_extent(pu.get_aabb(body))
        x = center[0]
        y = center[1]
        z = center[2] + EPSILON
        body_pose = pu.Pose([x,y,z], pu.euler_from_quat(rotation))
        goal_pose = pu.multiply(body_pose, ((0,0,0), pu.quat_from_axis_angle([1,0,0],np.pi)))
        res, path = self.motion_plan(body, goal_pose, attaching=False)
        return res, path

    def op_put_down(self, args):
        (a, obj, region, i, j) = args
        nop = tm.op_nop(self.scn)
        body = self.scn.bd_body[obj]
        region_ind = self.scn.bd_body[region]

        aabb = pu.get_aabb(region_ind)
        center_region = pu.get_aabb_center(aabb)
        extend_region = pu.get_aabb_extent(aabb)

        (point, rotation) = pu.get_pose(body)
        aabb_body = pu.get_aabb(body)
        extend_body = pu.get_aabb_extent(aabb_body)
        x = center_region[0] + int(i)*RESOLUTION
        y = center_region[1] + int(j)*RESOLUTION
        z = center_region[2] + extend_region[2]/2 + extend_body[2] + EPSILON
        
        body_pose = pu.Pose([x,y,z], pu.euler_from_quat(rotation))
        goal_pose = pu.multiply(body_pose, ((0,0,0), pu.quat_from_axis_angle([1,0,0],np.pi)))

        res, path = self.motion_plan(body, goal_pose, attaching=True)
        return res, path

    def op_cook_or_clean(self, args):
        [a, obj, region, i, j] = args
        color = p.getVisualShapeData(self.scn.bd_body[region])[0][-1]
        p.changeVisualShape(self.scn.bd_body[obj], -1, rgbaColor=color)
        return True, obj

class PDDLProblem(object):
    def __init__(self, scene, domain_name):
        self.scn = scene
        self.domain_name = domain_name
        self.name = 'clean-and-cook'
        self.objects = None
        self.init = None
        self.goal = None
        self.metric = None

        self.objects = self._gen_scene_objects()
        self._goal_state()
        self._init_state()
        logger.info(f"problem instantiated")

    def _gen_scene_objects(self):
        scene_objects = dict()
        
        # define stove, sink, locations
        stove = set()
        sink = set()
        locations = set()
        for region in self.scn.regions:
            (lower, upper) = pu.get_aabb(region)
            size = np.abs((upper -lower)[:2])
            xr = int(np.floor(size[0]/2/RESOLUTION))
            yr = int(np.floor(size[1]/2/RESOLUTION))
            for i in range(xr+1):
                for j in range(yr+1):
                    locations.add(tm.mangle(self.scn.bd_body[region], i, j))
                    locations.add(tm.mangle(self.scn.bd_body[region], -i, j))
                    locations.add(tm.mangle(self.scn.bd_body[region], i, -j))
                    locations.add(tm.mangle(self.scn.bd_body[region], -i, -j))
        scene_objects['location'] = locations

        # define movable objects
        movable = set()
        for b in self.scn.movable_bodies:
            movable.add(self.scn.bd_body[b])
        scene_objects['block'] = movable

        return scene_objects

    def _init_state(self):

        init = []
        # handempty
        init.append(['handempty'])

        movables = self.scn.movable_bodies
        # clear

        # ontable
        region_aabb=dict()
        for r in self.scn.regions:
            region_aabb[self.scn.bd_body[r]] = np.array(pu.get_aabb(r))[:,:2]
        occuppied = set()
        for m in movables:
            point = np.array(pu.get_point(m))[:2]
            for region, aabb in region_aabb.items():
                if pu.aabb_contains_point(point, aabb):
                    loc = (point - np.array(pu.get_point(self.scn.bd_body[region]))[:2])/RESOLUTION
                    location = tm.mangle(region, int(loc[0]), int(loc[1]))
                    init.append(['ontable', self.scn.bd_body[m], location])
                    # init.append(['not',['clear',location]])
                    # occuppied.add(location)
                    break
        # holding
        for m in movables:
            init.append(['not',['holding', self.scn.bd_body[m]]])

        # cleaned
        for m in movables:
            init.append(['not',['cleaned', self.scn.bd_body[m]]])

        # cooked
        for m in movables:
            init.append(['not', ['cooked', self.scn.bd_body[m]]])
        
        # issink and isstove
        for loc in self.objects['location']:
            if 'stove' in loc:
                init.append(['isstove', loc])
            if 'sink' in loc:
                init.append(['issink', loc])

        self.init = init

    def _goal_state(self):
        self.goal = []
        self.goal.append(['handempty'])
        self.goal.append(['cooked', 'box1'])
        self.goal.append(['cooked', 'box2'])

        # ontable = ['or']
        # for loc in self.objects['location']:
        #     if 'region2' in loc:
        #         ontable.append(['ontable','c1',loc])

def ExecutePlanNaive(scn, task_plan, motion_plan):

    robot = scn.robots[0]
    ind = -1
    for _,tp in task_plan.items():
        ind += 1
        tp_list = re.split(' |__', tp[1:-1])
        if 'put-down' == tp_list[0]:
            end_effector_link = pu.link_from_name(robot, pp.TOOL_FRAMES[pu.get_body_name(robot)])
            body = scn.bd_body[tp_list[1]]
            body_pose = pu.get_pose(body)
            tcp_pose = pu.get_link_pose(robot, end_effector_link)
            grasp_pose = pu.multiply(pu.invert(tcp_pose), body_pose)
            # grasp_pose = ((0,0,0.035),(1,0,0,0))
            approach_pose = (( 0. ,  0. , -0.02), (0.0, 0.0, 0.0, 1.0))
            grasp = pp.BodyGrasp(body, grasp_pose, approach_pose, robot, attach_link=end_effector_link)
            attachment = [grasp.attachment()]
            cmd = pp.Command([pp.BodyPath(robot, motion_plan[ind], attachments=attachment)])
        elif 'pick-up' == tp_list[0]:
            cmd = pp.Command([pp.BodyPath(robot, motion_plan[ind])])
        cmd.execute()
    time.sleep(1)
    scn.reset()

def main():
    tp_total_time = Timer(name='tp_total_time', text='', logger=logger.info)
    mp_total_time = Timer(name='mp_total_time', text='', logger=logger.info)
    total_time = 0

    visualization = True
    pu.connect(use_gui=visualization)
    scn = PlanningScenario()
    parser = PDDL_Parser()
    dirname = os.path.dirname(os.path.abspath(__file__))
    domain_filename = os.path.join(dirname, 'domain_idtmp_cook.pddl')
    parser.parse_domain(domain_filename)
    domain_name = parser.domain_name
    problem_filename = os.path.join(dirname, 'problem_idtmp_'+domain_name+'.pddl')
    problem = PDDLProblem(scn, parser.domain_name)
    parser.dump_problem(problem, problem_filename)
    logger.info(f"problem.pddl dumped")

    # initial domain semantics
    domain_semantics = UnpackDomainSemantics(scn)
    domain_semantics.activate()

    # IDTMP
    tp_total_time.start()
    tp = TaskPlanner(problem_filename, domain_filename, start_horizon=11, max_horizon=14)
    tp.incremental()
    tp.modeling()
    tp_total_time.stop()


    tm_plan = None
    t00 = time.time()
    while tm_plan is None:
        # ------------------- task plan ---------------------
        t0 = time.time()
        t_plan = None
        while t_plan is None:
            t_plan = tp.search_plan()
            if t_plan is None:
                logger.warning(f"task plan not found in horizon: {tp.horizon}")
                print(f'')
                tp_total_time.start()
                tp.incremental()
                tp.modeling()
                tp_total_time.stop()
                logger.info(f"search task plan in horizon: {tp.horizon}")
                global MOTION_TIMEOUT 
                MOTION_TIMEOUT += 10

        logger.info(f"task plan found, in horizon: {tp.horizon}")
        for h,p in t_plan.items():
            logger.info(f"{h}: {p}")

        # ------------------- motion plan ---------------------
        mp_total_time.start()
        res, m_plan = tm.motion_refiner(t_plan)
        mp_total_time.stop()

        scn.reset()
        if res:
            logger.info(f"task and motion plan found")
            break
        else: 
            t0 = time.time()
            logger.warning(f"motion refine failed")
            logger.info(f'')
            tp.add_constraint(m_plan)
            t_plan = None

    total_time = time.time()-t00
    all_timers = tp_total_time.timers
    print(f"all timers: {all_timers}")
    print("task plan time: {:0.4f} s".format(all_timers[tp_total_time.name]))
    print("motion refiner time: {:0.4f} s".format(all_timers[mp_total_time.name]))
    print(f"total planning time: {total_time}")
    print(f"task plan counter: {tp.counter}")

    embed()
    while True:
        ExecutePlanNaive(scn, t_plan, m_plan)
        time.sleep(1)

    pu.disconnect()

if __name__=="__main__":
    main()

