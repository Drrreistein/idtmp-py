from logging import PlaceHolder
from IPython import embed
import os, sys, time, re
from codetiming import Timer
from numpy.core.defchararray import array, mod
from numpy.core.fromnumeric import cumsum
from copy import deepcopy

import tmsmt as tm
import numpy as np
import z3

from pddl_parse.PDDL import PDDL_Parser
import pybullet_tools.utils as pu
import pybullet_tools.kuka_primitives3 as pk
from build_scenario import get_scn
from task_planner import TaskPlanner
from plan_cache import PlanCache
from logging_utils import *
logging.setLoggerClass(ColoredLogger)
logging.basicConfig(filename='./log/logging.log',
                            filemode='a',
                            format='%(asctime)s, %(name)s %(message)s',
                            datefmt='%H:%M:%S',
                            level=logging.DEBUG)
logger = logging.getLogger('MAIN')
from feasibility_check import FeasibilityChecker
EPSILON = 0.01
RESOLUTION = 0.1
DIR_NUM = 1
MOTION_ITERATION = 500
# for i in range(7):
#     custom_limits[i] = (-2*np.pi, 2*np.pi)

class DomainSemantics(object):
    def __init__(self, scene):
        self.scn = scene
        self.robot = self.scn.robots[0]
        self.movable_joints = pu.get_movable_joints(self.robot)
        self.all_bodies = self.scn.all_bodies
        self.max_distance = pk.MAX_DISTANCE
        self.self_collision = True
        self.approach_pose = (( 0. ,  0. , -0.02), (0.0, 0.0, 0.0, 1.0))
        self.end_effector_link = pu.link_from_name(self.robot, pk.TOOL_FRAMES[pu.get_body_name(self.robot)])
        self.sdg_motioner = pk.sdg_plan_free_motion(self.robot, self.all_bodies)

    def motion_plan(self, body, goal_pose, attaching=False):
        init_joints = pk.get_joint_positions(self.robot,self.movable_joints)
        # pu.draw_pose(goal_pose)

        if attaching:
            body_pose = pu.get_pose(body)
            tcp_pose = pu.get_link_pose(self.robot, self.end_effector_link)
            grasp_pose = pu.multiply(pu.invert(body_pose), tcp_pose)
            
            grasp = pk.BodyGrasp(body, grasp_pose, self.approach_pose, self.robot, attach_link=self.end_effector_link)
            obstacles = list(set(self.scn.all_bodies) - {body})
            attachment = [grasp.attachment()]
        else:
            obstacles = self.scn.all_bodies
            attachment = []

        # goal_joints = pu.inverse_kinematics_random(self.robot, self.end_effector_link, goal_pose, custom_limits=custom_limits)
        goal_joints = pu.inverse_kinematics_random(self.robot, self.end_effector_link, goal_pose, obstacles=obstacles,self_collisions=self.self_collision, 
        disabled_collisions=pk.DISABLED_COLLISION_PAIR, attachments=attachment,max_distance=self.max_distance)

        if goal_joints is None:
            return False, goal_joints
        # start_conf = pk.BodyConf(self.robot, pk.get_joint_positions(self.robot,self.movable_joints), self.movable_joints)
        goal_conf = pk.BodyConf(self.robot, goal_joints, self.movable_joints)
        
        # if attaching:
        #     embed()
        path = pu.plan_joint_motion(self.robot, self.movable_joints, goal_conf.configuration, obstacles=obstacles,self_collisions=self.self_collision, 
        disabled_collisions=pk.DISABLED_COLLISION_PAIR, attachments=attachment, max_distance=self.max_distance, iterations=MOTION_ITERATION)

        if path is None:
            # if attaching:
            #     embed()
            # logger.error(f"free motion planning failed")
            return False, path

        cmd = pk.Command([pk.BodyPath(self.robot, [path[0],path[-1]], joints=self.movable_joints, attachments=attachment)])
        cmd.execute()
        return True, path

class RegraspDomainSemantics(DomainSemantics):
    def activate(self):
        tm.bind_refine_operator(self.op_pick_up, "pick-up")
        tm.bind_refine_operator(self.op_put_down, "put-down")

    def op_pick_up(self, args):
        [a, obj, region, i, j, m,n,o,p] = args
        nop = tm.op_nop(self.scn)
        body = self.scn.bd_body[obj]
        (point, rotation) = pu.get_pose(body)
        center = pu.get_aabb_center(pu.get_aabb(body))
        extend = pu.get_aabb_extent(pu.get_aabb(body))
        x = center[0]
        y = center[1]
        z = center[2]
        
        offset = np.array([m,n,o], dtype=int) * (extend/2+np.array([EPSILON,EPSILON,EPSILON]))
        # body_pose = pu.Pose([x,y,z], pu.euler_from_quat(rotation))
        goal_point = np.array([x,y,z]) + offset
        angle_by_axis = np.array([m,n,o], dtype=int) * np.pi/2
        # goal_rot = pu.euler_from_quat(rotation) * pu.Euler(angle_by_axis[1], angle_by_axis[0], 0)
        goal_rot = pu.multiply_quats(rotation, pu.quat_from_euler((angle_by_axis[0], angle_by_axis[1], 0)))
        goal_rot = pu.multiply_quats(goal_rot, pu.quat_from_euler((np.pi, 0, 0)))
        # goal_rot = pu.multiply_quats(goal_rot, pu.quat_from_euler((0,0,int(p)/DIR_NUM*np.pi*2)))

        # pick_pose = pu.multiply(pu.Pose(goal_point, pu.euler_from_quat(goal_rot)), ((0,0,0), pu.quat_from_axis_angle([1,0,0],np.pi)))
        pick_pose = pu.Pose(goal_point, pu.euler_from_quat(goal_rot))
        # print(f"pick pose {pick_pose}")
        res, path = self.motion_plan(body, pick_pose, attaching=False)
        return res, path

    def op_put_down(self, args):
        (a, obj, region, i, j) = args
        nop = tm.op_nop(self.scn)
        body = self.scn.bd_body[obj]
        region_ind = self.scn.bd_body[region]

        aabb = pu.get_aabb(region_ind)
        center_region = pu.get_aabb_center(aabb)
        extend_region = pu.get_aabb_extent(aabb)

        (_, rotation) = pu.get_pose(body)
        aabb_body = pu.get_aabb(body)
        extend_body = pu.get_aabb_extent(aabb_body)
        x = center_region[0] + int(i)*RESOLUTION
        y = center_region[1] + int(j)*RESOLUTION
        z = center_region[2] + extend_region[2]/2 + extend_body[2]/2 + EPSILON
        body_pose = pu.get_pose(10)
        tcp_pose = pk.get_tcp_pose(0)
        body_to_tcp = pu.multiply(pu.invert(body_pose), tcp_pose)

        place_body_pose = pu.Pose([x,y,z], pu.euler_from_quat(rotation))
        goal_pose = pu.multiply(place_body_pose, body_to_tcp)
        # pu.draw_pose(place_body_pose)
        # pu.draw_pose(place_tcp_pose)

        res, path = self.motion_plan(body, goal_pose, attaching=True)
        return res, path

class PDDLProblem(object):
    def __init__(self, scene, domain_name):
        self.scn = scene
        self.domain_name = domain_name
        self.name = 'regrasp-block'
        self.objects = None
        self.init = None
        self.goal = None
        self.metric = None

        self.objects = self._gen_scene_objects()
        self._goal_state()
        self._init_state()
        self._plot_locations(self.objects['location'])
        self._real_goal_state()

    def _plot_locations(self, locations):
        for l in locations:
            [name, i, j] = tm.demangle(l)
            (point, rotation) = pu.get_pose(self.scn.bd_body[name])
            p = np.array([i,j,0], dtype=int) * RESOLUTION + np.array(point)
            pu.draw_pose((p, rotation), length=0.02)

    def _init_state(self):

        init = []
        # handempty
        init.append(['handempty'])

        movables = self.scn.movable_bodies
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

        # for loc in self.objects['location']:
        #     if loc not in occuppied:
        #         init.append(['clear', loc])
        self.init = init

    def _goal_state(self):
        self.goal = []
        self.goal.append(['handempty'])
        ontable = ['ontable','box1','region_shelf__1__0']

        self.goal.append(ontable)

    def _real_goal_state(self):
        self.real_goal = []
        self.real_goal.append(['handempty'])
        ontable = ['or']
        for loc in self.objects['location']:
            if 'shelf' in loc:
                ontable.append(['ontable','box1',loc])
        self.real_goal.append(ontable)

    def update_goal_in_formula(self, encoder):
        disconj = []
        for k,v in encoder.boolean_variables[encoder.horizon].items():
            if 'ontable_box1_region_shelf' in k:
                disconj.append(v)

        conj = []
        conj.append(z3.Or(disconj))
        conj.append(encoder.boolean_variables[encoder.horizon]['handempty'])

        return z3.And(conj)

    def _gen_scene_objects(self):
        scene_objects = dict()

        # define locations
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

        # define pick-up direction 
        direction = set()
        for i in range(DIR_NUM):
            direction.add(tm.mangle(1,0,0,i))
            direction.add(tm.mangle(-1,0,0,i))
            direction.add(tm.mangle(0,1,0,i))
            direction.add(tm.mangle(0,-1,0,i))
            direction.add(tm.mangle(0,0,1,i))
        scene_objects['direction'] = direction

        return scene_objects

def ExecutePlanNaive(scn, task_plan, motion_plan):

    robot = scn.robots[0]
    ind = -1
    for _,tp in task_plan.items():
        ind += 1
        tp_list = re.split(' |__', tp[1:-1])
        if 'put-down' == tp_list[0]:
            end_effector_link = pu.link_from_name(robot, pk.TOOL_FRAMES[pu.get_body_name(robot)])
            body = scn.bd_body[tp_list[1]]
            body_pose = pu.get_pose(body)
            tcp_pose = pu.get_link_pose(robot, end_effector_link)
            grasp_pose = pu.multiply(pu.invert(body_pose), tcp_pose)
            # grasp_pose = ((0,0,0.035),(1,0,0,0))
            approach_pose = (( 0. ,  0. , -0.02), (0.0, 0.0, 0.0, 1.0))
            grasp = pk.BodyGrasp(body, grasp_pose, approach_pose, robot, attach_link=end_effector_link)
            attachment = [grasp.attachment()]
            cmd = pk.Command([pk.BodyPath(robot, motion_plan[ind], attachments=attachment)])
        elif 'pick-up' == tp_list[0]:
            cmd = pk.Command([pk.BodyPath(robot, motion_plan[ind])])
        cmd.execute()
    time.sleep(1)
    scn.reset()

def SetState(scn, task_plan, motion_plan):
    robot = scn.robots[0]
    ind = -1
    for _,tp in task_plan.items():
        ind += 1
        if ind>=len(motion_plan):
            break
        path = [motion_plan[ind][0], motion_plan[ind][-1]]
        tp_list = re.split(' |__', tp[1:-1])
        if 'put-down' == tp_list[0]:
            end_effector_link = pu.link_from_name(robot, pk.TOOL_FRAMES[pu.get_body_name(robot)])
            body = scn.bd_body[tp_list[1]]
            body_pose = pu.get_pose(body)
            tcp_pose = pu.get_link_pose(robot, end_effector_link)
            grasp_pose = pu.multiply(pu.invert(body_pose), tcp_pose)
            approach_pose = (( 0. ,  0. , -0.02), (0.0, 0.0, 0.0, 1.0))
            grasp = pk.BodyGrasp(body, grasp_pose, approach_pose, robot, attach_link=end_effector_link)
            attachment = [grasp.attachment()]
            cmd = pk.Command([pk.BodyPath(robot, path, attachments=attachment)])
        elif 'pick-up' == tp_list[0]:
            cmd = pk.Command([pk.BodyPath(robot, path)])
        cmd.execute()

def motion_planning(scn, t_plan, path_cache=None, feasibility_checker=None):
    # check feasibility of task plan from learned model
    if feasibility_checker:
        isfeasible, failed_step = feasibility_checker.check_feasibility(t_plan)
        if not isfeasible:
            return isfeasible, None, failed_step

    # using plan cache to avoid to resample an known operator
    if path_cache is not None:
        depth, prefix_m_plans = path_cache.find_plan_prefixes(list(t_plan.values()))
        if depth>=0:
            t_plan_to_validate = deepcopy(t_plan)
            print(f"found prefixed operator")
            for _ in range(depth+1):
                min_key = min(t_plan_to_validate.keys())
                print(f"{min_key}: {t_plan_to_validate.pop(min_key)}"   )
            SetState(scn, t_plan, prefix_m_plans)

            res, post_m_plan, failed_step = tm.motion_refiner(t_plan_to_validate)
            m_plan = prefix_m_plans + post_m_plan
        else:
            res, m_plan, failed_step = tm.motion_refiner(t_plan)
        path_cache.add_feasible_motion(list(t_plan.values()), m_plan)
    else:
        res, m_plan, failed_step = tm.motion_refiner(t_plan)

    return res, m_plan, failed_step

def main():
    task_planning_timer = Timer(name='task_planning_timer', text='', logger=logger.info)
    motion_refiner_timer = Timer(name='motion_refiner_timer', text='', logger=logger.info)

    total_planning_timer = 0

    visualization = 1
    pu.connect(use_gui=visualization)
    PlanningScenario = get_scn(2)
    scn = PlanningScenario()
    parser = PDDL_Parser()
    dirname = os.path.dirname(os.path.abspath(__file__))
    domain_filename = os.path.join(dirname, 'domain_idtmp_regrasp.pddl')
    
    parser.parse_domain(domain_filename)
    domain_name = parser.domain_name
    problem_filename = os.path.join(dirname, 'problem_idtmp_'+domain_name+'.pddl')
    problem = PDDLProblem(scn, parser.domain_name)
    parser.dump_problem(problem, problem_filename)

    domain_semantics = RegraspDomainSemantics(scn)
    domain_semantics.activate()

    # IDTMP
    task_planning_timer.start()    
    tp = TaskPlanner(problem_filename, domain_filename, start_horizon=0, max_horizon=6)
    tp.incremental()
    goal_constraints = problem.update_goal_in_formula(tp.encoder)
    tp.formula['goal'] = goal_constraints
    tp.modeling()

    exceeding_horizon = False
    task_planning_timer.stop()
    path_cache = PlanCache()
    tm_plan = None
    t00 = time.time()
    while tm_plan is None:
        # ------------------- task plan ---------------------
        task_planning_timer.start()
        t_plan = None
        while t_plan is None:
            t_plan = tp.search_plan()
            if t_plan is None:
                logger.warning(f"task plan not found in horizon: {tp.horizon}")
                print(f'')
                if not tp.incremental():
                    break
                goal_constraints = problem.update_goal_in_formula(tp.encoder)
                tp.formula['goal'] = goal_constraints
                tp.modeling()
                # global MOTION_ITERATION
                # MOTION_ITERATION += 5
                logger.info(f"search task plan in horizon: {tp.horizon}")
        task_planning_timer.stop()

        if tp.horizon > tp.max_horizon:
            logger.error(f"exceeding task planner maximal horizon")
            break
        
        logger.info(f"task plan found, in horizon: {tp.horizon}")
        for h,p in t_plan.items():
            logger.info(f"{h}: {p}")
            print(f"{h}: {p}")
        # ------------------- motion plan ---------------------
        motion_refiner_timer.start()

        depth, prefix_m_plans = path_cache.find_plan_prefixes(list(t_plan.values()))
        if depth>=0:
            t_plan_to_validate = deepcopy(t_plan)
            for _ in range(depth+1):
                t_plan_to_validate.pop(min(t_plan_to_validate.keys()))
            print('found plan prefixed')
            SetState(scn, t_plan, prefix_m_plans)
            res, post_m_plan, failed_step = tm.motion_refiner(t_plan_to_validate)
            m_plan = prefix_m_plans + post_m_plan
        else:
            res, m_plan, failed_step = tm.motion_refiner(t_plan)
        motion_refiner_timer.stop()

        scn.reset()
        if res:
            logger.info(f"task and motion plan found")
            break
        else: 
            path_cache.add_feasible_motion(list(t_plan.values()), m_plan)
            t0 = time.time()
            logger.warning(f"motion refine failed")
            logger.info(f'')
            
            task_planning_timer.start()
            tp.add_constraint(failed_step, typ='general', cumulative=False)
            task_planning_timer.stop()
            t_plan = None

    total_planning_timer = time.time()-t00
    all_timers = task_planning_timer.timers
    print(f"all timers: {all_timers}")
    print("task_planning_time {:0.4f}".format(all_timers[task_planning_timer.name]))
    print("motion_refiner_time {:0.4f}".format(all_timers[motion_refiner_timer.name]))
    print(f"total_plan_time {total_planning_timer}")
    print(f"final_visits {tp.counter}")

    while True:
        ExecutePlanNaive(scn, t_plan, m_plan)
        time.sleep(1)

    pu.disconnect()

def test():
    task_planning_timer = Timer(name='task_planning_time', text='', logger=logger.info)
    motion_refiner_timer = Timer(name='motion_refiner_time', text='', logger=logger.info)
    total_planning_timer = Timer(name='total_planning_timer', text='', logger=logger.info)

    visualization = 1
    pu.connect(use_gui=visualization)
    PlanningScenario = get_scn(2)
    scn = PlanningScenario()
    save_world = pu.WorldSaver()
    parser = PDDL_Parser()
    dirname = os.path.dirname(os.path.abspath(__file__))
    domain_filename = os.path.join(dirname, 'domain_idtmp_regrasp.pddl')
    
    parser.parse_domain(domain_filename)
    domain_name = parser.domain_name
    problem_filename = os.path.join(dirname, 'problem_idtmp_'+domain_name+'.pddl')
    problem = PDDLProblem(scn, parser.domain_name)
    parser.dump_problem(problem, problem_filename)

    domain_semantics = RegraspDomainSemantics(scn)
    domain_semantics.activate() 

    # IDTMP
    for _ in range(1):

        task_planning_timer.reset()
        motion_refiner_timer.reset()
        total_planning_timer.reset()

        total_planning_timer.start()
        task_planning_timer.start()    
        tp = TaskPlanner(problem_filename, domain_filename, start_horizon=0, max_horizon=6)
        tp.incremental()
        goal_constraints = problem.update_goal_in_formula(tp.encoder)
        tp.formula['goal'] = goal_constraints
        tp.modeling()

        exceeding_horizon = False
        task_planning_timer.stop()

        tm_plan = None
        t00 = time.time()
        while tm_plan is None:
            # ------------------- task plan ---------------------
            task_planning_timer.start()
            t_plan = None
            while t_plan is None:
                t_plan = tp.search_plan()
                if t_plan is None:
                    logger.warning(f"task plan not found in horizon: {tp.horizon}")
                    print(f'')
                    if not tp.incremental():
                        break
                    goal_constraints = problem.update_goal_in_formula(tp.encoder)
                    tp.formula['goal'] = goal_constraints
                    tp.modeling()
                    global MOTION_ITERATION
                    MOTION_ITERATION += 5
                    logger.info(f"search task plan in horizon: {tp.horizon}")
            task_planning_timer.stop()

            if tp.horizon > tp.max_horizon:
                logger.error(f"exceeding task planner maximal horizon")
                break
            
            logger.info(f"task plan found, in horizon: {tp.horizon}")
            for h,p in t_plan.items():
                logger.info(f"{h}: {p}")
                print(f"{h}: {p}")
            # ------------------- motion plan ---------------------
            motion_refiner_timer.start()
            res, m_plan, failed_step = tm.motion_refiner(t_plan)
            motion_refiner_timer.stop()

            scn.reset()
            if res:
                logger.info(f"task and motion plan found")
                break
            else: 
                t0 = time.time()
                logger.warning(f"motion refine failed")
                logger.info(f'')
                
                task_planning_timer.start()
                tp.add_constraint(failed_step, typ='general', cumulative=False)
                task_planning_timer.stop()
                t_plan = None
        total_planning_timer.stop()
        if tp.horizon <= tp.max_horizon:
            all_timers = task_planning_timer.timers
            print(f"all timers: {all_timers}")
            print("task_planning_time {:0.4f}".format(all_timers[task_planning_timer.name]))
            print("motion_refiner_time {:0.4f}".format(all_timers[motion_refiner_timer.name]))
            print("total_planning_timer {:0.4f}".format(all_timers[total_planning_timer.name]))
            print(f"final_visits {tp.counter}")
        else:
            print(f"task and motion plan failed")
        scn.reset()
    os.system('spd-say -t female2 "hi lei! simulation done"')
    while True:
        ExecutePlanNaive(scn, t_plan, m_plan)
        time.sleep(1)
    pu.disconnect()

def multisim_plancache():
    task_planning_timer = Timer(name='task_planning_time', text='', logger=logger.info)
    motion_refiner_timer = Timer(name='motion_refiner_time', text='', logger=logger.info)
    total_planning_timer = Timer(name='total_planning_time', text='', logger=logger.info)

    pu.connect(use_gui=visualization)
    PlanningScenario = get_scn(2)
    scn = PlanningScenario()
    save_world = pu.WorldSaver()
    parser = PDDL_Parser()
    dirname = os.path.dirname(os.path.abspath(__file__))
    domain_filename = os.path.join(dirname, 'domain_idtmp_regrasp.pddl')

    if feasible_check:
        feasible_checker = FeasibilityChecker(scn, objects=scn.movable_bodies, resolution=RESOLUTION, model_file='../training_data/mlp_model.pk')
    else:
        feasible_checker = None

    parser.parse_domain(domain_filename)
    domain_name = parser.domain_name
    problem_filename = os.path.join(dirname, 'problem_idtmp_'+domain_name+'.pddl')
    problem = PDDLProblem(scn, parser.domain_name)
    parser.dump_problem(problem, problem_filename)

    domain_semantics = RegraspDomainSemantics(scn)
    domain_semantics.activate()

    # IDTMP
    for _ in range(max_sim):
        path_cache = PlanCache()
        task_planning_timer.reset()
        motion_refiner_timer.reset()
        total_planning_timer.reset()

        total_planning_timer.start()
        task_planning_timer.start()    
        tp = TaskPlanner(problem_filename, domain_filename, start_horizon=0, max_horizon=8)
        tp.incremental()
        goal_constraints = problem.update_goal_in_formula(tp.encoder)
        tp.formula['goal'] = goal_constraints
        tp.modeling()

        exceeding_horizon = False
        task_planning_timer.stop()

        tm_plan = None
        t00 = time.time()
        while tm_plan is None:
            # ------------------- task plan ---------------------
            task_planning_timer.start()
            t_plan = None
            while t_plan is None:
                t_plan = tp.search_plan()
                if t_plan is None:
                    logger.warning(f"task plan not found in horizon: {tp.horizon}")
                    print(f'')
                    if not tp.incremental():
                        break
                    goal_constraints = problem.update_goal_in_formula(tp.encoder)
                    tp.formula['goal'] = goal_constraints
                    tp.modeling()
                    # global MOTION_ITERATION
                    # MOTION_ITERATION += 5
                    logger.info(f"search task plan in horizon: {tp.horizon}")
            task_planning_timer.stop()

            if tp.horizon > tp.max_horizon:
                logger.error(f"exceeding task planner maximal horizon")
                break
            
            print(f"task plan found, in horizon: {tp.horizon}")
            for h,p in t_plan.items():
                logger.info(f"{h}: {p}")
                print(f"{h}: {p}")
            # ------------------- motion plan ---------------------
            motion_refiner_timer.start()
            res, m_plan, failed_step = motion_planning(scn, t_plan, path_cache=path_cache)
            # depth, prefix_m_plans = path_cache.find_plan_prefixes(list(t_plan.values()))
            # if depth>=0:
            #     t_plan_to_validate = deepcopy(t_plan)
            #     for _ in range(depth+1):
            #         t_plan_to_validate.pop(min(t_plan_to_validate.keys()))
            #     print('found plan prefixed')
            #     SetState(scn, t_plan, prefix_m_plans)
            #     res, post_m_plan, failed_step = tm.motion_refiner(t_plan_to_validate)
            #     m_plan = prefix_m_plans + post_m_plan
            # else:
            #     res, m_plan, failed_step = tm.motion_refiner(t_plan)
            motion_refiner_timer.stop()

            scn.reset()
            if res:
                logger.info(f"task and motion plan found")
                break
            else: 
                path_cache.add_feasible_motion(list(t_plan.values()), m_plan)
                t0 = time.time()
                logger.warning(f"motion refine failed")
                logger.info(f'')

                task_planning_timer.start()
                tp.add_constraint(failed_step, typ='general', cumulative=False)
                task_planning_timer.stop()
                t_plan = None
        total_planning_timer.stop()
        if tp.horizon <= tp.max_horizon:
            all_timers = task_planning_timer.timers
            print(f"all timers: {all_timers}")
            print("task_planning_time {:0.4f}".format(all_timers[task_planning_timer.name]))
            print("motion_refiner_time {:0.4f}".format(all_timers[motion_refiner_timer.name]))
            print("total_planning_time {:0.4f}".format(all_timers[total_planning_timer.name]))
            print(f"final_visits {tp.counter}")
        else:
            print(f"task and motion plan failed")
        path_cache.print_node(path_cache.root)
        embed()
        scn.reset()

    os.system('spd-say -t female2 "hi lei! simulation done"')

    pu.disconnect()

if __name__=="__main__":
    visualization = bool(int(sys.argv[1]))
    RESOLUTION = float(sys.argv[2])
    max_sim = int(sys.argv[3])
    MOTION_ITERATION = int(sys.argv[4])
    feasible_check = bool(int(sys.argv[5]))
    multisim_plancache()

