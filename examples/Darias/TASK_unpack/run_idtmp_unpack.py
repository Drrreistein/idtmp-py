from IPython import embed
import os, sys, time, re
from multiprocessing import Process

from codetiming import Timer
from copy import deepcopy
import tmsmt as tm
from tmsmt import save_plans, load_plans

import numpy as np
from utils.pybullet_tools.utils import WorldSaver
import z3

# `utils` in '~/tamp/idtmp'
import pybullet as p
from pddl_parse.PDDL import PDDL_Parser
import pybullet_tools.utils as pu
import pybullet_tools.kuka_primitives3 as pk
from build_scenario import PlanningScenario
from task_planner import TaskPlanner
from plan_cache import PlanCache
from feasibility_check import FeasibilityChecker, FeasibilityChecker_CNN, FeasibilityChecker_MLP

from logging_utils import *
logging.setLoggerClass(ColoredLogger)
logger = logging.getLogger('MAIN')

EPSILON = 0.005
RESOLUTION = 0.1
MOTION_ITERATION = 20

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
        sys.path.append('/home/lei/robotics/pybullet_manipulator/python')
        from Robot import Robot
        from Kinematics import Kinematics
        rm=Robot(self.robot)
        kin = Kinematics(rm)

    def motion_plan(self, body, goal_pose, attaching=False):

        if attaching:
            body_pose = pu.get_pose(body)
            tcp_pose = pu.get_link_pose(self.robot, self.end_effector_link)
            grasp_pose = pu.multiply(pu.invert(tcp_pose), body_pose)
            grasp = pk.BodyGrasp(body, grasp_pose, self.approach_pose, self.robot, attach_link=self.end_effector_link)
            obstacles = list(set(self.scn.all_bodies) - {body})
            attachment = [grasp.attachment()]
        else:
            obstacles = self.scn.all_bodies
            attachment = []
        # pu.draw_pose(goal_pose)
        # goal_joints = pk.inverse_kinematics(self.robot, self.end_effector_link, goal_pose)
        goal_joints = pu.inverse_kinematics_random(self.robot, self.end_effector_link, goal_pose, obstacles=obstacles,self_collisions=self.self_collision, 
        disabled_collisions = pk.DISABLED_COLLISION_PAIR, attachments=attachment, max_distance=self.max_distance)

        if goal_joints is None:
            return False, goal_joints
        goal_conf = pk.BodyConf(self.robot, goal_joints, self.movable_joints)
        path = pu.plan_joint_motion(self.robot, self.movable_joints, goal_conf.configuration, obstacles=obstacles,self_collisions=self.self_collision, 
        disabled_collisions=pk.DISABLED_COLLISION_PAIR, attachments=attachment,
        max_distance=self.max_distance, iterations=MOTION_ITERATION)
        # if attaching:
        #     embed()
        if path is None:
            # logger.error(f"free motion planning failed")
            return False, path
        cmd = pk.Command([pk.BodyPath(self.robot, [path[0],path[-1]], joints=self.movable_joints, attachments=attachment)])
        cmd.execute()
        return True, path

class UnpackDomainSemantics(DomainSemantics):
    def activate(self):
        tm.bind_refine_operator(self.op_pick_up, "pick-up")
        tm.bind_refine_operator(self.op_put_down, "put-down")

    def op_pick_up(self, args):
        [a, obj, region, i, j] = args
        nop = tm.op_nop(self.scn)
        body = self.scn.bd_body[obj]
        (point, rotation) = pu.get_pose(body)
        center = pu.get_aabb_center(pu.get_aabb(body))
        extend = pu.get_aabb_extent(pu.get_aabb(body))
        x = center[0]
        y = center[1]
        z = center[2] + extend[2]/2 + EPSILON
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
        z = center_region[2] + extend_region[2]/2 + extend_body[2] + EPSILON*2
        
        body_pose = pu.Pose([x,y,z], pu.euler_from_quat(rotation))
        goal_pose = pu.multiply(body_pose, ((0,0,0), pu.quat_from_axis_angle([1,0,0],np.pi)))

        res, path = self.motion_plan(body, goal_pose, attaching=True)
        return res, path

class PDDLProblem(object):
    def __init__(self, scene, domain_name):
        self.scn = scene
        self.domain_name = domain_name
        self.name = 'unpack-3blocks'
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
        ontable = ['ontable','c1','region2__0__0']

        self.goal.append(ontable)

    def _real_goal_state(self):
        self.real_goal = []
        self.real_goal.append(['handempty'])

        ontable = ['or']
        for loc in self.objects['location']:
            if 'region2' in loc:
                ontable.append(['ontable','c1',loc])

        self.real_goal.append(ontable)
        
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
        return scene_objects

    def update_goal_in_formula(self, encoder, formula):
        disconj = []
        for k,v in encoder.boolean_variables[encoder.horizon].items():
            if 'ontable_c1_region2' in k:
                disconj.append(v)

        conj = []
        conj.append(z3.Or(disconj))
        conj.append(encoder.boolean_variables[encoder.horizon]['handempty'])

        # formula['goal'] = z3.And(conj)
        return z3.And(conj)

def ExecutePlanNaive(scn, task_plan, motion_plan, time_step=0.01):
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
            grasp_pose = pu.multiply(pu.invert(tcp_pose), body_pose)
            # grasp_pose = ((0,0,0.035),(1,0,0,0))
            approach_pose = (( 0. ,  0. , -0.02), (0.0, 0.0, 0.0, 1.0))
            grasp = pk.BodyGrasp(body, grasp_pose, approach_pose, robot, attach_link=end_effector_link)
            attachment = [grasp.attachment()]
            cmd = pk.Command([pk.BodyPath(robot, motion_plan[ind], attachments=attachment)])
        elif 'pick-up' == tp_list[0]:
            cmd = pk.Command([pk.BodyPath(robot, motion_plan[ind])])
        cmd.execute(time_step=time_step)

def SetState(scn, task_plan, motion_plan):
    robot = scn.robots[0]
    ind = -1
    len_motion_path = len(motion_plan)
    for _,tp in task_plan.items():
        ind += 1
        if ind>=len_motion_path:
            break
        tp_list = re.split(' |__', tp[1:-1])
        tele_port_path = [motion_plan[ind][0], motion_plan[ind][-1]]
        if 'put-down' == tp_list[0]:
            end_effector_link = pu.link_from_name(robot, pk.TOOL_FRAMES[pu.get_body_name(robot)])
            body = scn.bd_body[tp_list[1]]
            body_pose = pu.get_pose(body)
            tcp_pose = pu.get_link_pose(robot, end_effector_link)
            grasp_pose = pu.multiply(pu.invert(tcp_pose), body_pose)
            # grasp_pose = ((0,0,0.035),(1,0,0,0))
            approach_pose = (( 0. ,  0. , -0.02), (0.0, 0.0, 0.0, 1.0))
            grasp = pk.BodyGrasp(body, grasp_pose, approach_pose, robot, attach_link=end_effector_link)
            attachment = [grasp.attachment()]
            cmd = pk.Command([pk.BodyPath(robot, tele_port_path, attachments=attachment)])
        elif 'pick-up' == tp_list[0]:
            cmd = pk.Command([pk.BodyPath(robot, tele_port_path)])
        cmd.execute()

grasp_directions = {0:(1,0,0),1:(-1,0,0),2:(0,1,0),3:(0,-1,0),4:(0,0,1),(1,0,0):0,(-1,0,0):1,(0,1,0):2,(0,-1,0):3,(0,0,1):4}
def check_feasibility(feasibility_checker, scn, t_plan):
    failed_step = None
    res = True
    init_world = pu.WorldSaver()
    for step, operator in t_plan.items():
        operator = operator[1:-1]
        if 'pick-up' in operator:
            op, obj, region, i, j = re.split(' |__', operator)
            target_body = scn.bd_body[obj]
            target_pose = pu.get_pose(target_body)
        elif 'put-down' in operator:
            op, obj, region, i, j = re.split(' |__', operator)
            target_body = scn.bd_body[obj]
            region_ind = scn.bd_body[region]
            aabb = pu.get_aabb(region_ind)
            center_region = pu.get_aabb_center(aabb)
            extend_region = pu.get_aabb_extent(aabb)

            aabb_body = pu.get_aabb(target_body)
            extend_body = pu.get_aabb_extent(aabb_body)

            x = center_region[0] + int(i)*RESOLUTION
            y = center_region[1] + int(j)*RESOLUTION
            z = center_region[2] + extend_region[2]/2 + extend_body[2]/2 + EPSILON
            target_pose = ([x,y,z], (0,0,0,1))
        else:
            print("unknown operator: feasible by default")
            continue
        if region == 'region_drawer':
            region = 'region_table'
        print(f"checking feasibility: {step}: {operator}")
        is_feasible = feasibility_checker.check_feasibility_simple(target_body, target_pose, grsp_dir=4)

        if not is_feasible:
            print(f"check feasibility: {step}: {operator}: infeasible")
            failed_step = step
            break
        else:
            print(f"check feasibility: {step}: {operator}: FEASIBLE")
            if op=='put-down':
                pu.set_pose(target_body, target_pose)
    init_world.restore()
    if is_feasible:
        print("current task plan is feasible")
    else:
        print("current task plan is infeasible")
    return is_feasible, failed_step

def motion_planning(scn, t_plan, path_cache=None, feasibility_checker=None):
    # check feasibility of task plan from learned model
    if feasibility_checker:
        isfeasible, failed_step = check_feasibility(feasibility_checker, scn, t_plan)
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

    if feasibility_checker:
        feasibility_checker.fc_statistic(res)
    return res, m_plan, failed_step

def load_and_execute(Scenario, dir, file=None, process=1, win_size=[640, 490]):
    def execute_output(filename):
        pu.connect(use_gui=1, options=f'--width={win_size[0]} --height={win_size[1]}')
        scn = Scenario()
        t_plan, m_plan = load_plans(filename)
        while True:
            ExecutePlanNaive(scn, t_plan, m_plan)
            time.sleep(1)
            scn.reset()

    assert os.path.exists(dir), f"no {dir} found"
    processes = []
    filelist = [ file for file in os.listdir(dir) if '.json' in file]
    for i in range(process):
        if file is None or not os.path.exists(os.path.join(dir, file)):
            if not filelist==[]:
                tmp = np.random.choice(filelist)
                filelist.remove(tmp)
        filename = os.path.join(dir, tmp)
        print(filename)
        processes.append(Process(target=execute_output, args=(filename,)))
        processes[-1].start()

def multi_sims_path_cache(visualization=0):
    # visualization = True
    pu.connect(use_gui=visualization)
    scn = PlanningScenario()
    saved_world = WorldSaver()
    parser = PDDL_Parser()
    dirname = os.path.dirname(os.path.abspath(__file__))
    domain_filename = os.path.join(dirname, 'domain_idtmp_unpack.pddl')
    parser.parse_domain(domain_filename)
    domain_name = parser.domain_name
    problem_filename = os.path.join(dirname, 'problem_idtmp_'+domain_name+'.pddl')
    problem = PDDLProblem(scn, parser.domain_name)
    parser.dump_problem(problem, problem_filename)
    domain_semantics = UnpackDomainSemantics(scn)
    domain_semantics.activate()
    if feasible_check==1:
        feasible_checker = FeasibilityChecker(scn, objects=scn.movable_bodies, resolution=RESOLUTION, model_file='../training_data_tabletop/mlp_model.pk')
    elif feasible_check==2:
        feasible_checker = FeasibilityChecker_CNN(scn, objects=scn.movable_bodies, model_file=args_global.model_file, obj_centered_img=True)
        # feasible_checker = FeasibilityChecker(scn, objects=scn.movable_bodies, resolution=RESOLUTION, model_file='../training_data_bookshelf/table_2b/mlp_model.pk')
    elif feasible_check==3:
        feasible_checker = FeasibilityChecker_MLP(scn, objects=scn.movable_bodies,
                    model_file='../training_cnn_simple/mlp_fv_dir4_nodir_32.model',
                    model_file_1box='../training_cnn_simple/mlp_fv_dir4_1box_nodir_40.model')
    else:
        feasible_checker = None

    i=0
    task_planning_timer = Timer(name='task_planning_timer', text='', logger=logger.info)
    motion_refiner_timer = Timer(name='motion_refiner_timer', text='', logger=logger.info)
    total_planning_timer = Timer(name='total_planning_timer', text='', logger=logger.info)

    while i<max_sim:
        path_cache = PlanCache()
        task_planning_timer.reset()
        motion_refiner_timer.reset()
        total_planning_timer.reset()
        i+=1
        
        total_planning_timer.start()
        task_planning_timer.start()
        tp = TaskPlanner(problem_filename, domain_filename, start_horizon=0, max_horizon=6)
        tp.incremental()
        goal_constraints = problem.update_goal_in_formula(tp.encoder, tp.formula)
        tp.formula['goal'] = goal_constraints
        tp.modeling()
        task_planning_timer.stop()

        t0 = time.time()
        while time.time()-t0<1000:
            # ------------------- task plan ---------------------
            t_plan = None
            task_planning_timer.start()
            while t_plan is None:
                t_plan = tp.search_plan()
                if t_plan is None:
                    logger.warning(f"task plan not found in horizon: {tp.horizon}")
                    print(f'')
                    if not tp.incremental():
                        print(f"exceed maximal task plan horizon: {tp.max_horizon}")
                        break
                    goal_constraints = problem.update_goal_in_formula(tp.encoder, tp.formula)
                    tp.formula['goal'] = goal_constraints
                    tp.modeling()
                    logger.info(f"search task plan in horizon: {tp.horizon}")
                    global MOTION_ITERATION
                    MOTION_ITERATION += 10
            task_planning_timer.stop()
            if tp.horizon>tp.max_horizon:
                break
            logger.info(f"task plan found, in horizon: {tp.horizon}")
            for h,t in t_plan.items():
                print(f"{h}: {t}")

            # ------------------- motion plan ---------------------
            motion_refiner_timer.start()
            res, m_plan, failed_step = motion_planning(scn, t_plan, path_cache=path_cache, feasibility_checker=feasible_checker)
            motion_refiner_timer.stop()
            scn.reset()
            if res:
                logger.info(f"task and motion plan found")
                break
            else:
                logger.warning(f"motion refine failed")
                logger.info(f'')
                task_planning_timer.start()
                tp.add_constraint(failed_step, typ='general', cumulative=False)
                task_planning_timer.stop()
                t_plan = None

        total_planning_timer.stop()
        if tp.horizon <= tp.max_horizon:
            if res:
                save_plans(t_plan, m_plan, 'output/'+output_dir+f'/tm_plan_{str(i).zfill(4)}.json')
            else:
                print(f"ERROR: no task motion plan found...")

            all_timers = task_planning_timer.timers
            print(f"all timers: {all_timers}")
            print("task_planning_time {:0.4f}".format(all_timers[task_planning_timer.name]))
            print("motion_refiner_time {:0.4f}".format(all_timers[motion_refiner_timer.name]))
            print("total_planning_time {:0.4f}".format(all_timers[total_planning_timer.name]))
            print(f"final_visits {tp.counter}")
        else:
            print(f"task and motion plan failed")

    # os.system('spd-say -t female2 "hi lei, simulation done"')
    # while True:
    #     ExecutePlanNaive(scn, t_plan, m_plan)
    #     saved_world.restore()
    #     time.sleep(1)
    pu.disconnect()

if __name__=="__main__":
    """ usage
    python3 run_idtmp_unpack.py 0 0.1 10 20 0 
    """
    import argparse
    parser = argparse.ArgumentParser(prog='idtmp-py')
    parser.add_argument('-v','--visualization', action='store_true', help='visualize the simulation process in pybullet')
    parser.add_argument('-r','--resolution', default=0.1, type=float,help='discretize the continuous region in this sampling step')
    parser.add_argument('-n','--num_simulation', type=int, default=1, help='number of the IDTMP simulation to run')
    parser.add_argument('-i','--iteration', type=int, default=20, help='motion planning RRT iteration')
    parser.add_argument('-c','--feasibility', type=int,default=2, help='choose which kind of feasibility checker, \n 1:SVM/MLP using scikit, 2:CNN, 3:MLP using tensorflow')
    parser.add_argument('-f','--model_file', type=str,default='', help='model file of feasibility checker')
    parser.add_argument('-o','--output_file', type=str, default='output/test', help='save generated tm plan to output file')
    parser.add_argument('-l', '--load_scene',type=str,default='', help='load scene from file or random a new scene')

    args_global = parser.parse_args()

    visualization = args_global.visualization
    RESOLUTION = args_global.resolution
    max_sim = args_global.num_simulation
    MOTION_ITERATION = args_global.iteration
    feasible_check = args_global.feasibility
    model_file = args_global.model_file
    output_dir = args_global.output_file

    # visualization = bool(int(sys.argv[1]))
    # RESOLUTION = float(sys.argv[2])
    # max_sim = int(sys.argv[3])
    # MOTION_ITERATION = int(sys.argv[4])
    # feasible_check = int(sys.argv[5])
    # output_dir = str(sys.argv[6])

    multi_sims_path_cache(visualization=visualization)
