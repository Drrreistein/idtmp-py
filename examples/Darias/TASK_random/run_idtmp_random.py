
from random import choice

from numpy.lib.type_check import real_if_close
from IPython import embed
import os, sys, time, re
from multiprocessing import Process

from codetiming import Timer
from copy import deepcopy
import tmsmt as tm
from tmsmt import save_plans, load_plans, PickPlaceDomainSemantics

import numpy as np
from utils.pybullet_tools.utils import WorldSaver
import z3

# `utils` in '~/tamp/idtmp'
import pybullet as p
from pddl_parse.PDDL import PDDL_Parser
import pybullet_tools.utils as pu
import pybullet_tools.kuka_primitives3 as pk
from build_scenario import *
from task_planner import TaskPlanner
from plan_cache import PlanCache
from feasibility_check import FeasibilityChecker, FeasibilityChecker_CNN, FeasibilityChecker_MLP
from logging_utils import *
logging.setLoggerClass(ColoredLogger)
logger = logging.getLogger('MAIN')

EPSILON = 0.005
RESOLUTION = 0.1
MOTION_ITERATION = 20
DIR_NUM=1

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
        # self._plot_locations(self.objects['location'])
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
        for region, bodies in self.scn.goal.items():
            for body in bodies:
                self.goal.append(['ontable',body,f'{region}__0__0'])

    def _real_goal_state(self):
        self.real_goal = []
        self.real_goal.append(['handempty'])

        for region, bodies in self.scn.goal.items():
            for body in bodies:
                ontable = ['or']
                # self.goal.append(['ontable',body,f'{region}__0__0'])
                for loc in self.objects['location']:
                    if region in loc:
                        ontable.append(['ontable',body,loc])
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

        # define pick-up direction 
        direction = set()
        for i in range(DIR_NUM):
            direction.add(tm.mangle(0,0,1,i))
            if self.scn.all_pick_dir:
                direction.add(tm.mangle(1,0,0,i))
                direction.add(tm.mangle(-1,0,0,i))
                direction.add(tm.mangle(0,1,0,i))
                direction.add(tm.mangle(0,-1,0,i))
        scene_objects['direction'] = direction

        return scene_objects

    def update_goal_in_formula(self, encoder, formula):
        conj = []
        for region, bodies in self.scn.goal.items():
            for body in bodies:
                disconj = []
                for k,v in encoder.boolean_variables[encoder.horizon].items():
                    if f'ontable_{body}_{region}' in k:
                        disconj.append(v)

                conj.append(z3.Or(disconj))
        conj.append(encoder.boolean_variables[encoder.horizon]['handempty'])

        # formula['goal'] = z3.And(conj)
        return z3.And(conj)

def run_scene_in_folder(folder):
    for file in os.listdir(folder):
        if '0_' in file:
            pu.connect(use_gui=1)

            scene_file = os.path.join('random_scenes',file)
            scn = Scene_random(json_file=scene_file)
            saved_world = WorldSaver()
            t_plan = scn.scene['tm_plan']['t_plan']
            m_plan = scn.scene['tm_plan']['m_plan']
            print(f"execute old tm_plan or find a new one")
            print(f"executing the command \n ExecutePlanNaive(scn, t_plan, m_plan, time_step=0.01)\nsaved_world.restore()")

            print(f"scene file: {file}")
            tm.ExecutePlanNaive(scn, t_plan, m_plan, time_step=0.01)
            saved_world.restore()
            embed()
            pu.disconnect()

def multi_sims_path_cache():
    pu.connect(use_gui=args_global.visualization)
    if not args_global.load_scene:
        scn = Scene_random()
    else:
        scene_file = os.path.join('random_scenes',args_global.load_scene)
        scn = Scene_random(json_file=scene_file)
        saved_world = WorldSaver()
        t_plan = scn.scene['tm_plan']['t_plan']
        m_plan = scn.scene['tm_plan']['m_plan']
        print(f"execute old tm_plan or find a new one")
        print(f"executing the command \n ExecutePlanNaive(scn, t_plan, m_plan, time_step=0.01)\nsaved_world.restore()")

        print(f"scene file: {scene_file}")
        # tm.ExecutePlanNaive(scn, t_plan, m_plan, time_step=0.01)
        # time.sleep(1)
        # saved_world.restore()
        # time.sleep(1)
        # embed()

        # if  t_plan=='None':
        #     print(f"no task and motion plan available")
        #     embed()
        # else:
        #     while True:
        #         tm.ExecutePlanNaive(scn, t_plan, m_plan, time_step=0.01)
        #         time.sleep(1)
        #         saved_world.restore()

    saved_world = WorldSaver()
    parser = PDDL_Parser()
    dirname = os.path.dirname(os.path.abspath(__file__))
    domain_filename = os.path.join(dirname, 'domain_idtmp_regrasp.pddl')
    parser.parse_domain(domain_filename)
    domain_name = parser.domain_name
    problem_filename = os.path.join(dirname, 'problem_idtmp_'+domain_name+'.pddl')
    problem = PDDLProblem(scn, parser.domain_name)
    parser.dump_problem(problem, problem_filename)
    global MOTION_ITERATION
    domain_semantics = PickPlaceDomainSemantics(scn, resolution=RESOLUTION, 
                                                epsilon=EPSILON, motion_iteration=MOTION_ITERATION)
    domain_semantics.activate()

    scn_objects = list(scn.movable_bodies) + list(scn.obstacle_bodies)
    if feasible_check==1:
        feasible_checker = FeasibilityChecker(scn, objects=scn_objects, resolution=RESOLUTION, model_file = args_global.model_file)
    elif feasible_check==2:
        feasible_checker = FeasibilityChecker_CNN(scn, objects=scn_objects, \
            model_file = args_global.model_file, obj_centered_img=True, threshold=args_global.threshold)
    elif feasible_check==3:
        feasible_checker = FeasibilityChecker_MLP(scn, objects=scn_objects,
                    model_file=args_global.model_file, threshold=args_global.threshold)
    else:
        feasible_checker = None

    i=0
    task_planning_timer = Timer(name='task_planning_timer', text='', logger=logger.info)
    motion_refiner_timer = Timer(name='motion_refiner_timer', text='', logger=logger.info)
    total_planning_timer = Timer(name='total_planning_timer', text='', logger=logger.info)

    ########################################### tm planning ##############################
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
                MOTION_ITERATION += 10
                domain_semantics.motion_iteration = MOTION_ITERATION
        task_planning_timer.stop()
        if tp.horizon>tp.max_horizon:
            break
        logger.info(f"task plan found, in horizon: {tp.horizon}")
        for h,t in t_plan.items():
            print(f"{h}: {t}")

        # ------------------- motion plan ---------------------
        motion_refiner_timer.start()
        res, m_plan, failed_step = tm.motion_planning(scn, t_plan, path_cache=path_cache, feasibility_checker=feasible_checker, resolution=RESOLUTION)
        motion_refiner_timer.stop()
        # scn.reset()
        saved_world.restore()
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

    feasible_checker.hypothesis_test()
    if tp.horizon <= tp.max_horizon:
        if res:
            pass
            # embed()
            # if len(list(t_plan.keys()))>=4:
            #     scn.save_scene_in_json()
            #     scn.update_scene_tm_plan(t_plan, m_plan)
            # save_plans(t_plan, m_plan, os.path.join(output_dir,f'/tm_plan_{str(i).zfill(4)}.json'))
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
    #     tm.ExecutePlanNaive(scn, t_plan, m_plan)
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
    parser.add_argument('-c','--feasibility', type=int,default=4, help='choose which kind of feasibility checker, \n 1:SVM/MLP using scikit, 2:CNN, 3:MLP using tensorflow')
    parser.add_argument('-f','--model_file', type=str,default='', help='model file of feasibility checker')
    parser.add_argument('-o','--output_file', type=str, default='output/test', help='save generated tm plan to output file')
    parser.add_argument('-l', '--load_scene',type=str,default='', help='load scene from file or random a new scene')
    parser.add_argument('-t','--threshold', default=0.5, type=float, help='probability threshold of feasible action for learned model')
    args_global = parser.parse_args()

    visualization = args_global.visualization
    RESOLUTION = args_global.resolution
    max_sim = args_global.num_simulation
    MOTION_ITERATION = args_global.iteration
    feasible_check = args_global.feasibility
    model_file = args_global.model_file
    output_dir = args_global.output_file

    file_list = os.listdir('random_scenes')
    for i in range(max_sim):
        args_global.load_scene = np.random.choice(file_list)
        try:
            multi_sims_path_cache()
        except:
            pass

    embed()
    for i in range(max_sim):
        multi_sims_path_cache()
