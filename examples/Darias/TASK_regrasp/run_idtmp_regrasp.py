from logging import PlaceHolder
from IPython import embed
import os, sys, time, re
from codetiming import Timer
from numpy.core.defchararray import array, mod
from numpy.core.fromnumeric import cumsum
from copy import deepcopy

import json
import tmsmt as tm
from tmsmt import save_plans, load_plans, PickPlaceDomainSemantics
import numpy as np
import z3

from pddl_parse.PDDL import PDDL_Parser
import pybullet_tools.utils as pu
import pybullet_tools.kuka_primitives3 as pk
from build_scenario import *
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
from feasibility_check import FeasibilityChecker, FeasibilityChecker_bookshelf, FeasibilityChecker_CNN
EPSILON = 0.01
RESOLUTION = 0.1
DIR_NUM = 1
MOTION_ITERATION = 500


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
                    break

        self.init = init

    def _goal_state(self):
        self.goal = []
        self.goal.append(['handempty'])
        ontable = ['ontable','box1','region_shelf__0__0']

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


def multisim_plancache():
    task_planning_timer = Timer(name='task_planning_time', text='', logger=print)
    motion_refiner_timer = Timer(name='motion_refiner_time', text='', logger=print)
    total_planning_timer = Timer(name='total_planning_time', text='', logger=print)

    pu.connect(use_gui=visualization)
    scn = PlanningScenario_4obs_1box()
    save_world = pu.WorldSaver()

    embed()
    parser = PDDL_Parser()
    dirname = os.path.dirname(os.path.abspath(__file__))
    domain_filename = os.path.join(dirname, 'domain_idtmp_regrasp.pddl')

    if feasible_check==1:
        feasible_checker = FeasibilityChecker(scn, objects=scn.movable_bodies, resolution=RESOLUTION, 
                            model_file=args_global.model_file)
    elif feasible_check==2:
        feasible_checker = FeasibilityChecker_CNN(scn, objects=scn.movable_bodies+scn.obstacle_bodies, 
                        model_file=args_global.model_file, obj_centered_img=True)
        # feasible_checker = FeasibilityChecker(scn, objects=scn.movable_bodies, resolution=RESOLUTION, model_file='../training_data_bookshelf/table_2b/mlp_model.pk')
    elif feasible_check==3:
        feasible_checker = FeasibilityChecker_MLP(scn, objects=scn.movable_bodies,
                    model_file='../training_cnn_simple/mlp_fv_dir4_nodir_32.model',
                    model_file_1box='../training_cnn_simple/mlp_fv_dir4_1box_nodir_40.model')
    else:
        feasible_checker = None

    parser.parse_domain(domain_filename)
    domain_name = parser.domain_name
    problem_filename = os.path.join(dirname, 'problem_idtmp_'+domain_name+'.pddl')
    problem = PDDLProblem(scn, parser.domain_name)
    parser.dump_problem(problem, problem_filename)
    domain_semantics = PickPlaceDomainSemantics(scn, resolution=RESOLUTION, 
                                                epsilon=EPSILON, motion_iteration=MOTION_ITERATION)
    domain_semantics.activate()
    # IDTMP
    for it in range(max_sim):
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
                    print(f"WARN: task plan not found in horizon: {tp.horizon}")
                    print(f'')
                    if not tp.incremental():
                        break
                    goal_constraints = problem.update_goal_in_formula(tp.encoder)
                    tp.formula['goal'] = goal_constraints
                    tp.modeling()
                    # global MOTION_ITERATION
                    # MOTION_ITERATION += 5
                    domain_semantics.motion_iteration = MOTION_ITERATION

                    print(f"search task plan in horizon: {tp.horizon}")
            task_planning_timer.stop()

            if tp.horizon > tp.max_horizon:
                print(f"WARN: exceeding task planner maximal horizon")
                break

            print(f"task plan found, in horizon: {tp.horizon}")
            for h,p in t_plan.items():
                print(f"{h}: {p}")

            # ------------------- motion plan ---------------------
            motion_refiner_timer.start()
            res, m_plan, failed_step = motion_planning(scn, t_plan, path_cache, feasible_checker)
            motion_refiner_timer.stop()

            scn.reset()
            if res:
                print(f"task and motion plan found")
                break
            else: 
                print(f"WARN: motion refine failed")
                print(f'')
                task_planning_timer.start()
                tp.add_constraint(failed_step, typ='general', cumulative=False)
                task_planning_timer.stop()
                t_plan = None

        total_planning_timer.stop()
        if tp.horizon <= tp.max_horizon:
            save_plans(t_plan, m_plan, 'output/'+output_dir+f'/tm_plan_{str(it).zfill(4)}.json')
            all_timers = task_planning_timer.timers
            print(f"all timers: {all_timers}")
            print("task_planning_time {:0.4f}".format(all_timers[task_planning_timer.name]))
            print("motion_refiner_time {:0.4f}".format(all_timers[motion_refiner_timer.name]))
            print("total_planning_time {:0.4f}".format(all_timers[total_planning_timer.name]))
            print(f"final_visits {tp.counter}")
        else:
            print(f"task and motion plan failed")
        path_cache.print_node(path_cache.root)
        scn.reset()

    os.system('spd-say -t female2 "hi lei! simulation done"')
    pu.disconnect()

if __name__=="__main__":
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

    args_global = parser.parse_args()

    visualization = args_global.visualization
    RESOLUTION = args_global.resolution
    max_sim = args_global.num_simulation
    MOTION_ITERATION = args_global.iteration
    feasible_check = args_global.feasibility
    model_file = args_global.model_file
    output_dir = args_global.output_file

    multisim_plancache()