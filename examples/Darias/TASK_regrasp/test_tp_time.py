from IPython import embed
import os, sys, time, re
from codetiming import Timer
from numpy.core.defchararray import array
from numpy.core.fromnumeric import cumsum

import tmsmt as tm
import numpy as np
import z3

from pddl_parse.PDDL import PDDL_Parser
import pybullet_tools.utils as pu
import pybullet_tools.kuka_primitives3 as pk
from build_scenario import get_scn
from task_planner import TaskPlanner
import matplotlib.pyplot as plt
from logging_utils import *
logging.setLoggerClass(ColoredLogger)
logger = logging.getLogger('MAIN')
logger.setLevel(logging.ERROR)

RESOLUTION=0.1
MOTION_TIMEOUT = 500
DIR_NUM=1

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
        disabled_collisions=pk.DISABLED_COLLISION_PAIR, attachments=attachment, max_distance=self.max_distance, iterations=MOTION_TIMEOUT)

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


visualization = False
pu.connect(use_gui=visualization)
PlanningScenario = get_scn(2)
scn = PlanningScenario()

def test():

    tp_total_time = Timer(name='tp_total_time', text='')

    parser = PDDL_Parser()
    dirname = os.path.dirname(os.path.abspath(__file__))
    domain_filename = os.path.join(dirname, 'domain_idtmp_regrasp.pddl')
    
    parser.parse_domain(domain_filename)
    domain_name = parser.domain_name
    problem_filename = os.path.join(dirname, 'problem_idtmp_'+domain_name+'.pddl')
    t0 = time.time()
    problem = PDDLProblem(scn, parser.domain_name)
    parser.dump_problem(problem, problem_filename)
    print(f"build pddl problem: {time.time()-t0}")

    # IDTMP
    tp_total_time.start()    
    tp = TaskPlanner(problem_filename, domain_filename, start_horizon=2, max_horizon=2)
    tp.incremental()
    goal_constraints = problem.update_goal_in_formula(tp.encoder)
    tp.formula['goal'] = goal_constraints
    tp.modeling()

    exceeding_horizon = False

    tm_plan = None
    t00 = time.time()
    while tm_plan is None:
        # ------------------- task plan ---------------------
        t_plan = None
        while t_plan is None:
            t_plan = tp.search_plan()
            if t_plan is None:
                logger.warning(f"task plan not found in horizon: {tp.horizon}")
                if not tp.incremental():
                    break
                goal_constraints = problem.update_goal_in_formula(tp.encoder)
                tp.formula['goal'] = goal_constraints
                tp.modeling()
                logger.info(f"search task plan in horizon: {tp.horizon}")

        if tp.horizon > tp.max_horizon:
            # logger.error(f"exceeding task planner maximal horizon")
            break

        logger.info(f"task plan found, in horizon: {tp.horizon}")

        tp.add_constraint(None, typ='negated', cumulative=False)
        t_plan = None
    tp_total_time.stop()
    results = dict(tp_total_time.timers)
    results['ground_actions'] = len(tp.encoder.action_variables[0])
    results['ground_states'] = len(tp.encoder.boolean_variables[0])
    print("task plan time: {:0.4f} s".format(results[tp_total_time.name]))
    print(f"task plan counter: {tp.counter}")
    return results

def plt_data(durations):

    resolutions = []
    dir_nums = []
    dauer = []
    actions = []
    states = []
    for k, v in durations.items():
        resolutions.append(k[0])
        dir_nums.append(k[1])
        dauer.append(v['tp_total_time'])
        actions.append(v['ground_actions'])
        states.append(v['ground_states'])

    import matplotlib.pyplot as plt
    fig = plt.figure()
    ax = fig.add_subplot(1,2,1,projection='3d')

    ax.scatter(resolutions, dir_nums, dauer,marker='o')
    ax.set_xlabel('resolution / m')
    ax.set_ylabel('pick direction number')
    ax.set_zlabel('task planning time / s')

    ax1 = fig.add_subplot(1,2,2)
    ax1.plot(actions, dauer)
    ax1.set_xlabel('ground actions')
    ax1.set_ylabel('task planning time / s')

    plt.savefig('tt_time_resolution.png')
    plt.show()

    
if __name__=='__main__':
    durations = dict()

    for reso in np.linspace(0.1,0.02,9):
        for dir in np.linspace(1, 4, 4):
            print(f"")
            print(f"timing resolution:{reso}, dir_num:{dir}")
            RESOLUTION = reso
            DIR_NUM = int(dir)
                # global RESOLUTION
            t = test()
            # except:
            #     print(f"test failed")
            #     # durations[(reso, dir)] = ()
            #     continue
            durations[((reso, dir))] = t

    embed() 

    tp_total_time = []
    ground_actions = []
    ground_states = []
    for k, t in durations.items():
        tp_total_time.append(t['tp_total_time'])
        ground_actions.append(t['ground_actions'])
        ground_states.append(t['ground_states'])

    import pickle
    with open('ttime_vs_resolution2.pk', 'wb') as f:
        pickle.dump(durations, f)

    plt_data(durations)
    
    embed() 