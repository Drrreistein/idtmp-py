from IPython import embed
import os, sys, time, re

import tmsmt as tm
import numpy as np

from pddl_parse.PDDL import PDDL_Parser
import pybullet as p
import pybullet_tools.utils as pu
import pybullet_tools.pr2_primitives as pp
import pybullet_tools.pr2_utils as ppu
from build_scenario import PlanningScenario
from task_planner import TaskPlanner

from logging_utils import *
logging.setLoggerClass(ColoredLogger)
logger = logging.getLogger('MAIN')

EPSILON = 0.01
RESOLUTION = 0.1
MOTION_TIMEOUT = 200

class DomainSemantics(object):
    def __init__(self, scene):
        self.scn = scene
        self.robot = self.scn.robots[0]
        self.movable_joints = pu.get_movable_joints(self.robot)
        self.all_bodies = self.scn.all_bodies
        self.max_distance = pp.MAX_DISTANCE
        self.self_collision = True
        self.approach_pose = (( 0. ,  0. , -0.02), (0.0, 0.0, 0.0, 1.0))

    def motion_plan(self, body, goal_pose, attaching=False):
        # if attaching and self.scn.bd_body[body]=='c2':
        #     embed()
        embed()
        # pu.draw_pose(goal_pose)
        goal_joints = pp.inverse_kinematics(self.robot, self.end_effector_link, goal_pose)

        start_conf = pp.BodyConf(self.robot, pp.get_joint_positions(self.robot,self.movable_joints), self.movable_joints)
        goal_conf = pp.BodyConf(self.robot, goal_joints, self.movable_joints)

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

        path = pu.plan_joint_motion(self.robot, self.movable_joints, goal_conf.configuration, obstacles=obstacles,self_collisions=self.self_collision, 
        disabled_collisions=pp.DISABLED_COLLISION_PAIR, attachments=attachment,
        max_distance=self.max_distance)

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
        z = center[2] + extend[2]/2 + EPSILON
        body_pose = pu.Pose([x,y,z], pu.euler_from_quat(rotation))
        goal_pose = pu.multiply(body_pose, ((0,0,0), pu.quat_from_axis_angle([1,0,0],np.pi)))
        embed()
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
    tp_time = 0
    mp_time = 0
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

    embed()
    # initial domain semantics
    domain_semantics = UnpackDomainSemantics(scn)
    domain_semantics.activate()

    # IDTMP
    tp = TaskPlanner(problem_filename, domain_filename, start_horizon=12)
    tp.incremental()
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
                tp.incremental()
                
                logger.info(f"search task plan in horizon: {tp.horizon}")
                # MOTION_TIMEOUT += 0.1
        tp_time += time.time() - t0

        logger.info(f"task plan found, in horizon: {tp.horizon}")
        for h,p in t_plan.items():
            logger.info(f"{h}: {p}")
        # ------------------- motion plan ---------------------
        t0 = time.time()
        res, m_plan = tm.motion_refiner(t_plan)
        mp_time += time.time() - t0
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
            tp_time += time.time()-t0

    total_time = time.time()-t00
    print(f"task plan time: {tp_time}")
    print(f"motion refiner time: {mp_time}")
    print(f"total planning time: {total_time}")
    print(f"task plan counter: {tp.counter}")

    embed()
    while True:
        ExecutePlanNaive(scn, t_plan, m_plan)
        time.sleep(1)

    pu.disconnect()

if __name__=="__main__":
    main()

