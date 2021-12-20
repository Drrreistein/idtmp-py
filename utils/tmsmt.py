
import re
import numpy as np
from IPython import embed
import logging
from logging_utils import ColoredLogger
logging.setLoggerClass(ColoredLogger)
logger = logging.getLogger('tmsmt')
from plan_cache import PlanCache
motion_timeout = 5
operator_bindings = dict()
import json, os
import utils.pybullet_tools.utils as pu
import utils.pybullet_tools.kuka_primitives3 as pk
from copy import deepcopy
grasp_directions = {0:(1,0,0),1:(-1,0,0),2:(0,1,0),3:(0,-1,0),4:(0,0,1),(1,0,0):0,(-1,0,0):1,(0,1,0):2,(0,-1,0):3,(0,0,1):4}

class PickPlaceDomainSemantics(object):
    def __init__(self, scene, resolution, epsilon, motion_iteration):

        self.scn = scene
        self.robot = self.scn.robots[0]
        self.movable_joints = pu.get_movable_joints(self.robot)
        self.all_bodies = self.scn.all_bodies
        self.max_distance = pk.MAX_DISTANCE
        self.self_collision = True
        self.approach_pose = (( 0. ,  0. , -0.02), (0.0, 0.0, 0.0, 1.0))
        self.end_effector_link = pu.link_from_name(self.robot, pk.TOOL_FRAMES[pu.get_body_name(self.robot)])
        self.sdg_motioner = pk.sdg_plan_free_motion(self.robot, self.all_bodies)

        self.resolution = resolution
        self.epsilon = epsilon
        self.motion_iteration = motion_iteration

    def activate(self):
        bind_refine_operator(self.op_pick_up, "pick-up")
        bind_refine_operator(self.op_put_down, "put-down")

    def motion_plan(self, body, goal_pose, attaching=False):
        init_joints = pk.get_joint_positions(self.robot,self.movable_joints)

        if attaching:
            body_pose = pu.get_pose(body)
            tcp_pose = pu.get_link_pose(self.robot, self.end_effector_link)
            grasp_pose = pu.multiply(pu.invert(body_pose), tcp_pose)
            
            grasp = pk.BodyGrasp(body, grasp_pose, self.approach_pose, self.robot, attach_link=self.end_effector_link)
            obstacles = list(set(self.scn.all_bodies) - {body} - {self.robot})
            attachment = [grasp.attachment()]
        else:
            obstacles = list(set(self.scn.all_bodies) - {self.robot})
            attachment = []

        goal_joints = pu.inverse_kinematics_random(self.robot, self.end_effector_link, goal_pose, obstacles=obstacles,self_collisions=self.self_collision, 
        disabled_collisions=pk.DISABLED_COLLISION_PAIR, attachments=attachment,max_distance=self.max_distance)

        if goal_joints is None:
            return False, goal_joints
        goal_conf = pk.BodyConf(self.robot, goal_joints, self.movable_joints)
        
        path = pu.plan_joint_motion(self.robot, self.movable_joints, goal_conf.configuration, 
                                    obstacles=obstacles,self_collisions=self.self_collision, 
                                    disabled_collisions=pk.DISABLED_COLLISION_PAIR, 
                                    attachments=attachment, max_distance=self.max_distance, 
                                    iterations=self.motion_iteration)

        if path is None:
            return False, path
        cmd = pk.Command([pk.BodyPath(self.robot, [path[0],path[-1]], joints=self.movable_joints, attachments=attachment)])
        cmd.execute()
        return True, path

    def op_pick_up(self, args):
        [a, obj, region, i, j, m,n,o,p] = args
        body = self.scn.bd_body[obj]
        (point, rotation) = pu.get_pose(body)
        center = pu.get_aabb_center(pu.get_aabb(body))
        extend = pu.get_aabb_extent(pu.get_aabb(body))
        x = center[0]
        y = center[1]
        z = center[2]

        offset = np.array([m,n,o], dtype=int) * (extend/2 + np.array([self.epsilon,self.epsilon,self.epsilon]))
        goal_point = np.array([x,y,z]) + offset

        # angle_by_axis = np.array([m,n,o], dtype=int) * np.pi/2
        # goal_rot = pu.multiply_quats(rotation, pu.quat_from_euler((angle_by_axis[0], angle_by_axis[1], 0)))
        # goal_rot = pu.multiply_quats(goal_rot, pu.quat_from_euler((np.pi, 0, 0)))

        m,n,o = int(m),int(n),int(o)
        if o==0:
            goal_rot = pu.multiply_quats(rotation, pu.quat_from_axis_angle((-n, m, 0), np.pi/2))
        else:
            goal_rot = rotation
        # goal_rot = pu.multiply_quats(rotation, pu.quat_from_axis_angle((-n, m, 0), np.pi/2))
        goal_rot = pu.multiply_quats(goal_rot, pu.quat_from_euler((np.pi, 0, 0)))
        goal_pose = pu.Pose(goal_point, pu.euler_from_quat(goal_rot))

        embed()
        res, path = self.motion_plan(body, goal_pose, attaching=False)

        return res, path

    def op_put_down(self, args):
        (a, obj, region, i, j) = args
        body = self.scn.bd_body[obj]
        region_ind = self.scn.bd_body[region]

        aabb = pu.get_aabb(region_ind)
        center_region = pu.get_aabb_center(aabb)
        extend_region = pu.get_aabb_extent(aabb)

        (_, rotation) = pu.get_pose(body)
        aabb_body = pu.get_aabb(body)
        extend_body = pu.get_aabb_extent(aabb_body)
        x = center_region[0] + int(i)*self.resolution
        y = center_region[1] + int(j)*self.resolution
        z = center_region[2] + extend_region[2]/2 + extend_body[2]/2 + self.epsilon
        body_pose = pu.get_pose(body)
        tcp_pose = pk.get_tcp_pose(self.robot)
        body_to_tcp = pu.multiply(pu.invert(body_pose), tcp_pose)

        place_body_pose = pu.Pose([x,y,z], pu.euler_from_quat(rotation))
        goal_pose = pu.multiply(place_body_pose, body_to_tcp)

        res, path = self.motion_plan(body, goal_pose, attaching=True)
        return res, path

def save_plans(t_plan, m_plan, filename):
    dirname = os.path.dirname(filename)
    if not os.path.exists(dirname):
        print('no output diretory create new one')
        os.makedirs(dirname)

    tm_plan = dict()
    tm_plan['t_plan'] = t_plan
    tm_plan['m_plan'] = m_plan

    with open(filename, 'w') as f:
        json.dump(tm_plan, f)

def ExecutePlanNaive(scn, task_plan, motion_plan,time_step=0.01):
    robot = scn.robots[0]
    len_motion_path = len(motion_plan)
    ind = -1
    for _,tp in task_plan.items():
        ind += 1
        if ind>=len_motion_path:
            break
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
        cmd.execute(time_step=time_step)

# def load_and_execute(Scenario, dir, file=None, process=1, win_size=[640, 510],time_step=0.1):
#     def execute_output(filename):
#         pu.connect(use_gui=1, options=f'--width={win_size[0]} --height={win_size[1]}')
#         scn = Scenario()
#         t_plan, m_plan = load_plans(filename)
#         while True:
#             ExecutePlanNaive(scn, t_plan, m_plan,time_step=time_step)
#             import time
#             time.sleep(1)
#             scn.reset()
#     import numpy as np
#     from multiprocessing import Process
#     assert os.path.exists(dir), f"no {dir} found"
#     processes = []
#     filelist = [ file for file in os.listdir(dir) if '.json' in file]
#     for i in range(process):
#         if file is None or not os.path.exists(os.path.join(dir, file)):
#             if not filelist==[]:
#                 tmp = np.random.choice(filelist)
#                 filelist.remove(tmp)
#         filename = os.path.join(dir, tmp)
#         print(filename)
#         processes.append(Process(target=execute_output, args=(filename,)))
#         processes[-1].start()

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


def load_plans(filename):
    assert os.path.exists(filename), f'no file named {filename} exists'
    with open(filename, 'r') as f:
        tm_plan = json.load(f)
    return tm_plan['t_plan'], tm_plan['m_plan']

def load_and_execute(Scenario, dir, file=None, process=1, win_size=[640, 490]):
    """
    load multiple scenario and execute the task_motion_plan
    """
    def execute_output(filename):
        pu.connect(use_gui=1, options=f'--width={win_size[0]} --height={win_size[1]}')
        scn = Scenario()
        t_plan, m_plan = load_plans(filename)
        while True:
            ExecutePlanNaive(scn, t_plan, m_plan)
            import time
            time.sleep(1)
            scn.reset()
    import numpy as np
    from multiprocessing import Process
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

def check_feasibility(feasibility_checker, scn, t_plan, resolution):
    failed_step = None
    init_world = pu.WorldSaver()
    for step, operator in t_plan.items():
        operator = operator[1:-1]
        if 'pick-up' in operator:
            op, obj, region, i, j, m, n, o, p = re.split(' |__', operator)
            target_body = scn.bd_body[obj]
            grsp_dir = (grasp_directions[tuple(np.array([m,n,o],dtype=int))])
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

            x = center_region[0] + int(i)*resolution
            y = center_region[1] + int(j)*resolution
            z = center_region[2] + extend_region[2]/2 + extend_body[2]/2 
            target_pose = ([x,y,z], (0,0,0,1))
        else:
            print("unknown operator: feasible by default")
            continue
        if region=='region_drawer':
            region = 'region_table'
        is_feasible = feasibility_checker.check_feasibility_simple(target_body, target_pose, grsp_dir)

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
    
def motion_planning(scn, t_plan, path_cache=None, feasibility_checker=None, resolution=None):
    # check feasibility of task plan from learned model
    if feasibility_checker:
        isfeasible, failed_step = check_feasibility(feasibility_checker, scn, t_plan, resolution)
        if not isfeasible:
            # if len(t_plan.keys())==4:
            #     embed()
            return isfeasible, None, failed_step

    # using plan cache to avoid to resample an known operator
    if path_cache is not None:
        depth, prefix_m_plans = path_cache.find_plan_prefixes(list(t_plan.values()))
        if depth>=0:
            t_plan_to_validate = deepcopy(t_plan)
            print(f"found prefixed operator")
            for _ in range(depth+1):
                min_key = min(t_plan_to_validate.keys())
                print(f"{min_key}: {t_plan_to_validate.pop(min_key)}")
            SetState(scn, t_plan, prefix_m_plans)

            res, post_m_plan, failed_step = motion_refiner(t_plan_to_validate)
            m_plan = prefix_m_plans + post_m_plan
        else:
            res, m_plan, failed_step = motion_refiner(t_plan)
        path_cache.add_feasible_motion(list(t_plan.values()), m_plan)
    else:
        res, m_plan, failed_step = motion_refiner(t_plan)

    return res, m_plan, failed_step

def mangle(*args, symbol='__'):
    s = ''
    for a in args:
        if s=='':
            s += str(a)
        else:
            s += symbol + str(a)
    return s

def demangle(s, symbol='__'):
    return s.split(sep=symbol)

def plan(operator):
    """
    Create a plan consisting of task-motion OPERATORS.
    """
    pass

def op_nop(scene):
    """
    Create a NO-OP task-motion operator.
    """
    pass

def op_motion(frame, goal):
    """
    Create a motion-plan task-motion operator.
    """
    pass

def op_cartesian(frame, goal):
    """
    Create a motion-plan task-motion operator.
    """
    pass

def op_reparent(parent, frame):
    """
    Create a reparent task-motion operator.
    """
    pass

def op_tf_abs(operator, frame):
    """
    return absolute pose of FRAME after operator
    """
    pass

def op_tf_rel(operator, parent, child):
    """
    return relative pose of child to parent after operator
    """
    pass

def collect_frame_type(scene, type):
    """
    return all frames in SCENE of the given type
    """
    pass

def PlanningFailure(value=None):
    """
    Create an exception indicating motion planning failure.
    """
    assert False, "planning failure, @value: {value}"


def bind_scene_state(func):
    pass

def bind_goal_state(func):
    pass

def bind_scene_object(func):
    pass

def bind_collision_constraint(func):
    pass

def bind_refine_operator(func, operator:str):
    operator_bindings[operator.lower()] = func
    
def motion_refiner(task_plan:dict):
    """
    validate task planner in motion refiner
    """
        
    paths = []
    for id, op in task_plan.items():
        op = op[1:-1]
        args = re.split(' |__', op)
        motion_func = operator_bindings[args[0].lower()]
        res, path = motion_func(args)
        if not res:
            # logger.error("motion refining failed")
            logger.error(f"failed operator:{id}, {op}")
            print(f"failed operator:{id}, {op}")
            return False, paths, id
        else:
            paths.append(tuple(path))
    return True, paths, 0
