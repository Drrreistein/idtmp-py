
import re

from IPython import embed
import logging
from logging_utils import ColoredLogger
logging.setLoggerClass(ColoredLogger)
logger = logging.getLogger('tmsmt')
from plan_cache import PlanCache
motion_timeout = 5
operator_bindings = dict()
import json, os

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

def load_plans(filename):
    assert os.path.exists(filename), f'no file named {filename} exists'
    with open(filename, 'r') as f:
        tm_plan = json.load(f)
    return tm_plan['t_plan'], tm_plan['m_plan']

def load_and_execute(Scenario, dir, file=None, process=1, win_size=[640, 490]):
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
