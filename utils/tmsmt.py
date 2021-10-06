
import re

from IPython import embed
import logging
from logging_utils import ColoredLogger
logging.setLoggerClass(ColoredLogger)
logger = logging.getLogger('tmsmt')

motion_timeout = 5
operator_bindings = dict()

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
            return False, id
        else:
            paths.append(path)
    return True, paths
