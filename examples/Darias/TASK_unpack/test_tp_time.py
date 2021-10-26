from run_idtmp import UnpackDomainSemantics, PDDLProblem, RESOLUTION
from IPython import embed
import os, sys, time, re

from IPython.core.magic_arguments import construct_parser
from codetiming import Timer

import tmsmt as tm
import numpy as np
import z3

from pddl_parse.PDDL import PDDL_Parser
import pybullet_tools.utils as pu
import pybullet_tools.kuka_primitives3 as pk
from build_scenario import PlanningScenario
from task_planner import TaskPlanner

from logging_utils import *
logging.setLoggerClass(ColoredLogger)
logger = logging.getLogger('MAIN')


def test():
    embed()

    visualization = False
    pu.connect(use_gui=visualization)
    scn = PlanningScenario()

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

    # IDTMP
    tp_total_time = Timer(name='tp_total_time', text='', logger=logger.info)
    mp_total_time = Timer(name='mp_total_time', text='', logger=logger.info)

    tp_total_time.start()
    tp = TaskPlanner(problem_filename, domain_filename, start_horizon=1)
    tp.incremental()
    goal_constraints = problem.update_goal_in_formula(tp.encoder, tp.formula)
    tp.formula['goal'] = goal_constraints
    tp.modeling()

    tp_total_time.stop()

    tm_plan = None
    t00 = time.time()
    while tm_plan is None:
        # ------------------- task plan ---------------------
        t_plan = None
        tp_total_time.start()
        while t_plan is None:
            t_plan = tp.search_plan()
            if t_plan is None:
                logger.warning(f"task plan not found in horizon: {tp.horizon}")
                print(f'')
                tp.incremental()
                goal_constraints = problem.update_goal_in_formula(tp.encoder, tp.formula)
                tp.formula['goal'] = goal_constraints
                tp.modeling()
                logger.info(f"search task plan in horizon: {tp.horizon}")
                global MOTION_ITERATION
                MOTION_ITERATION += 10
        tp_total_time.stop()

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
            logger.warning(f"motion refine failed")
            logger.info(f'')
            tp_total_time.start()
            tp.add_constraint(m_plan, typ='general', cumulative=False)
            tp_total_time.stop()
            t_plan = None

    all_timers = tp_total_time.timers
    print(all_timers)
    total_time = time.time()-t00
    print("task plan time: {:0.4f} s".format(all_timers[tp_total_time.name]))
    print("motion refiner time: {:0.4f} s".format(all_timers[mp_total_time.name]))
    print(f"total planning time: {total_time}")
    print(f"task plan counter: {tp.counter}")


    if __name__=="__main__":
        test()