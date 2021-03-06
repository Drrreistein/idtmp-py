from collections import defaultdict
import time

from IPython.core.magic_arguments import construct_parser
from codetiming import Timer

import translate
from planner import encoder, search, modifier, plan
from IPython import embed

import z3
from logging_utils import *
logging.setLoggerClass(ColoredLogger)
logger = logging.getLogger('t_planner')
logger.setLevel(logging.ERROR)

class TaskPlanner(object):
    def __init__(self, problem, domain, start_horizon=0, max_horizon=20):
        t0 = time.time()
        self.task = translate.pddl.open_file(problem, domain)
        self.max_horizon = max_horizon
        self.encoder = encoder.EncoderSMT(self.task, modifier.LinearModifier())
        self.search = search.SearchSMT(self.encoder, ub=max_horizon)
        self.horizon = start_horizon
        self.solver = z3.Solver()
        self.formula =  self.encoder.encode(self.horizon)
        self.counter = 0
        self.general_constraints = set()

        self.dura_initializing = time.time() - t0
        self.dura_formulation = 0
        self.dura_searching = 0
        self.dura_add_constraints = 0
        logger.info(f"task plan initialized")
    
    @Timer(name='tp_searching', text='')
    def search_plan(self):
        self.counter += 1
        res = self.solver.check()
        if res != z3.sat:
            return None
        self.model = self.solver.model()
        self.solution = plan.Plan(self.model, self.encoder)
        return self.solution.plan

    @Timer(name='tp_formulation', text='')
    def incremental(self, step=1):
        self.horizon += step
        if self.horizon > self.max_horizon:
            return False
        # t0 = time.time()
        self.formula = self.encoder.incrmental(self.horizon)
        # print(f"incremental formulation time: {time.time()-t0}")

        return True

    @Timer(name='tp_modeling', text='')
    def modeling(self):
        self.solver = z3.Solver()
        for k,v in self.formula.items():
            self.solver.add(v)
        self.solver.add(z3.And(self.general_constraints))

    @Timer(name='tp_add_constraints', text='')
    def add_constraint(self, failed_step, typ='negated', cumulative=True):
        """
        typ = ['negated','general','collision']
        """
        if typ=='negated':
            constraints = self.solution.negate_plan_constraints()
        elif typ=='general':
            constraints = self.solution.general_failure_constraints(self.model, self.encoder, self.solution.plan, failed_step)
        elif typ=='collision':
            constraints = self.solution.collision_generalization_constraints(self.model, self.encoder, self.solution.plan, failed_step)
        
        self.general_constraints = self.general_constraints.union(set(constraints))
        if not cumulative:
            self.solver.add(constraints)
        else:
            self.solver.add(z3.And(self.general_constraints))
        self.solver.push()

    def display_boolean_variables(self):
        true_boolean_variables = defaultdict(set)
        for step, variables in self.encoder.boolean_variables.items():
            for s, val in variables.items():
                if self.model.eval(val):
                    true_boolean_variables[step].add(val)
        return true_boolean_variables

    def display_action_variables(self):
        true_action_variables = defaultdict(set)
        for step, variables in self.encoder.action_variables.items():
            for s, val in variables.items():
                if self.model.eval(val):
                    true_action_variables[step].add(val)
        return true_action_variables

        # self.solver.add(z3.And(constraints))
        # self.solver.push()

        # constraints = self.solution.negate_plan_constraints()
        # self.solver.add(z3.And(constraints))
        # self.solver.push()