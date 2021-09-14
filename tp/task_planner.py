import translate
from planner import encoder, search, modifier, plan
import z3

class TaskPlanner(object):
    def __init__(self, problem, domain, max_horizon=10):
        
        self.task = translate.pddl.open_file(problem, domain)
        self.max_horizon = max_horizon
        self.encoder = encoder.EncoderSMT(self.task, modifier.LinearModifier())

        self.search = search.SearchSMT(self.encoder, ub=max_horizon)
        self.horizon = 0
        self.solver = z3.Solver()
        self.formula =  self.encoder.encode(self.horizon)
        self.counter = 0
        self.general_constraints = set()

    def search_plan(self):

        assert self.horizon<self.max_horizon, "exceeding plan horizon"
        self.counter += 1
        res = self.solver.check()
        if res != z3.sat:
            return None
        self.model = self.solver.model()
        self.solution = plan.Plan(self.model, self.encoder)
        return self.solution.plan

    def incremental(self):
        self.horizon += 1
        self.solver = z3.Solver()
        self.solver.add(z3.And(self.general_constraints))
        self.formula =  self.encoder.encode(self.horizon)
        for k,v in self.formula.items():
            self.solver.add(v)

    def add_constraint(self, failed_step):
        constraints = self.solution.general_failure_constraints(self.model, self.encoder, self.solution.plan, failed_step)
        # # constraints = self.solution.negate_plan_constraints()
        # self.solver.add(z3.And(constraints))
        # self.solver.push()
        self.general_constraints = self.general_constraints.union(set(constraints))
        self.solver.add(z3.And(self.general_constraints))
        self.solver.push()