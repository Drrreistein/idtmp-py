from task_planner import TaskPlanner
from codetiming import Timer
from IPython import embed
task_planning_timer = Timer(name='task_planning_timer')

def task_planning(problem_filename, domain_filename):
    tp = TaskPlanner(problem_filename, domain_filename, start_horizon=0, max_horizon=100)
    tp.incremental()
    tp.modeling()

    t_plan = None
    while t_plan is None:
        task_planning_timer.start()
        t_plan = tp.search_plan()
        task_planning_timer.stop()
        if t_plan is None:
            print(f"task plan not found in horizon: {tp.horizon}")
            print(f'')
            # if tp.horizon<19:
            num_constraints = 0
            for k, v, in tp.formula.items():
                if k != 'goal':
                    num_constraints += len(v)
            print(f'ground_actions {len(tp.encoder.action_variables)*len(tp.encoder.action_variables[0])}')
            print(f'ground_states {len(tp.encoder.boolean_variables)*len(tp.encoder.boolean_variables[0])}')
            print(f'number_constraints {num_constraints}')
            print(f"task planning time: {task_planning_timer.timers['task_planning_timer']}")
            # else:
            #     embed()
            task_planning_timer.start()
            if not tp.incremental():
                task_planning_timer.stop()
                print(f"{tp.max_horizon} exceeding, TAMP is failed")
                embed()
            tp.modeling()
            task_planning_timer.stop()
            print(f"search task plan in horizon: {tp.horizon}")

if __name__=='__main__':
    import sys

    domain = sys.argv[1]
    problem = sys.argv[2]
    task_planning(problem_filename=problem, domain_filename=domain)