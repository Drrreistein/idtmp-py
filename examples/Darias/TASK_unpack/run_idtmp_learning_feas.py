from pybullet_tools.utils import WorldSaver
from run_idtmp_unpack import *
import pickle
from feasibility_check import FeasibilityChecker

def multi_sims_path_cache(visualization=0):
    # visualization = True
    pu.connect(use_gui=visualization)
    scn = PlanningScenario()
    saved_world = WorldSaver()
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
    if feasible_check:
        feasible_checker = FeasibilityChecker(scn, objects=scn.movable_bodies, resolution=RESOLUTION, model_file='../training_data/mlp_model.pk')
    else:
        feasible_checker = None

    # IDTMP
    i=0
    task_planning_timer = Timer(name='task_planning_timer', text='', logger=logger.info)
    motion_refiner_timer = Timer(name='motion_refiner_timer', text='', logger=logger.info)
    total_planning_timer = Timer(name='total_planning_timer', text='', logger=logger.info)

    while i<max_sim:
        path_cache = PlanCache()
        task_planning_timer.reset()
        motion_refiner_timer.reset()
        total_planning_timer.reset()
        i+=1
        
        total_planning_timer.start()
        task_planning_timer.start()
        tp = TaskPlanner(problem_filename, domain_filename, start_horizon=0, max_horizon=10)
        tp.incremental()
        goal_constraints = problem.update_goal_in_formula(tp.encoder, tp.formula)
        tp.formula['goal'] = goal_constraints
        tp.modeling()

        task_planning_timer.stop()

        tm_plan = None

        while tm_plan is None:
            # ------------------- task plan ---------------------
            t_plan = None
            task_planning_timer.start()
            while t_plan is None:
                t_plan = tp.search_plan()
                if t_plan is None:
                    logger.warning(f"task plan not found in horizon: {tp.horizon}")
                    print(f'')
                    if not tp.incremental():
                        print(f"exceed maximal task plan horizon: {tp.max_horizon}")
                        break
                    goal_constraints = problem.update_goal_in_formula(tp.encoder, tp.formula)
                    tp.formula['goal'] = goal_constraints
                    tp.modeling()
                    logger.info(f"search task plan in horizon: {tp.horizon}")
                    global MOTION_ITERATION
                    MOTION_ITERATION += 10
            task_planning_timer.stop()

            if tp.horizon>tp.max_horizon:
                break
            logger.info(f"task plan found, in horizon: {tp.horizon}")
            for h,p in t_plan.items():
                print(f"{h}: {p}")

            # ------------------- motion plan ---------------------
            motion_refiner_timer.start()
            res, m_plan, failed_step = motion_planning(scn, t_plan, path_cache=path_cache, feasibility_checker=feasible_checker)
            motion_refiner_timer.stop()
            scn.reset()
            if res:
                logger.info(f"task and motion plan found")
                break
            else:
                logger.warning(f"motion refine failed")
                logger.info(f'')
                task_planning_timer.start()
                tp.add_constraint(failed_step, typ='general', cumulative=False)
                task_planning_timer.stop()
                t_plan = None
        if feasible_check:
            feasible_checker.hypothesis_test()

        if res:
            all_timers = task_planning_timer.timers
            print(all_timers)
            total_planning_timer.stop()
            print("task_planning_time {:0.4f}".format(all_timers[task_planning_timer.name]))
            print("motion_refiner_time {:0.4f}".format(all_timers[motion_refiner_timer.name]))
            print(f"total_planning_time {all_timers[total_planning_timer.name]}")
            print(f"final_visits {tp.counter}")
        else:
            print("TAMP is failed")
        saved_world.restore()

        embed()
    # os.system('spd-say -t female2 "hi lei, simulation done"')
    # while True:
    #     ExecutePlanNaive(scn, t_plan, m_plan)
    #     saved_world.restore()
    #     time.sleep(1)
    pu.disconnect()

if __name__=="__main__":
    """ usage
    python3 run_idtmp_unpack.py 0 0.1 10 20 1
    """
    visualization = bool(int(sys.argv[1]))
    RESOLUTION = float(sys.argv[2])
    max_sim = int(sys.argv[3])
    MOTION_ITERATION = int(sys.argv[4])
    feasible_check = bool(int(sys.argv[5]))
    multi_sims_path_cache(visualization=visualization)

    # multi_sims(visualization)

