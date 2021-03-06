#!/usr/bin/env python
from run_branch_regrasp import *
import os
from codetiming import Timer
#######################################################

def eTAMP_session():
    connect(use_gui=visualization)
    task_planning_timer = Timer(name='task_planning_time', text='')
    motion_refiner_timer = Timer(name='motion_refiner_time', text='')
    total_planning_timer = Timer(name='total_planning_timer', text='')

    PlanningScenario = get_scn(2)
    scn = PlanningScenario()

    pddlstream_problem = get_pddlstream_problem(scn, discrete=discret_stream)
    _, _, _, _, stream_info, action_info = pddlstream_problem

    st = time.time()
    total_planning_timer.start()
    task_planning_timer.start()
    sk_batch = solve_progressive2(pddlstream_problem,
                                  num_optms_init=80, target_sk=20)
    task_planning_timer.stop()
    e_root = ExtendedNode()

    concrete_plan = None
    num_attempts = 0
    thinking_time = 0
    while concrete_plan is None and thinking_time < 60 * 10:
        # Progressive Widening
        e_root.visits += 1
        # alpha = 0.3
        # need_expansion = np.floor(e_root.visits ** alpha) > np.floor(
        #     (e_root.visits - 1) ** alpha)
        flag_pw = e_root.visits > 9.5 * (len(e_root.children) ** 2)  # 8.5
        need_expansion = e_root.num_children < 1 or flag_pw
        need_expansion = need_expansion and (e_root.num_children < sk_batch.num_ap)
        # need_expansion = e_root.num_children < 1
        if need_expansion:
            task_planning_timer.start()
            op_plan = sk_batch.get_next_operatorPlan()
            task_planning_timer.stop()
            if op_plan is None:
                break
            motion_refiner_timer.start()
            skeleton_env = SkeletonEnv(e_root.num_children, op_plan,
                                       get_update_env_reward_fn(scn, action_info),
                                       stream_info, scn, use_bo=False)
            selected_branch = PlannerUCT(skeleton_env)
            e_root.add_child(selected_branch)
            motion_refiner_timer.stop()
        else:
            motion_refiner_timer.start()
            selected_branch = e_root.select_child_ucb()
            motion_refiner_timer.stop()
        motion_refiner_timer.start()
        concrete_plan = selected_branch.think(1, False)
        motion_refiner_timer.stop()
        # print('total_node: ', e_root.total_node)
        num_attempts += 1
        thinking_time = time.time() - st

        if (e_root.visits + 1) % 10 == 0:
            with open('ctype_to_constraints.pk', 'wb') as f:
                pk.dump(Constraint.ctype_to_constraints, f)
                for ctype, cs in Constraint.ctype_to_constraints.items():
                    print(f"#{ctype}# {cs[0]}: {len([c for c in cs if c.yg <= 0])}-{len([c for c in cs if c.yg > 0])}")
    total_planning_timer.stop()

    print('think time: ' + str(thinking_time))
    # e_root.save_the_tree(idx)
    
    disconnect()

    if concrete_plan is None:
        print('TAMP is failed.', concrete_plan)
        return -1, -1, thinking_time, -1
    print('TAMP is successful.', concrete_plan)
    all_timers = task_planning_timer.timers
    print(f"all timers: {all_timers}")
    print("task_planning_time {:0.4f}".format(all_timers[task_planning_timer.name]))
    print("motion_refiner_time {:0.4f}".format(all_timers[motion_refiner_timer.name]))
    print("total_planning_timer {:0.4f}".format(all_timers[total_planning_timer.name]))
    print(f"final_visits {e_root.num_total_child_visits}")
    return e_root.num_total_child_visits, e_root.total_node, thinking_time, len(e_root.children)


if __name__ == '__main__':
    import sys
    visualization = bool(int(sys.argv[1]))
    discret_stream = bool(int(sys.argv[2]))
    max_sim = int(sys.argv[3])
    MOTION_ITERATION = int(sys.argv[4])
    for _ in range(max_sim):
        result_vnts = eTAMP_session()
        print(result_vnts)
    # list_report_vnts = []
    # for i in range(100):
    #     print(f'exp {i} -------------------------------------------------------------------')
    #     result_vnts = eTAMP_session()
    #     list_report_vnts.append(result_vnts)

    #     print('======================')
    #     for vnt in list_report_vnts:
    #         print(vnt)
    #     print('======================')

    #     data1 = [c[0] for c in list_report_vnts]
    #     data2 = [c[1] for c in list_report_vnts]
    #     data3 = [c[2] for c in list_report_vnts]
    #     data4 = [c[3] for c in list_report_vnts]
    #     np.savetxt("result_vnts.csv", np.column_stack((data1, data2, data3, data4)), delimiter=",", fmt='%s')
