#!/usr/bin/env python
from run_branch_unpack import *
import os, sys
from collections import defaultdict
from codetiming import Timer

#######################################################

def eTAMP_session():
    visualization = 0
    connect(use_gui=visualization)

    scn = Scene_unpack3()

    pddlstream_problem = get_pddlstream_problem(scn)
    _, _, _, _, stream_info, action_info = pddlstream_problem

    st = time.time()

    sk_batch = solve_progressive2(pddlstream_problem,
                                  num_optms_init=80, target_sk=50)

    e_root = ExtendedNode()

    concrete_plan = None
    num_attempts = 0
    thinking_time = 0

    """Any skeleton starting with some essenceJam will be prevented from motion planning until this essenceJam is broken through."""
    sk_to_essenceJam = {}  # mapping from skeleton_id to its resultant essenceList

    essenceJam_to_sks = defaultdict(set)

    sk_id = -1

    while concrete_plan is None and thinking_time < 300:
        # Progressive Widening
        e_root.visits += 1
        # alpha = 0.3
        # need_expansion = np.floor(e_root.visits ** alpha) > np.floor(
        #     (e_root.visits - 1) ** alpha)
        flag_pw = e_root.visits > 6 * (len(e_root.children) ** 2)  # 8.5 6
        need_expansion = e_root.num_children < 1 or flag_pw
        need_expansion = need_expansion and (e_root.num_children < sk_batch.num_ap)
        # need_expansion = e_root.num_children < 1
        blocking_sk = None  # the skeleton that blocks other sks from motion planning
        op_plan = None
        if need_expansion:
            op_plan = sk_batch.get_next_operatorPlan()

        if op_plan is not None:
            sk_id += 1
            skeleton_env = SkeletonEnv(sk_id, op_plan,
                                       get_update_env_reward_fn(scn, action_info),
                                       stream_info, scn, use_bo=False)
            selected_branch = PlannerUCT(skeleton_env)
            for sk, essenceJam in sk_to_essenceJam.items():
                if sk != selected_branch and selected_branch.test_essenceJam(essenceJam):
                    essenceJam_to_sks[essenceJam].add(selected_branch)
                    blocking_sk = sk
            if blocking_sk is None:
                e_root.add_child(selected_branch)
            else:
                print('\n $$ Skeleton {} is blocked by {}. Continue the loop.\n'.format(selected_branch.env.skeleton_id,
                                                                                        blocking_sk.env.skeleton_id))
                continue
        else:
            selected_branch = e_root.select_child_ucb()

        concrete_plan = selected_branch.think(1, 0)
        if concrete_plan is None:
            essenceJam = selected_branch.get_essenceJam()
            old_essenceJam = tuple([])
            if selected_branch in sk_to_essenceJam:
                old_essenceJam = sk_to_essenceJam[selected_branch]
            if len(essenceJam) > len(old_essenceJam):
                to_remove = []
                for blocked_sk in essenceJam_to_sks[old_essenceJam]:
                    to_remove.append(blocked_sk)
                    if not blocked_sk.test_essenceJam(essenceJam):
                        """Unlock the sk"""
                        blocked_sk.init_from_other(selected_branch, len(old_essenceJam) - 1)
                        e_root.add_child(blocked_sk)
                for blocked_sk in to_remove:
                    essenceJam_to_sks[old_essenceJam].remove(blocked_sk)

            sk_to_essenceJam[selected_branch] = essenceJam

        # print('total_node: ', e_root.total_node)
        num_attempts += 1
        thinking_time = time.time() - st

        # if (e_root.visits + 1) % 10 == 0:
        #     with open('ctype_to_constraints.pk', 'wb') as f:
        #         pk.dump(Constraint.ctype_to_constraints, f)
        #         for ctype, cs in Constraint.ctype_to_constraints.items():
        #             print(f"#{ctype}# {cs[0]}: {len([c for c in cs if c.yg <= 0])}-{len([c for c in cs if c.yg > 0])}")

    print('think time: ' + str(thinking_time))
    # e_root.save_the_tree(idx)

    disconnect()

    if concrete_plan is None:
        print('TAMP is failed.', concrete_plan)
        return -1, -1, thinking_time, -1
    print('TAMP is successful.', concrete_plan)

    return e_root.num_total_child_visits, e_root.total_node, thinking_time, len(e_root.children)


def multisim():
    connect(use_gui=visualization)

    # scn = Scene_unpack3()
    task_planning_timer = Timer(name='task_planning_timer', text='')
    motion_refiner_timer = Timer(name='motion_refiner_timer', text='')
    total_planning_timer = Timer(name='total_planning_timer', text='')

    scn = PlanningScenario()
    saved_world = WorldSaver()

    pddlstream_problem = get_pddlstream_problem(scn, discret=stream_discret)
    _, _, _, _, stream_info, action_info = pddlstream_problem

    st = time.time()
    task_planning_timer.reset()
    motion_refiner_timer.reset()
    total_planning_timer.reset()

    total_planning_timer.start()
    task_planning_timer.start()
    sk_batch = solve_progressive2(pddlstream_problem,
                                num_optms_init=80, target_sk=50)
    task_planning_timer.stop()
    e_root = ExtendedNode()

    concrete_plan = None
    num_attempts = 0
    thinking_time = 0

    """Any skeleton starting with some essenceJam will be prevented from motion planning until this essenceJam is broken through."""
    sk_to_essenceJam = {}  # mapping from skeleton_id to its resultant essenceList

    essenceJam_to_sks = defaultdict(set)

    sk_id = -1

    while concrete_plan is None and thinking_time < 300:
        # Progressive Widening
        e_root.visits += 1
        # alpha = 0.3
        # need_expansion = np.floor(e_root.visits ** alpha) > np.floor(
        #     (e_root.visits - 1) ** alpha)
        flag_pw = e_root.visits > 6 * (len(e_root.children) ** 2)  # 8.5 6
        need_expansion = e_root.num_children < 1 or flag_pw
        need_expansion = need_expansion and (e_root.num_children < sk_batch.num_ap)
        # need_expansion = e_root.num_children < 1
        blocking_sk = None  # the skeleton that blocks other sks from motion planning
        op_plan = None
        task_planning_timer.start()
        if need_expansion:
            op_plan = sk_batch.get_next_operatorPlan()

        if op_plan is not None:
            sk_id += 1
            skeleton_env = SkeletonEnv(sk_id, op_plan,
                                    get_update_env_reward_fn(scn, action_info),
                                    stream_info, scn, use_bo=False)
            selected_branch = PlannerUCT(skeleton_env)
            for sk, essenceJam in sk_to_essenceJam.items():
                if sk != selected_branch and selected_branch.test_essenceJam(essenceJam):
                    essenceJam_to_sks[essenceJam].add(selected_branch)
                    blocking_sk = sk
            if blocking_sk is None:
                e_root.add_child(selected_branch)
            else:
                print('\n $$ Skeleton {} is blocked by {}. Continue the loop.\n'.format(selected_branch.env.skeleton_id,
                                                                                        blocking_sk.env.skeleton_id))
                task_planning_timer.stop()
                continue
        else:
            selected_branch = e_root.select_child_ucb()
        task_planning_timer.stop()
        motion_refiner_timer.start()
        concrete_plan = selected_branch.think(1, 0)
        motion_refiner_timer.stop()

        if concrete_plan is None:
            essenceJam = selected_branch.get_essenceJam()
            old_essenceJam = tuple([])
            if selected_branch in sk_to_essenceJam:
                old_essenceJam = sk_to_essenceJam[selected_branch]
            if len(essenceJam) > len(old_essenceJam):
                to_remove = []
                for blocked_sk in essenceJam_to_sks[old_essenceJam]:
                    to_remove.append(blocked_sk)
                    if not blocked_sk.test_essenceJam(essenceJam):
                        """Unlock the sk"""
                        blocked_sk.init_from_other(selected_branch, len(old_essenceJam) - 1)
                        e_root.add_child(blocked_sk)
                for blocked_sk in to_remove:
                    essenceJam_to_sks[old_essenceJam].remove(blocked_sk)

            sk_to_essenceJam[selected_branch] = essenceJam

        # print('total_node: ', e_root.total_node)
        num_attempts += 1
        thinking_time = time.time() - st

            # if (e_root.visits + 1) % 10 == 0:
            #     with open('ctype_to_constraints.pk', 'wb') as f:
            #         pk.dump(Constraint.ctype_to_constraints, f)
            #         for ctype, cs in Constraint.ctype_to_constraints.items():
            #             print(f"#{ctype}# {cs[0]}: {len([c for c in cs if c.yg <= 0])}-{len([c for c in cs if c.yg > 0])}")

    print('think time: ' + str(thinking_time))
    # e_root.save_the_tree(idx)
    total_planning_timer.stop()

    if concrete_plan is None:
        print('TAMP is failed.', concrete_plan)
    else:
        print('TAMP is successful.', concrete_plan)
        all_timers = task_planning_timer.timers
        print(f'task_planning_time {all_timers[task_planning_timer.name]}')
        print(f'motion_refiner_time {all_timers[motion_refiner_timer.name]}')
        print(f'total_planning_time {all_timers[total_planning_timer.name]}')
        print(f"final_visits {e_root.num_total_child_visits}")
        print(f"total_node_numbers {e_root.total_node}")

    disconnect()

if __name__ == '__main__':
    visualization = bool(int(sys.argv[1]))
    stream_discret = bool(int(sys.argv[2]))
    max_sim = int(sys.argv[3])
    for _ in range(max_sim):
        multisim()
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
