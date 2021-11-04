#!/usr/bin/env python
from run_branch_pack import *
from build_scenario import *
import sys
from codetiming import Timer
from etamp.block_skeleton import *
#######################################################

def multisim():
    connect(use_gui=visualization)

    # scn = Scene_unpack3()
    task_planning_timer = Timer(name='task_planning_timer', text='')
    motion_refiner_timer = Timer(name='motion_refiner_timer', text='')
    total_planning_timer = Timer(name='total_planning_timer', text='')

    scn = PlanningScenario4b()
    saved_world = WorldSaver()

    pddlstream_problem = get_pddlstream_problem(scn, discret=stream_discret)
    _, _, _, _, stream_info, action_info = pddlstream_problem

    for _ in range(max_sim):
        task_planning_timer.reset()
        motion_refiner_timer.reset()
        total_planning_timer.reset()

        total_planning_timer.start()
        task_planning_timer.start()
        st = time.time()

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

        id_to_actionEssence = {}
        id_to_branch = {}

        queue_candidate_sk = list(range(sk_batch.num_ap))

        while concrete_plan is None and thinking_time < 1000:
            e_root.visits += 1
            # Progressive Widening
            # alpha = 0.3
            # need_expansion = np.floor(e_root.visits ** alpha) > np.floor(
            #     (e_root.visits - 1) ** alpha)
            flag_pw = e_root.visits > 3 * (len(e_root.children) ** 2)  # 8.5 6
            need_expansion = e_root.num_children < 1 or flag_pw
            need_expansion = need_expansion and (len(queue_candidate_sk) > 0)
            # need_expansion = e_root.num_children < 1
            cur_branch = None
            task_planning_timer.start()
            if need_expansion:
                sk_id = queue_candidate_sk.pop(0)

                if sk_id not in id_to_actionEssence:
                    actionEssence = sk_batch.get_actionEssence(sk_id)
                    # assert actionEssence is not None
                    if actionEssence is None:
                        print("actionEssence is None")
                        task_planning_timer.stop()
                        break
                    id_to_actionEssence[sk_id] = actionEssence
                else:
                    actionEssence = id_to_actionEssence[sk_id]

                blocking_sk = None  # the skeleton that blocks other sks from motion planning
                for sk_id0, essenceJam in sk_to_essenceJam.items():
                    if sk_id0 != sk_id and test_essenceJam(actionEssence, essenceJam):
                        # essenceJam_to_sks[essenceJam].add(sk_id)
                        blocking_sk = sk_id0
                if blocking_sk is None:
                    # if sk_id not in id_to_branch:
                    #     op_plan = sk_batch.generate_operatorPlan(sk_id)
                    #     skeleton_env = SkeletonEnv(sk_id, op_plan,
                    #                                get_update_env_reward_fn(scn, action_info),
                    #                                stream_info, scn, use_bo=False)
                    #     cur_branch = PlannerUCT(skeleton_env)
                    #     id_to_branch[sk_id] = cur_branch
                    # else:
                    #     cur_branch = id_to_branch[sk_id]

                    op_plan = sk_batch.generate_operatorPlan(sk_id)
                    skeleton_env = SkeletonEnv(sk_id, op_plan,
                                            get_update_env_reward_fn(scn, action_info),
                                            stream_info, scn, use_bo=False)
                    cur_branch = PlannerUCT(skeleton_env)
                    id_to_branch[sk_id] = cur_branch
                    e_root.add_child(cur_branch)
                else:
                    e_root.visits -= 1
                    if sk_id not in queue_candidate_sk:
                        print(f"sk_id not in queue_candidate_sk")
                        task_planning_timer.stop()
                        break
                    # assert sk_id not in queue_candidate_sk
                    queue_candidate_sk.append(sk_id)
                    print('\n $$ Skeleton {} is blocked by {}.\n'.format(sk_id,
                                                                        blocking_sk))

            if cur_branch is None:
                # continue
                cur_branch = e_root.select_child_ucb()
            task_planning_timer.stop()

            motion_refiner_timer.start()
            concrete_plan = cur_branch.think(1, 0)
            motion_refiner_timer.stop()

            if concrete_plan is None:
                essenceJam = cur_branch.get_actionEssenceJam()
                """If this essenceJam has been met"""
                # if cur_branch.env.skeleton_id in sk_to_essenceJam:
                #     old_essenceJam = sk_to_essenceJam[cur_branch.env.skeleton_id]
                #     if len(essenceJam) > len(old_essenceJam):
                #         to_recover = []
                #         for blocked_id in essenceJam_to_sks[old_essenceJam]:
                #             if not test_essenceJam(id_to_actionEssence[blocked_id], essenceJam):
                #                 """Unlock the sk"""
                #                 to_recover.append(blocked_id)
                #         for blocked_id in to_recover:
                #             print('\n $$ Skeleton {} is unlocked by {}.\n'.format(blocked_id,
                #                                                                   cur_branch.env.skeleton_id))
                #             essenceJam_to_sks[old_essenceJam].remove(blocked_id)
                sk_to_essenceJam[cur_branch.env.skeleton_id] = essenceJam

            # print('total_node: ', e_root.total_node)
            num_attempts += 1
            thinking_time = time.time() - st

            # if (e_root.visits + 1) % 10 == 0:
            #     with open('ctype_to_constraints.pk', 'wb') as f:
            #         pk.dump(Constraint.ctype_to_constraints, f)
            #         for ctype, cs in Constraint.ctype_to_constraints.items():
            #             print(f"#{ctype}# {cs[0]}: {len([c for c in cs if c.yg <= 0])}-{len([c for c in cs if c.yg > 0])}")
        total_planning_timer.stop()

        print('think time: ' + str(thinking_time))
        # e_root.save_the_tree(idx)
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
        embed()

    disconnect()

if __name__ == '__main__':
    visualization = bool(int(sys.argv[1]))
    stream_discret = bool(int(sys.argv[2]))
    max_sim = int(sys.argv[3])
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
 