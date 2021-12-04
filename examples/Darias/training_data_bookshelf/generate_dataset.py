from random import random

from PIL.Image import init
from progress.bar import IncrementalBar

from numpy.core.getlimits import _register_known_types
from numpy.lib.npyio import save

from build_scenario import *
from IPython import embed
import csv, uuid, os, sys   
from multiprocessing import Process, process

import pybullet as p
from utils.pybullet_tools.utils import WorldSaver, connect, get_pose, set_joint_positions, set_pose, get_configuration, is_placement, \
    disconnect, get_bodies, create_box, remove_body, get_aabb   
import utils.pybullet_tools.utils as pu
import utils.pybullet_tools.kuka_primitives3 as pk
import numpy as np

EPSILON = 0.001

def get_jacobian():
    jaco_t, jaco_r = pu.compute_jacobian(0, scn.end_effector_link)
    jaco = np.r_[np.array(jaco_t).T, np.array(jaco_r).T]
    return jaco

def inverse_kinematics(scn):

    init_pose = ((0.6278852224349976, 0.14589789509773254, 0.9492117762565613),(0.7088728547096252,  -0.23877574503421783,0.5746845006942749,0.33199870586395264))
    init_joints = np.array([ 0.1, -1.4,  1. ,  1.7,  0. ,  0. ,  0. ])
    target_pose = ((0.6352361440658569, 0.15205420553684235, 0.9416102170944214),(0.710773766040802,-0.23820507526397705,0.5755693316459656,  0.32677048444747925))
    target_joints = np.array([ 0.11, -1.39,  1.01,  1.71,  0.01,  0.01,  0.01])

    pu.set_joint_positions(0, scn.movable_joints, init_joints)
    inv_init_pose = p.invertTransform(init_pose[0], init_pose[1])
    dx = p.multiplyTransforms(inv_init_pose[0], inv_init_pose[1], target_pose[0], target_pose[1])
    dt = dx[0]
    dr = np.array(p.getAxisAngleFromQuaternion(dx[1])[0])*p.getAxisAngleFromQuaternion(dx[1])[1]
    vb = np.r_[dt, dr]
    err = np.linalg.norm(vb)
    i =0
    tmp_joints = init_joints.copy()
    while i<50 and err>1e-3:
        jaco_t, jaco_r = pu.compute_jacobian(0, 8)

        tmp_joints += np.linalg.pinv(np.c_[jaco_t, jaco_r].T) @ vb  
        pu.set_joint_positions(0, scn.movable_joints, tmp_joints)
        i += 1
        tmp_pose = pu.forward_kinematics(0,scn.movable_joints, tmp_joints, scn.end_effector_link)
        inv_init_pose = p.invertTransform(tmp_pose[0], tmp_pose[1])
        dx = p.multiplyTransforms(inv_init_pose[0], inv_init_pose[1], target_pose[0], target_pose[1])
        dt = dx[0]
        dr = np.array(p.getAxisAngleFromQuaternion(dx[1])[0])*p.getAxisAngleFromQuaternion(dx[1])[1]
        vb = np.r_[dt, dr]
        err = np.linalg.norm(vb)
        print(err)
        time.sleep(0.1)


def save_dataset(filename, dataset):
    with open(filename, "a") as file:
        wr = csv.writer(file)
        wr.writerows(dataset)

def check_feasibility(scn, goal_pose):
    obstacles = list(set(scn.all_bodies))

    pu.draw_pose(goal_pose)
    start_joints = pu.get_joint_positions(scn.robot, scn.movable_joints)
    init_body_pose = pu.get_pose(scn.body_gripped)
    for _ in range(10):
        goal_joints = pu.inverse_kinematics_random(scn.robot, scn.end_effector_link, goal_pose, obstacles=obstacles,self_collisions=True, 
        disabled_collisions=pk.DISABLED_COLLISION_PAIR, attachments=[], max_distance=pk.MAX_DISTANCE)
        if goal_joints is None:
            # print(f"found no ik")
            continue
            # pu.set_joint_positions(scn.robot, scn.movable_joints, start_joints)
            # return False
        pu.set_joint_positions(scn.robot, scn.movable_joints, start_joints)
        pu.set_pose(scn.body_gripped,init_body_pose)
        return True
        pu.set_joint_positions(scn.robot, scn.movable_joints, goal_joints)

        attachment = attach_to_robot(scn, scn.body_gripped)
        obstacles = list(set(scn.all_bodies) - {attachment.child})
        goal_conf = pk.BodyConf(scn.robot, start_joints, scn.movable_joints)
        attachment.assign()
        path = pu.plan_joint_motion(scn.robot, scn.movable_joints, goal_conf.configuration, obstacles=obstacles,self_collisions=True, 
        disabled_collisions = pk.DISABLED_COLLISION_PAIR, attachments=[attachment],
        max_distance=pk.MAX_DISTANCE, iterations=1000)
        if path is None:
            # print(f"found no path")
            continue
        else:
            pu.set_joint_positions(scn.robot, scn.movable_joints, start_joints)
            attachment.assign()
            return True
    pu.set_joint_positions(scn.robot, scn.movable_joints, start_joints)
    pu.set_pose(scn.body_gripped,init_body_pose)
    return False

def get_dist_theta(pose1, pose2):
    dist = np.linalg.norm(np.array(pose1[0][:2])-np.array(pose2[0][:2]))
    theta = np.arctan2(pose1[0][0]-pose2[0][0], pose1[0][1]-pose2[0][1])
    return [dist, theta]

def random_box(scn):
    if region_ind==0:
        lwh = list(np.random.uniform([0.04,0.04,0.1],[0.1,0.1,0.3]))

    elif region_ind==1:
        lwh = list(np.random.uniform([0.04,0.04,0.1],[0.1,0.1,0.2]))
    body = create_box(lwh[0], lwh[1], lwh[2])
    return lwh, body

def attach_to_robot(scn, body):

    body_pose = pu.get_pose(body)
    tcp_pose = pu.get_link_pose(scn.robot, scn.end_effector_link)
    grasp_pose = pu.invert(pu.multiply(pu.invert(body_pose), tcp_pose))
    attachment = pk.Attachment(scn.robot, scn.end_effector_link, grasp_pose, body)
    return attachment

# def attach_to_robot(scn, body, lwh):
#     # TODO body pose may not right
#     body_pose = pu.multiply(scn.tcp_pose, ((0,0,lwh[2]/2+EPSILON),pu.quat_from_axis_angle((1,0,0),np.pi)))
#     # body_pose = pu.multiply(body_pose, ((0,0,0),))
#     scn.body_gripped = body
#     pu.set_pose(body, body_pose)
#     grasp_pose = pu.multiply(pu.invert(body_pose), scn.tcp_pose)
#     attachment = pk.Attachment(scn.robot, scn.end_effector_link, grasp_pose, body)
#     return attachment

def attach_to_table(scn, body, lwh, lower, upper):
    # scn.body_on_table = body
    xy = list(np.random.uniform(lower[:2] ,upper[:2]))
    pose = ((xy[0], xy[1], lwh[2]/2+upper[2]+EPSILON),(0,0,0.7071067811865475, 0.7071067811865476))
    pu.set_pose(body, pose)
    return xy, pose

def target_pose_vs_grsp_dir(center_pose, grsp_dir, extend):
    center_point, rotation = center_pose

    offset = np.array(grsp_dir, dtype=int) * (extend/2+np.array([EPSILON,EPSILON,EPSILON]))
    goal_point = tuple(np.array(center_point) + offset)

    angle_by_axis = np.array(grsp_dir, dtype=int) * np.pi/2

    goal_rot = pu.multiply_quats(rotation, pu.quat_from_euler((angle_by_axis[0], angle_by_axis[1], 0)))
    goal_rot = pu.multiply_quats(goal_rot, pu.quat_from_euler((np.pi, 0, 0)))
    target_pose = pu.Pose(goal_point, pu.euler_from_quat(goal_rot))
    # pu.draw_pose(target_pose)

    return target_pose

def sample_training_data():
    connect(use_gui=visualization)
    scn = TrainingScenario()

    aabb_table = pu.get_aabb(scn.table)
    aabb_shelf = pu.get_aabb(scn.shelf)
    global grasp_directions
    grasp_directions = {0:(1,0,0),1:(-1,0,0),2:(0,1,0),3:(0,-1,0),4:(0,0,1)}

    aabbs = [aabb_table, aabb_shelf]
    aabb = aabbs[region_ind]
    lower, upper = aabb
    (robot_position, _) = pu.get_link_pose(scn.robot, 0)
    robot_pose = (robot_position, (0,0,0,1))

    # pu.draw_pose(robot_pose)
    embed()
    if region_ind==0:
        region_str = 'table'
    else:
        region_str = 'shelf'
    filename_2b = f'./{region_str}_2b/' + str(uuid.uuid1()) + '.csv'
    filename_1b = f'./{region_str}_1b/' + str(uuid.uuid1()) + '.csv'

    bar = IncrementalBar('Countdown', max = 500)
    for _ in range(500):
        bar.next()
        pu.remove_body(scn.body_on_table)
        lwh2, scn.body_on_table = random_box(scn)

        pu.remove_body(scn.body_gripped)
        lwh1, scn.body_gripped = random_box(scn)
        xy1, pose1 = attach_to_table(scn, scn.body_gripped, lwh1, lower, upper)

        dataset_2b = []
        dataset_1b = []
        saved_world = WorldSaver()
        for dir_ind, grsp_dir in grasp_directions.items():
            if dir_ind == 1 or dir_ind == 4:
                continue
            saved_world.restore()

            pu.set_pose(scn.body_on_table, ((2,2,2),(0,0,0,1)))
            extend = pu.get_aabb_extent(pu.get_aabb(scn.body_gripped))
            target_pose = target_pose_vs_grsp_dir(pose1, grsp_dir, np.array(extend))
            dist_theta1 = get_dist_theta(robot_pose, pose1)
            feature_vector_1b = lwh1 + dist_theta1 + [dir_ind]
            isfeasible_1b = int(check_feasibility(scn, target_pose))
            dataset_1b.append(feature_vector_1b+[isfeasible_1b])
            # pu.draw_pose(target_pose)
            # print(f'1b feasible:{isfeasible_1b}')

            if isfeasible_1b:
                num_2b = 10
            else:
                num_2b = 1
            for _ in range(num_2b):
                saved_world.restore()
                xy2, pose2 = attach_to_table(scn, scn.body_on_table, lwh2, lower, upper)
                dist_theta2 = get_dist_theta(robot_pose, pose2)
                dist_theta12 = get_dist_theta(pose1, pose2)
                feature_vector_2b = lwh1 + lwh2 + dist_theta1 + dist_theta2 + dist_theta12 + [dir_ind]
                isfeasible_2b = isfeasible_1b and int(check_feasibility(scn, target_pose))
                dataset_2b.append(feature_vector_2b+[isfeasible_2b])
                # print(f'2b feasible:{isfeasible_2b}')
        save_dataset(filename_2b, dataset_2b)
        save_dataset(filename_1b, dataset_1b)
    bar.finish()

if __name__=='__main__':
    """ usage
    python3 generate_dataset 0 2 1
    """
    visualization = int(sys.argv[1])
    num_process = int(sys.argv[2])
    region_ind = int(sys.argv[3])

    # print(f"number of processes: {num_process}")
    assert num_process<=9, "CPU overworked"
    sample_training_data()
    processes = []
    for _ in range(num_process):
        processes.append(Process(target=sample_training_data, args=()))
        processes[-1].start()
        # processes[-1].join()
