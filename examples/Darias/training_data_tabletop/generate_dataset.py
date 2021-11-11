from random import random

from build_scenario import ScenarioForTraining
from IPython import embed
import csv, uuid, os, sys   
from multiprocessing import Process, process

import pybullet as p
from utils.pybullet_tools.utils import WorldSaver, connect, get_pose, set_joint_positions, set_pose, get_configuration, is_placement, \
    disconnect, get_bodies, create_box, remove_body, get_aabb   
import utils.pybullet_tools.utils as pu
import utils.pybullet_tools.kuka_primitives3 as pk
import numpy as np

EPSILON = 0.01

def save_dataset(filename, dataset):
    with open(filename, "a") as file:
        wr = csv.writer(file)
        wr.writerows(dataset)

def check_feasibility(scn, goal_pose, attachment):
    obstacles = list(set(scn.all_bodies) - {attachment.child})
    # pu.draw_pose(goal_pose)
    for _ in range(5):
        goal_joints = pu.inverse_kinematics_random(scn.robot, scn.end_effector_link, goal_pose, obstacles=obstacles,self_collisions=True, 
        disabled_collisions=pk.DISABLED_COLLISION_PAIR, attachments=[attachment], max_distance=pk.MAX_DISTANCE)
        if goal_joints is None:
            return False
        goal_conf = pk.BodyConf(scn.robot, goal_joints, scn.movable_joints)
        path = pu.plan_joint_motion(scn.robot, scn.movable_joints, goal_conf.configuration, obstacles=obstacles,self_collisions=True, 
        disabled_collisions=pk.DISABLED_COLLISION_PAIR, attachments=[attachment],
        max_distance=pk.MAX_DISTANCE, iterations=200)
        if path is None:
            continue
    pu.set_joint_positions(scn.robot, scn.movable_joints, goal_joints)
    return True

def get_dist_theta(pose1, pose2):
    dist = np.linalg.norm(np.array(pose1[0][:2])-np.array(pose2[0][:2]))
    theta = np.arctan2(pose1[0][0]-pose2[0][0], pose1[0][1]-pose2[0][1])
    return [dist, theta]

def random_box(scn):
    lwh = list(np.random.uniform([0.04,0.04,0.1],[0.2,0.2,0.5]))
    body = create_box(lwh[0], lwh[1], lwh[2])
    return lwh, body

def attach_to_robot(scn, body, lwh):
    # TODO body pose may not right
    body_pose = pu.multiply(scn.tcp_pose, ((0,0,lwh[2]/2+EPSILON),pu.quat_from_axis_angle((1,0,0),np.pi)))
    # body_pose = pu.multiply(body_pose, ((0,0,0),))
    scn.body_gripped = body
    pu.set_pose(body, body_pose)
    grasp_pose = pu.multiply(pu.invert(body_pose), scn.tcp_pose)
    attachment = pk.Attachment(scn.robot, scn.end_effector_link, grasp_pose, body)
    return attachment

def attach_to_table(scn, body, lwh):
    scn.body_on_table = body
    xy = list(np.random.uniform(lower[:2] ,upper[:2]))
    pose = ((xy[0], xy[1], lwh[2]/2+upper[2]+EPSILON),(0,0,0,1))
    pu.set_pose(body, pose)
    return xy, pose

def sample_training_data():
    connect(use_gui=visualization)
    scn = ScenarioForTraining()
    global lower, upper
    lower, upper = pu.get_aabb(scn.regions[0])
    (robot_position, _) = pu.get_link_pose(scn.robot, 0)
    robot_pose = (robot_position, (0,0,0,1))
    # pu.draw_pose(robot_pose)
    filename = './' + str(uuid.uuid1()) + '.csv'
    for _ in range(1000):
        pu.remove_body(scn.body_gripped)
        lwh1, scn.body_gripped = random_box(scn)
        attachment = attach_to_robot(scn, scn.body_gripped, lwh1)
        pu.remove_body(scn.body_on_table)
        lwh2, scn.body_on_table = random_box(scn)
        xy2, pose2 = attach_to_table(scn, scn.body_on_table, lwh2)
        dist_theta2 = get_dist_theta(robot_pose, pose2)
        dataset = []
        saved_world = WorldSaver()
        for _ in range(10):
            xy1 = list(np.random.uniform(lower[:2] ,upper[:2]))
            target_pose = ((xy1[0],xy1[1],upper[2]+lwh1[2]+2*EPSILON),pu.quat_from_axis_angle((1,0,0),np.pi))
            dist_theta1 = get_dist_theta(robot_pose, target_pose)
            dist_theta12 = get_dist_theta(target_pose, pose2)
            feature_vector = lwh1 + lwh2 + dist_theta1 + dist_theta2 + dist_theta12
            isfeasible = int(check_feasibility(scn, target_pose, attachment))
            dataset.append(feature_vector+[isfeasible])
            saved_world.restore()
        save_dataset(filename, dataset)

if __name__=='__main__':
    """ usage
    python3 generate_dataset 0 2
    """
    visualization = int(sys.argv[1])
    num_process = int(sys.argv[2])
    # print(f"number of processes: {num_process}")
    assert num_process<9, "CPU overworked"

    processes = []
    for _ in range(num_process):
        processes.append(Process(target=sample_training_data, args=()))
        processes[-1].start()
        # processes[-1].join()