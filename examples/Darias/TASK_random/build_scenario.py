#!/usr/bin/env python

from __future__ import print_function
import numpy as np

import time
import utils.pybullet_tools.utils as pu
import utils.pybullet_tools.kuka_primitives3 as pk
from utils.pybullet_tools.kuka_primitives3 import BodyPose, BodyConf, Register
from utils.pybullet_tools.utils import WorldSaver, connect, dump_world, get_pose, set_pose, Pose, \
    Point, set_default_camera, stable_z, disconnect, get_bodies, HideOutput, \
    create_box, set_color,\
    load_pybullet, step_simulation, Euler, get_links, get_link_info, get_movable_joints, set_joint_positions, \
    set_camera, get_center_extent, tform_from_pose, attach_viewcone, LockRenderer

from utils.pybullet_tools.body_utils import draw_frame
import json, uuid, os
from copy import copy
from IPython import embed
import pandas as pd
EPSILON = 0.005

class PlanningScenario(object):
    def __init__(self):
        with HideOutput():
            with LockRenderer():
                self.arm_left = load_pybullet("../darias_description/urdf/darias_L_primitive_collision.urdf",
                                              fixed_base=True)
                self.arm_base = load_pybullet("../darias_description/urdf/darias_base.urdf", fixed_base=True)

                self.bd_body = {
                    'floor': load_pybullet("../scenario_description/floor.urdf", fixed_base=True),
                    'cabinet_shelf': load_pybullet(
                        "../scenario_description/manipulation_worlds/urdf/cabinet_shelf.urdf",
                        fixed_base=True),
                    'drawer_shelf': load_pybullet(
                        "../scenario_description/manipulation_worlds/urdf/drawer_shelf.urdf",
                        fixed_base=True),
                    'pegboard': load_pybullet(
                        "../scenario_description/manipulation_worlds/urdf/pegboard.urdf",
                        fixed_base=True),
                    'region1': load_pybullet("../scenario_description/region.urdf", fixed_base=True),
                    'region2': load_pybullet("../scenario_description/region_big.urdf",fixed_base=True),
                    'c1': load_pybullet("../scenario_description/boxCm.urdf", fixed_base=False),
                    'c2': load_pybullet("../scenario_description/boxC.urdf", fixed_base=False),
                    'c3': load_pybullet("../scenario_description/boxCx.urdf", fixed_base=False),
                }
                color=[1,0,0]
                set_color(self.bd_body['region1'], color=color)
                set_color(self.bd_body['region2'], color=color)

                self.bd_body.update(dict((self.bd_body[k], k) for k in self.bd_body))

                self.drawer_links = get_links(self.bd_body['drawer_shelf'])
                cabinet_links = get_links(self.bd_body['cabinet_shelf'])

                set_pose(self.bd_body['cabinet_shelf'],
                         Pose(Point(x=-0.45, y=-0.8, z=stable_z(self.bd_body['cabinet_shelf'], self.bd_body['floor']))))
                set_pose(self.bd_body['drawer_shelf'],
                         Pose(Point(x=-0.45, y=0.8, z=stable_z(self.bd_body['drawer_shelf'], self.bd_body['floor']))))
                set_pose(self.bd_body['pegboard'],
                         Pose(Point(x=-0.60, y=0, z=stable_z(self.bd_body['pegboard'], self.bd_body['floor']))))
                set_pose(self.bd_body['region1'],
                         Pose(Point(x=0.35, y=0.9, z=stable_z(self.bd_body['region1'], self.bd_body['floor']))))
                set_pose(self.bd_body['region2'],
                         Pose(Point(x=0.05, y=0.8, z=stable_z(self.bd_body['region2'], self.bd_body['floor']))))

                self.movable_bodies = [self.bd_body['c1'], self.bd_body['c2'], self.bd_body['c3']]
                self.env_bodies = [self.arm_base, self.bd_body['floor'], self.bd_body['cabinet_shelf'],
                                   self.bd_body['drawer_shelf'], self.bd_body['pegboard']]
                self.regions = [self.bd_body['region1'], self.bd_body['region2']]

                self.all_bodies = list(set(self.movable_bodies) | set(self.env_bodies) | set(self.regions))

                self.sensors = []

                self.robots = [self.arm_left]

                self.dic_body_info = {}
                for b in self.movable_bodies:
                    obj_center, obj_extent = get_center_extent(b)
                    body_pose = get_pose(b)
                    body_frame = tform_from_pose(body_pose)
                    bottom_center = copy(obj_center)
                    bottom_center[2] = bottom_center[2] - obj_extent[2] / 2
                    bottom_frame = tform_from_pose((bottom_center, body_pose[1]))
                    relative_frame_bottom = np.dot(bottom_frame, np.linalg.inv(body_frame))  # from pose to bottom
                    center_frame = tform_from_pose((obj_center, body_pose[1]))
                    relative_frame_center = np.dot(center_frame, np.linalg.inv(body_frame))

                    self.dic_body_info[b] = (obj_extent, relative_frame_bottom, relative_frame_center)
                self.reset()

    def reset(self):
        with HideOutput():
            with LockRenderer():
                # initial_jts = np.array([0.8, 0.75, 0.4, -1.8, 0.8, -1.5, 0])
                initial_jts = np.array([0.1, 1.4, 1, 1.7, 0, 0, 0])
                config_left = BodyConf(self.arm_left, initial_jts)
                config_left.assign()

                movable_door = get_movable_joints(self.bd_body['cabinet_shelf'])
                set_joint_positions(self.bd_body['cabinet_shelf'], movable_door, [-0.])

                set_pose(self.bd_body['c1'],
                         Pose(Point(x=0.375, y=0.9, z=EPSILON+stable_z(self.bd_body['c1'], self.bd_body['region1']))))
                set_pose(self.bd_body['c2'],
                         Pose(Point(x=0.32, y=0.9, z=EPSILON+stable_z(self.bd_body['c2'], self.bd_body['region1']))))
                set_pose(self.bd_body['c3'],
                         Pose(Point(x=0.34, y=0.845, z=EPSILON+stable_z(self.bd_body['c3'], self.bd_body['region1']))))

                set_camera(150, -35, 1.6, Point(-0.1, 0.1, -0.1))

    def get_elemetns(self):
        self.reset()
        return self.arm_left, self.movable_bodies, self.regions

class ScenarioForTraining(object):
    def __init__(self):
        with HideOutput():
            with LockRenderer():
                self.arm_left = load_pybullet("../darias_description/urdf/darias_L_primitive_collision.urdf",
                                              fixed_base=True)
                self.arm_base = load_pybullet("../darias_description/urdf/darias_base.urdf", fixed_base=True)

                self.bd_body = {
                    'floor': load_pybullet("../scenario_description/floor.urdf", fixed_base=True),
                    # 'cabinet_shelf': load_pybullet(
                    #     "../scenario_description/manipulation_worlds/urdf/cabinet_shelf.urdf",
                    #     fixed_base=True),
                    # 'drawer_shelf': load_pybullet(
                    #     "../scenario_description/manipulation_worlds/urdf/drawer_shelf.urdf",
                    #     fixed_base=True),
                    # 'pegboard': load_pybullet(
                    #     "../scenario_description/manipulation_worlds/urdf/pegboard.urdf",
                    #     fixed_base=True),
                    'region1': load_pybullet("../scenario_description/training_region.urdf", fixed_base=True),
                    'c1': load_pybullet("../scenario_description/boxCm.urdf", fixed_base=False),
                    'c2': load_pybullet("../scenario_description/boxC.urdf", fixed_base=False),
                }
                self.bd_body.update(dict((self.bd_body[k], k) for k in self.bd_body))

                # self.drawer_links = get_links(self.bd_body['drawer_shelf'])
                # cabinet_links = get_links(self.bd_body['cabinet_shelf'])

                # set_pose(self.bd_body['cabinet_shelf'],
                #          Pose(Point(x=-0.45, y=-0.8, z=stable_z(self.bd_body['cabinet_shelf'], self.bd_body['floor']))))
                # set_pose(self.bd_body['drawer_shelf'],
                #          Pose(Point(x=-0.45, y=0.8, z=stable_z(self.bd_body['drawer_shelf'], self.bd_body['floor']))))
                # set_pose(self.bd_body['pegboard'],
                #          Pose(Point(x=-0.60, y=0, z=stable_z(self.bd_body['pegboard'], self.bd_body['floor']))))
                set_pose(self.bd_body['region1'],
                         Pose(Point(x=0.2, y=0., z=stable_z(self.bd_body['region1'], self.bd_body['floor']))))

                self.movable_bodies = [self.bd_body['c1'], self.bd_body['c2']]
                self.body_gripped = self.bd_body['c1']
                self.body_on_table = self.bd_body['c2']
                # self.movable_bodies = []
                # self.env_bodies = [self.arm_base, self.bd_body['floor'], self.bd_body['cabinet_shelf'],
                                #    self.bd_body['drawer_shelf'], self.bd_body['pegboard']]
                self.env_bodies = [self.arm_base, self.bd_body['floor']]

                self.regions = [self.bd_body['region1']]

                self.all_bodies = list(set(self.movable_bodies) | set(self.env_bodies) | set(self.regions))

                self.sensors = []

                self.robots = [self.arm_left]
                self.robot  =self.robots[0]
                self.movable_joints = pu.get_movable_joints(self.robot)
                self.dic_body_info = {}
                for b in self.movable_bodies:
                    obj_center, obj_extent = get_center_extent(b)
                    body_pose = get_pose(b)
                    body_frame = tform_from_pose(body_pose)
                    bottom_center = copy(obj_center)
                    bottom_center[2] = bottom_center[2] - obj_extent[2] / 2
                    bottom_frame = tform_from_pose((bottom_center, body_pose[1]))
                    relative_frame_bottom = np.dot(bottom_frame, np.linalg.inv(body_frame))  # from pose to bottom
                    center_frame = tform_from_pose((obj_center, body_pose[1]))
                    relative_frame_center = np.dot(center_frame, np.linalg.inv(body_frame))
                    self.dic_body_info[b] = (obj_extent, relative_frame_bottom, relative_frame_center)

                self.reset()
                self.end_effector_link = pu.link_from_name(self.robot, pk.TOOL_FRAMES[pu.get_body_name(self.robot)])
                self.tcp_pose = pu.get_link_pose(self.robot, self.end_effector_link)
                self.robot_pose = pu.get_pose(self.robot)

                set_camera(150, -35, 1.6, Point(-0.1, 0.1, -0.1))

    def reset(self):
        with HideOutput():
            with LockRenderer():
                # initial_jts = np.array([0.8, 0.75, 0.4, -1.8, 0.8, -1.5, 0])
                initial_jts = np.array([0.1, 0, 0, 1.7, 0, 0, 0])
                config_left = BodyConf(self.arm_left, initial_jts)
                config_left.assign()

                # movable_door = get_movable_joints(self.bd_body['cabinet_shelf'])
                # set_joint_positions(self.bd_body['cabinet_shelf'], movable_door, [-0.])

                set_pose(self.bd_body['c1'],
                         Pose(Point(x=0.375, y=0.9, z=stable_z(self.bd_body['c1'], self.bd_body['region1']))))
                set_pose(self.bd_body['c2'],
                         Pose(Point(x=0.32, y=0.9, z=stable_z(self.bd_body['c2'], self.bd_body['region1']))))

    def get_elemetns(self):
        self.reset()
        return self.arm_left, self.movable_bodies, self.regions

class Scene_unpack1(object):
    def __init__(self):
        with HideOutput():
            with LockRenderer():
                self.arm_left = load_pybullet("../darias_description/urdf/darias_L_primitive_collision.urdf",
                                              fixed_base=True)
                self.arm_base = load_pybullet("../darias_description/urdf/darias_base.urdf", fixed_base=True)

                self.bd_body = {
                    'floor': load_pybullet("../scenario_description/floor.urdf", fixed_base=True),
                    'cabinet_shelf': load_pybullet(
                        "../scenario_description/manipulation_worlds/urdf/cabinet_shelf.urdf",
                        fixed_base=True),
                    'drawer_shelf': load_pybullet(
                        "../scenario_description/manipulation_worlds/urdf/drawer_shelf.urdf",
                        fixed_base=True),
                    'pegboard': load_pybullet(
                        "../scenario_description/manipulation_worlds/urdf/pegboard.urdf",
                        fixed_base=True),
                    'region1': load_pybullet("../scenario_description/region.urdf", fixed_base=True),
                    'region2': load_pybullet("../scenario_description/region_big.urdf",
                                             fixed_base=True),
                    'c1': load_pybullet("../scenario_description/boxCm.urdf", fixed_base=False),
                }
                self.bd_body.update(dict((self.bd_body[k], k) for k in self.bd_body))

                self.drawer_links = get_links(self.bd_body['drawer_shelf'])
                cabinet_links = get_links(self.bd_body['cabinet_shelf'])

                set_pose(self.bd_body['cabinet_shelf'],
                         Pose(Point(x=-0.45, y=-0.8, z=stable_z(self.bd_body['cabinet_shelf'], self.bd_body['floor']))))
                set_pose(self.bd_body['drawer_shelf'],
                         Pose(Point(x=-0.45, y=0.8, z=stable_z(self.bd_body['drawer_shelf'], self.bd_body['floor']))))
                set_pose(self.bd_body['pegboard'],
                         Pose(Point(x=-0.60, y=0, z=stable_z(self.bd_body['pegboard'], self.bd_body['floor']))))
                set_pose(self.bd_body['region1'],
                         Pose(Point(x=0.35, y=0.9, z=stable_z(self.bd_body['region1'], self.bd_body['floor']))))
                set_pose(self.bd_body['region2'],
                         Pose(Point(x=0.05, y=0.8, z=stable_z(self.bd_body['region2'], self.bd_body['floor']))))

                self.movable_bodies = [self.bd_body['c1'], ]
                self.env_bodies = [self.arm_base, self.bd_body['floor'], self.bd_body['cabinet_shelf'],
                                   self.bd_body['drawer_shelf'], self.bd_body['pegboard']]
                self.regions = [self.bd_body['region1'], self.bd_body['region2']]

                self.all_bodies = list(set(self.movable_bodies) | set(self.env_bodies) | set(self.regions))

                self.sensors = []

                self.robots = [self.arm_left]

                self.dic_body_info = {}
                for b in self.movable_bodies:
                    obj_center, obj_extent = get_center_extent(b)
                    body_pose = get_pose(b)
                    body_frame = tform_from_pose(body_pose)
                    bottom_center = copy(obj_center)
                    bottom_center[2] = bottom_center[2] - obj_extent[2] / 2
                    bottom_frame = tform_from_pose((bottom_center, body_pose[1]))
                    relative_frame_bottom = np.dot(bottom_frame, np.linalg.inv(body_frame))  # from pose to bottom
                    center_frame = tform_from_pose((obj_center, body_pose[1]))
                    relative_frame_center = np.dot(center_frame, np.linalg.inv(body_frame))

                    self.dic_body_info[b] = (obj_extent, relative_frame_bottom, relative_frame_center)

                self.reset()

    def reset(self):
        with HideOutput():
            with LockRenderer():
                # initial_jts = np.array([0.8, 0.75, 0.4, -1.8, 0.8, -1.5, 0])
                initial_jts = np.array([0.1, 1.4, 1, 1.7, 0, 0, 0])
                config_left = BodyConf(self.arm_left, initial_jts)
                config_left.assign()

                movable_door = get_movable_joints(self.bd_body['cabinet_shelf'])
                set_joint_positions(self.bd_body['cabinet_shelf'], movable_door, [-0.])

                set_pose(self.bd_body['c1'],
                         Pose(Point(x=0.375, y=0.9, z=stable_z(self.bd_body['c1'], self.bd_body['region1']))))


                set_camera(150, -35, 1.6, Point(-0.1, 0.1, -0.1))

    def get_elemetns(self):
        self.reset()
        return self.arm_left, self.movable_bodies, self.regions

class Scene_unpack2(object):
    def __init__(self):
        with HideOutput():
            with LockRenderer():
                self.arm_left = load_pybullet("../darias_description/urdf/darias_L_primitive_collision.urdf",
                                              fixed_base=True)
                self.arm_base = load_pybullet("../darias_description/urdf/darias_base.urdf", fixed_base=True)

                self.bd_body = {
                    'floor': load_pybullet("../scenario_description/floor.urdf", fixed_base=True),
                    'cabinet_shelf': load_pybullet(
                        "../scenario_description/manipulation_worlds/urdf/cabinet_shelf.urdf",
                        fixed_base=True),
                    'drawer_shelf': load_pybullet(
                        "../scenario_description/manipulation_worlds/urdf/drawer_shelf.urdf",
                        fixed_base=True),
                    'pegboard': load_pybullet(
                        "../scenario_description/manipulation_worlds/urdf/pegboard.urdf",
                        fixed_base=True),
                    'region1': load_pybullet("../scenario_description/region.urdf", fixed_base=True),
                    'region2': load_pybullet("../scenario_description/region_big.urdf",
                                             fixed_base=True),
                    'c1': load_pybullet("../scenario_description/boxCm.urdf", fixed_base=False),
                    'c2': load_pybullet("../scenario_description/boxC.urdf", fixed_base=False),
                }
                self.bd_body.update(dict((self.bd_body[k], k) for k in self.bd_body))

                self.drawer_links = get_links(self.bd_body['drawer_shelf'])
                cabinet_links = get_links(self.bd_body['cabinet_shelf'])

                set_pose(self.bd_body['cabinet_shelf'],
                         Pose(Point(x=-0.45, y=-0.8, z=stable_z(self.bd_body['cabinet_shelf'], self.bd_body['floor']))))
                set_pose(self.bd_body['drawer_shelf'],
                         Pose(Point(x=-0.45, y=0.8, z=stable_z(self.bd_body['drawer_shelf'], self.bd_body['floor']))))
                set_pose(self.bd_body['pegboard'],
                         Pose(Point(x=-0.60, y=0, z=stable_z(self.bd_body['pegboard'], self.bd_body['floor']))))
                set_pose(self.bd_body['region1'],
                         Pose(Point(x=0.35, y=0.9, z=stable_z(self.bd_body['region1'], self.bd_body['floor']))))
                set_pose(self.bd_body['region2'],
                         Pose(Point(x=0.05, y=0.8, z=stable_z(self.bd_body['region2'], self.bd_body['floor']))))

                self.movable_bodies = [self.bd_body['c1'], self.bd_body['c2'], ]
                self.env_bodies = [self.arm_base, self.bd_body['floor'], self.bd_body['cabinet_shelf'],
                                   self.bd_body['drawer_shelf'], self.bd_body['pegboard']]
                self.regions = [self.bd_body['region1'], self.bd_body['region2']]

                self.all_bodies = list(set(self.movable_bodies) | set(self.env_bodies) | set(self.regions))

                self.sensors = []

                self.robots = [self.arm_left]

                self.dic_body_info = {}
                for b in self.movable_bodies:
                    obj_center, obj_extent = get_center_extent(b)
                    body_pose = get_pose(b)
                    body_frame = tform_from_pose(body_pose)
                    bottom_center = copy(obj_center)
                    bottom_center[2] = bottom_center[2] - obj_extent[2] / 2
                    bottom_frame = tform_from_pose((bottom_center, body_pose[1]))
                    relative_frame_bottom = np.dot(bottom_frame, np.linalg.inv(body_frame))  # from pose to bottom
                    center_frame = tform_from_pose((obj_center, body_pose[1]))
                    relative_frame_center = np.dot(center_frame, np.linalg.inv(body_frame))

                    self.dic_body_info[b] = (obj_extent, relative_frame_bottom, relative_frame_center)

                self.reset()

    def reset(self):
        with HideOutput():
            with LockRenderer():
                # initial_jts = np.array([0.8, 0.75, 0.4, -1.8, 0.8, -1.5, 0])
                initial_jts = np.array([0.1, 1.4, 1, 1.7, 0, 0, 0])
                config_left = BodyConf(self.arm_left, initial_jts)
                config_left.assign()

                movable_door = get_movable_joints(self.bd_body['cabinet_shelf'])
                set_joint_positions(self.bd_body['cabinet_shelf'], movable_door, [-0.])

                set_pose(self.bd_body['c1'],
                         Pose(Point(x=0.375, y=0.9, z=stable_z(self.bd_body['c1'], self.bd_body['region1']))))
                set_pose(self.bd_body['c2'],
                         Pose(Point(x=0.32, y=0.9, z=stable_z(self.bd_body['c2'], self.bd_body['region1']))))


                set_camera(150, -35, 1.6, Point(-0.1, 0.1, -0.1))

    def get_elemetns(self):
        self.reset()
        return self.arm_left, self.movable_bodies, self.regions

class Scene_unpack3(object):
    def __init__(self):
        with HideOutput():
            with LockRenderer():
                self.arm_left = load_pybullet("../darias_description/urdf/darias_L_primitive_collision.urdf",
                                              fixed_base=True)
                self.arm_base = load_pybullet("../darias_description/urdf/darias_base.urdf", fixed_base=True)

                self.bd_body = {
                    'floor': load_pybullet("../scenario_description/floor.urdf", fixed_base=True),
                    'cabinet_shelf': load_pybullet(
                        "../scenario_description/manipulation_worlds/urdf/cabinet_shelf.urdf",
                        fixed_base=True),
                    'drawer_shelf': load_pybullet(
                        "../scenario_description/manipulation_worlds/urdf/drawer_shelf.urdf",
                        fixed_base=True),
                    'pegboard': load_pybullet(
                        "../scenario_description/manipulation_worlds/urdf/pegboard.urdf",
                        fixed_base=True),
                    'region1': load_pybullet("../scenario_description/region.urdf", fixed_base=True),
                    'region2': load_pybullet("../scenario_description/region_big.urdf",
                                             fixed_base=True),
                    'c1': load_pybullet("../scenario_description/boxCm.urdf", fixed_base=False),
                    'c2': load_pybullet("../scenario_description/boxC.urdf", fixed_base=False),
                    'c3': load_pybullet("../scenario_description/boxCx.urdf", fixed_base=False),
                }
                self.bd_body.update(dict((self.bd_body[k], k) for k in self.bd_body))

                self.drawer_links = get_links(self.bd_body['drawer_shelf'])
                cabinet_links = get_links(self.bd_body['cabinet_shelf'])

                set_pose(self.bd_body['cabinet_shelf'],
                         Pose(Point(x=-0.45, y=-0.8, z=stable_z(self.bd_body['cabinet_shelf'], self.bd_body['floor']))))
                set_pose(self.bd_body['drawer_shelf'],
                         Pose(Point(x=-0.45, y=0.8, z=stable_z(self.bd_body['drawer_shelf'], self.bd_body['floor']))))
                set_pose(self.bd_body['pegboard'],
                         Pose(Point(x=-0.60, y=0, z=stable_z(self.bd_body['pegboard'], self.bd_body['floor']))))
                set_pose(self.bd_body['region1'],
                         Pose(Point(x=0.35, y=0.9, z=stable_z(self.bd_body['region1'], self.bd_body['floor']))))
                set_pose(self.bd_body['region2'],
                         Pose(Point(x=0.05, y=0.8, z=stable_z(self.bd_body['region2'], self.bd_body['floor']))))

                self.movable_bodies = [self.bd_body['c1'], self.bd_body['c2'], self.bd_body['c3']]
                self.env_bodies = [self.arm_base, self.bd_body['floor'], self.bd_body['cabinet_shelf'],
                                   self.bd_body['drawer_shelf'], self.bd_body['pegboard']]
                self.regions = [self.bd_body['region1'], self.bd_body['region2']]

                self.all_bodies = list(set(self.movable_bodies) | set(self.env_bodies) | set(self.regions))

                self.sensors = []

                self.robots = [self.arm_left]

                self.dic_body_info = {}
                for b in self.movable_bodies:
                    obj_center, obj_extent = get_center_extent(b)
                    body_pose = get_pose(b)
                    body_frame = tform_from_pose(body_pose)
                    bottom_center = copy(obj_center)
                    bottom_center[2] = bottom_center[2] - obj_extent[2] / 2
                    bottom_frame = tform_from_pose((bottom_center, body_pose[1]))
                    relative_frame_bottom = np.dot(bottom_frame, np.linalg.inv(body_frame))  # from pose to bottom
                    center_frame = tform_from_pose((obj_center, body_pose[1]))
                    relative_frame_center = np.dot(center_frame, np.linalg.inv(body_frame))

                    self.dic_body_info[b] = (obj_extent, relative_frame_bottom, relative_frame_center)

                self.reset()

    def reset(self):
        with HideOutput():
            with LockRenderer():
                # initial_jts = np.array([0.8, 0.75, 0.4, -1.8, 0.8, -1.5, 0])
                initial_jts = np.array([0.1, 1.4, 1, 1.7, 0, 0, 0])
                config_left = BodyConf(self.arm_left, initial_jts)
                config_left.assign()

                movable_door = get_movable_joints(self.bd_body['cabinet_shelf'])
                set_joint_positions(self.bd_body['cabinet_shelf'], movable_door, [-0.])

                set_pose(self.bd_body['c1'],
                         Pose(Point(x=0.375, y=0.9, z=stable_z(self.bd_body['c1'], self.bd_body['region1']))))
                set_pose(self.bd_body['c2'],
                         Pose(Point(x=0.32, y=0.9, z=stable_z(self.bd_body['c2'], self.bd_body['region1']))))
                set_pose(self.bd_body['c3'],
                         Pose(Point(x=0.34, y=0.845, z=stable_z(self.bd_body['c3'], self.bd_body['region1']))))

                set_camera(150, -35, 1.6, Point(-0.1, 0.1, -0.1))

    def get_elemetns(self):
        self.reset()
        return self.arm_left, self.movable_bodies, self.regions

class Scene_random(object):
    def __init__(self, dirname='random_scenes', json_file=None):
        self.dirname = dirname
        if not os.path.exists(self.dirname):
            print(f"exists no {dirname}")
            os.mkdir(self.dirname)
        print(f"going to save scene json file into {self.dirname}")
        
        self.initial_scene()
        if json_file is None:
            self.random_scene()
        else:
            self.load_scene_from_json(json_file)

    def initial_scene(self):
        with HideOutput():
            with LockRenderer():

                self.env_bodies_name = [
                    "arm_base", 
                    "robot", 
                    "floor", 
                    "cabinet_shelf", 
                    "drawer_shelf", 
                    "pegboard"
                    ]

                self.files_env_bodies = {
                    "robot": "../darias_description/urdf/darias_L_primitive_collision.urdf",
                    "arm_base": "../darias_description/urdf/darias_base.urdf",
                    'floor': "../scenario_description/floor.urdf",
                    # 'cabinet_shelf': "../scenario_description/manipulation_worlds/urdf/cabinet_shelf.urdf",
                    # 'drawer_shelf': "../scenario_description/manipulation_worlds/urdf/drawer_shelf.urdf",
                    # 'pegboard': "../scenario_description/manipulation_worlds/urdf/pegboard.urdf",
                }

                self.pose_env_bodies = {
                    "robot": ((0, 0, 0), (0.0, 0.0, 0.0, 1.0)),
                    "arm_base": ((0, 0, 0), (0.0, 0.0, 0.0, 1.0)),
                    "floor": ((0, 0, 0), (0.0, 0.0, 0.0, 1.0)),
                    # "cabinet_shelf": ((-0.45, -0.8, -0.998), (0.0, 0.0, 0.0, 1.0)),
                    # "drawer_shelf": ((-0.45, 0.8, -0.998), (0.0, 0.0, 0.0, 1.0)),
                    # "pegboard": ((-0.6, 0.0, -0.498), (0.0, 0.0, 0.0, 1.0))
                }

                self.env_bodies = set()
                self.bd_body = dict()
                for name, file in self.files_env_bodies.items():
                    self.bd_body[name] = load_pybullet(file, fixed_base=True)
                    pu.set_pose(self.bd_body[name], self.pose_env_bodies[name])
                    self.env_bodies.add(self.bd_body[name])
                pu.set_joint_positions(0, list(range(1,8)), [0.1, 1.4, 1, 1.7, 0, 0, 0])

                self.camera_setting = [1.6, 150, -35, -0.1, 0.1, -0.1]
                set_camera(self.camera_setting[1], self.camera_setting[2], self.camera_setting[0], 
                                    self.camera_setting[3:])

    def random_scene(self):
        # random scene
        self.fixed_or_random_boxZ()
        self.random_or_fixed_region_height()
        self.random_regions()
        self.random_boxes()
        self.random_boxes_location()
        self.bd_body.update(dict((self.bd_body[k], k) for k in self.bd_body))

        self.random_goal()
        self.hhhhhhhhhh()
        # self.save_scene_in_json()

    def random_or_fixed_region_height(self):
        # self.random_height = bool(np.random.randint(2))
        self.random_height = 0
        self.subregions = [((-0.1,0.5),(0.5,0.8)),
                    # ((0.2,-0.4),(0.7,0.4)),
                    # ((-0.2,-1.0),(0.7,-0.4))
                    ]
        if self.random_height:
            self.table_z_lower, self.table_z_upper = 0.1, 0.5
        else:
            self.table_z_lower, self.table_z_upper = 0.1, 0.1

    def all_or_top_pick_dir(self):
        self.all_pick_dir = bool(np.random.randint(2))

    def fixed_or_random_boxZ(self):
        self.fixed_boxZ = bool(np.random.randint(2))
        self.fixed_boxZ = True

    def random_regions(self):
        regions = np.array(self.subregions)
        tmp = regions[:,1]-regions[:,0]
        area = tmp[:,0]*tmp[:,1]
        prob_area = area/np.sum(area)
        def random_xy():
            while True:
                i = np.random.randint(len(self.subregions))
                if prob_area[i]>=np.random.rand(1)[0]:
                    break
            lower,upper = self.subregions[i]
            xy = np.random.uniform(lower, upper)
            # pu.draw_point((xy[0],xy[1],0.1))
            return xy

        self.region_num = np.random.randint(2,3)
        region_size = (0.3,0.3,0.001)
        self.regions = set()
        for i in range(self.region_num):
            tmp_reg = pu.create_box(region_size[0],region_size[1],region_size[2])
            self.regions.add(tmp_reg)
            self.bd_body[f"region{i}"] = tmp_reg
            pu.set_color(tmp_reg, [0.1,0.5,0.5,1])

            while True:
                # random region height
                table_z = np.random.uniform(self.table_z_lower, self.table_z_upper)
                xyz = list(random_xy())+ [table_z]
                pu.set_pose(tmp_reg, (xyz, (0,0,0,1)))
                if not pu.body_list_collision([tmp_reg], list(set(self.bd_body.values())-{tmp_reg})):
                    break

    def random_boxes(self):
        # self.boxes_num = np.random.randint(2,5)
        self.movable_box_num = np.random.randint(4,5)
        self.obstacle_box_num = np.random.randint(0,1)
        self.movable_bodies = set()
        self.obstacle_bodies = set()

        for i in range(self.movable_box_num):
            lower, upper = [0.04,0.04,0.05],[0.1,0.1,0.25]
            w,l,h = np.random.uniform(lower, upper)
            tmp_box = pu.create_box(w,l,h)
            self.movable_bodies.add(tmp_box)
            self.bd_body[f"box{i}"] = tmp_box
            pu.set_color(tmp_box, [1,0,0,1])

        for i in range(self.obstacle_box_num):
            lower, upper = [0.05,0.05,0.1],[0.12,0.12,0.2]
            w,l,h = np.random.uniform(lower, upper)
            tmp_obs = pu.create_box(w,l,h)
            self.obstacle_bodies.add(tmp_obs)
            self.bd_body[f"obs{i}"] = tmp_obs
            pu.set_color(tmp_obs, [0.5,0.5,0.5,1])

    def random_boxes_location(self):
        for body in self.movable_bodies.union(self.obstacle_bodies):
            region = np.random.choice(list(self.regions))
            region_lower, region_upper = pu.get_aabb(region)
            region_w, region_h, _ = pu.get_aabb_extent((region_lower, region_upper))
            box_wlh = pu.get_aabb_extent(pu.get_aabb(body))
            box_z = [region_upper[2]+box_wlh[2]/2+EPSILON]

            # sampling box location on specific region and ensure no collision detected
            while True:
                xy_norm = np.random.uniform([0.1,0.1],[0.9,0.9])
                box_xy = list(np.array([region_w,region_h]) * np.array(xy_norm) + np.array(region_lower[:2]))

                if self.fixed_boxZ:
                    box_pose = (tuple(box_xy+box_z), (0,0,0,1))
                else:
                    rot = np.random.uniform()
                    quat = pu.quat_from_axis_angle((0,0,1), rot)
                    box_pose = (tuple(box_xy+box_z), quat)

                pu.set_pose(body, box_pose)
                if not pu.body_list_collision([body], list(set(self.bd_body.values())-{body})):
                    break
                else:
                    print(f'collision detected when sampling box{body}')

    def random_goal(self):
        # ? body on ? region
        target_body = list(self.movable_bodies)[0]
        pu.set_color(target_body, [0,0,1,1])
        region_of_target_body = pu.get_region_of_body(target_body, self.regions)
        goal_region = np.random.choice(list(self.regions-{region_of_target_body}))
        pu.set_color(goal_region, [0,0,1,1])
        self.goal = {self.bd_body[goal_region]: [self.bd_body[target_body]]}

    def save_scene_in_json(self):
        self.scene_save_name = f'{self.dirname}/' + str(uuid.uuid1()) + '.json'
        print(f"saving current scene to {self.scene_save_name}")
        
        # robot: filename, base_pose, init_config
        self.scene = {}
        self.scene['bodies'] = []   
        self.scene['bodies'].append({
            'name':'robot',
            'urdf':'../darias_description/urdf/darias_L_primitive_collision.urdf',
            'shape': 'None',
            'pose':((0,0,0),(0,0,0,1)),
            'movable_joints':[1,2,3,4,5,6,7],
            'init_conf': [0.1, 1.4, 1, 1.7, 0, 0, 0]
        })
        self.scene['bodies'].append({
            'name':'arm_base',
            'urdf':'../darias_description/urdf/darias_base.urdf',
            'shape': 'None',
            'pose':((0,0,0),(0,0,0,1)),
        })
        self.scene['bodies'].append({
            'name':'floor',
            'urdf':'../scenario_description/floor.urdf',
            'shape': 'None',
            'pose':((0,0,0),(0,0,0,1)),
        })

        for body_ind in self.movable_bodies.union(self.obstacle_bodies.union(self.regions)):
            tmp_body = dict()
            tmp_body['name'] = self.bd_body[body_ind]
            tmp_body['urdf'] = "None"
            tmp_body['shape'] = 'box' # sphere / cylinder / capsule / plane
            tmp_body['size'] = list(pu.get_vertical_placed_box_size(body_ind))
            tmp_body['pose'] = pu.get_pose(body_ind)
            tmp_body['color'] = pu.get_visual_data(body_ind)[0].rgbaColor
            self.scene['bodies'].append(tmp_body)

        self.scene['camera'] = {
            'distance': self.camera_setting[0],
            'yaw': self.camera_setting[1],
            'pitch': self.camera_setting[2],
            'target_point': list(self.camera_setting[3:])
        }
        self.scene['goal'] = self.goal

        self.scene['tm_plan'] = {
            't_plan': "None",
            'm_plan': "None",
        }

        with open(self.scene_save_name, 'w') as file:
            json.dump(self.scene, file, indent=4)

    def update_scene_tm_plan(self, t_plan, m_plan):
        tm_plan = dict()
        tm_plan['t_plan'] = t_plan
        tm_plan['m_plan'] = m_plan
        self.scene['tm_plan'] = tm_plan
        with open(self.scene_save_name, 'w') as file:
            json.dump(self.scene, file, indent=4)

    def load_scene_from_json(self, filename):
        with open(filename, 'r') as file:
            self.scene = json.load(file)

        self.movable_bodies = set()
        self.obstacle_bodies = set()
        self.regions = set()
        for body in self.scene['bodies']:
            if body['shape'] not in ['box','sphere','cylinder']:
                continue
            if body['shape']=='box':
                tmp_ind = pu.create_box(*body['size'], color=body['color'])
                self.bd_body[body['name']] = tmp_ind
                pu.set_pose(tmp_ind, body['pose'])
            else:
                raise NotImplementedError

            if 'obs' in body['name']:
                self.obstacle_bodies.add(tmp_ind)
            elif 'box' in body['name']:
                self.movable_bodies.add(tmp_ind)
            elif 'region' in body['name']:
                self.regions.add(tmp_ind)
        self.bd_body.update(dict((self.bd_body[k], k) for k in self.bd_body))

        self.goal = self.scene['goal']
        
        self.hhhhhhhhhh()

    def hhhhhhhhhh(self):
        self.robots = [self.bd_body['robot']]
        self.all_bodies = self.env_bodies.union(
                            self.movable_bodies.union(
                                self.obstacle_bodies.union(
                                    self.regions)))
        self.movable_joints = list(range(1,8))

    def reset(self):
        with HideOutput():
            with LockRenderer():
                # initial_jts = np.array([0.8, 0.75, 0.4, -1.8, 0.8, -1.5, 0])
                initial_jts = np.array([0.1, 1.4, 1, 1.7, 0, 0, 0])
                config_left = BodyConf(self.arm_left, initial_jts)
                config_left.assign()
                # movable_door = get_movable_joints(self.bd_body['cabinet_shelf'])
                # set_joint_positions(self.bd_body['cabinet_shelf'], movable_door, [-0.])
                set_camera(150, -35, 1.6, Point(-0.1, 0.1, -0.1))

    def get_elemetns(self):
        self.reset()
        return self.arm_left, self.movable_bodies, self.regions

    def clear_scene(self):
        
        pass
#######################################################

def display_scenario():
    connect(use_gui=True)

    scn = PlanningScenario()

    for i in range(10000):
        step_simulation()
        time.sleep(0.1)

    disconnect()
    print('Finished.')


if __name__ == '__main__':
    display_scenario()
