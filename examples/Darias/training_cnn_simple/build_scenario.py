#!/usr/bin/env python

from __future__ import print_function
import numpy as np

import time
from utils.pybullet_tools.kuka_primitives3 import BodyPose, BodyConf, Register
from utils.pybullet_tools.utils import WorldSaver, connect, dump_world, get_pose, set_pose, Pose, \
    Point, set_default_camera, stable_z, disconnect, get_bodies, HideOutput, \
    create_box, \
    load_pybullet, step_simulation, Euler, get_links, get_link_info, get_movable_joints, set_joint_positions, \
    set_camera, get_center_extent, tform_from_pose, attach_viewcone, LockRenderer

from utils.pybullet_tools.body_utils import draw_frame
import utils.pybullet_tools.utils as pu
import utils.pybullet_tools.kuka_primitives3 as pk
from copy import copy

class TrainingScenario(object):
    def __init__(self):
        with HideOutput():
            with LockRenderer():
                self.arm_left = load_pybullet("../darias_description/urdf/darias_L_primitive_collision.urdf",
                                              fixed_base=True)
                # set_pose(self.arm_left, Pose(Point(x=0.25, y=0.10, z=0.0), Euler(0, 0, 30 * np.pi / 180)))
                self.arm_base = load_pybullet("../darias_description/urdf/darias_base.urdf", fixed_base=True)
                # set_pose(self.arm_base, Pose(Point(x=0.25, y=0.10, z=0.0), Euler(0, 0, 30 * np.pi / 180)))

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

                    'region_shelf': create_box(0.35, 0.44, 0.001, color=[0,0,1,0.5]),
                    'region_table': create_box(0.8, 0.6, 0.001, color=[0,0,1,0.5]),
                    # load_pybullet("../scenario_description/train_shelf_region.urdf", fixed_base=True),
                    # 'region_table': load_pybullet("../scenario_description/training_bs_region.urdf", fixed_base=True),
                    'c1': load_pybullet("../scenario_description/boxA.urdf", fixed_base=False),
                    'c2': load_pybullet("../scenario_description/boxA.urdf", fixed_base=False),
                    # 'region1': load_pybullet("../scenario_description/region.urdf", fixed_base=True),
                    # 'region2': load_pybullet("../scenario_description/region_big.urdf",fixed_base=True),
                }
                self.all_bodies = [b for b in self.bd_body.values()]
                self.bd_body.update(dict((self.bd_body[k], k) for k in self.bd_body))

                self.drawer_links = get_links(self.bd_body['drawer_shelf'])
                cabinet_links = get_links(self.bd_body['cabinet_shelf'])

                self.id_to_body = {}

                set_pose(self.bd_body['cabinet_shelf'],
                         Pose(Point(x=-0.45, y=-0.8, z=stable_z(self.bd_body['cabinet_shelf'], self.bd_body['floor']))))
                set_pose(self.bd_body['drawer_shelf'],
                         Pose(Point(x=-0.45, y=0.8, z=stable_z(self.bd_body['drawer_shelf'], self.bd_body['floor']))))
                set_pose(self.bd_body['pegboard'],
                         Pose(Point(x=-0.60, y=0, z=stable_z(self.bd_body['pegboard'], self.bd_body['floor']))))
                set_pose(self.bd_body['region_shelf'],Pose(Point(x=-0.45, y=0.8, z=0.774)))
                set_pose(self.bd_body['region_table'],Pose(Point(x=0.2, y=0.8, z=stable_z(self.bd_body['region_table'], self.bd_body['floor']))))
                # set_pose(self.bd_body['region1'],
                #          Pose(Point(x=0.35, y=0.9, z=stable_z(self.bd_body['region1'], self.bd_body['floor']))))
                # set_pose(self.bd_body['region2'],
                #          Pose(Point(x=0.05, y=0.8, z=stable_z(self.bd_body['region2'], self.bd_body['floor']))))
        self.body_gripped = self.bd_body['c1']
        self.body_on_table = self.bd_body['c2']
        self.table = self.bd_body['region_table']
        self.shelf = self.bd_body['region_shelf']

        self.movable_bodies = [self.bd_body['c1'], self.bd_body['c2']]

        self.regions = [self.bd_body['region_shelf'], self.bd_body['region_table']]

        self.env_bodies = [b for b in self.all_bodies if (b not in self.regions) and (b not in self.movable_bodies)]

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

        """If the body should be attached to the region when it is placed."""
        self.dic_body_info[self.bd_body['region_shelf']] = (False,)
        self.dic_body_info[self.bd_body['region_table']] = (False,)
        self.end_effector_link = pu.link_from_name(self.robot, pk.TOOL_FRAMES[pu.get_body_name(self.robot)])
        self.tcp_pose = pu.get_link_pose(self.robot, self.end_effector_link)
        self.robot_pose = pu.get_pose(self.robot)
        
        self.reset()

    def reset(self):
        with HideOutput():
            with LockRenderer():
                # initial_jts = np.array([0.8, 0.75, 0.4, -1.8, 0.8, -1.5, 0])
                # initial_jts = np.array([-0.5, 1.4, 1.5, 1.3, 0, 0, 0])
                # initial_jts = np.array([0, 0, 0, 0, 0, 0, 0])
                initial_jts = np.array([0.1, -1.4, 1, 1.7, 0, 0, 0])

                config_left = BodyConf(self.arm_left, initial_jts)
                config_left.assign()

                movable_door = get_movable_joints(self.bd_body['cabinet_shelf'])
                set_joint_positions(self.bd_body['cabinet_shelf'], movable_door, [-0.])

                set_pose(self.bd_body['c1'],Pose(Point(x=-0.02, y=0.9, z=0.1), Euler(yaw=1 * np.pi / 2)))
                set_pose(self.bd_body['c2'], Pose(Point(x=-0.02, y=0.72, z=0.1)))

                p = get_pose(self.bd_body['c1'])

                set_camera(160, -35, 1.8, Point())

    def get_elemetns(self):
        self.reset()
        return self.arm_left, self.movable_bodies, self.regions

    # def show_view_cone(self, lifetime=10):
    #     r = Register(self.bd_body['camera1'], self.bd_body['box1'])
    #     attach_viewcone(self.bd_body['camera1'], depth=1.5)
    #     draw_frame(tform_from_pose(get_pose(self.bd_body['camera1'])), None)

    #     r.show()

class TrainingScenario_obj_centered(object):
    def __init__(self):
        with HideOutput():
            with LockRenderer():
                self.arm_left = load_pybullet("../darias_description/urdf/darias_L_primitive_collision.urdf",
                                              fixed_base=True)
                # set_pose(self.arm_left, Pose(Point(x=0.25, y=0.10, z=0.0), Euler(0, 0, 30 * np.pi / 180)))
                self.arm_base = load_pybullet("../darias_description/urdf/darias_base.urdf", fixed_base=True)
                # set_pose(self.arm_base, Pose(Point(x=0.25, y=0.10, z=0.0), Euler(0, 0, 30 * np.pi / 180)))

                self.bd_body = {
                    # 'floor': load_pybullet("../scenario_description/floor.urdf", fixed_base=True),
                    # 'cabinet_shelf': load_pybullet(
                    #     "../scenario_description/manipulation_worlds/urdf/cabinet_shelf.urdf",
                    #     fixed_base=True),
                    # 'drawer_shelf': load_pybullet(
                    #     "../scenario_description/manipulation_worlds/urdf/drawer_shelf.urdf",
                    #     fixed_base=True),
                    # 'pegboard': load_pybullet(
                    #     "../scenario_description/manipulation_worlds/urdf/pegboard.urdf",
                    #     fixed_base=True),

                    'region_shelf': create_box(0.35, 0.44, 0.001, color=[0,0,1,0.5]),
                    'region_table': create_box(0.4, 0.4, 0.001, color=[0,0,1,0.5]),
                    # load_pybullet("../scenario_description/train_shelf_region.urdf", fixed_base=True),
                    # 'region_table': load_pybullet("../scenario_description/training_bs_region.urdf", fixed_base=True),
                    'c1': load_pybullet("../scenario_description/boxA.urdf", fixed_base=False),
                    'c2': load_pybullet("../scenario_description/boxA.urdf", fixed_base=False),
                    # 'region1': load_pybullet("../scenario_description/region.urdf", fixed_base=True),
                    # 'region2': load_pybullet("../scenario_description/region_big.urdf",fixed_base=True),
                }
                self.all_bodies = [b for b in self.bd_body.values()]
                self.bd_body.update(dict((self.bd_body[k], k) for k in self.bd_body))

                # self.drawer_links = get_links(self.bd_body['drawer_shelf'])
                # cabinet_links = get_links(self.bd_body['cabinet_shelf'])

                self.id_to_body = {}

                # set_pose(self.bd_body['cabinet_shelf'],
                #          Pose(Point(x=-0.45, y=-0.8, z=stable_z(self.bd_body['cabinet_shelf'], self.bd_body['floor']))))
                # set_pose(self.bd_body['drawer_shelf'],
                #          Pose(Point(x=-0.45, y=0.8, z=stable_z(self.bd_body['drawer_shelf'], self.bd_body['floor']))))
                # set_pose(self.bd_body['pegboard'],
                #          Pose(Point(x=-0.60, y=0, z=stable_z(self.bd_body['pegboard'], self.bd_body['floor']))))
                set_pose(self.bd_body['region_shelf'],Pose(Point(x=-0.45, y=0.8, z=1.774)))
                set_pose(self.bd_body['region_table'],Pose(Point(x=0.2, y=0.8, z=0)))
                    # z=stable_z(self.bd_body['region_table'], self.bd_body['floor']))))
                # set_pose(self.bd_body['region1'],
                #          Pose(Point(x=0.35, y=0.9, z=stable_z(self.bd_body['region1'], self.bd_body['floor']))))
                # set_pose(self.bd_body['region2'],
                #          Pose(Point(x=0.05, y=0.8, z=stable_z(self.bd_body['region2'], self.bd_body['floor']))))
        self.body_gripped = self.bd_body['c1']
        self.body_on_table = self.bd_body['c2']
        self.table = self.bd_body['region_table']
        self.shelf = self.bd_body['region_shelf']

        self.movable_bodies = [self.bd_body['c1'], self.bd_body['c2']]

        self.regions = [self.bd_body['region_shelf'], self.bd_body['region_table']]

        self.env_bodies = [b for b in self.all_bodies if (b not in self.regions) and (b not in self.movable_bodies)]

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

        """If the body should be attached to the region when it is placed."""
        self.dic_body_info[self.bd_body['region_shelf']] = (False,)
        self.dic_body_info[self.bd_body['region_table']] = (False,)
        self.end_effector_link = pu.link_from_name(self.robot, pk.TOOL_FRAMES[pu.get_body_name(self.robot)])
        self.tcp_pose = pu.get_link_pose(self.robot, self.end_effector_link)
        self.robot_pose = pu.get_pose(self.robot)
    
        self.reset()

    def reset(self):
        with HideOutput():
            with LockRenderer():
                # initial_jts = np.array([0.8, 0.75, 0.4, -1.8, 0.8, -1.5, 0])
                # initial_jts = np.array([-0.5, 1.4, 1.5, 1.3, 0, 0, 0])
                # initial_jts = np.array([0, 0, 0, 0, 0, 0, 0])
                initial_jts = np.array([0.1, -1.4, 1, 1.7, 0, 0, 0])

                config_left = BodyConf(self.arm_left, initial_jts)
                config_left.assign()

                # movable_door = get_movable_joints(self.bd_body['cabinet_shelf'])
                # set_joint_positions(self.bd_body['cabinet_shelf'], movable_door, [-0.])

                set_pose(self.bd_body['c1'],Pose(Point(x=-0.02, y=0.9, z=0.1), Euler(yaw=1 * np.pi / 2)))
                set_pose(self.bd_body['c2'], Pose(Point(x=-0.02, y=0.72, z=0.1)))

                p = get_pose(self.bd_body['c1'])

                set_camera(160, -35, 1.8, Point())

    def get_elemetns(self):
        self.reset()
        return self.arm_left, self.movable_bodies, self.regions

if __name__ == '__main__':
    display_scenario()
