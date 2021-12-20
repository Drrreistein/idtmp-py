#!/usr/bin/env python

from __future__ import print_function
import numpy as np

import time

from numpy import random
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
        while True:
        # random scene
            self.fixed_or_random_boxZ()
            self.random_or_fixed_region_height()
            self.random_regions()
            res = self.random_regions_location()
            self.random_boxes()
            res = res and self.random_boxes_location()
            if not res:
                pu.remove_bodies(list(self.regions)+list(self.movable_bodies)+list(self.obstacle_bodies))
                continue
            else:
                self.bd_body.update(dict((self.bd_body[k], k) for k in self.bd_body))
                self.random_goal()
                self.hhhhhhhhhh()
                self.all_or_top_pick_dir()
                break

        # self.save_scene_in_json()
        self.scene_save_name = f'{self.dirname}/' + str(uuid.uuid1()) + '.json'

    def random_or_fixed_region_height(self):
        # self.random_height = bool(np.random.randint(2))
        self.random_height = 0
        self.subregions = [((-0.2,0.5),(0.7,0.9)),
                    # ((0.2,-0.4),(0.7,0.4)),
                    # ((-0.2,-1.0),(0.7,-0.4))
                    ]
        if self.random_height:
            self.table_z_lower, self.table_z_upper = 0.1, 0.5
        else:
            self.table_z_lower, self.table_z_upper = 0.1, 0.1

    def all_or_top_pick_dir(self):
        self.all_pick_dir = bool(np.random.randint(2))
        # self.all_pick_dir = 1
        print(f"pick direction from {self.all_pick_dir}")

    def fixed_or_random_boxZ(self):
        self.fixed_boxZ = bool(np.random.randint(2))
        self.fixed_boxZ = True

    def random_regions(self):
        self.region_num = np.random.randint(2,3)
        region_size = (0.25,0.25,0.001)
        self.regions = set()
        for i in range(self.region_num):
            tmp_reg = pu.create_box(region_size[0],region_size[1],region_size[2])
            self.regions.add(tmp_reg)
            self.bd_body[f"region{i}"] = tmp_reg
            pu.set_color(tmp_reg, [0.1,0.5,0.5,1])

    def random_regions_location(self):
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

        random_region_location_times = 0
        for tmp_reg in self.regions:
            while True:
                # random region height
                table_z = np.random.uniform(self.table_z_lower, self.table_z_upper)
                xyz = list(random_xy())+ [table_z]
                pu.set_pose(tmp_reg, (xyz, (0,0,0,1)))
                if not pu.body_list_collision([tmp_reg], list(set(self.bd_body.values())-{tmp_reg})):
                    break
                random_region_location_times += 1
                if random_region_location_times>50:
                    return False
        return True

    def random_boxes(self):
        # self.boxes_num = np.random.randint(2,5)
        self.movable_box_num = np.random.randint(2,5)
        self.obstacle_box_num = np.random.randint(2,4)
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
            random_box_location_times = 0
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
                random_box_location_times += 1
                if random_box_location_times>50:
                    return False
        return True
                

    def random_goal(self):
        # ? body on ? region
        target_bodies = list(self.movable_bodies)[:2]
        [pu.set_color(bd, [0,0,1,1]) for bd in target_bodies]
        region_of_target_body = pu.get_region_of_body(target_bodies[0], self.regions)
        goal_region = np.random.choice(list(self.regions-{region_of_target_body}))
        pu.set_color(goal_region, [0,0,0.5,1])
        self.goal = {self.bd_body[goal_region]: [self.bd_body[bd] for bd in target_bodies]}

    def save_scene_in_json(self):
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
        self.scene['all_pick_dir'] = self.all_pick_dir

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
        self.scene_save_name = filename
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

        try:
            self.all_pick_dir = self.scene['all_pick_dir']
        except:
            self.all_pick_dir = 0

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

