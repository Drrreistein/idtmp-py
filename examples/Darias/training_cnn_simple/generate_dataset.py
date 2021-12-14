from random import random

from PIL.Image import init
from progress.bar import IncrementalBar
import itertools

from numpy.core.getlimits import _register_known_types
from numpy.lib.npyio import save

from build_scenario import *
from IPython import embed
import csv, uuid, os, sys   
from multiprocessing import Process, process

import pybullet as p
from utils.pybullet_tools.utils import WorldSaver, connect, get_pose, is_fixed_base, pixel_from_ray, set_joint_positions, set_pose, get_configuration, is_placement, \
    disconnect, get_bodies, create_box, remove_body, get_aabb   
import utils.pybullet_tools.utils as pu
import utils.pybullet_tools.kuka_primitives3 as pk
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
from PIL import Image
EPSILON = 0.005
max_height = 0.512
pixel_size=0.002

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

    start_joints = pu.get_joint_positions(scn.robot, scn.movable_joints)
    init_body_pose = pu.get_pose(scn.body_gripped)
    for _ in range(10):
        goal_joints = pu.inverse_kinematics_random(scn.robot, scn.end_effector_link, goal_pose, obstacles=obstacles,self_collisions=True, 
        disabled_collisions=pk.DISABLED_COLLISION_PAIR, attachments=[], max_distance=pk.MAX_DISTANCE)
        if goal_joints is None:
            continue

        pu.set_joint_positions(scn.robot, scn.movable_joints, start_joints)
        pu.set_pose(scn.body_gripped,init_body_pose)

        pu.set_joint_positions(scn.robot, scn.movable_joints, goal_joints)

        attachment = attach_to_robot(scn, scn.body_gripped)
        obstacles = list(set(scn.all_bodies) - {attachment.child})
        goal_conf = pk.BodyConf(scn.robot, start_joints, scn.movable_joints)
        attachment.assign()
        path = pu.plan_joint_motion(scn.robot, scn.movable_joints, goal_conf.configuration, obstacles=obstacles,self_collisions=True, 
        disabled_collisions = pk.DISABLED_COLLISION_PAIR, attachments=[attachment], direct=True,
        max_distance=pk.MAX_DISTANCE, iterations=1000)
        if path is None:
            # print(f"found no path")
            continue
        else:
            pu.set_joint_positions(scn.robot, scn.movable_joints, start_joints)
            attachment.assign()
            pu.set_pose(scn.body_gripped,init_body_pose)
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
        lwh = list(np.random.uniform([0.04,0.04,0.05],[0.12,0.12,0.25]))
        # lwh = list(np.random.uniform([0.05,0.05,0.05],[0.05,0.05,0.3]))
        # lwh = [0.05,0.05,np.random.choice([0.1,0.15,0.2])]

    elif region_ind==1:
        lwh = list(np.random.uniform([0.04,0.04,0.05],[0.1,0.1,0.2]))
    body = create_box(lwh[0], lwh[1], lwh[2])
    return lwh, body

def attach_to_robot(scn, body):

    body_pose = pu.get_pose(body)
    tcp_pose = pu.get_link_pose(scn.robot, scn.end_effector_link)
    grasp_pose = pu.invert(pu.multiply(pu.invert(body_pose), tcp_pose))
    attachment = pk.Attachment(scn.robot, scn.end_effector_link, grasp_pose, body)
    return attachment

def attach_to_table(scn, body, lwh, lower, upper, ref=None):
    # scn.body_on_table = body
    if ref is None:
        xy = list(np.random.uniform(lower[:2] ,upper[:2]))
    else:
        while True:
            xy = ref[:2] + np.random.randn(2) * 0.3
            xy = np.clip(xy, lower[:2], upper[:2])
            if np.linalg.norm(xy[:2]-ref[:2])>0.03 or np.random.rand(1)>0.7:
                break

    pose = ((xy[0], xy[1], lwh[2]/2+upper[2]+EPSILON),(0,0,0.7071067811865475, 0.7071067811865476))
    pu.set_pose(body, pose)
    return xy, pose

def target_pose_vs_grsp_dir(center_pose, grsp_dir, extend):
    center_point, rotation = center_pose

    offset = np.array(grsp_dir, dtype=int) * (extend/2 + np.array([EPSILON,EPSILON,EPSILON]))
    goal_point = tuple(np.array(center_point) + offset)
    # angle_by_axis = np.array(grsp_dir, dtype=int) * np.pi/2

    if grsp_dir[2]==0:
        goal_rot = pu.multiply_quats(rotation, 
                # pu.quat_from_euler((angle_by_axis[0], angle_by_axis[1], 0))
                pu.quat_from_axis_angle((-grsp_dir[1], grsp_dir[0], 0), np.pi/2))
    else:
        goal_rot = rotation

    goal_rot = pu.multiply_quats(goal_rot, pu.quat_from_euler((np.pi, 0, 0)))
    target_pose = pu.Pose(goal_point, pu.euler_from_quat(goal_rot))
    # pu.draw_pose(target_pose)

    return target_pose

def bodies_in_region(scn, region):
    pass

def png_mat_gripped(scn, region_aabb, max_height=0.512):
    lower, upper = region_aabb
    length, width = np.array(np.abs(((upper-lower)/pixel_size)[:2]), dtype=int)
    # mat_full = np.ones((length, width)) * upper[2]
    mat_targ = np.ones((length, width,1)) * upper[2]/pixel_size
    mat_targ = np.array(mat_targ, dtype=np.uint8)
    l, u = pu.get_aabb(scn.body_gripped)
    h = np.abs(int((u[2]-l[2])*256/max_height))
    x0,y0 = np.array(((l-lower)/pixel_size)[:2], dtype=int)
    x1,y1 = np.array(((u-lower)/pixel_size)[:2], dtype=int)
    mat_targ[max(0,x0):x1,max(0,y0):y1] = h

def png_mat(body, mat, lower, upper):
    mat_targ = mat.copy()
    l, u = pu.get_aabb(body)
    h = np.abs(int((u[2]-l[2])*256/max_height))
    x0,y0 = np.array(((l-lower)/pixel_size)[:2], dtype=int)
    x1,y1 = np.array(((u-lower)/pixel_size)[:2], dtype=int)
    mat_targ[max(0,x0):x1,max(0,y0):y1] = h
    return mat_targ

def scene_to_mat(scn, region_aabb, max_height=0.512):
    lower, upper = region_aabb
    pixel_size=0.002
    length, width = np.array(np.abs(((upper-lower)/pixel_size)[:2]), dtype=int)
    # mat_full = np.ones((length, width)) * upper[2]
    mat_targ = np.ones((length, width,1)) * upper[2]/pixel_size
    mat_targ = np.array(mat_targ, dtype=np.uint8)
    l, u = pu.get_aabb(scn.body_gripped)
    h = np.abs(int((u[2]-l[2])*256/max_height))
    x0,y0 = np.array(((l-lower)/pixel_size)[:2], dtype=int)
    x1,y1 = np.array(((u-lower)/pixel_size)[:2], dtype=int)
    mat_targ[max(0,x0):x1,max(0,y0):y1] = h

    mat_full = mat_targ.copy()
    l, u = pu.get_aabb(scn.body_on_table)
    h = np.abs(int((u[2]-l[2])*256/max_height))
    x0,y0 = np.array(((l-lower)/pixel_size)[:2], dtype=int)
    x1,y1 = np.array(((u-lower)/pixel_size)[:2], dtype=int)
    mat_full[max(0,x0):x1, max(y0,0):y1] = h
    return Image.fromarray( np.concatenate((mat_targ,mat_full), axis=2))
    # return mat_full, mat_targ

def save_data(mat_targ, mat_full_list, isfeasible_1b, isfeasible_2b, pic_name, label_name, render_image=False):
    label_name += '.txt'
    for i in range(len(mat_full_list)):
        pic_name_final = f"{pic_name}_{i}"
        if render_image:
            im_gray = Image.fromarray( np.array(np.concatenate((mat_targ[:,:,:1],mat_full_list[i][:,:,:1]), axis=-1), dtype=np.uint8))
            im_depth = Image.fromarray(np.array(np.concatenate((mat_targ[:,:,1:],mat_full_list[i][:,:,1:]), axis=-1)*255, dtype=np.uint8))
            pic_name_gray = f"{pic_name}_{i}_gray.png"
            pic_name_depth = f"{pic_name}_{i}_depth.png"
            im_gray.save(pic_name_gray)
            im_depth.save(pic_name_depth)
        else:
            im = Image.fromarray( np.concatenate((mat_targ,mat_full_list[i]), axis=2))
            im.save(pic_name_final + ".png")

        labels = pic_name_final.split('/')[-1] + ' ' +  ' '.join(list(np.array(isfeasible_1b, dtype=str)))+ ' ' + ' '.join(list(np.array(isfeasible_2b[i], dtype=str)))
        # write label
        with open(label_name, "a") as file:
            # wr = csv.writer(file)
            # wr.writerows(labels)
            file.write(labels)
            file.write('\n')

camera_max_dist = 2.56
def get_render_image():
    # https://github.com/bulletphysics/bullet3/blob/master/examples/pybullet/examples/eglRenderTest.py
    # # for both table and shelf
    # pixelWidth = 540
    # pixelHeight = 720
    # viewMatrix = p.computeViewMatrix(
    #     cameraEyePosition=[1, 0.8, 2],
    #     cameraTargetPosition=[0.3, 0.8, 0],
    #     cameraUpVector=[-1, 0, 0])
    # projectionMatrix = p.computeProjectionMatrixFOV(
    #     fov=45.0,
    #     aspect=0.5,
    #     nearVal=0.1,
    #     farVal=4)

    # only for table
    pixelWidth = 540
    pixelHeight = 540
    viewMatrix = p.computeViewMatrix(
        cameraEyePosition=[0.6, 0.8, 2],
        cameraTargetPosition=[0.2, 0.8, 0],
        cameraUpVector=[-1, 0, 0])
    projectionMatrix = p.computeProjectionMatrixFOV(
        fov=35.0,
        aspect=0.6,
        nearVal=1.5,
        farVal=2.5)

    # start = time.time()
    width, height, rgba_pixels, depth_pixels, seg_pixels = p.getCameraImage(pixelWidth,
                               pixelHeight,
                               viewMatrix=viewMatrix,
                               projectionMatrix=projectionMatrix,
                               shadow=1,
                               lightDirection=[1, 1, 1])
    # print(f"rendering duration: {time.time()-start}")
    # img = Image.fromarray(rgba_pixels)
    # img.show()
    image = np.concatenate((rgba_pixels[:,:,:1], depth_pixels.reshape(pixelHeight, pixelWidth,1)), axis=-1)
    return image

def sample_training_data():
    """
    sampling training data
    generate image of fixed area
    """
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

    if region_ind==0:
        region_str = 'table_dirall_newrep'
    else:
        region_str = 'shelf'
    filename = f'{region_str}/' + str(uuid.uuid1())
    # filename_1b = f'./{region_str}_1b/' + str(uuid.uuid1()) + '.csv'
    
    lower, upper = pu.get_aabb(scn.bd_body['region_table'])
    # sampling_lower = (-0.15,0.7,0.001)
    # sampling_upper = (0.5,1.05,0.002)
    length, width = np.array(np.abs(((upper-lower)/pixel_size)[:2]), dtype=int)
    mat_init = np.array(np.ones((length, width,1)) * upper[2]/pixel_size, dtype=np.uint8)
    # max_sim = 5000
    bar = IncrementalBar('Countdown', max = max_sim)
    for i in range(max_sim):
        bar.next()
        pu.remove_body(scn.body_on_table)
        lwh2, scn.body_on_table = random_box(scn)

        pu.remove_body(scn.body_gripped)
        lwh1, scn.body_gripped = random_box(scn)
        xy1, pose1 = attach_to_table(scn, scn.body_gripped, lwh1, lower, upper)
        # pu.set_pose(scn.body_gripped, ((0.375, 0.9, 0.056012659751127014), (0.0, 0.0, 0.0, 1.0)))
        mat_targ = png_mat(scn.body_gripped, mat_init, lower, upper)
        # mat_targ =  get_render_image()

        saved_world = WorldSaver()
        isfeasible_1b = np.zeros(len(grasp_directions), dtype=np.uint8)

        pose_list = []
        mat_full_list = []
        num_2b = 5
        isfeasible_2b = np.zeros((num_2b, len(grasp_directions)),dtype=np.uint8)
        for dir_ind, grsp_dir in grasp_directions.items():
            # if not dir_ind == 4:
            #     continue
            saved_world.restore()
            pu.set_pose(scn.body_on_table, ((2,2,2),(0,0,0,1)))
            extend = pu.get_aabb_extent(pu.get_aabb(scn.body_gripped))
            target_pose = target_pose_vs_grsp_dir(pose1, grsp_dir, np.array(extend))
            # pu.draw_pose(target_pose)
            # print(target_pose)
            isfeasible_1b[dir_ind] = int(check_feasibility(scn, target_pose))

            for jj in range(num_2b):
                xy2, pose2 = attach_to_table(scn, scn.body_on_table, lwh2, lower, upper)
                pose_list.append(pose2)

            for jj in range(num_2b):
                saved_world.restore()
                pu.set_pose(scn.body_on_table, pose_list[jj])
                if len(mat_full_list)<num_2b:
                    mat_full = png_mat(scn.body_on_table, mat_init, lower, upper)
                    # mat_full = get_render_image()
                    mat_full_list.append(mat_full)
                isfeasible_2b[jj, dir_ind] = isfeasible_1b[dir_ind] and int(check_feasibility(scn, target_pose))
                # if dir_ind==4 and not isfeasible_2b[jj,dir_ind]:
                    # print(isfeasible_2b[jj,dir_ind])
        #         print(f"\nfeasible: \n{isfeasible_1b[dir_ind]} {isfeasible_2b[jj, dir_ind]}")
        # print(f'\n1b feasible:{isfeasible_1b}')
        # print(f'2b feasible:{isfeasible_2b}')
        if not debug:
            save_data(mat_targ, mat_full_list, isfeasible_1b, isfeasible_2b, filename+'_'+str(i).zfill(4), filename, render_image=False)
        # save_dataset(filename_2b, dataset_2b)
        # save_dataset(filename_1b, dataset_1b)
    bar.finish()

def get_body_size_pos_aabb(body):
    aabb = pu.get_aabb(body)
    return pu.get_aabb_extent(aabb), pu.get_point(body), aabb

def sample_training_data_obj_centered():
    """
    sampling training data
    generate image of fixed area
    """
    img_shape = (400,400)
    pixel_size = 0.4/img_shape[0]
    subregions = [((-0.6,0.4),(0.7,1.1)),
                    ((0.2,-0.4),(0.7,0.4)),
                    ((-0.2,-1.0),(0.7,-0.4))]
    regions = np.array(subregions)
    tmp = regions[:,1]-regions[:,0]
    area = tmp[:,0]*tmp[:,1]
    prob_area = area/np.sum(area)
    def random_xy():
        while True:
            i = np.random.randint(len(subregions))
            if prob_area[i]>=np.random.rand(1)[0]:
                break
        lower,upper = subregions[i]
        xy = np.random.uniform(lower, upper)
        # pu.draw_point((xy[0],xy[1],0.1))
        return xy

    def random_targ_box_loc(body):
        l,u = pu.get_aabb(body)
        tl,tu=aabb_table

        xy = random_xy()
        table_z = np.random.uniform(-0.15,0.95)
        # table_z = 0.0055
        box_z = table_z + (u[2]-l[2]+tu[2]-tl[2])/2 + EPSILON
        # box_z = np.random.uniform(-0.1,1)
        xyz = (xy[0],xy[1],box_z)
        dir = np.random.uniform(0,0)

        body_pose = (xyz, pu.quat_from_euler((0,0,dir)))
        pu.set_pose(body, body_pose)
        table_pose = (tuple(np.array(xyz)-np.array([0,0,u[2]-l[2]+tu[2]-tl[2]])/2-EPSILON), (0,0,0,1))
        pu.set_pose(scn.table, table_pose)
        return xyz, dir, body_pose

    def random_neigbor_box(body, region):
        region_size, region_pos, region_aabb = get_body_size_pos_aabb(region)
        xy = list(np.random.uniform(region_aabb[0][:2], region_aabb[1][:2]))
        dir = np.random.uniform(0, 0)
        body_size, body_pos, body_aabb = get_body_size_pos_aabb(body)
        z=region_pos[2]+region_size[2]/2+body_size[2]/2+EPSILON
        xyz = (xy[0],xy[1],z)
        body_pose = (xyz,pu.quat_from_euler((0,0,dir)))
        return xyz, dir, body_pose

    def get_non_zero_pixel_args_old(xmin,xmax,ymin,ymax, dir):
        args = np.array(tuple(itertools.product(range(xmin, xmax), range(ymin,ymax))))
        mid_arg = np.array([(xmin+xmax)/2, (ymin+ymax)/2])
        args = args - mid_arg
        # rotate matrix
        args_rot = np.array([[np.cos(dir),-np.sin(dir)],
                             [np.sin(dir),np.cos(dir)]]) @ args.T
        # delete point out of bound [0,199]
        args_rot = np.clip(np.array(np.round(args_rot.T + mid_arg), dtype=int), 0, img_shape[0]-1)
        return args_rot

    def get_args_inside_bounds(args_rot:list):

        args = []
        for args1 in args_rot[:2]:
            for args2 in args_rot[2:]:
                for a1 in args1:
                    for a2 in args2:
                        if a1[1]==a2[1]:
                            args.extend(list(np.array(tuple(itertools.product(range(a1[0],a2[0]+1),range(a1[1],a1[1]+1))))))
        return np.array(args)

    def get_non_zero_pixel_args_new(xmin,xmax,ymin,ymax,dir):
        mid_arg = np.array([(xmin+xmax)/2, (ymin+ymax)/2])
        args_left = np.array(tuple(itertools.product(range(xmin,xmin+1), range(ymin+1,ymax+1)))) 
        args_right = np.array(tuple(itertools.product(range(xmax,xmax+1), range(ymin,ymax)))) 
        args_top = np.array(tuple(itertools.product(range(xmin+1,xmax+1), range(ymax,ymax+1)))) 
        args_bottom = np.array(tuple(itertools.product(range(xmin,xmax), range(ymin,ymin+1)))) 
        args = [args_left, args_top, args_right, args_bottom]
        # rotate matrix
        rot_matrix = np.array([[np.cos(dir),-np.sin(dir)],
                             [np.sin(dir),np.cos(dir)]])
        args_rot = []
        for i in range(4):
            args[i] = args[i]-mid_arg
            args_rot.append(np.array((rot_matrix @ args[i].T).T, dtype=int))
        args_rot = get_args_inside_bounds(args_rot)

        # delete point out of bound [0,199]
        args_rot = np.clip(np.array(args_rot + mid_arg, dtype=int), 0, img_shape[0]-1)
        return args_rot

    def png_mat_object_centered(body, xyz, dir, body_ref = None):
        xyz = np.array(xyz)
        
        mat_targ = np.zeros(img_shape, dtype=np.uint8)
        
        h = np.abs(int(xyz[2]*256/max_height))
        x_len, y_len = np.array((xyz[:2]/pixel_size/2)[:2])
        mid_arg = (img_shape[0]-1)/2

        if body_ref is None:
            xmin,xmax,ymin,ymax = np.array([mid_arg-x_len,mid_arg+x_len,
                                            mid_arg-y_len,mid_arg+y_len], dtype=int)
        else:
            body_center = np.array(pu.get_point(body))
            center = np.array(pu.get_point(body_ref))
            x0, y0 = np.array(((body_center-center)/pixel_size)[:2])
            xmin,xmax,ymin,ymax = np.array(np.round([mid_arg+x0-x_len, mid_arg+x0+x_len, 
                                        mid_arg+y0-y_len, mid_arg+y0+y_len]), dtype=int)
            xmin,xmax,ymin,ymax = tuple(np.clip([xmin,xmax,ymin,ymax], 0, img_shape[0]-1))
        
        args_rot = get_non_zero_pixel_args_new(xmin, xmax, ymin, ymax,dir)
        mat_targ[args_rot[:,0],args_rot[:,1]] = h

        downsampling=int(mat_targ.shape[0]/200)
        mat_targ = mat_targ[::downsampling,::downsampling]

        assert mat_targ.shape==(200,200)
        return mat_targ.reshape((200,200,1))

    connect(use_gui=visualization)
    scn = TrainingScenario_obj_centered()
    aabb_table = pu.get_aabb(scn.table)
    global grasp_directions
    grasp_directions = {0:(1,0,0),1:(-1,0,0),2:(0,1,0),3:(0,-1,0),4:(0,0,1)}

    (robot_position, _) = pu.get_link_pose(scn.robot, 0)
    robot_pose = (robot_position, (0,0,0,1))

    if region_ind==0:
        region_str = 'table_3d_all_dir_no_rotZ'
    else:
        region_str = 'shelf'
    if not os.path.exists(region_str):
        os.mkdir(region_str)
    else:
        print(f'{region_str} already exists')
    print(f"will save data into: {region_str}")   
    filename = f'{region_str}/' + str(uuid.uuid1())
    # filename_1b = f'./{region_str}_1b/' + str(uuid.uuid1()) + '.csv'
    t0 = time.time()
    bar = IncrementalBar('Countdown', max = max_sim)
    for hhhhhhhhhh in range(max_sim):
        bar.next()
        pu.remove_body(scn.body_on_table)
        lwh2, scn.body_on_table = random_box(scn)

        pu.remove_body(scn.body_gripped)
        lwh1, scn.body_gripped = random_box(scn)
        
        xyz1, dir1, pose1 = random_targ_box_loc(scn.body_gripped)
        mat_targ = png_mat_object_centered(scn.body_gripped, lwh1,dir1)
        saved_world = WorldSaver()
        isfeasible_1b = np.zeros(len(grasp_directions), dtype=np.uint8)

        pose_list = []
        dir_list = []
        mat_full_list = []
        num_2b = 2
        isfeasible_2b = np.zeros((num_2b, len(grasp_directions)),dtype=np.uint8)
        for dir_ind, grsp_dir in grasp_directions.items():
            # if not dir_ind == 4:
            #     continue
            saved_world.restore()
            pu.set_pose(scn.body_on_table, ((2,2,2),(0,0,0,1)))
            extend = pu.get_aabb_extent(pu.get_aabb(scn.body_gripped))
            target_pose = target_pose_vs_grsp_dir(pose1, grsp_dir, np.array(extend))
            # pu.draw_pose(target_pose)   
            # embed()
            # print(target_pose)
            isfeasible_1b[dir_ind] = int(check_feasibility(scn, target_pose))

            for jj in range(num_2b):
                xyz2, dir2, pose2 = random_neigbor_box(scn.body_on_table, scn.table)
                pose_list.append(pose2)
                dir_list.append(dir2)

            for jj in range(num_2b):
                saved_world.restore()
                pu.set_pose(scn.body_on_table, pose_list[jj])
                if len(mat_full_list)<num_2b:
                    mat_full = png_mat_object_centered(scn.body_on_table, lwh2, dir_list[jj], scn.body_gripped)
                    mat_full_list.append(mat_full)
                isfeasible_2b[jj, dir_ind] = isfeasible_1b[dir_ind] and int(check_feasibility(scn, target_pose))
                # print(f"feas: {isfeasible_1b[dir_ind]} {isfeasible_2b[jj, dir_ind]}")
                # embed()

        lower, upper = pu.get_aabb(scn.body_gripped)
        d1_vector = list(np.round(xyz1[:2],4)) + [np.round(lower[2],4)] +list(isfeasible_1b)

        # if not debug:
        for i in range(num_2b):
            tmp = list(np.round(list(lwh1) + list(xyz1) + [dir1] + \
                    list(lwh2) + list(xyz2[:2]) + [dir_list[i]],3)) +\
                    list(isfeasible_1b) + list(isfeasible_2b[i])
            tmp = list(np.array(tmp, dtype=str))
            string = ' '.join(tmp)
            with open(f'{filename}_feat_vec.txt','a') as f:
                f.write(string)
                f.write('\n')

        save_data(mat_targ, mat_full_list, d1_vector, isfeasible_2b, 
                    filename+'_'+str(hhhhhhhhhh).zfill(5), filename, render_image=False)

    bar.finish()
    print(f"duration: {time.time()-t0}")

if __name__=='__main__':
    """ usage
    python3 generate_dataset 0 2 1
    """
    visualization = int(sys.argv[1])
    num_process = int(sys.argv[2])
    region_ind = int(sys.argv[3])
    max_sim = int(sys.argv[4])
    debug = int(sys.argv[5])
    assert num_process<=16, "CPU overworked"
    if debug:
        sample_training_data_obj_centered()
    processes = []
    print(f"start time: {datetime.now()}")
    for _ in range(num_process):
        processes.append(Process(target=sample_training_data_obj_centered, args=()))
        processes[-1].start()
