import numpy as np
import itertools
import matplotlib.pyplot as plt
from numpy.core.defchararray import center

def get_args_inside_bounds(args_list:list):
    """
    given: a list of bounds:
        first two: left bounds
        last two: right bounds
    return args of all point inside the bounds
    """
    args=[]
    edge_nums = len(args_list)
    for i in range(edge_nums):
        args1 = args_list[i]
        for j in range(i+1,edge_nums):
            args2 = args_list[j]
            for a1 in args1:
                for a2 in args2:
                    if a1[1] == a2[1]:
                        min_arg = min(a1[0],a2[0])
                        max_arg = max(a1[0],a2[0])
                        args.extend(list(np.array(tuple(itertools.product(range(min_arg, max_arg+1),
                                                                        range(a1[1],a1[1]+1))))))
    return np.array(args)

def get_boundery_args(xmin,xmax,ymin,ymax):
    args_left = np.array(tuple(itertools.product(range(xmin,xmin+1), range(ymin+1,ymax+1)))) 
    args_right = np.array(tuple(itertools.product(range(xmax,xmax+1), range(ymin,ymax)))) 
    args_top = np.array(tuple(itertools.product(range(xmin+1,xmax+1), range(ymax,ymax+1)))) 
    args_bottom = np.array(tuple(itertools.product(range(xmin,xmax), range(ymin,ymin+1)))) 
    args = [args_left, args_top, args_right, args_bottom]
    return args

def rotate_args(args, rotation_angle, center_arg=None, boundery=None):
    """
    """
    if center_arg is None:
        center_arg = np.mean(args, axis=0)
    rot_matrix = np.array([[np.cos(rotation_angle),-np.sin(rotation_angle)],
                            [np.sin(rotation_angle),np.cos(rotation_angle)]])
    args = args - center_arg
    args_rot = np.array((rot_matrix @ args.T).T, dtype=int)
    args_rot = args_rot + center_arg
    if boundery is not None:
        args_rot = np.clip(np.array(args_rot + center_arg, dtype=int), boundery[0], boundery[1])
    return args_rot


def get_non_zero_pixel_args(xmin,xmax,ymin,ymax, dir, center_arg=None, boundery=None):
    """
    given: 
        x,y dimension of axis aligned rectangle
        dir: rotation angle
    return:
        all args of rotated point inside bounds
    """

    if center_arg is None:
        center_arg = np.array([(xmin+xmax)/2, (ymin+ymax)/2])
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
        args[i] = args[i]-center_arg
        args_rot.append(np.array((rot_matrix @ args[i].T).T, dtype=int))
    args_rot = get_args_inside_bounds(args_rot)

    args_rot = np.array(args_rot + center_arg, dtype=int)

    if boundery is not None:
        args_rot = np.clip(np.array(args_rot + center_arg, dtype=int), boundery[0], boundery[1])
    return args_rot

