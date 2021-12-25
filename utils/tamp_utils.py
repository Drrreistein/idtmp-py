import utils.pybullet_tools.utils as pu
import numpy as np
import general_utils as gu
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from IPython import embed

class TextDisplayer(object):
    def __init__(self, textPosition):
        self.textPosition = textPosition
        self.max_lines = 10

    def display_text(text):
        p.addUserDebugText(text)
        pass

    def roll_lines():
        pass

    def remove_all_text():
        pass


def pickup_rotation_from_labels(rotation, direction=(1,0,0)):
    m,n,o = direction
    if o==0:
        goal_rot = pu.multiply_quats(rotation, 
                   pu.quat_from_axis_angle((-n, m, 0), np.pi/2))
    else:
        goal_rot = rotation
    # goal_rot = pu.multiply_quats(rotation, pu.quat_from_axis_angle((-n, m, 0), np.pi/2))
    goal_rot = pu.multiply_quats(goal_rot, pu.quat_from_euler((np.pi, 0, 0)))

    return goal_rot

def plot_obj_centered_img(image, xyz, dir, label):
    if image.max()<1:
        image *= 256
    lower,upper = np.array(((-0.8,-1.2), (0.9,1.3)))
    pixel_size = 0.001
    resolution = np.array((upper - lower)/pixel_size, dtype=int)
    robot_arm_plot = np.ones(resolution) * 10

    # arm base
    origin_point = np.array((np.zeros(2)-lower)/pixel_size, dtype=int)
    arm_base_size = np.array([0.2,0.2])
    arm_base_reso = np.array(arm_base_size/pixel_size/2, dtype=int)
    boundery_arm_base = gu.get_boundery_args(origin_point[0]-arm_base_reso[0],origin_point[0]+arm_base_reso[0], origin_point[1]-arm_base_reso[1], origin_point[1]+arm_base_reso[1])
    boundery_arm_base_rotated = [gu.rotate_args(arg, np.pi/4,center_arg=origin_point) for arg in boundery_arm_base]
    args_arm_base_rotated = gu.get_args_inside_bounds(boundery_arm_base_rotated)
    robot_arm_plot[args_arm_base_rotated[:,0], args_arm_base_rotated[:,1]]=50

    # region 
    image_size = np.array((0.4, 0.4))
    region_center = np.array((np.array(xyz[:2])-lower)/pixel_size, dtype=int)

    # boxes
    shape = image.shape[:2]
    image_center = np.array(region_center, dtype=int)
    image_reso = np.array(image_size/pixel_size/2, dtype=int)
    upsampling = np.array(image_size/pixel_size/np.array(shape), dtype=int)
    reshape_image = image.repeat(upsampling[0], axis=0).repeat(upsampling[1], axis=1)

    _,_,ch = image.shape
    assert ch==2, 'image channels not equal 2'
    # channel 1
    target_box_args = np.argwhere(reshape_image[:,:,0]>0)
    target_box_size = np.max(target_box_args,axis=0)-np.min(target_box_args, axis=0)

    plt.subplot(2,1,1)
    back_ground = robot_arm_plot.copy()
    back_ground[image_center[0]-image_reso[0]:image_center[0]+image_reso[0],image_center[1]-image_reso[1]:image_center[1]+image_reso[1]] = reshape_image[:,:,0]
    plt.imshow(back_ground, cmap='gray')
    plt.title(f"box_height: {np.round(reshape_image[:,:,0].max()/256*0.512,3)}, region location: {np.round(xyz,2)}")
    if dir==0:
        plt.text(region_center[1],region_center[0]+target_box_size[0]/2+20,label, fontsize='xx-small', color='w')
    if dir==1:
        plt.text(region_center[1],region_center[0]-target_box_size[0]/2,label, fontsize='xx-small', color='w')
    if dir==2:
        plt.text(region_center[1]+target_box_size[1]/2,region_center[0],label, fontsize='xx-small', color='w')
    if dir==3:
        plt.text(region_center[1]-target_box_size[1]/2-15,region_center[0],label, fontsize='xx-small', color='w')
    if dir==4:
        plt.text(region_center[1],region_center[0],label[4], fontsize='xx-small')

    plt.subplot(2,1,2)
    back_ground = robot_arm_plot.copy()
    back_ground[image_center[0]-image_reso[0]:image_center[0]+image_reso[0],image_center[1]-image_reso[1]:image_center[1]+image_reso[1]] = reshape_image[:,:,0] + reshape_image[:,:,1]
    plt.imshow(back_ground, cmap='gray')
    plt.title(f"box_height: {np.round(reshape_image[:,:,1].max()/256*0.512,3)}, region location: {np.round(xyz,2)}")
    if dir==0:
        plt.text(region_center[1],region_center[0]+target_box_size[0]/2+20,label, fontsize='xx-small', color='w')
    if dir==1:
        plt.text(region_center[1],region_center[0]-target_box_size[0]/2,label, fontsize='xx-small', color='w')
    if dir==2:
        plt.text(region_center[1]+target_box_size[1]/2,region_center[0],label, fontsize='xx-small', color='w')
    if dir==3:
        plt.text(region_center[1]-target_box_size[1]/2-15,region_center[0],label, fontsize='xx-small', color='w')
    if dir==4:
        plt.text(region_center[1],region_center[0],label, fontsize='xx-small')
    # manager = plt.get_current_fig_manager()
    # manager.full_screen_toggle()
    # plt.show()
    plt.savefig(f'images2/image{self.image_num}')
    print(f'images2/image{self.image_num}')
