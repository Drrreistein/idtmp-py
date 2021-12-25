import pickle
from numpy.core.defchararray import array

from numpy.lib.type_check import real
import pybullet_tools.utils as pu
import numpy as np
from codetiming import Timer
import re
from IPython import embed
import tensorflow as tf
from tensorflow.keras import models
import matplotlib.pyplot as plt
import itertools
import time, os
from tmsmt import bind_scene_object
import tamp_utils as tu
import general_utils as gu

from utils.pybullet_tools.utils import dimensions_from_camera_matrix, is_fixed_base

class FeasibilityChecker(object):

    def __init__(self, scn, objects, resolution, model_file):
        self.scn = scn
        self.resolution = resolution
        self.objects = set(objects)
        self.model_file = model_file
        (robot_position, _) = pu.get_link_pose(scn.robots[0], 0)
        self.robot_pose = (robot_position, (0,0,0,1))
        self.object_properties = dict()
        for bd in self.objects:
            lower, upper = pu.get_aabb(bd)
            self.object_properties[bd] = list(np.array(upper)-np.array(lower))

        self.model = self.load_ml_model(self.model_file)
        # do some statistics
        self.call_times = 0
        self.feasible_call = 0
        self.infeasible_call = 0
        self.current_feasibility = 0
        self.false_infeasible = np.zeros(9)
        self.false_feasible = np.zeros(9)
        self.true_feasible = np.zeros(9)
        self.true_infeasible = np.zeros(9)

    def fc_statistic(self, real_feasibility):
        if real_feasibility:
            self.true_feasible += np.array(self.current_feasibility==1, dtype=int)
            self.false_infeasible += np.array(self.current_feasibility==0, dtype=int)
        if not real_feasibility:
            self.true_infeasible += np.array(self.current_feasibility==0, dtype=int)
            self.false_feasible += np.array(self.current_feasibility==1, dtype=int)
        # if real_feasibility and self.current_feasibility:
        #     self.true_feasible += 1
        # if (not real_feasibility) and (not self.current_feasibility):
        #     self.true_infeasible += 1
        # if (not real_feasibility) and self.current_feasibility:
        #     self.false_feasible += 1
        # if real_feasibility and (not self.current_feasibility):
        #     self.false_infeasible += 1

    def hypothesis_test(self):
        call_times = self.true_feasible + self.true_infeasible + self.false_infeasible + self.false_feasible

        perc_tf = np.round(self.true_feasible/call_times, 4)
        perc_ti = np.round(self.true_infeasible/call_times, 4)
        perc_fi = np.round(self.false_infeasible/call_times, 4)
        perc_ff = np.round(self.false_feasible/call_times, 4)
        print(f"true_feasible {self.true_feasible}")
        print(f"true_infeasible {self.true_infeasible}")
        print(f"false_infeasible {self.false_infeasible}")
        print(f"false_feasible {self.false_feasible}")
        print(f"TF {np.round(perc_tf, 5)}")
        print(f"TI {np.round(perc_ti, 5)}")
        print(f"FI {np.round(perc_fi, 5)}")
        print(f"FF {np.round(perc_ff, 5)}")

        # print(f'\t\t\tTrue Feasibility')
        # print(f'\t\t\tfeas\tinfeas')
        # print(f'Classifier\tfeas\t{perc_tf}\t{perc_ff}')
        # print(f'Feasibility\tinfeas\t{perc_fi}\t{perc_ti}')

    def load_ml_model(self, model_file):
        if os.path.isdir(model_file):
            model = models.load_model(model_file)
        else:
            with open(model_file, 'rb') as file:
                model = pickle.load(file)
        return model

    def _get_dist_theta(self, pose1, pose2):
        dist = np.linalg.norm(np.array(pose1[0][:2])-np.array(pose2[0][:2]))
        theta = np.arctan2(pose1[0][0]-pose2[0][0], pose1[0][1]-pose2[0][1])
        return [dist, theta]

    def _get_feature_vector(self, target_body, target_pose):
        feature_vectors = []
        dist_theta1 = self._get_dist_theta(self.robot_pose, target_pose)

        for bd in self.objects-{target_body}:
            bd_pose = pu.get_pose(bd)
            tmp = self.object_properties[target_body] + self.object_properties[bd]
            tmp += dist_theta1
            tmp += self._get_dist_theta(self.robot_pose, bd_pose)
            tmp += self._get_dist_theta(target_pose, bd_pose)
            # tmp += [4]
            feature_vectors.append(tmp)
        return np.array(feature_vectors)

    @Timer(name='feasible_checking_timer', text='')
    def check_feasibility(self, task_plan):
        self.call_times += 1
        failed_step = None
        res = True
        init_world = pu.WorldSaver()
        for step, op in task_plan.items():
            op = op[1:-1]
            op, obj, region, i, j = re.split(' |__', op)
            target_body = self.scn.bd_body[obj]
            if op=='pick-up':
                target_pose = pu.get_pose(target_body)
            elif op=='put-down':
                region_ind = self.scn.bd_body[region]
                aabb = pu.get_aabb(region_ind)
                center_region = pu.get_aabb_center(aabb)
                extend_region = pu.get_aabb_extent(aabb)

                aabb_body = pu.get_aabb(target_body)
                extend_body = pu.get_aabb_extent(aabb_body)

                x = center_region[0] + int(i)*self.resolution
                y = center_region[1] + int(j)*self.resolution
                z = center_region[2] + extend_region[2]/2 + extend_body[2]/2 
                target_pose = ([x,y,z], (0,0,0,1))
            else:
                print("unknown operator: feasible by default")
                continue
            feature_vectors = self._get_feature_vector(target_body, target_pose)
            is_feasible = self.model.predict(feature_vectors)
            if not np.all(is_feasible):
                res = False
                print(f"check feasibility: {step}: {op}: infeasible")
                failed_step = step
                break
            else:
                print(f"check feasibility: {step}: {op}: FEASIBLE")
                if op=='put-down':
                    pu.set_pose(target_body, target_pose)
        init_world.restore()
        if res:
            self.feasible_call += 1
            print("current task plan is feasible")
        else:
            self.infeasible_call += 1
            print("current task plan is infeasible")
        self.current_feasibility = res
        return res, failed_step

    @Timer(name='feasible_checking_simple_timer', text='')
    def check_feasibility_simple(self, target_body, target_pose, grsp_dir):
        self.call_times += 1
        feature_vectors = self._get_feature_vector(target_body, target_pose)

        probs = self.model.predict_proba(feature_vectors)[:,1]
        feas = probs>0.5
        is_feasible = np.all(feas)
        print(f"probs: {probs}, feas: {feas}")
        print(feature_vectors)

        if not np.all(is_feasible):
            res = False
        else:
            res = True

        if res:
            self.feasible_call += 1
            print("current task plan is feasible")
        else:
            self.infeasible_call += 1
            print("current task plan is infeasible")

        return res

class FeasibilityChecker_bookshelf(FeasibilityChecker):
    def __init__(self, scn, objects):

        self.scn = scn
        self.objects = set(objects)

        self.models = dict()
        # self.model_1b_table = None
        # self.model_mb_table = None
        # self.model_1b_shelf = None
        # self.model_mb_shelf = None
        (robot_position, _) = pu.get_link_pose(scn.robots[0], 0)
        self.robot_pose = (robot_position, (0,0,0,1))
        self.object_properties = dict()
        for bd in self.objects:
            lower, upper = pu.get_aabb(bd)
            self.object_properties[bd] = list(np.array(upper)-np.array(lower))

        # do some statistics
        self.call_times = 0
        self.feasible_call = 0
        self.infeasible_call = 0
        self.current_feasibility = 0
        self.false_infeasible = 0
        self.false_feasible = 0
        self.true_feasible = 0
        self.true_infeasible = 0

    def _pose_in_same_region(self, pose1, pose2):
        return np.abs(pose1[0][2]-pose2[0][2])<0.2

    def _get_feature_vector(self, target_body, target_pose, region, grsp_dir):
        
        feature_vectors = []
        dist_theta1 = self._get_dist_theta(self.robot_pose, target_pose)

        for bd in self.objects-{target_body}:
            bd_pose = pu.get_pose(bd)
            if not self._pose_in_same_region(bd_pose, target_pose):
                continue
            tmp = self.object_properties[target_body] + self.object_properties[bd]
            tmp += dist_theta1
            tmp += self._get_dist_theta(self.robot_pose, bd_pose)
            tmp += self._get_dist_theta(target_pose, bd_pose)
            tmp += [grsp_dir]
            feature_vectors.append(tmp)
        if feature_vectors==[]:
            tmp = self.object_properties[target_body] + dist_theta1 + [grsp_dir]
            feature_vectors.append(tmp)
        
        if len(tmp)>10:
            bd_num = 2
        else:
            bd_num = 1

        return feature_vectors, self.models[(region, bd_num)]

    def check_feasibility_simple(self, target_body, target_pose, region, grsp_dir):
        self.call_times += 1
        feature_vectors, model = self._get_feature_vector(target_body, target_pose, region, grsp_dir)
        is_feasible = model.predict(feature_vectors)
        print(f"body: {target_body}, region: {region}, dir: {grsp_dir}, feas: {is_feasible}")
        if not np.all(is_feasible):
            res = False
        else:
            res = True

        if res:
            self.feasible_call += 1
        else:
            self.infeasible_call += 1
        return res

class FeasibilityChecker_MLP(FeasibilityChecker):
    def __init__(self, scn, objects, model_file, threshold=0.5):
        self.scn = scn
        self.objects = set(objects)
        self.model_file = model_file
        self.threshold = threshold

        # self.model_file_1box = model_file_1box
        self.object_properties = dict()
        for bd in self.objects:
            lower, upper = pu.get_aabb(bd)
            self.object_properties[bd] = list(np.array(upper)-np.array(lower))

        self.load_ml_model()

        # do some statistics
        self.call_times = 0
        self.feasible_call = 0
        self.infeasible_call = 0
        self.current_feasibility = 0
        self.false_infeasible = 0
        self.false_feasible = 0
        self.true_feasible = 0
        self.true_infeasible = 0

    def load_ml_model(self):
        self.model = models.load_model(self.model_file)
        # self.model_1box = models.load_model(self.model_file_1box)
    
    def _get_feature_vector(self, target_body, target_pose, grsp_dir):
        def inside_capture_region(xy_center, xy_neig, region_bounds=[0.2,0.2]):
            return np.all(np.abs(np.array(xy_neig[:2])-np.array(xy_center[:2]))<np.array(region_bounds))
        feature_vectors = []
        lwh1 = self.object_properties[target_body]
        xyz1 = list(target_pose[0])
        feat1 = lwh1 + xyz1 + [pu.euler_from_quat(target_pose[1])[2]]

        for bd in self.objects-{target_body}:
            xyz2, rot = pu.get_pose(bd)
            lwh2 = self.object_properties[bd]
            # feat2 = lwh2 + list(xyz2[:2]) + [pu.euler_from_quat(rot)[2]] + [grsp_dir]
            if inside_capture_region(xyz1, xyz2):
                lwh2 = self.object_properties[bd]
                feat2 = lwh2 + list(xyz2[:2]) + [pu.euler_from_quat(rot)[2]] + [grsp_dir]
            else:
                feat2 = [0,0,0,0,0,0, grsp_dir]
            # tmp += [4]
            feature_vectors.append(feat1 + feat2)
        return np.array(feature_vectors)

    def check_feasibility_simple(self, target_body, target_pose, grsp_dir):
        self.call_times += 1
        feature_vectors = self._get_feature_vector(target_body, target_pose, grsp_dir)
        self.len_feature_vector = len(feature_vectors)
        probs = self.model.predict(feature_vectors)
        # probs = []

        # for i in range(len(feature_vectors)):
        #     if len(feature_vectors[i]) == self.model.input_shape[1]:
        #         prob = self.model.predict(feature_vectors[i:i+1])[0,0]
            # elif len(feature_vectors[i]) == self.model_1box.input_shape[1]:
            #     prob = self.model_1box.predict(feature_vectors[i:i+1])[0,0]
            # else:
                # assert False, 'wrong model or feature vector'
        # probs.append(probs)
        # probs = np.array(probs)

        infeasible = probs>=self.threshold
        print(f'prob: {probs[:,0]}, infeasible: {infeasible}')
        # print(f"{feature_vectors}")

        res = np.all(infeasible)
        print(f"body: {self.scn.bd_body[target_body]}, dir: {grsp_dir}, feas: {res}")

        if res:
            self.feasible_call += 1
            print("current task plan is feasible")
        else:
            self.infeasible_call += 1
            print("current task plan is infeasible")
        # self.current_feasibility = res
        self.current_feasibility = np.all(probs>=np.array(range(1,10))*0.1, axis=0)
        return res

class FeasibilityChecker_CNN(FeasibilityChecker_bookshelf):
    def __init__(self, scn, objects, model_file, model_file_1box=None,
    threshold=0.5, obj_centered_img=False):
        self.scn = scn
        self.objects = set(objects)
        self.model_file = model_file
        self.model_file_1box = model_file_1box
        self.obj_centered_img = obj_centered_img
        self._load_cnn_model()
        self.region_bounds = (np.array([-0.2 ,  0.5 ,  0.001]), np.array([0.6 , 1.1  , 0.002]))
        self.max_height = 0.512
        self.pixel_size = 0.002
        self.downsampling_ratio = 10
        self.threshold = threshold

        # do some statistics
        self.call_times = 0
        self.feasible_call = 0
        self.infeasible_call = 0
        self.current_feasibility = 0
        self.false_infeasible = 0
        self.false_feasible = 0
        self.true_feasible = 0
        self.true_infeasible = 0

        self.image_num = 0

        self.start_plot = False

    def _load_cnn_model(self):
        self.model = models.load_model(self.model_file)
        self.model.summary()
        if self.obj_centered_img:
            self.input_shape = self.model.input_shape[0][1:]
        else:
            self.input_shape = self.model.input_shape[1:]
    
    def display_images(self, images, labels, feat=None):
        for i, image in enumerate(images):
            plt.subplot(1,2,1)
            plt.imshow(image[:,:,0], cmap='gray')
            if feat is None:
                plt.title(f"box1: {np.max(image[:,:,0])}")
            else:
                plt.title(f"box1: {np.max(image[:,:,0])} \n feat: {feat[i]}")


            plt.subplot(1,2,2)
            plt.imshow(image[:,:,1]+image[:,:,0], cmap='gray')
            if feat is None:
                plt.title(f"{labels[i]}\nbox2: {np.max(image[:,:,1]-image[:,:,0])}")
            else:
                plt.title(f"{labels[i]}\nbox2: {np.max(image[:,:,1]-image[:,:,0])} \n feat: {feat[i]}")

            plt.savefig(f'images2/image{self.image_num}')
            print(f'images2/image{self.image_num}')
            self.image_num += 1
            # print(f"box1: {np.max(image[:,:,0])}, box2: {np.max(image[:,:,1]-image[:,:,0])}")
            # print(f"feasible: {labels[i]}")
            # time.sleep(1)

    def plot_obj_centered_img(self, images, features, probs):
        if images.max()<1:
            images *= 256
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

        for i, image in enumerate(images):
            xyz =features[i,:-1]
            dir = features[i,-1]
            label = probs[i]
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
            plt.title(f"box_height: {np.round(reshape_image[:,:,0].max()/256*0.512,3)}, \n region location: {np.round(xyz,2)} \n dir: {dir}, prob: {np.round(label,3)}")
            # if dir==0:
            #     plt.text(region_center[1],region_center[0]+target_box_size[0]/2+20,label, fontsize='xx-small', color='w')
            # if dir==1:
            #     plt.text(region_center[1],region_center[0]-target_box_size[0]/2,label, fontsize='xx-small', color='w')
            # if dir==2:
            #     plt.text(region_center[1]+target_box_size[1]/2,region_center[0],label, fontsize='xx-small', color='w')
            # if dir==3:
            #     plt.text(region_center[1]-target_box_size[1]/2-15,region_center[0],label, fontsize='xx-small', color='w')
            # if dir==4:
            #     plt.text(region_center[1],region_center[0],label[4], fontsize='xx-small')

            plt.subplot(2,1,2)
            back_ground = robot_arm_plot.copy()
            back_ground[image_center[0]-image_reso[0]:image_center[0]+image_reso[0],image_center[1]-image_reso[1]:image_center[1]+image_reso[1]] = reshape_image[:,:,0] + reshape_image[:,:,1]
            plt.imshow(back_ground, cmap='gray')
            # plt.title(f"box_height: {np.round(reshape_image[:,:,1].max()/256*0.512,3)}, region location: {np.round(xyz,2)}")
            # if dir==0:
            #     plt.text(region_center[1],region_center[0]+target_box_size[0]/2+20,label, fontsize='xx-small', color='w')
            # if dir==1:
            #     plt.text(region_center[1],region_center[0]-target_box_size[0]/2,label, fontsize='xx-small', color='w')
            # if dir==2:
            #     plt.text(region_center[1]+target_box_size[1]/2,region_center[0],label, fontsize='xx-small', color='w')
            # if dir==3:
            #     plt.text(region_center[1]-target_box_size[1]/2-15,region_center[0],label, fontsize='xx-small', color='w')
            # if dir==4:
            #     plt.text(region_center[1],region_center[0],label, fontsize='xx-small')
            # manager = plt.get_current_fig_manager()
            # manager.full_screen_toggle()
            # plt.show()
            plt.savefig(f'images2/image{self.image_num}')
            plt.close()
            print(f'images2/image{self.image_num}')
            self.image_num += 1

    def _png_mat(self, body, mat, lower, upper):
        mat_targ = mat.copy()
        l, u = pu.get_aabb(body)
        h = np.abs((u[2]-l[2])/self.max_height)
        x0,y0 = np.array(((l-lower)/self.pixel_size)[:2], dtype=int)
        x1,y1 = np.array(((u-lower)/self.pixel_size)[:2], dtype=int)
        mat_targ[max(0,x0):x1,max(0,y0):y1] = h
        return mat_targ

    def _get_images(self, target_body, target_pose):
        init_pose = pu.get_pose(target_body)
        pu.set_pose(target_body, target_pose)
        self.images = []
        length, width = np.array(((self.region_bounds[1]-self.region_bounds[0])/self.pixel_size)[:2], dtype=int)
        mat_init = np.array(np.zeros((length, width,1)) * self.region_bounds[1][2]/self.pixel_size/256, dtype=np.float)
        mat_targ = self._png_mat(target_body, mat_init, self.region_bounds[0], self.region_bounds[1])

        for bd in self.objects-{target_body}:
            # bd_pose = pu.get_pose(bd)
            # if not self._pose_in_same_region(bd_pose, target_pose):
            #     continue
            mat_full = self._png_mat(bd, mat_init, self.region_bounds[0], self.region_bounds[1])
            tmp = np.concatenate((mat_targ, mat_full), axis=2)
            self.downsampling_ratio = int(tmp.shape[0]/self.input_shape[0])
            tmp = tmp[::self.downsampling_ratio, ::self.downsampling_ratio, :]
            assert tmp.shape == self.input_shape, 'shape of the generated image not equals the model input shape'
            self.images.append(tmp)
        self.images = np.array(self.images)
        pu.set_pose(target_body, init_pose)
        return self.images

    def _png_mat_obj_centered(self, body, lwh, dir, body_ref = None):
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

        lwh = np.array(lwh)
        img_shape = (400, 400)
        self.pixel_size = 0.4/img_shape[0]
        mat_targ = np.zeros(img_shape, dtype=np.uint8)
        h = np.abs(int(lwh[2]*256/self.max_height))
        x_len, y_len = np.array((lwh[:2]/self.pixel_size/2)[:2])
        mid_arg = (img_shape[0]-1)/2

        if body_ref is None:
            xmin,xmax,ymin,ymax = np.array([mid_arg - x_len, mid_arg + x_len,
                                            mid_arg - y_len, mid_arg + y_len], dtype=int)
        else:
            body_center = np.array(pu.get_point(body))
            center = np.array(pu.get_point(body_ref))
            x0, y0 = np.array(((body_center-center)/self.pixel_size)[:2])
            xmin,xmax,ymin,ymax = np.array(np.round([mid_arg+x0-x_len, mid_arg+x0+x_len, 
                                        mid_arg+y0-y_len, mid_arg+y0+y_len]), dtype=int)
            xmin,xmax,ymin,ymax = tuple(np.clip([xmin,xmax,ymin,ymax], 0, img_shape[0]-1))

        # args = np.array(tuple(itertools.product(range(xmin, xmax), range(ymin,ymax))))
        # mid_arg = np.array([(xmin+xmax)/2, (ymin+ymax)/2])
        # args = args - mid_arg
        # # rotate matrix
        # args_rot = np.array([[np.cos(dir),-np.sin(dir)],
        #                      [np.sin(dir),np.cos(dir)]]) @ args.T
        # # delete point out of bound [0,199]
        # args_rot = np.clip(np.array(np.round(args_rot.T + mid_arg), dtype=int), 0, img_shape[0]-1)

        if (xmin<=0 and xmax<=0) or (ymin<=0 and ymax<=0) or (xmin>=399 and xmax>=399) or (ymin>=399 and ymax>=399):
            pass
        else:
            args_rot = get_non_zero_pixel_args_new(xmin, xmax, ymin, ymax, dir)
            mat_targ[args_rot[:,0],args_rot[:,1]] = h
        # downsampling=int(mat_targ.shape[0]/img_shape[0])
        downsampling = int(img_shape[0]/100)
        mat_targ = mat_targ[::downsampling,::downsampling]/256
        # assert mat_targ.shape==(200,200)
        return mat_targ.reshape((100,100,1))

    def _get_vertical_placed_box_size(self, box):
        point, rot = pu.get_pose(box)
        pu.set_pose(box, (point, (0,0,0,1)))
        lwh = pu.get_aabb_extent(pu.get_aabb(box))
        pu.set_pose(box, (point, rot))
        return lwh

    def _get_input_obj_centered(self, target_body, target_pose, grsp_dir):
        init_pose = pu.get_pose(target_body)
        pu.set_pose(target_body, target_pose)
        dir_target = pu.euler_from_quat(target_pose[1])[2]
        lwh_targ = self._get_vertical_placed_box_size(target_body)
        l,_ = pu.get_aabb(target_body)
        self.images = []
        
        mat_targ = self._png_mat_obj_centered(target_body,lwh_targ,dir_target)
        for bd in self.objects-{target_body}:
            bd_pose = pu.get_pose(bd)
            if np.abs(pu.get_aabb(bd)[0][2] - l[2])>0.1:
                print(f"two body not in the same height, give up checking feasibility")
                continue
            lwh_bd = self._get_vertical_placed_box_size(bd)
            dir_bd = pu.euler_from_quat(bd_pose[1])[2]

            # mat_neigh = self._png_mat_obj_centered(bd, lwh_bd, dir_bd, target_body)

            tmp_bd_h = bd_pose[0][2]+lwh_bd[2]/2-(target_pose[0][2]-lwh_targ[2]/2)
            if tmp_bd_h>0:
                lwh_bd[2] = tmp_bd_h
                mat_neigh = self._png_mat_obj_centered(bd, lwh_bd, dir_bd, target_body)
            else:
                mat_neigh = mat_targ - mat_targ
            tmp = np.concatenate((mat_targ, mat_neigh), axis=-1)
            assert tmp.shape == self.input_shape, 'shape of the generated image not equals the model input shape'
            self.images.append(tmp)
        if len(self.images) == 0:
            tmp = np.concatenate((mat_targ, mat_targ-mat_targ), axis=-1)
            self.images.append(tmp)

        self.images = np.array(self.images)
        features = np.repeat([[target_pose[0][0], target_pose[0][1], l[2], grsp_dir]], 
                                len(self.images),axis=0)
        pu.set_pose(target_body, init_pose)
        return self.images, features

    def check_feasibility_simple(self, target_body, target_pose, grsp_dir):
        self.call_times += 1
        if self.obj_centered_img:
            # learned model with object centered image
            try:
                image, feat = self._get_input_obj_centered(target_body, target_pose, grsp_dir)
                prob = np.round(self.model.predict([image, feat]), 5)
            except:
                embed()
        else:
            # learned model using image with fixed size
            image = self._get_images(target_body, target_pose)
            prob = np.round(self.model.predict(image),3)
        self.len_feature_vector = len(image)
        labels = prob>=self.threshold
        # self.display_images(image, prob, feat)
        # if not self.start_plot and target_pose[0][0]>0.1:
        #     self.start_plot = True
        # if self.start_plot:
        #     self.plot_obj_centered_img(image, feat, prob[:,0])
        is_feasible = labels[:,-1]
        print(f"body: {target_body}, dir: {grsp_dir}, feas: {prob[:,0]}")

        if not np.all(is_feasible):
            res = False
        else:
            res = True

        if res:
            self.feasible_call += 1
        else:
            self.infeasible_call += 1
        self.current_feasibility = np.all(prob>np.array(range(1,10))*0.1, axis=0)
        return res
