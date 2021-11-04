import pickle

from numpy.lib.type_check import real
import pybullet_tools.utils as pu
import numpy as np
from codetiming import Timer
import re
from IPython import embed

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

        self.load_ml_model(self.model_file)
        # do some statistics
        self.call_times = 0
        self.feasible_call = 0
        self.infeasible_call = 0
        self.current_feasibility = 0
        self.false_infeasible = 0
        self.false_feasible = 0
        self.true_feasible = 0
        self.true_infeasible = 0

    def fc_statistic(self, real_feasibility):
        if real_feasibility and self.current_feasibility:
            self.true_feasible += 1
        if (not real_feasibility) and (not self.current_feasibility):
            self.true_infeasible += 1
        if (not real_feasibility) and self.current_feasibility:
            self.false_feasible += 1
        if real_feasibility and (not self.current_feasibility):
            self.false_infeasible += 1

    def hypothesis_test(self):
        perc_tf = round(self.true_feasible/self.call_times, 4)
        perc_ti = round(self.true_infeasible/self.call_times, 4)
        perc_fi = round(self.false_infeasible/self.call_times, 4)
        perc_ff = round(self.false_feasible/self.call_times, 4)
        print(f'\t\t\tTrue Feasibility')
        print(f'\t\t\tfeas\t\tinfeas')
        print(f'Classifier\tfeas\t{perc_tf}\t{perc_fi}')
        print(f'Feasibility\tinfeas\t{perc_ff}\t{perc_ti}')

    def load_ml_model(self, model_file):
        with open(model_file, 'rb') as file:
            self.model = pickle.load(file)

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
            feature_vectors.append(tmp)
        return feature_vectors

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
