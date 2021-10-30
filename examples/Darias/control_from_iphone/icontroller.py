from random import random
import time
from build_scenario import ScenarioForiPhoneController
from IPython import embed
import csv, uuid, os, sys   
import threading
import pybullet as p
import pybullet_tools.utils as pu
import pybullet_tools.kuka_primitives3 as pk
import numpy as np

class Sensor(object):
    def __init__(self, filename, init_robot_pose, sampling_rate):
        self.filename = filename
        self.init_tcp_pose = init_robot_pose
        self.sensor_calibration()
        self.sampling_rate = sampling_rate
        self.dt = 1/self.sampling_rate

        self.current_robot_position = self.init_tcp_pose[0]
        self.current_robot_rotation = self.init_tcp_pose[1]
        self.current_lin_vel = 0
        self.delta_position = 0

        self.num_hist_data = 1
        self.sensor_data_queue = np.zeros((self.num_hist_data,9))
        self.sensor_data_queue_pointer = 0
    
    def sensor_data_filter(self, val):
        self.sensor_data_queue[self.sensor_data_queue_pointer%self.num_hist_data] = np.array(val)
        self.sensor_data_queue_pointer += self.sensor_data_queue_pointer+1
        return np.mean(self.sensor_data_queue[:min(self.sensor_data_queue_pointer,self.num_hist_data)], 0)

    def sensor_data_filter_advanced(self):
        pass

    def read_sensor_data(self):
        while True:
            try:
                arr = np.loadtxt(self.filename)
                assert len(arr)==9
                break
            except:
                time.sleep(0.001)
        acc = arr[:3]
        ang_v = arr[3:6]
        rot = arr[6:]
        return acc, ang_v, rot

    def sensor_calibration(self):
        print(f"sensor calibration")
        print(f"wait ...")
        sum_acc, sum_ang_v, sum_rot = np.zeros(3),np.zeros(3),np.zeros(3)
        num_sample = 100
        for _ in range(num_sample):
            tmp_acc, tmp_ang_v, tmp_rot = self.read_sensor_data()
            sum_acc += tmp_acc
            sum_ang_v += tmp_ang_v
            sum_rot += tmp_rot
            time.sleep(0.02)
        self.init_sensor_gravity, self.init_sensor_ang_v, self.init_sensor_rotation = sum_acc/num_sample, sum_ang_v/num_sample, sum_rot/num_sample

    def get_rotation(self):
        d_theta = (self.sensor_rot - self.init_sensor_rotation) * np.pi/180
        ans = pu.multiply_quats(self.init_tcp_pose[1], pu.quat_from_euler([d_theta[1],d_theta[2], d_theta[0]]))
        return ans

    def gravity_to_linearAcc(self):
        init_robot_rotation_mat = pu.matrix_from_quat(self.init_tcp_pose[1])
        current_robot_rotation_mat = pu.matrix_from_quat(self.current_robot_rotation)
        linacc_world = np.linalg.pinv(init_robot_rotation_mat) @ current_robot_rotation_mat @ self.sensor_gravity
        return linacc_world-np.array(self.init_sensor_gravity)

    def get_position(self):
        self.linear_acc = self.gravity_to_linearAcc()
        self.delta_position += self.current_lin_vel * self.dt + 0.5*self.linear_acc*self.dt**2
        self.current_lin_vel += self.linear_acc * self.dt
        return self.init_tcp_pose[0] + self.delta_position

    def get_pose(self):
        tmp_acc, tmp_ang_v, tmp_rot = self.read_sensor_data()
        self.sensor_filter_data = self.sensor_data_filter(list(tmp_acc)+list(tmp_ang_v)+list(tmp_rot))
        self.sensor_gravity, _, self.sensor_rot = self.sensor_filter_data[:3], self.sensor_filter_data[3:6], self.sensor_filter_data[6:]
        self.current_robot_rotation = tuple(self.get_rotation())
        self.current_robot_position = tuple(self.get_position())
        return (self.current_robot_position, self.current_robot_rotation)

class KeyBoardController(object):
    def __init__(self):
        self.pressed_key = []
        # define key dict
        self.key_dict = dict()
        self.key_dict[65295] = 'LEFT'
        self.key_dict[65296] = 'RIGHT'
        self.key_dict[65297] = 'UP'
        self.key_dict[65298] = 'DOWN'
        self.key_dict[65309] = 'ENTER'
        self.key_dict[65306] = 'SHIFT'
        self.key_dict[65307] = 'CTRL'
        self.key_dict[65308] = 'OPTION'

        self.move_linear = [0,0,0]
        self.speed_ratio_list = np.linspace(0.0001,0.01,100)
        self.speed_ratio_ind = 10

    def get_pressed_key(self):
        self.pressed_key=set()
        self.bullet_events = p.getKeyboardEvents()
        key_codes = self.bullet_events.keys()
        for key in key_codes:
            if key != 65308:
                self.pressed_key.add(key)

    def listen_keyboard(self):
        print(f"start listening pybullet keyboard")
        while True:
            x, y, z = 0,0,0
            self.get_pressed_key()
            if len(self.pressed_key)>0:
                # change speed ratio
                if 45 in self.pressed_key:
                    self.speed_ratio_ind = max(0,self.speed_ratio_ind-1)
                    print(f"change speed ratio to: {self.speed_ratio_list[self.speed_ratio_ind]}")
                if 61 in self.pressed_key:
                    self.speed_ratio_ind = min(99,self.speed_ratio_ind+1)
                    print(f"change speed ratio to: {self.speed_ratio_list[self.speed_ratio_ind]}")

                # move linear
                if 65307 in self.pressed_key:
                    if 65297 in self.pressed_key:
                        print(f"move +z")
                        z += 1
                    elif 65298 in self.pressed_key:
                        print(f"move -z")
                        z -= 1
                else:
                    if 65297 in self.pressed_key:
                        print(f"move -y")
                        y -= 1
                    elif 65298 in self.pressed_key:
                        print(f"move +y")
                        y += 1
                    elif 65296 in self.pressed_key:
                        print(f"move -x")
                        x -= 1
                    elif 65295 in self.pressed_key:
                        print(f"move +x")   
                        x += 1

            time.sleep(0.05)    
            self.move_linear =  np.array([x, y, z]) * self.speed_ratio_list[self.speed_ratio_ind]

if __name__=='__main__':
    """ usage
    python3 generate_dataset 0 2
    """
    # load robot from pybullet
    visualization = int(sys.argv[1])
    pu.connect(use_gui=visualization)
    scn = ScenarioForiPhoneController()
    kb_controller = KeyBoardController()
    kb_listener = threading.Thread(target=kb_controller.listen_keyboard, args=())
    kb_listener.start()

    # robot initial pose
    init_robot_pose = pu.get_link_pose(scn.robot, scn.end_effector_link)
    filename = 'sensor_data.txt'
    sensor = Sensor(filename, init_robot_pose, sampling_rate=10)

    # start control
    print(f"################### start robot control ######################")
    t0 = time.time()
    while True:
        # tmp_acc, tmp_ang_v, tmp_rot = read_sensor_data()
        # d_theta = (tmp_rot - init_sensor_rot) * np.pi/180
        # new_position = init_robot_pose[0] 
        new_position = np.array(pu.get_link_pose(scn.robot, scn.end_effector_link)[0]) + kb_controller.move_linear
        _, new_rotation = sensor.get_pose()
        new_pose = (new_position, new_rotation)

        ik=None
        ik = pu.inverse_kinematics_naive(scn.robot, scn.end_effector_link, new_pose)
            # ik = pu.inverse_kinematics_random(scn.robot, scn.end_effector_link, new_pose, \
            #                     obstacles=[], attachments=[], self_collisions=True, disabled_collisions=[], max_distance=0)
        if ik is not None:
            # print(f"ik: {ik}")
            pu.set_joint_positions(scn.robot, scn.movable_joints, ik)
            scn.attachment.assign()
        time.sleep(0.001)
