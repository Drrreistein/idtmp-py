from random import random
import time

from PIL.Image import init
from numpy.core.defchararray import array
from build_scenario import ScenarioForiPhoneController
from IPython import embed
import csv, uuid, os, sys   
import threading
import pybullet as p
import pybullet_tools.utils as pu
import pybullet_tools.kuka_primitives3 as pk
import numpy as np
import requests, json
import matplotlib.pyplot as plt

class SensorIPhone(object):
    def __init__(self, url='172.20.10.1'):
        self.url = url
        self.clock = 'full'

        self.linear_acc_full = []
        self.acc_time_full = []
        self.gyro_full = []
        self.gyro_time_full = []

        self.init_gyro = np.zeros(3)
        self.init_acc = np.zeros(3)

        self.linear_vel = np.zeros(3)
        self.linear_position = np.zeros(3)
        self.euler_angle = np.zeros(3)

        self.linear_vel_full = [self.linear_vel]
        self.linear_position_full = [self.linear_position]
        self.euler_angle_full = [self.euler_angle]

    def get_acc(self):
        url = f'http://{self.url}/get?lin_accX={self.clock}|lin_acc_time&lin_acc_time={self.clock}&lin_accY={self.clock}|lin_acc_time&lin_accZ={self.clock}|lin_acc_time'
        r = requests.get(url)
        tmp_dic = r.json()
        r.close()
        accX = tmp_dic['buffer']['lin_accX']['buffer'][-1]
        accY = tmp_dic['buffer']['lin_accY']['buffer'][-1]
        accZ = tmp_dic['buffer']['lin_accZ']['buffer'][-1]
        acc_time = tmp_dic['buffer']['lin_acc_time']['buffer'][-1]
        self.acc_time_full.append(acc_time)
        acc = np.array([accX, accY, accZ])
        self.linear_acc_full.append(acc-self.init_acc)

    def get_gyro(self):
        url = f'http://{self.url}/get?gyroX={self.clock}|gyro_time&gyro_time={self.clock}&gyroY={self.clock}|gyro_time&gyroZ={self.clock}|gyro_time'
        r = requests.get(url)
        tmp_dic = json.loads(r.text)
        r.close()
        gyroX = tmp_dic['buffer']['gyroX']['buffer'][-1]
        gyroY = tmp_dic['buffer']['gyroY']['buffer'][-1]
        gyroZ = tmp_dic['buffer']['gyroZ']['buffer'][-1]
        gyro_time = tmp_dic['buffer']['gyro_time']['buffer'][-1]
        self.gyro_time_full.append(gyro_time)
        gyro = np.array([gyroX, gyroY, gyroZ])
        self.gyro_full.append(gyro-self.init_gyro)

    def calc_euler_angle(self):
        if len(self.gyro_time_full)<2:
            self.euler_angle = np.zeros(3)
        else:
            dt = self.gyro_time_full[-1]-self.gyro_time_full[-2]
            mean_a = (np.array(self.gyro_full[-1]) + np.array(self.gyro_full[-2]))/2
            self.euler_angle += dt * mean_a
        self.euler_angle_full.append(list(self.euler_angle))
        return self.euler_angle

    def calc_linear_vel_pos(self):
        if len(self.acc_time_full)<2:
            self.linear_vel = np.zeros(3)
            self.linear_position = np.zeros(3)
        else:
            dt = self.acc_time_full[-1]-self.acc_time_full[-2]
            mean_a = (np.array(self.linear_acc_full[-1]) + np.array(self.linear_acc_full[-2]))/2
            self.linear_position += self.linear_vel * dt + 0.5 * mean_a * dt**2
            self.linear_vel += dt * mean_a
        self.linear_vel_full.append(list(self.linear_vel))
        self.linear_position_full.append(list(self.linear_position))
        return self.linear_position

    def start_sensing(self):
        while True:
            self.sense_once()
        # except:
        #     print(f"get sensor data failed, retry")
        #     time.sleep(0.001)

    def sense_once(self):
        self.get_acc()
        self.get_gyro()
        self.calc_euler_angle()
        self.calc_linear_vel_pos()

    def sensor_calibration(self, init_robot_pose):
        self.init_robot_pose = init_robot_pose
        print(f"sensor calibration")
        print(f"wait ...")
        sum_acc, sum_ang_v, sum_rot = np.zeros(3),np.zeros(3),np.zeros(3)
        num_sample = 50
        for _ in range(num_sample):
            self.sense_once()

        self.init_position = np.sum(self.linear_position_full[-num_sample:], 0)/num_sample
        self.init_rotation = np.sum(self.euler_angle_full[-num_sample:], 0)/num_sample
        self.init_acc = np.sum(self.linear_acc_full[-num_sample:], 0)/num_sample
        self.init_gyro = np.sum(self.gyro_full[-num_sample:], 0)/num_sample

        self.linear_acc_full = []
        self.acc_time_full = []
        self.gyro_full = []
        self.gyro_time_full = []

        self.linear_vel = np.zeros(3)
        self.linear_position = np.zeros(3)
        self.euler_angle = np.zeros(3)

        self.linear_vel_full = [self.linear_vel]
        self.linear_position_full = [self.linear_position]
        self.euler_angle_full = [self.euler_angle]

    def _get_position(self):
        return tuple(np.array(self.init_robot_pose[0]) + (np.array(self.linear_position_full[-1]) - np.array(self.init_position)))

    def _get_rotation(self):
        d_theta = np.array(self.euler_angle_full[-1]) - np.array(self.init_rotation)
        return tuple(pu.multiply_quats(self.init_robot_pose[1], pu.quat_from_euler([-d_theta[0],d_theta[1], -d_theta[2]])))

    def get_pose(self):
        return (self._get_position(), self._get_rotation())

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
    # TODO, signal filter in real time
    # load robot from pybullet
    pu.connect(use_gui=1)
    scn = ScenarioForiPhoneController()
    # robot initial pose
    init_robot_pose = pu.get_link_pose(scn.robot, scn.end_effector_link)

    # keyboard listener
    kb_controller = KeyBoardController()
    kb_listener = threading.Thread(target=kb_controller.listen_keyboard, args=())
    kb_listener.start()

    # sensor data from an iphone APP, named phyphox
    pp_sensor = SensorIPhone()
    pp_listener = threading.Thread(target=pp_sensor.start_sensing, args=())
    pp_sensor.sensor_calibration(init_robot_pose)
    pp_listener.start()

    # sensor data from matlab synchronization
    # filename = 'sensor_data.txt'
    # sensor = Sensor(filename, init_robot_pose, sampling_rate=10)
    
    # start control
    print(f"################### start robot control ######################")
    t0 = time.time()
    while True:
        # tmp_acc, tmp_ang_v, tmp_rot = read_sensor_data()
        # d_theta = (tmp_rot - init_sensor_rot) * np.pi/180
        # new_position = init_robot_pose[0] 
        new_position = np.array(pu.get_link_pose(scn.robot, scn.end_effector_link)[0]) + kb_controller.move_linear
        # _, new_rotation = sensor.get_pose()
        # new_pose = (new_position, new_rotation)
        new_pose = pp_sensor.get_pose()
        new_pose = (new_position, new_pose[1])
        print(f"new pose {new_pose}")

        ik=None
        ik = pu.inverse_kinematics_naive(scn.robot, scn.end_effector_link, new_pose)
            # ik = pu.inverse_kinematics_random(scn.robot, scn.end_effector_link, new_pose, \
            #                     obstacles=[], attachments=[], self_collisions=True, disabled_collisions=[], max_distance=0)
        if ik is not None:
            # print(f"ik: {ik}")
            pu.set_joint_positions(scn.robot, scn.movable_joints, ik)
            scn.attachment.assign()
        time.sleep(0.01)
