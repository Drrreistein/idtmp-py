# from PIL.Image import merge
import itertools
from PIL import Image
from matplotlib import image
import matplotlib.pyplot as plt
import pickle
from itertools import repeat
from numpy.core.fromnumeric import argsort

from numpy.ma.core import choose

from IPython import embed
import csv
import uuid
import os
import sys
import numpy as np
from multiprocessing import Process, Manager, pool, process
import multiprocessing as mps
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.ensemble import BaggingClassifier, RandomForestClassifier
from sklearn.model_selection import cross_val_score
from sklearn.neural_network import MLPClassifier

import tensorflow as tf
from tensorflow.keras import datasets, layers, models
from tensorflow.keras import regularizers
import pandas as pd
from PIL import Image
import random
import time
import sparse

from utils.pybullet_tools.utils import read, spaced_colors

downsampling = 4

class CNN_FilterVisualizer(object):
    def __init__(self, model, input_shape=None) -> None:
        self.model = model
        if input_shape is None:
            self.input_shape = tuple(self.model.input.shape[1:])
        else:
            self.input_shape = input_shape

    def create_image(self):
        return tf.random.uniform(self.input_shape, minval=-0.5, maxval=0.5)

    def plot_image(self, image, title='title'):
        image = image - tf.math.reduce_min(image)
        image = image / tf.math.reduce_max(image)
        m, n, c = image.shape
        if c == 2:
            image_tmp = np.concatenate((image, np.zeros((m, n, 1))), axis=-1)
        else:
            image_tmp = image
        plt.ion()
        fig = plt.figure()
        plt.imshow(image_tmp)
        plt.xticks([])
        plt.yticks([])
        plt.title(title)
        # plt.show()

    def get_submodel(self, layer_name):
        return tf.keras.models.Model(
            self.model.input,
            self.model.get_layer(layer_name).output)

    def visualize(self, layer_name, f_index=None, iters=20, delay=0.001):
        submodel = self.get_submodel(layer_name)
        num_filters = submodel.output.shape[-1]

        if f_index is None:
            f_index = random.randint(0, num_filters-1)
        assert num_filters > f_index, 'f_index is out of bounds'

        image = self.create_image()
        verbose_step = int(iters/5)

        for i in range(0, iters):
            with tf.GradientTape() as tape:
                tape.watch(image)
                out = submodel(tf.expand_dims(image, axis=0))[:, :, :, f_index]
                loss = tf.math.reduce_mean(out)
            grads = tape.gradient(loss, image)
            grads = tf.math.l2_normalize(grads)
            image += grads * 10

            if (i+1) % verbose_step == 0:
                print(f'iteration: {i+1}, loss: {loss.numpy():.4f}')
                self.plot_image(
                    image, f'{layer_name}, layer {f_index}, iter {i+1}')
                time.sleep(delay)

def merge_csv(dirname):
    all_filess = [f for f in os.listdir(dirname) if 'csv' in f]
    dataset = []
    for filename in all_filess:
        with open(os.path.join(dirname, filename), "r") as file:
            reader = csv.reader(file)
            for row in reader:
                dataset.append(row)
    # train_dat_len = int(len(dataset)*0.8)
    # return np.array(dataset[:train_dat_len], dtype=float), np.array(dataset[train_dat_len:], dtype=float)
    return np.array(dataset, dtype=float)

class CustomGen(tf.keras.utils.Sequence):
    def __init__(self, df:list, labels, pick_dir=[4], batch_size = 32, choose_ch2=2, dtype='tensorflow'):
        self.df = df
        self.labels = labels
        self.dtype = dtype
        if self.dtype=='FV_CNN' or self.dtype=='image_edge':
            self.n = len(df[0])
        else:
            self.n = len(df)
        self.batch_size = batch_size
        self.batch_num = self.n // self.batch_size
        self.choose_ch2 = choose_ch2
        if pick_dir == 'all':
            self.pick_dir = [0,1,2,3,4]
        else:
            self.pick_dir = pick_dir
        self.timer = 0

    def __getitem__(self, index):
        m, n = index*self.batch_size, (index+1)*self.batch_size
        if self.dtype=='COO':
            # scipy sparse matrix
            return self.df[m:n].todense(), self.labels[m:n]
        elif self.dtype=='tensorflow':
            # tensorflow sparse matrix
            return tf.sparse.to_dense(tf.sparse.concat(0,self.df[m:n])), self.labels[m:n]

        elif self.dtype=='FV_CNN':
            # for cases using objected center images
            # if self.dir_ind is None:
            #     self.dir_ind = 0
            # else:
            #     self.dir_ind = (self.dir_ind+1)%5
            # self.dir_ind = np.random.randint(0,5)
            # self.dir_ind = 4
            # empty_channel_2  = np.random.randint(2)
            # choose one pick-up/put-down direction
            # print(f'direction index: {self.dir_ind}')
            # feat = np.concatenate((self.df[1][m:n], np.ones((self.batch_size,1))*self.dir_ind), axis=-1)
            # mat1 = tf.sparse.to_dense(tf.sparse.concat(0,self.df[0][m:n])).numpy()/256
            # if empty_channel_2:
            #     input_shape = mat1[:,:,:,:1].shape
            #     mat2 = np.concatenate((mat1[:,:,:,:1],np.zeros(input_shape)),axis=-1)
            #     return [mat2, feat], self.labels[m:n, self.dir_ind]
            # else:
            #     return [mat1, feat], self.labels[m:n, self.dir_ind+5]
            mat1 = tf.sparse.to_dense(tf.sparse.concat(0,self.df[0][m:n])).numpy()
            if np.max(mat1)>1:
                mat1 = mat1/256
            input_shape = mat1[:,:,:,:1].shape
            mat2 = np.concatenate((mat1[:,:,:,:1],np.zeros(input_shape)),axis=-1)
            
            for dir_ind in self.pick_dir:
                tmp_feat = np.concatenate((self.df[1][m:n], np.ones((self.batch_size,1))*dir_ind), axis=-1)
                
                if 'features' not in locals():
                    if self.choose_ch2==2:
                        features = np.concatenate((tmp_feat, tmp_feat), axis=0)
                        # duplicate two-channel images with 2th channel empty, bin-pick target box without neighbor
                        images = np.concatenate((mat1, mat2), axis=0)
                        labels = np.concatenate((self.labels[m:n, dir_ind+5], self.labels[m:n, dir_ind]),axis=0)
                    elif self.choose_ch2==1:
                        features = tmp_feat
                        # duplicate two-channel images with 2th channel empty, bin-pick target box without neighbor
                        images = mat1
                        labels = self.labels[m:n, dir_ind+5]
                else:
                    if self.choose_ch2==2:
                        features = np.concatenate((features, np.concatenate((tmp_feat, tmp_feat), axis=0)), axis=0)
                        # duplicate two-channel images with 2th channel empty, bin-pick target box without neighbor
                        images = np.concatenate((images, np.concatenate((mat1, mat2), axis=0)), axis=0)
                        labels = np.concatenate((labels, np.concatenate((self.labels[m:n, dir_ind+5], self.labels[m:n, dir_ind]),axis=0)), axis=0)
                    elif self.choose_ch2==1:
                        features = np.concatenate((features, tmp_feat), axis=0)
                        # duplicate two-channel images with 2th channel empty, bin-pick target box without neighbor
                        images = np.concatenate((images, mat1), axis=0)
                        labels = np.concatenate((labels, self.labels[m:n, dir_ind+5]), axis=0)
            return [images, features], labels

        elif self.dtype=='image_edge':
            t0=time.time()
            mat1 = np.zeros((n-m,100,100,self.df[0].shape[1]))
            for num in range(n-m):
                for i in range(self.df[0].shape[1]):
                    edges = self.df[0][m+num,i]
                    for edge in edges:
                        xmin,xmax,ymin,ymax,height = edge
                        mat1[num,xmin:xmax+1:,ymin:ymax+1,i] = height/256

            if np.max(mat1)>1:
                mat1 = mat1/256
            input_shape = mat1[:,:,:,:1].shape
            mat2 = np.concatenate((mat1[:,:,:,:1],np.zeros(input_shape)),axis=-1)

            for dir_ind in self.pick_dir:
                tmp_feat = np.concatenate((self.df[1][m:n], np.ones((self.batch_size,1))*dir_ind), axis=-1)
                if 'features' not in locals():
                    if self.choose_ch2==2:
                        features = np.concatenate((tmp_feat, tmp_feat), axis=0)
                        # duplicate two-channel images with 2th channel empty, bin-pick target box without neighbor
                        images = np.concatenate((mat1, mat2), axis=0)
                        labels = np.concatenate((self.labels[m:n, dir_ind+5], self.labels[m:n, dir_ind]),axis=0)
                    elif self.choose_ch2==1:
                        features = tmp_feat
                        # duplicate two-channel images with 2th channel empty, bin-pick target box without neighbor
                        images = mat1
                        labels = self.labels[m:n, dir_ind+5]
                else:
                    if self.choose_ch2==2:
                        features = np.concatenate((features, np.concatenate((tmp_feat, tmp_feat), axis=0)), axis=0)
                        # duplicate two-channel images with 2th channel empty, bin-pick target box without neighbor
                        images = np.concatenate((images, np.concatenate((mat1, mat2), axis=0)), axis=0)
                        labels = np.concatenate((labels, np.concatenate((self.labels[m:n, dir_ind+5], self.labels[m:n, dir_ind]),axis=0)), axis=0)
                    elif self.choose_ch2==1:
                        features = np.concatenate((features, tmp_feat), axis=0)
                        # duplicate two-channel images with 2th channel empty, bin-pick target box without neighbor
                        images = np.concatenate((images, mat1), axis=0)
                        labels = np.concatenate((labels, self.labels[m:n, dir_ind+5]), axis=0)
            self.timer += time.time()-t0
            return [images, features], labels

    def __len__(self):
        return self.batch_num

def get_data(dirname, num=100000, height=120):

    all_files = os.listdir(dirname)
    txt_files = []
    png_names = []
    for f in all_files:
        if 'txt' in f:
            txt_files.append(f)

    dataset = dict()
    for f in txt_files:
        with open(os.path.join(dirname, f), 'r') as file:
            for s in file:
                tmp = s[:-1].split(' ')
                dataset[tmp[0]] = np.array(tmp[1:], dtype=np.uint8)
    for k, v in dataset.items():
        img_mat = np.asarray(Image.open(os.path.join(dirname, k+'.png')))
        m,n,_ = img_mat.shape
        print(f'input shape: {img_mat.shape}')
        break

    i = 1
    downsampling = int(m/height)
    m, n = int(m/downsampling), int(n/downsampling)
    assert m==height, 'image resolution not consistent'
    shape = tuple([1] + [m,n,2])

    filenames = np.array([dict()])
    i=0
    for _ in range(15):
        filenames = np.concatenate((filenames,np.array([dict()])))
    for f in txt_files:
        with open(os.path.join(dirname, f), 'r') as file:
            for s in file:
                tmp = s[:-1].split(' ')
                ind = np.random.randint(16)
                filenames[ind][tmp[0]] = np.array(tmp[1:], dtype=np.uint8)
                i += 1
                if i>num:
                    break
        if i>num:
            break

    images = []
    labels = np.zeros((num, 5))
    for k, v in dataset.items():
        img_mat = np.asarray(Image.open(os.path.join(dirname, k+'.png')))
        img_mat = img_mat - np.array(img_mat<2, dtype=np.uint8)
        images.append(tf.sparse.from_dense(np.reshape(img_mat[::downsampling, ::downsampling, :]/256, shape)))
        # images[i-1] = img_mat[::downsampling, ::downsampling, :]/256
        # labels.append(v[5:])
        labels[i-1] = v[5:]
        if i >= num:
            break
        i += 1

    # images = np.array(images)
    train_data_len = int(len(images)*0.8)
    test_data_len = int(len(images)*0.9)
    # images = tf.sparse.from_dense(images)
    # return images, labels
    train_images, train_labels, validate_images, validate_labels, test_images, test_labels = images[:train_data_len], labels[
        :train_data_len], images[train_data_len:test_data_len], labels[train_data_len:test_data_len], \
            images[test_data_len:], labels[test_data_len:]
    return train_images, train_labels, validate_images, validate_labels, test_images, test_labels

def read_image(dataset:dict, shape, downsampling, labels_ind=-5, scale=256):
    images_labels=dict()
    for k, v in dataset.items():
        # if not np.any(v[-10:]):
        #     continue
        img_mat = np.asarray(Image.open(k+'.png'))
        if np.min(img_mat)>0:
            img_mat = img_mat - np.array(img_mat<2, dtype=np.uint8)
        if scale != 1:
            img_tf = tf.sparse.from_dense(np.reshape(img_mat[::downsampling, ::downsampling, :]/scale, shape))
        else:
            img_tf = tf.sparse.from_dense(np.reshape(img_mat[::downsampling, ::downsampling, :], shape))
        images_labels[img_tf] = v[labels_ind:]
    return images_labels

def read_image_edges(dataset:dict, downsampling=1, labels_ind=-5):
    images = []
    labels = []

    for k, v in dataset.items():
        # if not np.any(v[-10:]):
        #     continue
        img_mat = np.asarray(Image.open(k+'.png'))
        if np.min(img_mat)>0:
            img_mat = img_mat - np.array(img_mat<2, dtype=np.uint8)
        img_resized = img_mat[::downsampling, ::downsampling, :]
        # for AABB, box edge can be expressed as [xmin, xmax, ymin, ymax]
        image_edges = []
        for ch in range(img_resized.shape[2]):
            unq_values = np.unique(img_resized[:,:,ch])
            channel_edge = []
            for val in unq_values:
                if val <5:
                    continue
                args = np.argwhere(img_resized[:,:,ch]==val)
                xmin,ymin = np.min(args, axis=0)
                xmax,ymax = np.max(args, axis=0)
                channel_edge.append((xmin,xmax,ymin,ymax,val))
            image_edges.append(channel_edge)

        # for general expression, we should log all args of the edge pixels
        # image_edges = []
        # for ch in range(img_resized.shape[2]):
        #     unq_values = np.unique(img_resized[:,:,ch])
        #     channel_edge = []
        #     for val in unq_values:
        #         if val <5:
        #             continue
        #         args = np.argwhere(img_resized[:,:,ch]==val)
        #         sort_args = args[np.argsort(args[:,1])]
        #         y0 = sort_args[0,1]
        #         xmin, xmax = sort_args[0,0], sort_args[0,0]
        #         for i,y in enumerate(sort_args[:,1]):
        #             if y0 == y:
        #                 xmin = min(xmin, sort_args[i,0])
        #                 xmax = max(xmax, sort_args[i,0])
        #             else:
        #                 channel_edge.append((xmin,y0))
        #                 channel_edge.append((xmax,y0))
        #                 y0 = y
        #                 xmin, xmax = sort_args[i,0], sort_args[i,0]
        #         channel_edge.append((xmin,y0))
        #         channel_edge.append((xmax,y0))
        #     image_edges.append(channel_edge)

        images.append(image_edges)
        labels.append(v[labels_ind:])
    return images, labels

def repair_images(dirname, image_names:list):
    for image_name in image_names:
        filename = os.path.join(dirname, image_name+'.png')
        img = Image.open(filename)
        pixels = img.load()
        img_mat = np.asarray(img)
        ch = img_mat.shape[2]

        for i in range(ch):
            pixel_val = np.max(img_mat[:,:,i])
            mat = img_mat[:,:,i]
            args = np.argwhere(mat>0)
            for jj in set(args[:,1]):
                old_args_x_in_line = args[:,0][np.argwhere(args[:,1]==jj)]
                x_min = np.min(old_args_x_in_line)
                x_max = np.max(old_args_x_in_line)
                new_args_in_line = itertools.product(range(x_min, x_max+1), 
                                                    range(jj,jj+1))
                for arg in new_args_in_line:
                    if i==0:
                        pixels[int(arg[1]),int(arg[0])] = (pixel_val,pixels[int(arg[1]),int(arg[0])][1])
                    else:
                        pixels[int(arg[1]),int(arg[0])] = (pixels[int(arg[1]),int(arg[0])][0],pixel_val)
            for jj in set(args[:,0]):
                old_args_y_in_line = args[:,1][np.argwhere(args[:,0]==jj)]
                y_min = np.min(old_args_y_in_line)
                y_max = np.max(old_args_y_in_line)
                new_args_in_line = itertools.product(range(jj,jj+1),
                                                    range(y_min, y_max+1))
                for arg in new_args_in_line:
                    if i==0:
                        pixels[int(arg[1]),int(arg[0])] = (pixel_val,pixels[int(arg[1]),int(arg[0])][1])
                    else:
                        pixels[int(arg[1]),int(arg[0])] = (pixels[int(arg[1]),int(arg[0])][0],pixel_val)
        img.save(filename)

def repair_images_multi_proc(dirname, proc_num=16):
    all_files = os.listdir(dirname)
    txt_files = []
    for f in all_files:
        if 'txt' in f and 'feat' not in f:
            txt_files.append(f)

    filenames = []
    for f in txt_files:
        with open(os.path.join(dirname, f), 'r') as file:
            for s in file:
                tmp = s[:-1].split(' ')
                filenames.append(tmp[0])

    processes = []
    num_per_proc = int(np.ceil(len(filenames)/16))
    for i in range(proc_num):
        proc = Process(target = repair_images, 
                args=(dirname, filenames[num_per_proc*i:num_per_proc*(i+1)]))
        proc.start()
        processes.append(proc)

    for proc in processes:
        proc.join()

def get_data_mp(dirnames, num=100000, height=120, batch_size=50):
    txt_files = []
    for dirname in dirnames:
        all_files = os.listdir(dirname)
        for f in all_files:
            if 'txt' in f:
                txt_files.append(f)

    filenames = np.array([dict()])
    i=0
    num_batch = int(np.ceil(num/batch_size))
    for _ in range(num_batch-1):
        filenames = np.concatenate((filenames,np.array([dict()])))
    for f in txt_files:
        with open(f, 'r') as file:
            for s in file:
                tmp = s[:-1].split(' ')
                dirname = f.split('/')[0]
                dir_file = os.path.join(dirname, tmp[0])
                ind = np.random.randint(num_batch)
                filenames[ind][dir_file] = np.array(tmp[1:], dtype=np.uint8)
                i += 1
                if i>=num:
                    break
        if i>=num:
            break

    for k, v in filenames[0].items():
        img_mat = np.asarray(Image.open(k+'.png'))
        m,n,_ = img_mat.shape
        print(f'input shape: {img_mat.shape}')
        break

    i = 1
    downsampling = int(m/height)
    m, n = int(m/downsampling), int(n/downsampling)
    assert m==height, 'image resolution not consistent'
    shape = tuple([1] + [m,n,2])
    
    pool = mps.Pool(processes=16)
    images_labels = dict()
    for i in range(int(np.ceil(num_batch/16))):
        tmp = pool.starmap(read_image, zip(filenames[i*16:i*16+16],repeat(shape),repeat(downsampling)))
        for t in tmp:
            images_labels.update(t)

    images = list(images_labels.keys())
    labels = np.array(list(images_labels.values()))

    train_data_len = int(len(images)*0.8)
    test_data_len = int(len(images)*0.9)
    train_images, train_labels, validate_images, validate_labels, test_images, test_labels = images[:train_data_len], labels[
        :train_data_len], images[train_data_len:test_data_len], labels[train_data_len:test_data_len], \
            images[test_data_len:], labels[test_data_len:]
    return train_images, train_labels, validate_images, validate_labels, test_images, test_labels

def get_render_images(dirname, num=100000, height=270):

    all_files = os.listdir(dirname)
    txt_files = []
    png_names = []
    for f in all_files:
        if 'txt' in f:
            txt_files.append(f)

    dataset = dict()
    for f in txt_files:
        with open(os.path.join(dirname, f), 'r') as file:
            for s in file:
                tmp = s[:-1].split(' ')
                dataset[tmp[0]] = np.array(tmp[1:], dtype=np.uint8)

    for k, v in dataset.items():
        img_mat = np.asarray(Image.open(os.path.join(dirname, k+'_depth.png')))
        m,n,_ = img_mat.shape
        print(f'input shape: {img_mat.shape}')
        break

    i = 1
    downsampling = int(m/height)
    m, n = int(m/downsampling), int(n/downsampling)
    assert m==height, 'image resolution not consistent'
    shape = tuple([1] + [m,n,2])
    labels = np.zeros((num, 5))
    images = np.zeros((num,m,n,4))
    for k, v in dataset.items():
        img_gray = np.asarray(Image.open(os.path.join(dirname, k+'_gray.png')))[::downsampling, ::downsampling, :]/256
        img_depth = np.asarray(Image.open(os.path.join(dirname, k+'_depth.png')))[::downsampling, ::downsampling, :]/256
        img_mat = np.concatenate((img_gray, img_depth), axis=-1)
        # images.append(tf.sparse.from_dense(np.reshape(img_mat[::downsampling, ::downsampling, :]/256, shape)))
        images[i-1] = img_mat
        # labels.append(v[5:])
        labels[i-1] = v[5:]
        if i >= num:
            break
        i += 1
    # images = np.array(images)
    train_data_len = int(len(images)*0.8)
    test_data_len = int(len(images)*0.9)
    # images = tf.sparse.from_dense(images)
    # return images, labels
    train_images, train_labels, validate_images, validate_labels, test_images, test_labels = images[:train_data_len], labels[
        :train_data_len], images[train_data_len:test_data_len], labels[train_data_len:test_data_len], \
            images[test_data_len:], labels[test_data_len:]
    return train_images, train_labels, validate_images, validate_labels, test_images, test_labels

def plot_image_by_channels(images):
    
    _,_,ch = images.shape
    row=2
    col = int(np.ceil(ch/row))
    plt.ion()
    ind = 0
    for i in range(col):
        for j in range(row):
            plt.subplot(col, row, i*row+j+1)
            plt.imshow(images[:,:,ind], cmap='gray')
            ind += 1

def get_balance_data(X,y):
    unique_y = np.unique(y)
    min_num_y = len(np.argwhere(y==unique_y[0]))
    min_y = unique_y[0]
    y_num = dict()
    for hh in unique_y:
        y_num[hh] = len(np.argwhere(y==hh))
    min_num_y = np.min(list(y_num.values()))
    for hh,num in y_num.items():
        step = int(num/min_num_y)
        if 'args' not in locals():
            args = np.argwhere(y==hh)[::step]
        else:
            args = np.concatenate([args, np.argwhere(y==hh)[::step]], axis=0)
    args = args[:,0]
    tmp_X = X[args]
    tmp_y = y[args]

    return tmp_X, tmp_y, args

def get_data0(dirnames, num=100000, height=120):
    txt_files = []
    for dirname in dirnames:
        all_files = os.listdir(dirname)
        for f in all_files:
            if 'txt' in f:
                txt_files.append(os.path.join(dirname, f))

    dataset = dict()
    for f in txt_files:
        with open(f, 'r') as file:
            for s in file:
                dirname = f.split('/')[0]
                tmp = s[:-1].split(' ')
                dir_file = os.path.join(dirname, tmp[0])
                dataset[dir_file] = np.array(tmp[1:], dtype=np.uint8)
    for k, v in dataset.items():
        img_mat = np.asarray(Image.open(k +'.png'))
        m,n,_ = img_mat.shape
        print(f'input shape: {img_mat.shape}')
        break

    i = 1
    downsampling = int(m/height)
    m, n = int(m/downsampling), int(n/downsampling)
    assert m==height, 'image resolution not consistent'
    # images = []
    images = np.zeros((num, m,n,2))
    shape = tuple([1] + [m,n,2])
    labels = np.zeros((num, 5))
    for k, v in dataset.items():
        img_mat = np.asarray(Image.open(k+'.png'))
        # images.append(tf.sparse.from_dense(np.reshape(img_mat[::downsampling, ::downsampling, :]/256, shape)))
        images[i-1] = img_mat[::downsampling, ::downsampling, :]/256
        labels[i-1] = v[5:]
        if i >= num:
            break
        i += 1

    # images = np.array(images)
    train_data_len = int(len(images)*0.8)
    test_data_len = int(len(images)*0.9)
    # images = tf.sparse.from_dense(images)
    # return images, labels
    train_images, train_labels, validate_images, validate_labels, test_images, test_labels = images[:train_data_len], labels[
        :train_data_len], images[train_data_len:test_data_len], labels[train_data_len:test_data_len], \
            images[test_data_len:], labels[test_data_len:]
    return train_images, train_labels, validate_images, validate_labels, test_images, test_labels

def plot_images(dirname, num):
    all_files = os.listdir(dirname)
    txt_files = []
    for f in all_files:
        if 'txt' in f:
            txt_files.append(f)

    dataset = dict()
    for f in txt_files:
        with open(os.path.join(dirname, f), 'r') as file:
            for s in file:
                tmp = s[:-1].split(' ')
                dataset[tmp[0]] = np.array(tmp[1:], dtype=np.uint8)

    for i in range(num):
        k = np.random.choice(list(dataset.keys()))
        v = dataset[k]
        img_mat = np.asarray(Image.open(os.path.join(dirname, k+'.png')))/256*0.512
        plt.subplot(1,2,1)
        plt.title(f'feasible: {v[:5]}\nbox1: {np.max(img_mat[:,:,0])}')
        plt.imshow(img_mat[:,:,0], cmap='gray')
        plt.subplot(1,2,2)
        plt.title(f'feasible: {v[:5]}\nbox2: {np.max(img_mat[:,:,1]-img_mat[:,:,0])}')
        plt.imshow(img_mat[:,:,1], cmap='gray')
        plt.show()

def png_to_vector(images):
    vectors = []
    for i in range(len(images)):
        size_box1 = []
        size_box2 = None
        pos_box1 = None
        pos_box2 = None
        tmp = size_box1 + size_box2 + pos_box1 + pos_box2
        vectors.append(np.array(tmp))
    return np.array(vectors)

def create_model(input_shape):
    model = models.Sequential()
    model.add(layers.Conv2D(4, (3,3), padding='same',input_shape=input_shape))
    # model.add(layers.BatchNormalization(axis=-1))
    # model.add(layers.Dropout(0.2))
    model.add(layers.ReLU())
    model.add(layers.MaxPooling2D(pool_size=(4, 4)))

    model.add(layers.Conv2D(8, (3,3),padding='same'))
    # model.add(layers.BatchNormalization(axis=-1))
    # model.add(layers.Dropout(0.2))
    model.add(layers.ReLU())
    model.add(layers.MaxPooling2D(pool_size=(4, 4)))

    model.add(layers.Conv2D(16, (3, 3),padding='same'))
    # model.add(layers.Dropout(0.2))
    # model.add(layers.BatchNormalization(axis=-1))
    model.add(layers.ReLU())
    model.add(layers.MaxPooling2D(pool_size=(4, 4)))

    model.add(layers.Flatten())
    # model.add(layers.Dropout(0.2))
    model.add(layers.Dense(200, activation='relu'))
    model.add(layers.Dense(5, activation='sigmoid'))
    model.summary()
    return model

def get_data_mp_obj_centered(dirnames, nums=[100000, 10000], height=200, batch_size=50, proc_num=16, sparse_data=True):
    all_txt_files = []
    for dirname in dirnames:
        txt_files = []
        all_files = os.listdir(dirname)
        for f in all_files:
            if 'txt' in f and 'feat' not in f:
                txt_files.append(f)
        all_txt_files.append(txt_files)

    filenames = np.array([dict()])
    num_batch = int(np.ceil(np.sum(nums)/batch_size))
    for _ in range(num_batch-1):
        filenames = np.concatenate((filenames,np.array([dict()])))

    for mm, dirname in enumerate(dirnames):
        txt_files = all_txt_files[mm]
        i=0
        for f in txt_files:
            with open(os.path.join(dirname, f), 'r') as file:
                for s in file:
                    tmp =  s[:-1].split(' ')
                    # dirname = f.split('/')[0]
                    dir_file = os.path.join(dirname, tmp[0])
                    ind = np.random.randint(num_batch)
                    filenames[ind][dir_file] = np.array(tmp[1:], dtype=float)
                    i += 1
                    if i>=nums[mm]:
                        break
            if i>=nums[mm]:
                break

    for k, v in filenames[0].items():
        img_mat = np.asarray(Image.open(k+'.png'))
        m, n, _ = img_mat.shape
        print(f'input shape: {img_mat.shape}')
        break

    i = 1
    downsampling = int(m/height)
    m, n = int(m/downsampling), int(n/downsampling)
    assert m == height, 'image resolution not consistent'
    shape = tuple([1] + [m,n,2])
    
    if sparse_data:
        pool = mps.Pool(processes=proc_num)
        images_labels = dict()
        for i in range(int(np.ceil(num_batch/proc_num))):
            tmp = pool.starmap(read_image, zip(filenames[i*proc_num:i*proc_num+proc_num],repeat(shape),repeat(downsampling), repeat(0), repeat(1)))
            for t in tmp:
                images_labels.update(t)
        images = list(images_labels.keys())
        labels = np.array(list(images_labels.values()))
    else:
        pool = mps.Pool(processes=proc_num)
        images = []
        labels = []
        for i in range(int(np.ceil(num_batch/proc_num))):
            tmp = pool.starmap(read_image_edges, zip(filenames[i*proc_num:i*proc_num+proc_num],repeat(downsampling), repeat(0)))
            for t in tmp:
                images.extend(t[0])
                labels.extend(t[1])
        images = np.array(images)
        labels = np.array(labels)

    train_data_len = int(len(images)*0.8)
    test_data_len = int(len(images)*0.9)

    feat_num = 3
    features = labels[:,:feat_num]
    labels = np.array(labels[:,feat_num:], dtype=int)
    train_input = [images[:train_data_len],features[:train_data_len]]
    validate_input = [images[train_data_len:test_data_len], features[train_data_len:test_data_len]]
    test_input = [images[test_data_len:], features[test_data_len:]]
    train_labels = labels[:train_data_len]
    validate_labels = labels[train_data_len:test_data_len]
    test_labels = labels[test_data_len:]
    return train_input, train_labels, validate_input, validate_labels, test_input, test_labels

def create_model_obj_centered(input_shape, filters=[4,8,16], filter_size=(4,4),pool_size=(2,2), stride=1, dense_layers=[200], dropout=0.2):

    inputs = layers.Input(shape=input_shape)
    inputs2 = layers.Input(shape=4)
    for i,f in enumerate(filters):
        if i==0:
            x = inputs
        x = layers.Conv2D(f,filter_size,padding='same', strides=stride)(x)
        x = layers.Activation('relu')(x)
        # x = layers.BatchNormalization(axis=-1)(x)
        x = layers.MaxPooling2D(pool_size=pool_size)(x)

    x = layers.concatenate([layers.Flatten()(x), inputs2])
    x = layers.Dropout(dropout)(x)
    for i,num in enumerate(dense_layers):
        x = layers.Dense(num, activation='relu')(x)
        x = layers.Dropout(dropout)(x)
    x = layers.Dense(1, activation='sigmoid')(x)

    model = models.Model([inputs, inputs2], x)
    model.summary()
    return model

def freeze_conv2d(model):
    for layer in model.layers:
        if 'conv2d' in layer.name:
            layer.trainable = False
    model.summary()
    return model

def train_cnn_model(train_images, train_labels, validate_images, validate_labels, test_images, test_labels):
    threshold = 0.5
    def overall_accuracy(y_true, y_pred):
        m, n = y_pred.shape
        try:
            ans = int(tf.reduce_sum(tf.cast(tf.equal(tf.cast(y_pred >= threshold, dtype=tf.uint8), tf.cast(
                y_true, dtype=tf.uint8)), dtype=tf.uint8))) / m / n
        except:
            pass
        return ans

    # Create a new model instance
    input_shape = tuple(np.array(train_images[0].shape[1:]))
    # input_shape = train_images[0].shape
    model = create_model(input_shape)
    model.compile(
        optimizer = tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss = tf.keras.losses.MeanSquaredError())

    train_images, train_labels, validate_images, validate_labels, test_images,\
    test_labels = get_data_mp('table_dirall_newrep/', 160000, 200)

    train_gen = CustomGen(train_images, train_labels, batch_size=100)
    valid_gen = CustomGen(validate_images, validate_labels, batch_size=100)
    test_gen = CustomGen(test_images, test_labels, batch_size=100)
    for i in range(10):
        history = model.fit(train_gen, epochs=1, batch_size = train_gen.batch_size, validation_data=valid_gen)
    # evaluate_cnn_model(model, test_images[test_data_len:], test_labels[test_data_len:], norm=True)
        model.save(f"cnn_200_150_21945_dir4_new_rep{i*2+2}.model")
        evaluate_cnn_model(model, test_gen=test_gen, norm=True)
        evaluate_cnn_model(model, test_gen=train_gen, norm=True)

    train_input, train_labels, validate_input, validate_labels, test_input,\
    test_labels = get_data_mp_obj_centered(['table_3d2/'], [100000], 100, 50, 16)

    train_gen = CustomGen(train_input, train_labels, pick_dir=[4], choose_ch2=2,batch_size=10, dtype='FV_CNN')
    valid_gen = CustomGen(validate_input, validate_labels, pick_dir=[4], choose_ch2=2,batch_size=10, dtype='FV_CNN')
    test_gen = CustomGen(test_input, test_labels, pick_dir=[4], choose_ch2=2,batch_size=10, dtype='FV_CNN')

    input_shape = tuple(np.array(train_input[0][0].shape[1:]))
    model = create_model_obj_centered(input_shape, filters=[8,16], filter_size=(5,5),pool_size=(2,2), stride=2, dense_layers=[100], dropout=0.)
    model.compile(
        optimizer = tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss = tf.keras.losses.BinaryCrossentropy(from_logits=False),
        # loss=tf.keras.losses.MeanSquaredError(),
        metrics = [
                 'accuracy', 
                #  tf.keras.metrics.Accuracy(name='acc'),
                #  tf.keras.metrics.FalseNegatives(name='FN', dtype=tf.int32),
                #  tf.keras.metrics.FalsePositives(name='FP', dtype=tf.int32),
                 tf.keras.metrics.Precision(name='P'),
                 tf.keras.metrics.Recall(name='R'),
                #  tf.keras.metrics.AUC(num_thresholds=100, curve='PR', name='AUC_PR'),
                #  tf.keras.metrics.AUC(num_thresholds=100, curve='ROC',name='AUC_ROC')
                #  'recall', 
                #  'precision',
                 ])

    # objected centered model
    epochs=2
    for i in range(10):
        # for jj in range(train_gen.batch_num):
        #     tmp_input, tmp_labels = train_gen.__getitem__(jj)
        #     model.fit(tmp_input, tmp_labels,epochs=1, batch_size = train_gen.batch_size)
        train_batch_size = train_gen.batch_size * len(train_gen.pick_dir) * train_gen.choose_ch2
        history = model.fit(train_gen, epochs=epochs, batch_size = train_batch_size, validation_data=valid_gen)
    # evaluate_cnn_model(model, test_images[test_data_len:], test_labels[test_data_len:], norm=True)
        model.save(f"cnn_fv_100_100_31597_dir4_{(i+1)*epochs}.model")


        # # TODO
        # evaluate_cnn_model(model, test_gen=test_gen, norm=True)
        # evaluate_cnn_model(model, test_gen=train_gen, norm=True)

    # # The history.history["loss"] entry is a dictionary with as many values as epochs that the
    # # model was trained on.
    # df_loss_acc = pd.DataFrame(history.history)
    # df_loss = df_loss_acc[['loss', 'val_loss']]
    # df_loss.rename(
    #     columns={'loss': 'train', 'val_loss': 'validation'}, inplace=True)
    # # df_acc= df_loss_acc[['accuracy','val_accuracy']]
    # # df_acc.rename(columns={'accuracy':'train','val_accuracy':'validation'},inplace=True)
    # df_loss.plot(title='Model loss', figsize=(12, 8)).set(
    #     xlabel='Epoch', ylabel='Loss')
    # # df_acc.plot(title='Model Accuracy',figsize=(12,8)).set(xlabel='Epoch',ylabel='Accuracy')

    return model

def evaluate_cnn_model(model, test_images=None, test_labels=None, test_gen=None, threshold=0.5, norm=False):
    if test_gen is not None:
        test_labels_pred = np.array(model.predict(test_gen) > threshold, dtype=int)
        m,n = test_labels_pred.shape
        test_labels = np.array(test_gen.labels[:m], dtype=np.uint)
    else:
        test_labels_pred = np.array(model.predict(test_images) > threshold, dtype=int)
        test_labels = np.array(test_labels, dtype=int)
    assert test_labels_pred.shape == test_labels.shape, 'labels shape not consistent'
    m,n = test_labels_pred.shape
    overall_acc = np.sum(test_labels == test_labels_pred) / (m*n)
    direction_acc = np.sum(test_labels == test_labels_pred, axis=0) / m
    print('################ binary accuracy ################')
    print(f'positive threshold: {threshold}')
    print(f"overall accuracy: {overall_acc}")
    directions = ['right', 'left', 'front', 'back', 'top']
    if len(test_labels[0])==5:
        for i in range(5):
            print(f"accuracy in {directions[i]} direction: {direction_acc[i]}")

    false_args = np.argwhere(test_labels_pred != test_labels)
    true_args = np.argwhere(test_labels_pred == test_labels)
    false_neg_num = 0
    false_pos_num = 0
    true_neg_num = 0
    true_pos_num = 0
    for arg in false_args:
        if test_labels_pred[tuple(arg)]:
            false_pos_num += 1
        else:
            false_neg_num += 1
    for arg in true_args:
        if test_labels_pred[tuple(arg)]:
            true_pos_num += 1
        else:
            true_neg_num += 1

    print(f"true positive: {true_pos_num/m/n}")
    print(f"true negative: {true_neg_num/m/n}")
    print(f"false negative: {false_neg_num/m/n}")
    print(f"false positive: {false_pos_num/m/n}")

    R = true_pos_num/(true_pos_num+false_neg_num)
    P = true_pos_num/(true_pos_num+false_pos_num)
    f_score = 2*P*R/(P+R)
    print(f"recall rate: {R}")
    print(f"precision rate: {P}")
    print(f"f_score: {f_score}")

    return 

def test_accuracy(test_y, real_y):
    num_test = len(real_y)
    res = {"true_pos": 0, "true_neg": 0, "false_pos": 0, "false_neg": 0}

    for i in range(num_test):
        if test_y[i] == real_y[i]:
            if test_y[i] == 1:
                res["true_pos"] += 1
            else:
                res["true_neg"] += 1
        else:
            if test_y[i] == 1:
                res["false_pos"] += 1
            else:
                res["false_neg"] += 1
    for key, val in res.items():
        res[key] = val/num_test
    return res
















###################################### feature based model #################################


def train_svm_model(train, test):

    # clf = make_pipeline(StandardScaler(), SVC(C=30000, gamma=0.1))
    clf = SVC(C=30000, gamma=0.1)
    clf.fit(train[:, :-1], train[:, -1])
    clf.score(test[:, :-1], test[:, -1])

    # bagging
    clf = BaggingClassifier(SVC(C=3000,gamma=0.1, kernel='rbf'), n_jobs=15)
    clf.fit(train[:, :-1], train[:, -1])
    clf.score(test[:, :-1], test[:, -1])

    prob_train = clf.predict_proba(train[:, :-1], train[:, -1])
    prob_test = clf.predict_proba(train[:, :-1], train[:, -1])
    pred_train_y = prob_train > 0.5
    pred_test_y = prob_test > 0.5
    # acc_test = clf.score(test[:, :-1], test[:, -1])
    # acc_train = clf.score(train[:, :-1], train[:, -1])
    
    # pred_train_y = clf.predict(train[:, :-1])
    # pred_test_y = clf.predict(test[:, :-1])
    # print(f"acc_train: {acc_train}, acc_test: {acc_test}")
    print(f"real_pos_train: {sum(train[:,-1])/len(train)}")
    print(f"real_pos_test: {sum(test[:,-1])/len(test)}")
    print(test_accuracy(pred_train_y, train[:, -1]))
    print(test_accuracy(pred_test_y, test[:, -1]))

    with open("./svm_model.pk", 'wb') as file:
        pickle.dump(clf, file)

    return clf

def predict_proba(model, data_x, threshold=0.5):
    """ bigger threshold, smaller false negative
    """
    return model.predict_proba(data_x)[:, 0] < threshold

def plot_statistic(model, data):
    """
    plot recall and precision
    """

    thres = []
    recalls = []
    precisions = []
    overall_acc = []
    false_neg = []
    f_scores = []
    for i in 0.1+np.array(range(10))*0.1:
        thres.append(i)
        pred_y = predict_proba(model, data[:, :-1], threshold=i)
        test_res = test_accuracy(pred_y, data[:, -1])
        overall_acc.append(sum(pred_y == data[:, -1])/len(pred_y))
        false_neg.append(test_res['false_neg'])
        R = test_res['true_pos']/(test_res['true_pos']+test_res['false_neg'])
        P = test_res['true_pos']/(test_res['true_pos']+test_res['false_pos'])
        f_score = 2*P*R/(P+R)
        recalls.append(R)
        precisions.append(P)
        f_scores.append(f_score)

    plt.subplot(1, 2, 1)
    plt.plot(thres, 90*np.array([false_neg, overall_acc]).T)
    plt.xlim([0.1, 0.9])
    plt.ylabel('percentage [%]')
    plt.xlabel('feasibility confidence threshold')
    plt.legend(['false negative', 'overall accuracy'])
    plt.subplot(1, 2, 2)
    plt.plot(thres, 90*np.array([recalls, precisions, f_scores]).T)
    plt.xlim([0.1, 0.9])
    plt.ylabel('percentage [%]')
    plt.xlabel('feasibility confidence threshold')
    plt.legend(['recall', 'precision', 'f_score'])
    plt.show()

def create_tf_mlp_model(input_shape, neurons=[20,20,20], dropout=0):
    """
    create TF MLP model
    """
    input = layers.Input(shape=input_shape)

    for i in range(len(neurons)):
        if i==0:
            x = tf.keras.layers.Dense(neurons[i], activation='relu')(input)
        else:
            x = tf.keras.layers.Dense(neurons[i], activation='relu')(x)
        x = tf.keras.layers.Dropout(dropout)(x)
    x = layers.Dense(1, activation='sigmoid')(x)
    model = models.Model(input, x)
    model.summary()
    return model

def get_data_for_mlp(dirname, data_num='all',pick_dir=[4], duplicate=2, balance_dataset=False):
    for f in os.listdir(dirname):
        if 'feat_vec' in f:
            if 'dataset' not in locals():
                dataset = np.loadtxt(os.path.join(dirname,f))
            else:
                dataset = np.concatenate((dataset,np.loadtxt(os.path.join(dirname,f))), axis=0)
    if not data_num=='all':
        dataset = dataset[:data_num]
    if balance_dataset:
        args = np.argwhere(np.any(dataset[:,13:]==1, axis=1))
        dataset = dataset[args[:,0]]

    features = dataset[:,:13]
    labels = dataset[:,13:]
    if pick_dir=='all':
        pick_dir = [0,1,2,3,4]

    if duplicate==2:
        pick_dir_list = []
        for direction in pick_dir:
            pick_dir_list.extend([direction, direction+5])
    elif duplicate == 1:
        pick_dir_list = [direction+5 for direction in pick_dir]

    repeat_num  = len(pick_dir) * duplicate
    feature_dir = np.repeat([pick_dir], features.shape[0], \
                        axis=0).reshape((len(pick_dir)*features.shape[0],1))
    feature_dir = np.repeat(feature_dir, duplicate, axis=0)

    features = np.repeat(features, repeat_num, axis=0)
    labels = labels[:,pick_dir_list].reshape((features.shape[0],1))

    assert features.shape[0] == feature_dir.shape[0], "feature size not consistent"
    assert features.shape[0] == labels.shape[0], "feature size not consistent"

    if duplicate==2:
        features[::2,7:]=0
    features = np.concatenate((features, feature_dir), axis=-1)
    train_len = int(len(features)*0.8)
    dataset_processed = np.concatenate((features, labels), axis=-1)
    # return features[:train_len], labels[:train_len], features[train_len:], labels[train_len:]
    return dataset_processed[:train_len], dataset_processed[train_len:]

def train_tf_mlp_model():
    # train_x, train_y, test_x, test_y = get_data_for_mlp('table_3d_')
    # train_x_1box, test_x_1box = train_x[:,:7], test_x[:,:7]
    train, test  = get_data_for_mlp( 'table_3d_all_dir_no_rotZ/', duplicate=2, 
                                    pick_dir='all', balance_dataset=True)
    neurons = 100
    model = create_tf_mlp_model(train.shape[1]-1, neurons=[neurons,neurons,neurons], dropout=0)
    # model_1box = create_tf_mlp_model(train_x_1box.shape[1], neurons=[20,20,20])
    model.compile(
        optimizer = tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss = tf.keras.losses.BinaryCrossentropy(from_logits=False),
        metrics = ['accuracy', 
                 tf.keras.metrics.Precision(name='P'),
                 tf.keras.metrics.Recall(name='R'),
                #  tf.keras.metrics.AUC(num_thresholds=100, curve='PR', name='AUC_PR'),
                #  tf.keras.metrics.AUC(num_thresholds=100, curve='ROC',name='AUC_ROC')
                 ])

    # model_1box.compile(
    #     optimizer = tf.keras.optimizers.Adam(learning_rate=1e-3),
    #     loss = tf.keras.losses.BinaryCrossentropy(from_logits=False),
    #     metrics = ['accuracy', 
    #              tf.keras.metrics.Precision(name='P'),
    #              tf.keras.metrics.Recall(name='R'),
    #              tf.keras.metrics.AUC(num_thresholds=100, curve='PR', name='AUC_PR'),
    #              tf.keras.metrics.AUC(num_thresholds=100, curve='ROC',name='AUC_ROC')
    #              ])

    # objected centered model
    epochs = 2
    for i in range(10):
        history = model.fit(train[:,:-1], train[:,-1], epochs=epochs, batch_size = 64, validation_data=(test[:,:-1],test[:,-1]))
        # model_1box.fit(train_x_1box, train_y[:,4], epochs=epochs, batch_size = 100, validation_data=(test_x_1box,test_y[:,4]))
    # evaluate_cnn_model(model, test_images[test_data_len:], test_labels[test_data_len:], norm=True)
        model.save(f"mlp_tf_dirall_1161_{(i+1)*epochs}.model")
        # model_1box.save(f"mlp_fv_dir4_1box{(i+1)*epochs}.model")

def training_mlp_log_wandb():
    """
    sweep hyperparameter with wandb
    """
    import wandb
    from wandb.keras import WandbCallback
    wandb.login()

    def make_model(config):
        neurons = int(config["dense_layer"])
        model = create_tf_mlp_model(train.shape[1]-1, neurons=[neurons, neurons, neurons], dropout=config["dropout"])
        model.compile(
            optimizer = tf.keras.optimizers.Adam(config["learning_rate"]),
            loss = tf.keras.losses.BinaryCrossentropy(from_logits=False),
            metrics = ['accuracy',
                    tf.keras.metrics.Precision(name='P'),
                    tf.keras.metrics.Recall(name='R'),
                    # tf.keras.metrics.AUC(num_thresholds=100, curve='PR', name='AUC_PR'),
                    # tf.keras.metrics.AUC(num_thresholds=100, curve='ROC',name='AUC_ROC')
                    ])
        return model

    def train_mlp():
        with wandb.init(project="tamp_fc_mlp") as run:
            config = wandb.config
            model = make_model(config)
            for _ in range(10):
                logging_callback = WandbCallback(log_evaluation=True)
                embed()
                history = model.fit(train[:1000,:-1], train[:1000,-1:], epochs=1, batch_size=100,\
                validation_data=[test[:100000,:-1:], test[:100000,-1:]], callbacks=[logging_callback])  # your model training code here
            wandb.log(history.history)

    sweep_config = {
    "name" : "my-sweep",
    "entity": "lei_xu_43",

    "method" : "grid",
    "parameters" : {
        "dropout": {
            "values":[0, 0.1, 0.2, 0.3],
            # "min": 0.,
            # "max": 0.9,
        },

        "dense_layer":{
            "values":[10, 40, 160],
            # "min": 10,
            # "max": 200,
        },

        "learning_rate" :{
            "values":[0.0001, 0.001, 0.01],
        # "min": 0.0001,
        # "max": 0.1
        }
    }
    }

    sweep_id = wandb.sweep(sweep_config)
    wandb.agent(sweep_id, function=train_mlp)
    wandb.finish()

def train_mlp_model(train, test):
    # training
    classifier = MLPClassifier(alpha=1e-05, solver='adam', hidden_layer_sizes=(
        10, 5), random_state=1, max_iter=10, warm_start=True)
    clf_mlp = make_pipeline(StandardScaler(), classifier)
    train_acc_list = []
    test_acc_list = []
    for i in range(100):
        clf_mlp.fit(train[:, :-1], train[:, -1])
        # predict
        pred_train_y = clf_mlp.predict(train[:, :-1])
        pred_test_y = clf_mlp.predict(test[:, :-1])
        acc_train = np.sum(pred_train_y == train[:, -1])/len(pred_train_y)
        acc_test = np.sum(pred_test_y == test[:, -1])/len(pred_test_y)
        train_acc_list.append(acc_train)
        test_acc_list.append(acc_test)
        print(f"acc_train: {acc_train}, acc_test: {acc_test}")
    plt.plot(train_acc_list)
    plt.plot(test_acc_list)
    plt.legend(['train acc', 'test acc'])
    plt.show()
    print(test_accuracy(pred_train_y, train[:, -1]))
    print(test_accuracy(pred_test_y, test[:, -1]))
    with open("./mlp_model.pk", 'wb') as file:
        pickle.dump(clf_mlp, file)
    return clf_mlp

def model_evaluation_vs_threshold_mlp(model, x,y):
    """
    model, tf model
    x,
    y,

    return:
        overall_acc,
        false_positive_rate,
        false_negative_rate,
    """

    data_length = len(y)
    pred_y = model.predict(x)
    pred_labels = np.array(pred_y >= np.array(list(range(1,10)))*0.1, dtype=int)
    args_real_pos = np.argwhere(y==1)
    args_real_neg = np.argwhere(y==0)

    tp = np.sum(pred_labels[args_real_pos[:,0]] == np.array(y[args_real_pos], dtype=int), axis=0) / data_length
    fn = len(args_real_pos) / data_length - tp

    tn = np.sum(pred_labels[args_real_neg[:,0]] == np.array(y[args_real_neg], dtype=int), axis=0) / data_length
    fp = len(args_real_neg) / data_length - tn

    dat = {'true_feasible':tp,'true_infeasible':tn,'false_feasible':fp,'false_infeasible':fn}
    return pd.DataFrame(dat)

def model_evaluation_vs_threshold_cnn(model, data_gen):

    """
    model, tf model
    x,
    y,

    return:
        overall_acc,
        false_positive_rate,
        false_negative_rate,
    """

    y = []
    pred_y = []
    for i in range(data_gen.batch_num):
        tmp_x, tmp_y = data_gen.__getitem__(i)
        y += list(tmp_y)
        pred_y += list(model.predict(tmp_x)[:,0])
    y = np.array(y, dtype=np.int8)
    data_length = len(y)

    pred_y = np.array(pred_y).reshape((data_length, 1))

    pred_labels = np.array(pred_y >= np.array(list(range(1,10)))*0.1, dtype=int)
    args_real_pos = np.argwhere(y==1)
    args_real_neg = np.argwhere(y==0)
    
    tp = np.sum(pred_labels[args_real_pos[:,0]] == np.array(y[args_real_pos], dtype=int), axis=0) / data_length
    fn = len(args_real_pos) / data_length - tp

    tn = np.sum(pred_labels[args_real_neg[:,0]] == np.array(y[args_real_neg], dtype=int), axis=0) / data_length
    fp = len(args_real_neg) / data_length - tn

    dat = {'true_feasible':tp,'true_infeasible':tn,'false_feasible':fp,'false_infeasible':fn}
    return pd.DataFrame(dat)

def merge_tf_history(h1, h2, force_merge=False):
    if h1.keys() == h2.keys() or force_merge:
        results = dict()
        for k,v in h1.items():
            try:
                results[k] = v + h2[k]
            except:
                pass
    else:
        results = None
    return results

def save_model(model):
    with open("./svm_model.pk", 'wb') as file:
        pickle.dump(model, file)

if __name__ == "__main__":
    dataset = merge_csv()
    len_dat = int(len(dataset)*0.8)
    train = dataset[:len_dat]
    test = dataset[len_dat:]

    # train with support vector machine
    train_svm_model(train, test)

    # train with multiple layer perceptons / neural network
    train_mlp_model(train, test)
