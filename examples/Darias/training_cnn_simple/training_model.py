# from PIL.Image import merge
from PIL import Image
import matplotlib.pyplot as plt
import pickle
from numpy.core.fromnumeric import size

from numpy.lib.histograms import histogram_bin_edges
from numpy.lib.type_check import imag

from IPython import embed
import csv
import uuid
import os
import sys
import numpy as np
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.model_selection import cross_val_score
from sklearn.neural_network import MLPClassifier

import tensorflow as tf
from tensorflow.keras import datasets, layers, models
from tensorflow.keras import regularizers
import pandas as pd
from PIL import Image
import random
import random
import time
import sparse
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
    def __init__(self, df:list, labels, batch_size = 32, dtype='tensorflow'):
        
        self.df = df
        self.labels = labels


        self.n = len(df)
        self.batch_size = batch_size
        self.dtype = dtype
        # self.input_size = df.shape[1:]

    def __getitem__(self, index):
        m, n = index*self.batch_size, (index+1)*self.batch_size
        if self.dtype=='COO':
            return self.df[m:n].todense(), self.labels[m:n]
        elif self.dtype=='tensorflow':
            return tf.sparse.to_dense(tf.sparse.concat(0,self.df[m:n])), self.labels[m:n]

    def __len__(self):
        return self.n // self.batch_size

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
    images = []
    shape = tuple([1] + [m,n,2])
    labels = np.zeros((num, 5))
    for k, v in dataset.items():
        img_mat = np.asarray(Image.open(os.path.join(dirname, k+'.png')))
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

def get_data0(dirname, num=100000, height=120):

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
    # images = []
    images = np.zeros((num, m,n,2))
    shape = tuple([1] + [m,n,2])
    labels = np.zeros((num, 5))
    for k, v in dataset.items():
        img_mat = np.asarray(Image.open(os.path.join(dirname, k+'.png')))
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
    model.add(layers.Conv2D(4, (3,3), input_shape=input_shape))
    # model.add(layers.BatchNormalization(axis=-1))
    # model.add(layers.Dropout(0.2))
    model.add(layers.ReLU())
    model.add(layers.MaxPooling2D(pool_size=(3, 3)))

    model.add(layers.Conv2D(8, (3,3)))
    # model.add(layers.BatchNormalization(axis=-1))
    # model.add(layers.Dropout(0.2))
    model.add(layers.ReLU())
    model.add(layers.MaxPooling2D(pool_size=(3, 3)))

    model.add(layers.Conv2D(16, (3, 3)))
    # model.add(layers.Dropout(0.2))
    # model.add(layers.BatchNormalization(axis=-1))
    model.add(layers.ReLU())
    model.add(layers.MaxPooling2D(pool_size=(3, 3)))

    # model.add(layers.Conv2D(32, (3, 3)))
    # # model.add(layers.Dropout(0.2))
    # # model.add(layers.BatchNormalization(axis=-1))
    # model.add(layers.ReLU())
    # model.add(layers.MaxPooling2D(pool_size=(4, 4), strides=(2, 2)))

    model.add(layers.Flatten())
    model.add(layers.Dropout(0.2))
    model.add(layers.Dense(125, activation='relu'))
    model.add(layers.Dense(1, activation='sigmoid'))
    model.summary()
    return model

def train_cnn_model(train_images, train_labels, validate_images, validate_labels, test_images, test_labels):

    threshold = 0.5

    def overall_accuracy(y_true, y_pred):
        # print(y_true.shape, type(y_true))
        # print(y_pred.shape, type(y_pred))
        m, n = y_pred.shape
        # int(tf.reduce_sum(tf.cast(tf.equal(tf.cast(y_pred>=threshold, dtype=tf.uint8), y_true), dtype=tf.uint8)))
        try:
            ans = int(tf.reduce_sum(tf.cast(tf.equal(tf.cast(y_pred >= threshold, dtype=tf.uint8), tf.cast(
                y_true, dtype=tf.uint8)), dtype=tf.uint8))) / m / n
        except:
            pass
        return ans

    # Create a new model instance
    input_shape = np.array(train_images[0].shape[1:])
    input_shape = train_images[0].shape
    model = create_model(input_shape)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss=tf.keras.losses.MeanSquaredError())

    train_gen = CustomGen(train_images, train_labels, batch_size=100)
    valid_gen = CustomGen(validate_images, validate_labels, batch_size=100)
    test_gen = CustomGen(test_images, test_labels, batch_size=100)
    history = model.fit(train_gen, epochs=10, batch_size = train_gen.batch_size, validation_data=valid_gen)
    # evaluate_cnn_model(model, test_images[test_data_len:], test_labels[test_data_len:], norm=True)
    evaluate_cnn_model(model, test_gen=test_gen, norm=True)

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

    # test_loss, test_acc = model.evaluate(test_images,  test_labels[:,4:], verbose=2)
    return model

def evaluate_cnn_model(model, test_images=None, test_labels=None, test_gen=None, threshold=0.5, norm=False):
    if test_gen is not None:
        test_labels_pred = test_gen.labels.copy()
        for i in range(test_gen.__len__()):
            test_images, test_labels = test_gen.__getitem__(i)
            m, n = i*test_gen.batch_size, (i+1)*test_gen.batch_size
            test_labels_pred[m:n] = np.array(model.predict(test_images) > threshold, dtype=int)
        test_labels = test_gen.labels
    # if norm and (not np.mean(test_images[:30])<1):
    #     test_images = test_images / 256
    # binary accuracy
    else:
        test_labels_pred = np.array(model.predict(test_images) > threshold, dtype=int)
    m,n = test_labels_pred.shape
    overall_acc = np.sum(test_labels == test_labels_pred) / (m*n)
    direction_acc = np.sum(test_labels == test_labels_pred,
                           axis=0)/len(test_labels_pred)
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

    print(f"true positive: {true_pos_num/len(test_labels_pred)/5}")
    print(f"true negative: {true_neg_num/len(test_labels_pred)/5}")
    print(f"false negative: {false_neg_num/len(test_labels_pred)/5}")
    print(f"false positive: {false_pos_num/len(test_labels_pred)/5}")

    R = true_pos_num/(true_pos_num+false_neg_num)
    P = true_pos_num/(true_pos_num+false_pos_num)
    f_score = 2*P*R/(P+R)
    print(f"recall rate: {R}")
    print(f"precision rate: {P}")
    print(f"f_score: {f_score}")

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


def train_svm_model(train, test):
    # clf = make_pipeline(StandardScaler(), SVC(C=30000, gamma=0.1))
    clf = SVC(C=30000, gamma=0.1)
    clf.fit(train[:, :-1], train[:, -1])
    pred_train_y = clf.predict(train[:, :-1])
    pred_test_y = clf.predict(test[:, :-1])
    acc_train = clf.score(train[:, :-1], train[:, -1])
    acc_test = clf.score(test[:, :-1], test[:, -1])
    print(f"acc_train: {acc_train}, acc_test: {acc_test}")
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
