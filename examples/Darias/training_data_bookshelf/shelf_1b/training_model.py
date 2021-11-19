from asyncio.transports import DatagramTransport
from PIL.Image import merge
import matplotlib.pyplot as plt
import pickle
from IPython import embed
import csv, uuid, os, sys   
import numpy as np
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.model_selection import cross_val_score
from sklearn.neural_network import MLPClassifier


def merge_csv(dirname):
    file_lists = [f for f in os.listdir(dirname) if 'csv' in f]
    dataset = []
    for filename in file_lists:
        with open(os.path.join(dirname, filename), "r") as file:
            reader = csv.reader(file)
            for row in reader:
                dataset.append(row)
    # train_dat_len = int(len(dataset)*0.8)
    # return np.array(dataset[:train_dat_len], dtype=float), np.array(dataset[train_dat_len:], dtype=float)
    return np.array(dataset, dtype=float)

def test_accuracy(test_y, real_y):
    num_test = len(real_y)
    res = {"true_pos":0,"true_neg":0, "false_pos":0, "false_neg":0}
    
    for i in range(num_test):
        if test_y[i]==real_y[i]:
            if test_y[i]==1:
                res["true_pos"]+=1
            else:
                res["true_neg"]+=1
        else:
            if test_y[i]==1:
                res["false_pos"]+=1
            else:
                res["false_neg"]+=1
    for key, val in res.items():
        res[key] = val/num_test
    return res

def train_svm_model(train, test):
    # clf = make_pipeline(StandardScaler(), SVC(C=30000, gamma=0.1))
    clf = SVC(C=30000, gamma=0.1)
    clf.fit(train[:,:-1], train[:,-1])
    pred_train_y = clf.predict(train[:,:-1])
    pred_test_y = clf.predict(test[:,:-1])
    acc_train = clf.score(train[:,:-1], train[:,-1])
    acc_test = clf.score(test[:,:-1], test[:,-1])
    print(f"acc_train: {acc_train}, acc_test: {acc_test}")
    print(test_accuracy(pred_train_y, train[:,-1]))
    print(test_accuracy(pred_test_y, test[:,-1]))
    with open("./svm_model.pk", 'wb') as file:
        pickle.dump(clf, file)
    return clf

def train_mlp_model(train, test):
    # training  
    classifier = MLPClassifier(alpha=1e-05, solver='adam', hidden_layer_sizes=(10,10), random_state=1, max_iter=10, warm_start=True)
    clf_mlp = make_pipeline(StandardScaler(), classifier)
    train_acc_list = []
    test_acc_list = []
    for i in range(100):
        clf_mlp.fit(train[:,:-1], train[:,-1])
        # predict
        pred_train_y = clf_mlp.predict(train[:,:-1])
        pred_test_y = clf_mlp.predict(test[:,:-1])
        acc_train = np.sum(pred_train_y==train[:,-1])/len(pred_train_y)
        acc_test = np.sum(pred_test_y==test[:,-1])/len(pred_test_y)
        train_acc_list.append(acc_train)
        test_acc_list.append(acc_test)
        print(f"acc_train: {acc_train}, acc_test: {acc_test}")
    plt.plot(train_acc_list)
    plt.plot(test_acc_list)
    plt.legend(['train acc','test acc'])
    plt.show()
    print(test_accuracy(pred_train_y, train[:,-1]))
    print(test_accuracy(pred_test_y, test[:,-1]))
    with open("./mlp_model.pk", 'wb') as file:
        pickle.dump(clf_mlp, file)
    return clf_mlp

def save_model(model):
    with open("./svm_model.pk", 'wb') as file:
        pickle.dump(model, file)

if __name__=="__main__":
    dataset = merge_csv()
    len_dat = int(len(dataset)*0.8)
    train = dataset[:len_dat]
    test = dataset[len_dat:]

    # train with support vector machine
    train_svm_model(train, test)

    # train with multiple layer perceptons / neural network
    train_mlp_model(train, test)