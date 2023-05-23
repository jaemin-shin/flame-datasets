import os
import numpy as np

### LOAD DATA FILES ###

train_X_f = open('UCI HAR Dataset/train/X_train.txt')
train_X_f_lines = train_X_f.readlines()

train_Y_f = open('UCI HAR Dataset/train/y_train.txt')
train_Y_f_lines = train_Y_f.readlines()

train_subject_f = open('UCI HAR Dataset/train/subject_train.txt')
train_subject_f_lines = train_subject_f.readlines()

test_X_f = open('UCI HAR Dataset/test/X_test.txt')
test_X_f_lines = test_X_f.readlines()

test_Y_f = open('UCI HAR Dataset/test/y_test.txt')
test_Y_f_lines = test_Y_f.readlines()

### READ FILE CONTENTS ###

train_X_tmp = []
for line in train_X_f_lines:
    tmp = line.strip().split(' ')
    new_tmp = []
    for item in tmp:
        if item == '':
            continue
        else:
            new_tmp.append(float(item))
    train_X_tmp.append(new_tmp)

test_X_tmp = []
for line in test_X_f_lines:
    tmp = line.strip().split(' ')
    new_tmp = []
    for item in tmp:
        if item == '':
            continue
        else:
            new_tmp.append(float(item))
    test_X_tmp.append(new_tmp)

train_Y_tmp = []
for line in train_Y_f_lines:
    train_Y_tmp.append(int(line.strip()) - 1)

test_Y_tmp = []
for line in test_Y_f_lines:
    test_Y_tmp.append(int(line.strip()) - 1)

train_subject_f_tmp = []
for line in train_subject_f_lines:
    train_subject_f_tmp.append(int(line.strip()))

train_users_list = []
for subject in train_subject_f_tmp:
    if str(subject) not in train_users_list:
        train_users_list.append(str(subject))

### PARSE TRAIN USER DATA ###

for k, client in enumerate(train_users_list):
    client_x = []
    client_y = []
    for i in range(len(train_subject_f_tmp)):
        if train_subject_f_tmp[i] == int(client):
            #train_output_user_data[client]['x'].append(train_X_tmp[i]+[0.0]*15)
            client_x.append(train_X_tmp[i])
            client_y.append(train_Y_tmp[i])
    client_x = np.array(client_x)
    client_y = np.array(client_y)
    np.savez(f'./train_user{k}.npz', train_x=client_x, train_y=client_y)

### PARSE TEST USER DATA INTO ONE DATASET ###

test_client_x = []
test_client_y = []
for i in range(len(test_X_tmp)):
    test_client_x.append(test_X_tmp[i])
    test_client_y.append(test_Y_tmp[i])
test_client_x = np.array(test_client_x)
test_client_y = np.array(test_client_y)

np.savez(f'./all_val.npz', test_x=test_client_x, test_y=test_client_y)
