import os
import numpy as np
import pandas as pd

# load a single file as a numpy array
def load_file(filepath):
	dataframe = pd.read_csv(filepath, header=None, delim_whitespace=True)
	return dataframe.values

# load a list of files into a 3D array of [samples, timesteps, features]
def load_group(filenames, prefix=''):
	loaded = list()
	for name in filenames:
		data = load_file(prefix + name)
		loaded.append(data)
	# stack group so that features are the 3rd dimension
	loaded = np.dstack(loaded)
	return loaded

# load a dataset group, such as train or test
def load_dataset_group(group, prefix=''):
	filepath = prefix + group + '/Inertial Signals/'
	# load all 9 files as a single array
	filenames = list()
	# total acceleration
	filenames += ['total_acc_x_'+group+'.txt', 'total_acc_y_'+group+'.txt', 'total_acc_z_'+group+'.txt']
	# body acceleration
	filenames += ['body_acc_x_'+group+'.txt', 'body_acc_y_'+group+'.txt', 'body_acc_z_'+group+'.txt']
	# body gyroscope
	filenames += ['body_gyro_x_'+group+'.txt', 'body_gyro_y_'+group+'.txt', 'body_gyro_z_'+group+'.txt']
	# load input data
	X = load_group(filenames, filepath)
	# load class output
	y = load_file(prefix + group + '/y_'+group+'.txt')
	return X, y

# load the dataset, returns train and test X and y elements
def load_dataset(prefix=''):
	# load all train
	trainX, trainy = load_dataset_group('train', prefix + './UCI HAR Dataset/')
	print(trainX.shape, trainy.shape)
	# load all test
	testX, testy = load_dataset_group('test', prefix + './UCI HAR Dataset/')
	print(testX.shape, testy.shape)
	# zero-offset class values
	trainy = trainy - 1
	testy = testy - 1
	# one hot encode y
	# trainy = to_categorical(trainy)
	# testy = to_categorical(testy)
	print(trainX.shape, trainy.shape, testX.shape, testy.shape)
	return trainX, trainy, testX, testy

### LOAD DATA FILES ###

trainX, trainy, testX, testy = load_dataset()
# trainX_flat = trainX.reshape((7352,1152))
# testX_flat = testX.reshape((2947,1152))

train_subject_f = open('./UCI HAR Dataset/train/subject_train.txt')
train_subject_f_lines = train_subject_f.readlines()

test_subject_f = open('./UCI HAR Dataset/test/subject_test.txt')
test_subject_f_lines = test_subject_f.readlines()

train_subject_f_tmp = []
for line in train_subject_f_lines:
    train_subject_f_tmp.append(int(line.strip()))

train_users_list = []
for subject in train_subject_f_tmp:
    if str(subject) not in train_users_list:
        train_users_list.append(str(subject))

### PARSE TRAIN USER DATA ###

for k, client in enumerate(train_users_list):
    print(k, client)
    client_x = []
    client_y = []
    for i in range(len(train_subject_f_tmp)):
        if train_subject_f_tmp[i] == int(client):
            #train_output_user_data[client]['x'].append(train_X_tmp[i]+[0.0]*15)
            client_x.append(np.expand_dims(trainX[i], axis=0))
            client_y.append(int(trainy[i][0]))
    client_x = np.array(client_x)
    client_y = np.array(client_y)
    np.savez(f'./train_user{k+1}.npz', train_x=client_x, train_y=client_y)

### PARSE TEST USER DATA INTO ONE DATASET ###

test_client_x = []
test_client_y = []
for i in range(len(testX)):
    test_client_x.append(np.expand_dims(testX[i], axis=0))
    test_client_y.append(int(testy[i][0]))
test_client_x = np.array(test_client_x)
test_client_y = np.array(test_client_y)

np.savez(f'./all_val.npz', test_x=test_client_x, test_y=test_client_y)
print("DONE")