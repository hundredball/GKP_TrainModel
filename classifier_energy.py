# -*- coding: utf-8 -*-
"""
Created on Fri Mar 22 16:15:49 2019

@author: John
"""

import numpy as np
import re
import matplotlib.pyplot as plt
from scipy.signal import butter, lfilter, filtfilt
from scipy.integrate import simps
import scipy
from sklearn.preprocessing import normalize, scale
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
import functools
from plot_confusion_matrix import plot_confusion_matrix

def load_data(file):
    data = []
    with open(file,'r') as fin:
        for row in fin:
            val = [float(i) for i in row.split(' ')[0:-1]]
            #row = np.array(row, dtype='float')
            data.append(val)
            
        data = np.array(data, dtype='float')
        data = np.delete(data,0,0)
    return data
    
def load_log(file):
    logdata = []
    #input is byte, convert to string
    with open(file,'rb') as flog:
        for row in flog:
            try:
                t = row.decode('utf-8').replace('\x00','')
                logdata.append(t)
            except:
                continue
    return logdata
        
#the time is at the beginning of a trial
def get_event_and_time(logdata):
    trial_event = []
    trial_time = []
    
    for row in logdata:
        if "num:" in row:
            trial_event.append(int(row[6]))
        if "OnsetTime:" in row:
            trial_time.append(int(re.split('\r| ', row)[1]))
            
    start_time = trial_time[-2]
    end_time = trial_time[-1]
    trial_time = trial_time[0:-2]
    
    #all_time minus start_time
    offset_trial_time = []
    for t in trial_time:
        offset_trial_time.append(t-start_time)
    
    return trial_event, offset_trial_time 

def split_data(data, time):
    rate = 125
    
    # take data from startPoint~startPoint+interval seconds
    startPoint = int(rate*0)
    interval = rate*6
    
    fine_data = []
    for i in range(len(time)):
        trig = int(round(time[i]/1000*rate))
        #minus the reference (0.6s) before task start
        m = np.mean(data[int(trig - 0.6*rate) : trig], axis=0)
        fine_data.append(data[trig+startPoint:trig+startPoint+interval]-m)
        
#        fine_data.append(data[trig+startPoint:trig+startPoint+interval])
    
    return np.array(fine_data)

def butter_bandpass(lowcut, highcut, fs, order = 5):
    nyq = 0.5*fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    return b, a

def butter_bandpass_filter(data, lowcut, highcut, fs, order = 5):
    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    y = lfilter(b,a,data, axis=0)
    return y

def butter_highpass(cutoff, fs, order = 5):
    nyq = 0.5*fs
    normal_cutoff = cutoff/nyq
    b, a = butter(order, normal_cutoff, btype='high', analog=False)
    return b, a

def butter_highpass_filter(data, cutoff, fs, order = 5):
    b, a = butter_highpass(cutoff, fs, order=order)
    y = lfilter(b, a, data, axis = 0)
    return y

def save(data):
    with open('fine_data.txt', 'w') as outfile:
        for slice_2d in data:
            np.savetxt(outfile, slice_2d)

def show_f(data):
    N = 750
    T = 1.0/125
    #x = np.linspace(0.0, N*T, N)
    yf = scipy.fft(data)
    xf = np.linspace(0.0, 1.0/(2.0*T), N/2)
    
    fig, ax = plt.subplots()
    ax.plot(xf, 2.0/N * np.abs(yf[:N//2]))
    plt.show()
    
def split_windows(x,y):
    '''split trials into frames (2s duration (250 points), 50 pointsX_train, split_datax
 gap)'''
    duration = 250
    gap = 50
    X_train = []
    Y_train = []
    
    for i in range(x.shape[0]):
        for j in range(0, x.shape[1]-duration+1, gap):
            X_train.append(x[i, j:j+duration, :])
            Y_train.append(y[i])
            
    X_train = np.array(X_train)
    Y_train = np.array(Y_train)

    return X_train, Y_train

def SVMClassify(X_train, Y_train, X_val, Y_val):
    clf = SVC(kernel = 'poly', degree=3)
    clf.fit(X_train, Y_train)
    pred_train = clf.predict(X_train)
    pred_val = clf.predict(X_val)
    
    correct_num = sum(pred_val == Y_val)
    val_accuracy = correct_num / len(Y_val)
    print('Validation')
    print('Total:', len(Y_val), ' | Correct:', correct_num)
    print('Accuracy:', val_accuracy)
    
    correct_num = sum(pred_train == Y_train)
    print('Train')
    print('Total:', len(Y_train), ' | Correct:', correct_num)
    print('Accuracy:', correct_num/len(Y_train))
    
    pred_val = (pred_val-1).astype('int')
    Y_val = (Y_val-1).astype('int')
    plot_confusion_matrix(Y_val, pred_val, np.array(['left', 'right', 'rest']), title='SVM accuracy: {0}%'.format(val_accuracy*100), normalize=True)
    
def XGBClassify(X_train, Y_train, X_val, Y_val):
    XG = XGBClassifier(
            colsample_bytree= 0.9, gamma= 2, max_delta_step= 5, max_depth= 8,
            min_child_weight= 2, n_estimators= 50)
    
    XG.fit(X_train,Y_train,verbose=True)
    pred_train = XG.predict(X_train)
    pred_val = XG.predict(X_val)
    
    correct_num = sum(pred_val == Y_val)
    val_accuracy = correct_num/len(Y_val)
    print('Validation')
    print('Total:', len(Y_val), ' | Correct:', correct_num)
    print('Accuracy:', val_accuracy)
    
    correct_num = sum(pred_train == Y_train)
    print('Train')
    print('Total:', len(Y_train), ' | Correct:', correct_num)
    print('Accuracy:', correct_num/len(Y_train))

    pred_val = (pred_val-1).astype('int')
    Y_val = (Y_val-1).astype('int')
    plot_confusion_matrix(Y_val, pred_val, np.array(['left', 'right', 'rest']), title='XGBoost accuracy: {0}%'.format(val_accuracy*100), normalize=True)
# ------------------------ output data for EEGNet ---------------------
def test(dataName, logName, saveParaName):
    # load data
    data = load_data(dataName)
    log = load_log(logName)
    event, fine_time = get_event_and_time(log)
    
    # split data with respect to fine_time , sample rate 125Hz, remove baseline => 90x750x5
    splited_data = split_data(data, fine_time)
    
    # split data into training and validation => X_train(80x750x5), X_val(10x750x5)
    X_train, X_val, Y_train, Y_val = train_test_split(splited_data, event, test_size = 0.1, random_state=42)
    
    # split windows of 2 sec => X_train(800x750x5), X_val(100x750x5)
    X_train, Y_train = split_windows(X_train, Y_train)
    X_val, Y_val = split_windows(X_val, Y_val)

    # bandpass filter 1-50
    for i in range(X_train.shape[0]):
        X_train[i] = butter_bandpass_filter(X_train[i], 1, 50, 125, 5)
    for i in range(X_val.shape[0]):
        X_val[i] = butter_bandpass_filter(X_val[i], 1, 50, 125, 5)
        
    # standardize data using training data, take data point in 0,2,4 secs
    X_train_specificPoint = []
    time_point = 0
    for i in range(len(X_train)):
        if time_point == 0 or time_point == 5 or time_point == 10:
            X_train_specificPoint.append(X_train[i])
            if time_point == 10:
                time_point = -1
            
        time_point += 1
    
    mean = np.mean(X_train_specificPoint)
    std = np.std(X_train_specificPoint)
    X_train = (X_train-mean)/std
    X_val = (X_val-mean)/std
    
    # randomize the data
    X_train, t, Y_train, tt = train_test_split(X_train,Y_train,test_size=0.0,random_state=52)
    X_val, t, Y_val, tt = train_test_split(X_val,Y_val,test_size=0.0,random_state=40)
    
    with open(saveParaName, 'w') as f:
        f.write(str(mean)+'\n')
        f.write(str(std))
    
    return X_train, Y_train, X_val, Y_val

def load_testData(dataName, logName, loadParaName):
    data = load_data(dataName)
    log = load_log(logName)
    event, fine_time = get_event_and_time(log)
    
    splited_data = split_data(data, fine_time)
    
    # split windows of 2 sec
    X_test, Y_test = split_windows(splited_data, event)
    
    # bandpass filter 1-50
    for i in range(len(X_test)):
        X_test[i] = butter_bandpass_filter(X_test[i], 1, 50, 125, 5)
    
    # standardize data
    with open(loadParaName, 'r') as f:
        mean = float(f.readline())
        std = float(f.readline())

#    print(filt_split_data.shape)
    X_test = (X_test-mean)/std
    
    return X_test, Y_test

def output_testData(dataName, logName):
    data = load_data(dataName)
    log = load_log(logName)
    event, fine_time = get_event_and_time(log)
    
    splited_data = split_data(data, fine_time)
    
    return splited_data, event

def getBandPower(X):
    # Define sampling rate and window length
#    X = filt_split_data
    fs = 125
#    time = np.arange(X.shape[1])/fs
    multiplier = 2/0.5
    win = multiplier*fs
    freq_res = 1/multiplier
    
    X_bandpower = np.zeros((X.shape[0], 4, X.shape[2]))
    simps_dx = functools.partial(simps, dx=freq_res)
    
    for sample in range(X.shape[0]):
        for channel in range(X.shape[2]):
            freqs, psd = scipy.signal.welch(X[sample,:,channel], fs, nperseg=win)
            [idx_delta, idx_theta, idx_alpha, idx_beta] = map(
                    np.logical_and, [freqs>=1, freqs>=4, freqs>=9, freqs>=13], [freqs<=4, freqs<=8, freqs<=13, freqs<=30])
            [X_bandpower[sample,0,channel], X_bandpower[sample,1,channel], X_bandpower[sample,2,channel], X_bandpower[sample,3,channel]] = map(
                    simps_dx, [psd[idx_delta], psd[idx_theta], psd[idx_alpha], psd[idx_beta]])
            
            
#    plt.figure(figsize=(8,4))
#    plt.plot(freqs, psd, color='k', lw=2)
#    plt.xlabel('Frequency(HZ)')
#    plt.ylabel('Power spectral density (V^2 / HZ)')
    
    return X_bandpower
    
def getStandardPSD(dataName, logName, saveParaName, standardize=True, test_size=0.1):
    # load data
    data = load_data(dataName)
    log = load_log(logName)
    event, fine_time = get_event_and_time(log)
    
    # split data with respect to fine_time , sample rate 125Hz, remove baseline => 90x750x5
    splited_data = split_data(data, fine_time)
    
    # split data into training and validation => X_train(80x750x5), X_val(10x750x5)
    X_train, X_val, Y_train, Y_val = train_test_split(splited_data, event, test_size = test_size, random_state=80)
    
    
    
    # split windows of 2 sec => X_train(800x750x5), X_val(100x750x5)
    X_train, Y_train = split_windows(X_train, Y_train)
    X_val, Y_val = split_windows(X_val, Y_val)
    
    X_train_before = X_train.copy()
    # highpass filter: cutoff, 1HZ
    '''
    for i in range(X_train.shape[0]):
        X_train[i] = butter_highpass_filter(X_train[i], 1, 125, 5)
    for i in range(X_val.shape[0]):
        X_val[i] = butter_highpass_filter(X_val[i], 1, 125, 5)
    '''
    
    X_train = X_train[:,10:,:]
    X_val = X_val[:,10:,:]
    X_trian_org = X_train.copy()
    # get psd of each sample
    fs = 125
    multiplier = 2/1
    win = multiplier*fs
    freq_res = 1/multiplier
    upper_freq = 50
    X_train_psd = np.zeros((X_train.shape[0], upper_freq-1, X_train.shape[2]))
    X_val_psd = np.zeros((X_val.shape[0], upper_freq-1, X_val.shape[2]))
    simps_dx = functools.partial(simps, dx=freq_res)
    
    for i, X in enumerate([X_train, X_val]):
        for sample in range(X.shape[0]):
            for channel in range(X.shape[2]):
                freqs, psd = scipy.signal.welch(X[sample, :, channel], fs, nperseg=win)
                for freq_i in range(upper_freq-1):
                    if i==0:
                        X_train_psd[sample, freq_i, channel] = simps_dx(psd[2*freq_i:2*freq_i+3])
                    else:
                        X_val_psd[sample, freq_i, channel] = simps_dx(psd[2*freq_i:2*freq_i+3])

    # standardize data by mean and std of training data
    if standardize:
        mean = np.mean(X_train_psd)
        std = np.std(X_train_psd)
        X_train_psd = (X_train_psd-mean)/std
        X_val_psd = (X_val_psd-mean)/std
    
    # randomize training data
    X_train_psd, t, Y_train, tt = train_test_split(X_train_psd,Y_train,test_size=0.0,random_state=52)
    
    return X_train_psd, Y_train, X_val_psd, Y_val

def compareEnergy(X_train, Y_train, X_val, Y_val):
    
    # subtract each channels of validation data from means of rest training data
    rest_data = X_train[Y_train==3]
    mean = np.mean(rest_data, axis=(0,1))
    for i in range(X_val.shape[2]):
        X_val[:,:,i] -= mean[i]
    
    pred = np.array([0]*X_val.shape[0])

    correct = 0
    num_left_right = 0
    start_frequency = 2
    assert start_frequency >= 1, 'in X_val, first element is 1 HZ'
    for sample in range(X_val.shape[0]):
        left_energy = np.sum(X_val[sample, 5:, :2])
        right_energy = np.sum(X_val[sample, 5:, 3:])
        
        if left_energy > right_energy:
            pred[sample] = 1
        else:
            pred[sample] = 2
        
        if Y_val[sample] == 1 or Y_val[sample] == 2:
            num_left_right += 1
            if pred[sample] == Y_val[sample]:
                correct += 1
    
    accuracy = correct/num_left_right
    print('Accuracy of left and right: {0}%'.format(accuracy*100))
    pred = (pred-1).astype('int')
    Y_val = (Y_val-1).astype('int')
    plot_confusion_matrix(Y_val, pred, np.array(['left', 'right', 'rest']), title='Left/right Accuracy {0}%'.format(accuracy*100), normalize=True)

def compareMax(dataName, logName):
    # load data
    data = load_data(dataName)
    log = load_log(logName)
    event, fine_time = get_event_and_time(log)
    
    # split data with respect to fine_time , sample rate 125Hz, remove baseline => 90x750x5
    splited_data = split_data(data, fine_time)
    
    # split windows of 2 sec
    X, Y = split_windows(splited_data, event)

    # highpass filter: cutoff, 1HZ
    for i in range(X.shape[0]):
        X[i] = butter_highpass_filter(X[i], 1, 125, 5)

    # compare the maximum of each trial
    pred = np.zeros(len(Y))
    num_left_right = 0
    correct = 0
    for sample in range(len(X)):
        
        left_max = np.max(X[sample,:,:2])
        right_max = np.max(X[sample,:,3:])
        if left_max > right_max:
            pred[sample] = 1
        else:
            pred[sample] = 2
            
        if Y[sample] == 1 or Y[sample] == 2:
            num_left_right += 1
            if pred[sample] == Y[sample]:
                correct += 1
                
    accuracy = correct / num_left_right
    pred = (pred-1).astype('int')
    Y = (Y-1).astype('int')
    plot_confusion_matrix(Y,pred, np.array(['left','right','rest']), title='CompareMax, Left/Right Accuracy {0}%'.format(accuracy*100), normalize=True)

if __name__ == '__main__':
    subject_session = '11-2'
    date = '0424'
    
    dataFile = date + '/tongue_move_5channel_' + subject_session + '.txt'
    eventFile = date + '/GKP_Exp' + date + '.txt'
    paramFile = date + '/param_' + date + '.txt'
    
    # get standardized data
#    X_train, train_label, X_val, test_label = test(dataFile, eventFile, paramFile)
    
    # get standardized PSD
    X_train, train_label, X_val, test_label = getStandardPSD(dataFile, eventFile, paramFile, standardize=False, test_size=0.9)
    
    '''Compare left and right energy'''
    compareEnergy(X_train, train_label, X_val, test_label)
    
    '''Compare left and right max value'''
    #compareMax(dataFile, eventFile)

    '''SVM, XGB'''
#    X_train = X_train.reshape(X_train.shape[0], -1)
#    X_val = X_val.reshape(X_val.shape[0], -1)
#    print('------------SVM-------------')
#    SVMClassify(X_train, train_label, X_val, test_label)
#    print('------------XGB-------------')
#    XGBClassify(X_train, train_label, X_val, test_label)
        
    
    
