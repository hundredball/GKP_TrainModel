#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov 16 17:40:59 2018

@author: hundredball
"""

import h5py

subject_session = '11-2'

fileName = 'recording' + subject_session + '.hdf5'
f1 = h5py.File(fileName,'r')
a_group_key = list(f1.keys())[0]
data = list(f1[a_group_key])
#print(len(data[0]))

fileName = 'tongue_move_5channel_' + subject_session + '.txt'
with open(fileName, 'w') as f:
    for time_point in range(len(data[0])):
        for channel in range(5):
            f.write("%s " % data[channel][time_point])
        f.write("\n")