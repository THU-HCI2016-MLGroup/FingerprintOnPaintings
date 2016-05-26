# -*- coding: utf-8 -*-
# preview lmdb data
# V1.0 by Ma Yidong this version is incomplete now
# May. 2016 THU HCI

import lmdb
import caffe
from caffe.proto import caffe_pb2
import numpy as np

dataPath = 'data'
map_size=604857600
print_size = 1

env=lmdb.open(dataPath,map_size=map_size)

txn = env.begin()
cursor = txn.cursor()
i = 0
for key, value in cursor:
    i = i+1
    datum = caffe_pb2.Datum.FromString(value)
    arr = caffe.io.datum_to_array(datum)
    #im = np.fromstring(datum.data)
    sampleImg = datum.data
    x = datum.height
    y = datum.width
    #print(datum.data)
    if(i > print_size):
        break;