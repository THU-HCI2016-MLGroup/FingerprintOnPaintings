# -*- coding: utf-8 -*-
'''
Process training info to lmdb or txt
For final project of THU HCI 2016

Created on May. 2016 

@author     maajor
@author     Li Mengdi
@version    1.3         Add style/artist option
'''
import numpy as np
import pandas as pd
import lmdb
from PIL import Image
import random
import sys
import os
import caffe

#change this flag to 'artist' to get train info for artist.
label_flag='artist'
resize_flag=False

#prepross data
NBYTES=2000000
#change to your data paths
TRAIN_IMAGE_DIR='/data/train/'
TEST_IMAGE_DIR='/data/test/'
TARGET_IMAGE_DIR='/data/image/'
labelDf=pd.read_csv('./data/train_info.csv')
unique_artist=labelDf.artist.unique()
unique_style=labelDf['style'].unique()
artist_dict={}
style_dict={}
for i,artist in enumerate(unique_artist):
    artist_dict[artist]=i
for i,style in enumerate(unique_style):
    style_dict[style]=i
labelDf['cat_artist']=labelDf.artist.apply(lambda x:artist_dict[x])
labelDf['cat_style']=labelDf['style'].apply(lambda x:style_dict[x])
train_image_list=os.listdir(TRAIN_IMAGE_DIR)
test_image_list=os.listdir(TEST_IMAGE_DIR)
train_saved_list = []
test_saved_list = []

if(resize_flag):
    for imgname in train_image_list:
        img_path = TRAIN_IMAGE_DIR + '/' + imgname
        img = Image.open(img_path)
        try:
            img.load()
            train_saved_list.append(imgname)
        except IOError:
            pass # You can always log it to logger
        width, height = img.size
        targetsize = width if height > width else height
        img=img.crop((0,0,targetsize,targetsize)).resize((256,256),Image.ANTIALIAS )
        img.convert('RGB').save(TARGET_IMAGE_DIR + '/' + imgname)
        
    for imgname in test_image_list:
        img_path = TEST_IMAGE_DIR + '/' + imgname
        img = Image.open(img_path)
        try:
            img.load()
            test_saved_list.append(imgname)
        except IOError:
            print img_path
            pass # You can always log it to logger
        width, height = img.size
        targetsize = width if height > width else height
        img=img.crop((0,0,targetsize,targetsize)).resize((256,256),Image.ANTIALIAS )
        img.convert('RGB').save(TARGET_IMAGE_DIR + '/' + imgname)
    
#create train txt
train_info = open("./data/train_" + label_flag + ".txt", "w")
styl_count_dict = {}
train_sub_data = labelDf[labelDf.filename.isin(train_saved_list if resize_flag else train_image_list)]
PATH = TARGET_IMAGE_DIR if resize_flag else TRAIN_IMAGE_DIR
for row in train_sub_data.values :
    train_info.write(os.path.abspath('.') + PATH + '/' + row[0] + ' ' + str(row[7] if label_flag == 'style' else row[6]) + '\n')
train_info.close()

#create test txt
test_info = open("./data/test_" + label_flag + ".txt", "w")
test_sub_data = labelDf[labelDf.filename.isin(test_saved_list if resize_flag else test_image_list)]
PATH = TARGET_IMAGE_DIR if resize_flag else TRAIN_IMAGE_DIR
for row in test_sub_data.values:
    test_info.write(os.path.abspath('.') + PATH + '/' + row[0] + ' ' + str(row[7] if label_flag == 'style' else row[6]) + '\n')
test_info.close()
    
'''
#style label for current sub data set
train_style_label_info = open("train_style_labels.txt", "w")
for style, label in sub_style_dict.items():
    train_style_label_info.write(str(label) + ' ' + str(style)+ '\n')  
train_style_label_info.close()
'''
#style and artist label for whole data set
style_label_info = open("./data/style_labels.txt", "w")
artist_label_info = open("./data/artist_labels.txt", "w")
for artist, label in artist_dict.items():
    artist_label_info.write(str(label) + '\t' + artist+ '\n')
for style, label in style_dict.items():
    style_label_info.write(str(label) + '\t' + str(style)+ '\n')    
style_label_info.close()
artist_label_info.close()


##create train lmdb
#map_size=904857600
#env=lmdb.open('fingerprint_train',map_size=map_size)
#with env.begin(write=True) as txn:
#    for i, row in enumerate(train_sub_data.values):
#        img_path = TRAIN_IMAGE_DIR + '/' + row[0]
#        label = row[8]
#        img = Image.open(img_path)
#        width, height = img.size
#        targetsize = width if height > width else height
#        im=np.array(img.crop((0,0,targetsize,targetsize)).resize((256,256),Image.ANTIALIAS ))
#        Dtype=im.dtype
#        if len(im.shape) == 2:
#                print('here')
#                (row, col) = im.shape
#                im3 = np.zeros([row, col, 3], Dtype)
#                for i in range(3):
#                    im3 [:, :, i] = im
#                im = im3
#                print('here')
#        if len(im.shape)!=3:
#            continue
#        im = im.transpose((2,0,1))
#        datum=caffe.proto.caffe_pb2.Datum()
#        datum.channels=im.shape[0]
#        datum.height=im.shape[1]
#        datum.width=im.shape[2]
#        print(str(i)+ ' ' + str(label))
#        datum.data=im.tobytes()
#        datum.label=label
#        str_id = '{:08}'.format(i)
#        # label=labelDf[labelDf.filename==image].artist
#        txn.put(str_id.encode('ascii'), datum.SerializeToString())
#        print 'Temp'
#print 'Finish'
#
##create test lmdb
#map_size=604857600
#env=lmdb.open('fingerprint_test',map_size=map_size)
#with env.begin(write=True) as txn:
#    j = 0
#    for i, row in enumerate(test_sub_data.values):
#        style = row[3]
#        if(style in sub_style_dict):
#            label = sub_style_dict[style]
#            img_path = TEST_IMAGE_DIR + '/' + row[0]
#            img = Image.open(img_path)
#            width, height = img.size
#            targetsize = width if height > width else height
#            im=np.array(img.crop((0,0,targetsize,targetsize)).resize((256,256),Image.ANTIALIAS ))
#            Dtype=im.dtype
#            if len(im.shape) == 2:
#                    print('here')
#                    (row, col) = im.shape
#                    im3 = np.zeros([row, col, 3], Dtype)
#                    for i in range(3):
#                        im3 [:, :, i] = im
#                    im = im3
#                    print('here')
#            if len(im.shape)!=3:
#                continue
#            im = im.transpose((2,0,1))
#            datum=caffe.proto.caffe_pb2.Datum()
#            datum.channels=im.shape[0]
#            datum.height=im.shape[1]
#            datum.width=im.shape[2]
#            print(str(j)+ ' ' + str(label))
#            datum.data=im.tobytes()
#            datum.label=label
#            str_id = '{:08}'.format(j)
#            # label=labelDf[labelDf.filename==image].artist
#            txn.put(str_id.encode('ascii'), datum.SerializeToString())
#            print 'Temp'
#            j = j+1
#print 'Finish'
