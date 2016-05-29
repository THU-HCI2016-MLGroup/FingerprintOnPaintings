# -*- coding: utf-8 -*-
# process training info to lmdb or txt
# V1.0 Li Mengdi
# V1.1 Ma Yidong add train txt
# V1.2 Ma Yidong add crop and resize to imgs and create lmdb
# May. 2016 THU HCI
import numpy as np
import pandas as pd
import lmdb
from PIL import Image
import random
import sys
import os
import caffe

#prepross data
NBYTES=2000000
#change to your data path
TRAIN_IMAGE_DIR='D:/Repo/FingerprintOnPaintings/FingerprintOnPaintings_StyleClassifier/data/train'
TEST_IMAGE_DIR='D:/Repo/FingerprintOnPaintings/FingerprintOnPaintings_StyleClassifier/data/test'
labelDf=pd.read_csv('train_info.csv')
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

#create train txt
train_info = open("train.txt", "w")
styl_count_dict = {}
#find current dataset as subset of labelDf
train_sub_data = labelDf[labelDf.filename.isin(train_image_list)]
sub_unique_style=train_sub_data['style'].unique()
sub_style_dict={}
for i,style in enumerate(sub_unique_style):
    sub_style_dict[style]=i
train_sub_data['sub_uni_style']=train_sub_data['style'].apply(lambda x:sub_style_dict[x])
for row in train_sub_data.values :
    train_info.write(TRAIN_IMAGE_DIR + '/' + row[0] + ' ' + str(row[8]) + '\n')
train_info.close()

#create test txt
test_info = open("test.txt", "w")
test_sub_data = labelDf[labelDf.filename.isin(test_image_list)]
for row in test_sub_data.values:
    style = row[3]
    if(style in sub_style_dict):
        test_info.write(TEST_IMAGE_DIR + '/' + row[0] + ' ' + str(sub_style_dict[style]) + '\n')
test_info.close()

#style label for current sub data set
train_style_label_info = open("train_style_labels.txt", "w")
for style, label in sub_style_dict.items():
    train_style_label_info.write(str(label) + ' ' + str(style)+ '\n')  
train_style_label_info.close()

#style and artist label for whole data set
style_label_info = open("style_labels.txt", "w")
artist_label_info = open("artist_labels.txt", "w")
for artist, label in artist_dict.items():
    artist_label_info.write(str(label) + ' ' + artist+ '\n')
for style, label in style_dict.items():
    style_label_info.write(str(label) + ' ' + str(style)+ '\n')    
style_label_info.close()
artist_label_info.close()


#create train lmdb
map_size=904857600
env=lmdb.open('fingerprint_train',map_size=map_size)
with env.begin(write=True) as txn:
    for i, row in enumerate(train_sub_data.values):
        img_path = TRAIN_IMAGE_DIR + '/' + row[0]
        label = row[8]
        img = Image.open(img_path)
        width, height = img.size
        targetsize = width if height > width else height
        im=np.array(img.crop((0,0,targetsize,targetsize)).resize((256,256),Image.ANTIALIAS ))
        Dtype=im.dtype
        if len(im.shape) == 2:
                print('here')
                (row, col) = im.shape
                im3 = np.zeros([row, col, 3], Dtype)
                for i in range(3):
                    im3 [:, :, i] = im
                im = im3
                print('here')
        if len(im.shape)!=3:
            continue
        im = im.transpose((2,0,1))
        datum=caffe.proto.caffe_pb2.Datum()
        datum.channels=im.shape[0]
        datum.height=im.shape[1]
        datum.width=im.shape[2]
        print(str(i)+ ' ' + str(label))
        datum.data=im.tobytes()
        datum.label=label
        str_id = '{:08}'.format(i)
        # label=labelDf[labelDf.filename==image].artist
        txn.put(str_id.encode('ascii'), datum.SerializeToString())
        print 'Temp'
print 'Finish'

#create test lmdb
map_size=604857600
env=lmdb.open('fingerprint_test',map_size=map_size)
with env.begin(write=True) as txn:
    j = 0
    for i, row in enumerate(test_sub_data.values):
        style = row[3]
        if(style in sub_style_dict):
            label = sub_style_dict[style]
            img_path = TEST_IMAGE_DIR + '/' + row[0]
            img = Image.open(img_path)
            width, height = img.size
            targetsize = width if height > width else height
            im=np.array(img.crop((0,0,targetsize,targetsize)).resize((256,256),Image.ANTIALIAS ))
            Dtype=im.dtype
            if len(im.shape) == 2:
                    print('here')
                    (row, col) = im.shape
                    im3 = np.zeros([row, col, 3], Dtype)
                    for i in range(3):
                        im3 [:, :, i] = im
                    im = im3
                    print('here')
            if len(im.shape)!=3:
                continue
            im = im.transpose((2,0,1))
            datum=caffe.proto.caffe_pb2.Datum()
            datum.channels=im.shape[0]
            datum.height=im.shape[1]
            datum.width=im.shape[2]
            print(str(j)+ ' ' + str(label))
            datum.data=im.tobytes()
            datum.label=label
            str_id = '{:08}'.format(j)
            # label=labelDf[labelDf.filename==image].artist
            txn.put(str_id.encode('ascii'), datum.SerializeToString())
            print 'Temp'
            j = j+1
print 'Finish'
