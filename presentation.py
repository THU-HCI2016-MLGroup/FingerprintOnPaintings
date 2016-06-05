# -*- coding: utf-8 -*-
"""
Created on Thu Jun 02 21:02:25 2016
Presentation of Fingerprint on Paintings
@author: yz
"""

###This file should be run from root of the project or set PROJ_ROOT

#import numpy for numerical routines, and matplotlib for plotting
import numpy as np
import matplotlib.pyplot as plt
# display plots in this notebook
# %matplotlib inline
import caffe

# set display defaults
plt.rcParams['figure.figsize'] = (10, 10)        # large images
plt.rcParams['image.interpolation'] = 'nearest'  # don't interpolate: show square pixels
plt.rcParams['image.cmap'] = 'gray'  # use grayscale output rather than a (potentially misleading) color heatmap

#config PATH
PROJ_ROOT = './'     #This is the the root of Fingerprint on Paintings Project
MODEL_FILE = PROJ_ROOT + 'models/fingerprint_caffenet_style/deploy.prototxt'
WEIGHTS_FILE = PROJ_ROOT + 'models/fingerprint_style_caffenet.caffemodel'
TEST_FILE = PROJ_ROOT + 'data/test_style.txt'
LABELS_FILE = PROJ_ROOT + 'data/style_labels.txt'
MEAN_FILE = PROJ_ROOT + 'models/ilsvrc_2012_mean.npy'
LOG_PREFIX = PROJ_ROOT + 'log/train-20160603-caffenet.log'
NET_IMG_FILE = PROJ_ROOT + 'images/fingerprint_caffenet_style.png'

## load a trained caffemodel
#caffe.set_mode_gpu()

net = caffe.Net(MODEL_FILE,      # defines the structure of the model
                WEIGHTS_FILE,  # contains the trained weights
                caffe.TEST)   # use test mode (e.g., don't perform dropout)   


## Net structure
#Net image
image = caffe.io.load_image(NET_IMG_FILE)
plt.imshow(image)

#layer parameters
#Layers' activation shapes: each layer's neuron shapes(num of neurons = multiplying elements of the vector))
#with (batch_size, channel_dim, height, width)
for layer_name, blob in net.blobs.iteritems():
    print layer_name + '\t' + str(blob.data.shape)
#Layers' parameter shapes: convolution kernel/filter shapes(num of parameters = multiplying elements of the vector, and num of filters = output_channels*input_channels)
#with (output_channels, input_channels, filter_height, filter_width) and the 1-dimensional shape (output_channels,)
for layer_name, param in net.params.iteritems():
    print layer_name + '\t' + str(param[0].data.shape), str(param[1].data.shape)   #[0] for weights and [1] for biases

## loss and accuracy
#first python ./tools/parse_log.py log_file out_dir can parse train and test loss and acc into files
#Load parsed data
f1 = open(LOG_PREFIX+'.train')
f2 = open(LOG_PREFIX+'.test')
header1 = f1.readline().rstrip().split(',')
header2 = f2.readline().rstrip().split(',')
x1 = []
x2 = []
y1 = []
y2 = []
y3 = []
y4 = []
for s in f1:
    if s != [''] and s != '\n':
        l = s.rstrip('\n\r').split(',')
        if float(l[0])%1000 != 0:
            continue
        x1.append(l[0])
        y1.append(l[3])
for s in f2: 
    if s != [''] and s != '\n':
        l = s.rstrip('\n\r').split(',')
        x2.append(l[0])
        y2.append(l[3])      
        y3.append(l[4])
        y4.append(l[5])
#output accuracy
print 'Initial test set accuracy with pretrained net = ' + y2[0]
print 'Initial test set accuracy/Top-5 with pretrained net = ' + y3[0]
print 'Final test set accuracy = ' + y2[-1]
print 'Fianl test set accuracy/Top-5 = ' + y3[-1]
#moving average of training
y1_copy = y1[:]
y1 = y1[:3]
ma3 = y1_copy[0]
ma2 = y1_copy[1]
ma1 = y1_copy[2]
for y in y1_copy[3:]:
    ma4 = ma3
    ma3 = ma2
    ma2 = ma1   
    ma1 = y
    y1.append(str((float(ma4)+float(ma3)+float(ma2))/3))      
#Plot learning curve
plt.figure(figsize = (10,6))
plt.plot(x1, y1,'r', label = 'train '+header1[3])
plt.plot(x2, y4,'b', label = 'test '+header2[5])
plt.legend(loc = 'best')
plt.title('Learning curve')
plt.xlabel('iteration')
plt.ylabel('loss')
plt.show()
#Plot test accuracy
plt.figure(figsize = (10,6))
plt.plot(x2, y2, 'g', label = 'test '+header2[3])
plt.plot(x2, y3, 'b', label ='test '+header2[4])
plt.legend(loc = 'best')
plt.title('Test accuracy')
plt.xlabel('iteration')
plt.ylabel('accuracy')
plt.show()

## Cassificaiton
#Load testset(image links and label number)
testset = np.loadtxt(TEST_FILE, str, delimiter=' ')
#Load labels(label number and label) so as to output labels
#labels = np.loadtxt(LABELS_FILE, str, delimiter='\t')
labels = {}
for line in open(LABELS_FILE, 'r'):
    key, label = line.decode('utf-8').rstrip('\n\r').split('\t')
    labels[key] = label
# set the size of the input(different batchsize can be set for batching)
net.blobs['data'].reshape(50,        # batch size
                          3,         # 3-channel (BGR) images
                          227, 227)  # image size is 227x227

#Training images input to the net are transformed by caffe to fit the net defined in MODEL_FILE like
#  transform_param {
#    mirror: false
#    crop_size: 227
#    mean_file: "imagenet_mean.binaryproto"
#So we first define image transformer
# load the mean ImageNet image (as distributed with Caffe) for subtraction
mu = np.load(MEAN_FILE)
mu = mu.mean(1).mean(1)  # average over pixels to obtain the mean (BGR) pixel values
print 'mean-subtracted values:', zip('BGR', mu)

# create transformer for the input called 'data'
transformer = caffe.io.Transformer({'data': net.blobs['data'].data.shape})

transformer.set_transpose('data', (2,0,1))  # move image channels to outermost dimension
transformer.set_mean('data', mu)            # subtract the dataset-mean value in each channel
transformer.set_raw_scale('data', 255)      # rescale from [0, 1] to [0, 255]
transformer.set_channel_swap('data', (2,1,0))  # swap channels from RGB to BGR

#Load one image from testset
image = caffe.io.load_image(testset[0][0])
plt.imshow(image)
#Transform
transformed_image = transformer.preprocess('data', image)
# print correct label
correct_label = labels[str(testset[0][1])]
print 'Correct label is: ' + correct_label

# copy the image data into the memory allocated for the net
net.blobs['data'].data[...] = transformed_image
# perform classification
output = net.forward()
output_prob = output['prob'][0]  # the output probability vector for the first image in the batch
# sort top five predictions from softmax output
top_inds = output_prob.argsort()[::-1][:5]  # reverse sort and take five largest items
print 'Predicted category is', labels[str(output_prob.argmax())]
print 'with labels and probabilities:'
for ind in top_inds:
    print labels[str(ind)] + ': ' + str(output_prob[ind])

##Visulization of extracted layer features
#define a func for visualizing sets of rectangular heatmaps.
def vis_square(data):
    """Take an array of shape (n, height, width) or (n, height, width, 3)
       and visualize each (height, width) thing in a grid of size approx. sqrt(n) by sqrt(n)"""
    # normalize data for display
    data = (data - data.min()) / (data.max() - data.min())  
    # force the number of filters to be square
    n = int(np.ceil(np.sqrt(data.shape[0])))
    padding = (((0, n ** 2 - data.shape[0]),
               (0, 1), (0, 1))                 # add some space between filters
               + ((0, 0),) * (data.ndim - 3))  # don't pad the last dimension (if there is one)
    data = np.pad(data, padding, mode='constant', constant_values=1)  # pad with ones (white)    
    # tile the filters into an image
    data = data.reshape((n, n) + data.shape[1:]).transpose((0, 2, 1, 3) + tuple(range(4, data.ndim + 1)))
    data = data.reshape((n * data.shape[1], n * data.shape[3]) + data.shape[4:])
    
    plt.imshow(data); plt.axis('off')
    
# the first layer filters, conv1 - the parameters are a list of [weights, biases]
filters = net.params['conv1'][0].data
vis_square(filters.transpose(0, 2, 3, 1))
#The first layer output, conv1 (rectified responses of the filters above, first 36 only)
feat = net.blobs['conv1'].data[0,:36]
vis_square(feat)
#The fifth layer after pooling, pool5(first 36 only))
feat = net.blobs['pool5'].data[0,:36]
vis_square(feat)
#The first fully connected layer, fc6 (rectified) - the output values and the histogram of the positive values
feat = net.blobs['fc6'].data[0]
plt.subplot(2, 1, 1)
plt.plot(feat.flat)
plt.subplot(2, 1, 2)
_ = plt.hist(feat.flat[feat.flat > 0], bins=100)
#Hist of the final probability output('prob' layer) distribution
feat = net.blobs['prob'].data[0]
plt.figure(figsize=(15, 3))
plt.plot(feat.flat)