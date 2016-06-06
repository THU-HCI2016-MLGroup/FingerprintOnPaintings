'''
draw compare image between finetune and non-finetune training

@version    1.0
@author     maajor
@date       Jun. 5th, 2016
'''

import numpy as np
import matplotlib.pyplot as plt
%matplotlib inline

#select two log to compare. first finetuned and second non-finetuned
LOGS = [
        'train-20160603-caffenet.log',
        'train_20160606_caffenet_style_nonfinetune.log'
        ]

iter_num = [[],[]]
train_loss = [[],[]]
test_loss = [[],[]]
test_accu1 = [[],[]]
test_accu5 = [[],[]]
for log_id in range(0,2):
    thislog_train = open(LOGS[log_id]+'.train')
    thislog_test = open(LOGS[log_id]+'.test')
    #read header
    header_train = thislog_train.readline().rstrip().split(',')
    header_test = thislog_test.readline().rstrip().split(',')
    #read train log
    for line in thislog_train:
        if line != [''] and line != '\n':
            l = line.rstrip('\n\r').split(',')
            if float(l[0])%1000 != 0:
                continue
            iter_num[log_id].append(l[0])
            train_loss[log_id].append(l[3])
    for line in thislog_test:
        if line != [''] and line != '\n':
            l = line.rstrip('\n\r').split(',')
            if float(l[0])%1000 != 0:
                continue
            test_loss[log_id].append(l[5])
            test_accu1[log_id].append(l[3])
            test_accu5[log_id].append(l[4])
iter_len = len(iter_num[0]) if len(iter_num[0]) < len(iter_num[1]) else len(iter_num[1])

    
#Plot learning curve
plt.style.use('ggplot')
plt.figure(figsize = (10,6))
plt.plot(iter_num[0][:iter_len], train_loss[0][:iter_len], lw=3,label = 'train '+header_train[3] + ' (finetune)')
plt.plot(iter_num[0][:iter_len], test_loss[0][:iter_len], lw=3,label = 'test '+header_test[5] + ' (finetune)')
plt.plot(iter_num[0][:iter_len], train_loss[1][:iter_len], lw=2,label = 'train '+header_train[3] + ' (non-finetune)')
plt.plot(iter_num[0][:iter_len], test_loss[1][:iter_len], lw=2,label = 'test '+header_test[5] + ' (non-finetune)')
plt.legend(loc = 'best')
plt.title('Learning curve')
plt.xlabel('iteration')
plt.ylabel('loss')
plt.grid(True)
plt.show()

#Plot accuracy curve
plt.figure(figsize = (10,6))
plt.plot(iter_num[0][:iter_len], test_accu1[0][:iter_len], lw=3,label = 'test '+header_test[3] + ' (finetune)')
plt.plot(iter_num[0][:iter_len], test_accu5[0][:iter_len], lw=3,label = 'test '+header_test[4] + ' (finetune)')
plt.plot(iter_num[0][:iter_len], test_accu1[1][:iter_len], lw=2,label = 'test '+header_test[3] + ' (non-finetune)')
plt.plot(iter_num[0][:iter_len], test_accu5[1][:iter_len], lw=2,label = 'test '+header_test[4] + ' (non-finetune)')
plt.legend(loc = 'best')
plt.title('Test accuracy')
plt.xlabel('iteration')
plt.ylabel('accuracy')
plt.grid(True)
plt.show()
