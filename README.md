# Fingerprint On Paintings
Recognize Painting's Artist and Style using Convolution Neural Network  
Finetune Caffenet and Googlenet to do the job.  
###Prerequisites
Please make sure [caffe](http://caffe.berkeleyvision.org/installation.html) and caffe python wrapper are ready to run the scripts.

###Data
Images used in this project mainly comes from [WikiArt](http://www.wikiart.org/). You can also download [the dataset in .csv format](https://www.kaggle.com/c/painter-by-numbers/data) prepared by Small Yellow Duck (Kiri Nichol) on Kaggle 
###Train
#####1. Prepare Data  
  a. Copy train images into `data/train` folder and test images into `data/test` folder.   
  b. Open `data/convert_data.py`, change working directory to the project's root folder.  
  c. Change `label_flag` and `resize_flag` according to your task.  
  d. Run the script to create `data/train_{label_flag}.txt` and `data/test_{label_flag}.txt`.  
#####2. Modify Network  
  a. Network definitions are inside `models/{net_name}/train_val_{net_name}.prototxt`. To finetune caffenet, I changed input Imagadata Layer and fc_8 layer's num_output according to the task. As for googlenet, it have three fc layer: `loss3/classifier`, `loss2/classifier`, `loss1/classifier`, I changed them all accordingly. You can change parameters but the nets in this project should work fine.
#####3. Train the Net
  a. Find or download `bvlc_reference_caffenet.caffemodel`, usually it should be inside your  `%caffe_root%/models/bvlc_reference_caffenet`. If not you should download it from [CaffeModel](http://dl.caffe.berkeleyvision.org/bvlc_reference_caffenet.caffemodel).  For Googlenet you can download from [GooglenetModel](http://dl.caffe.berkeleyvision.org/bvlc_googlenet.caffemodel).  
    You can either copy this model into `./models` or change the weights parameter in `train.sh`  
  b. Change directory to this project folder and run the shell. For windows users, please use [Cygwin](https://cygwin.com/index.html) Terminal to run the scripts.
  ```shell
    cd %your_project_folder%
    sh train_{net_name}.sh
  ```
###Prediction
Run `Fingerprint on Paintings.ipynb` to
* show the structure of the trained net
* show the learning curve to give a clue to the training process
* classify a test image
* visualize the net
