# FingerprintOnPaintings
Recognize Painting's Artist and Style using Convolution Neural Network
###Prerequisites
Please make sure [caffe](http://caffe.berkeleyvision.org/installation.html) and caffe python wrapper are ready to run the scripts.

###Setup
#####1. Prepare Data  
  a. Copy train images into `data/train` folder and test images into `data/test` folder. For example, I put img 1-2000 into train folder and img 2000-3000 into test folder  
  b. Open `data/convert_data.py`, dont forget to change your data path in this file.  
  c. Run the script to create `data/train.txt` and `data/test.txt`.   
  d. Count labels using scripts like `len(sub_style_dict)` in previous python file.
#####2. Modify Network  
  a. Open `train_val.prototxt`, find line 361 and change `num_output` to the amount of labels created in previous step.  
#####3. Train the Net
  a. Find or download `bvlc_reference_caffenet.caffemodel`, usually it should be inside your `%caffe_root%/models/bvlc_reference_caffenet`. If not you should download it from [CaffeModel](http://dl.caffe.berkeleyvision.org/bvlc_reference_caffenet.caffemodel).  
    You can either copy this model into this project folder or change the weights parameter in `train.sh`  
  b. Change directory to this project folder and run the shell. For windows users, please use [Cygwin](https://cygwin.com/index.html) Terminal to run the scripts.
  ```shell
    cd %your_project_folder%
    sh train.sh
  ```
#####4. Python Version?  
  Probably you find `train_val.py`, this script is working in progress, please make it work :)
  
###Prediction
Run Fingerprint on Paintings.ipynb to
* show the structure of the trained net
* show the learning curve to give a clue to the training process
* classify a test image
* visualize the net
