### Deep Adversarial Quantization  Network for Cross-modal  Retrieval 

##### 1. Environment 
This is a demo on MIRFlickr dataset for our paper.  We finish experiments on a server with one NVIDIA GeForce 1080Ti GPU.  
The python version is 3.5.2.  These python packages are used for our experiments:  

 ```
    -opencv-python          3.4.2.16
    -tensorflow-gpu         1.4.0 
    -numpy                  1.16.2  
 ```
 You can directly run the python file _train_scripts.py_ to get results after relevant data prepared.

##### 2. Relevant data and setup 

Please preprocess dataset to appropriate input format and modify the partition of dataset in _data_handler.py_  
Or you can download the data we preprocessed from the pan.baidu.com.  

```
    a. modify the path of pre-trained VGGnet weights in bone_net.py 
    link: https://pan.baidu.com/s/1vag9Cag40zAxySMKt0i9lg  
    password: ij9g  
    
    b. modify the  path and partition  in data_handler.py   
    MIR-FLICKR:  
    link: https://pan.baidu.com/s/1ea-TvNZAcG4e6IWZRzeB9Q  pas
    password: a1cv
```

##### 3. Thanks
We should thank to these kind researchers, who unreservedly share their source code and advice to us.  
Including but not limited to these:

```
    1. QingYuan Jiang,  Nanjing          University,  P.R. China
    2. Mingsheng Long,  Tsinghua         University,  P.R. China
    3. Yue        Cao,  Tsinghua         University,  P.R. China
    4. Zhikai      Hu,  HongKong Baptist University,  P.R. China
```

##### 4. Contact
If you have any question, don't be hesitate to contact Yu Zhou at 18990848997@163.com.  
If you are a Chinese, you can surely write an e-mail in Chinese. 
If you find DAQN useful in your research, please consider citing us: 

```
    We will update the bib information after our paper is published. 
```
