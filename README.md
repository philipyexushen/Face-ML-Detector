# Face-ML-Detector  
+ A toy in my deep learning study

+ **March.18.2018**  
Add yolo9000 algorithm. I have modified some source code from https://github.com/allanzelener/YAD2K. May be I will make a pull request when I finish the code that can read data from voc directly.

+ **March.17.2018**   
I have found why the accuracy of mobilenet is so low. It is because my way to train the model is wrong. The right way need to train the model with small dataset and high learning rate first, then enlarge the datset and decrease the learning rate.

+ **March.6.2018**   
I have implemented faster-rcnn with mobilenet, the keras-faster-rcnn original version is from https://github.com/yhenon/keras-frcnn.    
But I don't know why the accuracy of this model is so low, while the accuracy of resnet50 is much higher