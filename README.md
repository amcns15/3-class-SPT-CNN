# 3-class-SPT-CNN
Uses short microscopy videos capturing the dynamics of RNAP within E. Coli to classify three diffusive modes using a CNN  

Input format is greyscale, 16px x 44px, 5 frame, TIFF files. The network uses 3D convolutional layers to extract spatio-temporal dynamics particular to each class. 



## SIMULATION PIPELINE
Uses Matlab 2022b
*SMeagol and smoldyn must also be installed to simulate data*


There is a separate script to generate the confined state

## TRAINING PIPELINE
There are two pipelines, a three way classifier with softmax -> argmax classification, and a two stage hierarchical classifier using a sigmoid function that first identifies if the molecule is free, and if not passes it to a second classifier that identifies if it is bound or confined.


## INFERENCE
The saved model needs to be copied to somewhere on the local drive - not from a server.
