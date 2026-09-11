# 3-class-SPT-CNN
Uses short microscopy videos capturing the dynamics of RNAP within E. Coli to classify three diffusive modes using a CNN  

Input format is greyscale, 16px x 44px, 5 frame, TIFF files. The network uses 3D convolutional layers to extract spatio-temporal dynamics particular to each class. 



## SIMULATION PIPELINE
Uses Matlab 2022b
*SMeagol and smoldyn must also be installed to simulate data*

There are separate scripts for each state to generate the smoldyn trajectory simulation, (Labelled A, AB, B).  
Outputting all these trajectories into the same folder and then letting the script labelled C sort them and reformat them is easiest.  

The script labelled D generates the smeagol files, this takes ~30mins for 100x 100 frame videos, for each state.  

The script E cuts these into 5 frame chunks and deposits them into a separate training data file.  


## TRAINING PIPELINE
There are two pipelines, a three way classifier with softmax -> argmax classification, and a two stage hierarchical classifier using a sigmoid function that first identifies if the molecule is free, and if not passes it to a second classifier that identifies if it is bound or confined.  

The three way classifier is much better.


## INFERENCE
The saved model needs to be copied to somewhere on the local drive - not from a server.  
The contrast of the input you are predicting on will skew the results. SNR ratio needs to be somewhat similar to the simulated data.  

There are a few lines you can comment or uncomment in the run_model script to determine if you want to use some simple post processing and change obviously erroneous labels.  

You can also apply a 'sliding window' prediction to predicit on 5 frames, shift one, and predict on the next set of 5 frames.  

There are several ways to visualise the data. The easiest way is using kymographs (in some of the scripts it is spelt as khymographs), these can be displayed with or without a bar graph below showing the confidence in each prediction. Without the bar graph it is saved as a tiff file, so easier to analyse.

Or you can label the video itself, by tinting it, or displaying a letter on the screen. 

The labels can also be saved to a csv


