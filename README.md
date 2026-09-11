# 3-class-SPT-CNN
Uses short microscopy videos capturing the dynamics of RNAP within E. Coli to classify three diffusive modes using a CNN  

Input format is **greyscale, 16px x 44px, 5 frame,** TIFF files. The network uses 3D convolutional layers to extract spatio-temporal dynamics particular to each class. 


## SIMULATION PIPELINE
Uses Matlab 2022b
*SMeagol and smoldyn must also be installed to simulate data*

There are separate scripts for each state to generate the smoldyn trajectory simulation, **(Labelled A, AB, B)**.  

Diffusion coefficients are selected from a Poisson distribution using parameters found in previous experiments. 

Outputting all these trajectories into the same folder and then letting the **script labelled C sort and reformat them** is easiest.  

The script labelled **D generates the smeagol files**, for each state this takes ~30mins for 100x 100 frame videos. 

The script **E cuts these into 5 frame chunks** and deposits them into a separate training data file.  


## TRAINING PIPELINE
There are two pipelines, a three way classifier with softmax -> argmax classification, and a two stage hierarchical classifier using a sigmoid function that first identifies if the molecule is free, and if not passes it to a second classifier that identifies if it is bound or confined.  

The three way classifier is much better, the script for this is just called **train_model**

There is also a two state model for comparison.


## INFERENCE /  RUNNING THE MODEL
The saved model needs to be copied to somewhere on the local drive - not from a server.  
To run the model it also needs ti have a copy of the architecture of the model in the same folder, so that it can read in the custom convolutional layer defined in the model script. 
The contrast of the input you are predicting on will skew the results. SNR ratio needs to be somewhat similar to the simulated data.    

**run_model.py** is the primary script. You can comment and uncomment the various methods of visualisng the data.   

There are a few lines you can comment or uncomment in the run_model script to determine if you want to use some simple post processing and change obviously erroneous labels.  

You can also apply a 'sliding window' prediction to predicit on 5 frames, shift one, and predict on the next set of 5 frames.  

There are several ways to visualise the data. The easiest way is using kymographs (in some of the scripts it is spelt as khymographs, sorry), these can be displayed with or without a bar graph below showing the confidence in each prediction. Without the bar graph it is saved as a tiff file, so easier to analyse. You need to provide the kymograph of the model in a separate folder and point the function to that folder, by going in to edit **colour_khyograph.py**

Or you can label the video itself, by tinting it, or displaying a letter on the screen.   

## ANALYSIS
The predictions on larger amounts of molecules can be investigated by comparing the models labels with other methods we might use to classify the motion of the particles.

From a locoli file containing diffusion coefficients for each frame, a histogram can be plotted for each label to compare the distribution of diffusion coefficients.

Additionally, the size of confinement are can be estimated. One video of a molecule (with predictions) is split up along each state transition. Each section of frames that are continuously given the same label are treated separately. For each frame the molecule is localised (currently using a Gaussian fit). The estimated region of the frame where the molecule is present is masked, and stamped/ integrated over all the frames in a segment. We therefore get an image that tries to display where the molecule has moved in, e.g., the period of time where it was free. The area of this is calculated and plotted as a KDE histogram for each label.   

## FINE TUNING

With some labelled experimental data, the last few layers of the model can be fine tuned. The SNR of the fine tuning data again needs to be similar to the simulated data and the experimental data.

Script A creates the inputs for fine tuning.
**Script B** is the main script here. You can alter the augmentations and there is also the option to use a weighted loss function, but I did not find this generalised very well, and the results would change massively with a small change in the weights. 



