from os import name

import model
from model import model_simple

import keras
import tensorflow as tf
#from keras import ops

import tqdm
from tqdm.keras import TqdmCallback
import numpy as np
from pathlib import Path
from PIL  import Image, ImageSequence
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from pretty_confusion_matrix import pp_matrix, pp_matrix_from_data
from sklearn.metrics import confusion_matrix

EPOCHS = 30
BATCH_SIZE = 8
FRAMES = 5
HEIGHT = 44
WIDTH = 16
CHANNELS = 1

def load_segment(path): # load a segment of frames from a tif file and convert to numpy array
    path = path.numpy().decode("utf-8")
    with Image.open(path) as img:
        frames = [np.array(frame, dtype=np.float32) for frame in ImageSequence.Iterator(img)]

    if len(frames) != FRAMES:
        raise ValueError(f"Expected {FRAMES} frames, but found {len(frames)} in {path}")

    arr = np.stack(frames, axis = 0)

    arr = np.expand_dims(arr, axis=-1)  # add channel dimension
    arr /= 255.0  # normalize to [0, 1]

    return arr

def tf_load_segment(path, label): # sandwich the load_segment function in a tf.py_function to use in a tf.data.Dataset
    arr = tf.py_function(
        func = load_segment, 
        inp=[path], 
        Tout = tf.float32
    )    

    arr.set_shape([FRAMES, HEIGHT, WIDTH, CHANNELS])

    return arr, label

if __name__ == "__main__":

    root = Path(r"\\rivendell.physics.ox.ac.uk\user\students\2024\jesu4837\summer_internship\raw_training_data")

    class_names = sorted([d.name for d in root.iterdir()])
    class_to_idx = {name: i for i, name in enumerate(class_names)} # make a dictionary of class names and their corresponding indices

    file_paths = []
    labels = []

    for class_name in class_names: # create two matching lists of class directories and labels in a dictionary
        for f in (root / class_name).glob("*.tif"):
            file_paths.append(str(f))
            labels.append(class_to_idx[class_name])

    print(class_names)
    print(len(file_paths))


    path_ds = tf.data.Dataset.from_tensor_slices((file_paths, labels)) # create dataset from the two lists of file paths and labels
    
    total = len(file_paths)
    val_size = int(0.2 * total)

    path_ds = (
        path_ds
        .shuffle(buffer_size=len(file_paths), seed=293, reshuffle_each_iteration=False)
    ) # shuffle entire dataset

    val_ds = path_ds.take(val_size)
    train_ds = path_ds.skip(val_size)

    # map path to load images as arrays
    train_ds = train_ds.map(tf_load_segment, num_parallel_calls = tf.data.AUTOTUNE)
    val_ds = val_ds.map(tf_load_segment, num_parallel_calls = tf.data.AUTOTUNE)

    train_ds = (
        train_ds
        .shuffle(buffer_size=(total - val_size), seed=811, reshuffle_each_iteration=True)
        .batch(BATCH_SIZE)
        .prefetch(tf.data.AUTOTUNE)
    ) # reshuffle training dataset and batch it for training

    val_ds = (
        val_ds
        .batch(BATCH_SIZE)
        .prefetch(tf.data.AUTOTUNE)
    ) # batch validation dataset for evaluation

    print("validation and training sets created !!")

    for x, y in train_ds.take(1):
        print(f"Shape of x: {x.shape}")
        print(f"Shape of y: {y.shape}")

    model = model_simple(frame = FRAMES, height = HEIGHT, width = WIDTH, channels = CHANNELS)
    model.compile(optimizer = keras.optimizers.Adam(learning_rate=1e-4), loss = keras.losses.SparseCategoricalCrossentropy(), metrics = ["accuracy"])

    model.summary() 

    callbacks = [ # help prevent overfitting
        keras.callbacks.ModelCheckpoint( # keep best model parameters based on validation loss
            "best_model.keras",
            save_best_only = True,
            monitor = "val_loss",
        )]
    #     keras.callbacks.EarlyStopping( # stop if validatin loss stops improving
    #         monitor = "val_loss",
    #         patience = 5,
    #         restore_best_weights = True
    #     )
    # ]

    history = model.fit(
        train_ds, 
        validation_data = val_ds, 
        epochs = EPOCHS,
        callbacks = callbacks
        )

    model.save(r"\\rivendell.physics.ox.ac.uk\user\students\2024\jesu4837\summer_internship\models\resimulated_three_state.keras")

    # --- Confusion matrix --- # 
    # gather true labels and predictions from the validation set 
    y_true = [] 
    y_pred = [] 
    for x_batch, y_batch in val_ds: 
        preds = model.predict(x_batch, verbose=0) # shape (batch, 3) softmax probs 
        y_pred.extend(np.argmax(preds, axis=1)) # convert to class indices 
        y_true.extend(y_batch.numpy()) 

    y_true = np.array(y_true) 
    y_pred = np.array(y_pred) 
    cm = confusion_matrix(y_true, y_pred) 

    df_cm = pd.DataFrame(cm, index=class_names, columns=class_names) 
    pp_matrix(df_cm, cmap="Blues")

