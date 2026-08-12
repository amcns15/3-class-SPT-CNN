import keras
import numpy as np
import tifffile as tiff
import pandas as pd
import tensorflow as tf
from pathlib import Path

from model_info import Conv2Plus1D

def preprocess_chunk(chunk):

    chunk = chunk.astype("float32") #Set data type
    chunk /= 255.0 #Normalise values
    chunk = chunk[..., None] #Channel number axis
    chunk = chunk[None, ...] #Batch size axis

    return chunk

def predict_states(video, model, name, chunk_size = 5, class_names = ["free", "bound", "confined"]):
    
    n_frames = video.shape[0]
    n_chunks = n_frames // chunk_size
    remainder = n_frames % chunk_size

    if remainder != 0:
        print(f"Warning: {n_frames} frames is not divisible by {chunk_size}. "
              f"Last {remainder} frame(s) will be dropped.")

    results = []

    for i in range(n_chunks):
        start = i * chunk_size
        end = start + chunk_size
        chunk = video[start:end]  # (5, H, W, C)

        input_arr = preprocess_chunk(chunk)
        output = model.predict(input_arr, verbose=0)  # shape (1, n_classes)

        pred_idx = int(output.argmax(axis=1)[0])
        confidence = float(output[0, pred_idx])
        label = class_names[pred_idx]

        results.append({
            "start_frame": start,
            "end_frame": end - 1,
            "predicted_label": label,
            "confidence": round(confidence, 4),
        })

        print(f"Frames {start}-{end - 1} in {name}: {label} (confidence {confidence:.3f})")

    return pd.DataFrame(results)




if __name__ == "__main__":

    input_dir = Path(r"\\rivendell.physics.ox.ac.uk\user\students\2024\jesu4837\summer_internship\real_data\scaled")

    input_video = tiff.imread(r"\\rivendell.physics.ox.ac.uk\user\students\2024\jesu4837\summer_internship\real_data\scaled\69_scaled.tif")

    model_loaded = keras.saving.load_model(r"C:\Users\jesu4837\Downloads\model_local_store\initial_three_state.keras",
                                           custom_objects = {"Conv2Plus1D": Conv2Plus1D} )

    for vid in input_dir.glob("*.tif"):
        vid_stem = vid.stem
        input_video = tiff.imread(vid)
        predict_states(input_video, model_loaded, vid_stem)
