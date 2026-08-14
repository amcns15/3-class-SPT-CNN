import keras
import numpy as np
import tifffile as tiff
import pandas as pd
import tensorflow as tf
from pathlib import Path

from model_info import Conv2Plus1D
from video_output import add_caption 
from video_output import build_captioned_video
from graph_output import plot_states
from khymograph_output import colour_khymograph

def preprocess_chunk(chunk):

    chunk = chunk.astype("float32") #Set data type
    chunk /= 255.0 #Normalise values
    chunk = chunk[..., None] #Channel number axis
    chunk = chunk[None, ...] #Batch size axis

    return chunk

def predict_states(video, model_one, model_two, name, chunk_size = 5, class_names = ["free", "bound", "confined"]):
    
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

        # ----- Stage 1: free vs not free -----
        stage_one_output = model_one.predict(input_arr, verbose=0)
        free_prob = float(stage_one_output[0, 0])

        if free_prob >= 0.5:
            # 1 = free
            label = "free"
            confidence = free_prob

        else:
            # ----- Stage 2: confined vs bound -----
            stage_two_output = model_two.predict(input_arr, verbose=0)
            confined_prob = float(stage_two_output[0, 0])

            if confined_prob >= 0.5:
                # 1 = confined
                label = "confined"
                confidence = confined_prob
            else:
                # 0 = bound
                label = "bound"
                confidence = 1.0 - confined_prob

        results.append({
            "start_frame": start,
            "end_frame": end - 1,
            "predicted_label": label,
            "confidence": round(confidence, 4),
        })

       # print(f"Frames {start}-{end - 1} in {name}: {label} (confidence {confidence:.3f})")

    return pd.DataFrame(results)




if __name__ == "__main__":

    input_dir = Path(r"\\rivendell.physics.ox.ac.uk\user\students\2024\jesu4837\summer_internship\real_data\scaled_and_full")

    #input_video = tiff.imread(r"\\rivendell.physics.ox.ac.uk\user\students\2024\jesu4837\summer_internship\real_data\scaled\69_scaled.tif")

    model_one = keras.saving.load_model(r"C:\Users\jesu4837\Downloads\model_local_store\stage_one.keras",
                                           custom_objects = {"Conv2Plus1D": Conv2Plus1D} )
    model_two = keras.saving.load_model(r"C:\Users\jesu4837\Downloads\model_local_store\stage_two.keras",
                                               custom_objects = {"Conv2Plus1D": Conv2Plus1D} )
    

    for vid in input_dir.glob("*.tif"):
        vid_stem = vid.stem
        input_video = tiff.imread(vid)
        results_df = predict_states(input_video, model_one, model_two, vid_stem)

        captioned_video = build_captioned_video(input_video, results_df)
        tiff.imwrite(rf"\\rivendell.physics.ox.ac.uk\user\students\2024\jesu4837\summer_internship\run_model\labelled_videos\{vid_stem}_labelled_video.tif", captioned_video, photometric="rgb")

       # plot_states(results_df)
        colour_khymograph(vid_stem.split("_")[0], results_df)

        print(results_df)



