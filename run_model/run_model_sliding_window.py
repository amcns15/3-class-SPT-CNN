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
from khymograph_output import colour_khymograph_sliding_window

def preprocess_chunk(chunk):

    # print("chunk percentiles pre processing", np.percentile(chunk, [10, 25, 50, 75, 90]))

   # chunk = chunk.astype(np.float32)
    chunk = chunk / 255.0
    chunk = chunk[..., None] #Channel number axis
    chunk = chunk[None, ...] #Batch size axis

    # print("chunk shape:", chunk.shape)
    # print("chunk dtype:", chunk.dtype)
    # print("chunk percentiles", np.percentile(chunk, [10, 25, 50, 75, 90]))
    # print("chunk min/max:", chunk.min(), chunk.max())

    return chunk

def predict_states(video, model, store_frame_labels ,num, chunk_size = 5, class_names = ["free", "bound", "confined"]):


    print("Video shape:", video.shape)
    print("Video dtype:", video.dtype)
    print("Video min/max:", video.min(), video.max())
    print("Model input shape:", model.input_shape)
    print("Model output shape:", model.output_shape)
    
    n_frames = video.shape[0]
    n_chunks = n_frames // chunk_size
    remainder = n_frames % chunk_size

    if remainder != 0:
        print(f"Warning: {n_frames} frames is not divisible by {chunk_size}. "
              f"Last {remainder} frame(s) will be dropped.")

    results = []

    for i in range(n_chunks):
        start = i * chunk_size + num
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

        for frame in np.arange(start, end, 1):
            store_frame_labels.iloc[frame, num] = label

       # print(f"Frames {start}-{end - 1} in {name}: {label} (confidence {confidence:.3f})")

    result_df = pd.DataFrame(results)

    # OPTIONAL SCRIPT TO HOMOGENISE RESULTS
    # for chunk in range(1, len(results) - 1):
    #     if results.iloc[chunk - 1]["predicted_label"] == results.iloc[chunk + 1]["predicted_label"]:
    #         results.loc[chunk, "predicted_label"] = results.iloc[chunk - 1]["predicted_label"]        

    return result_df, store_frame_labels



if __name__ == "__main__":

    input_dir = Path(r"\\rivendell.physics.ox.ac.uk\user\students\2024\jesu4837\summer_internship\real_data\scaled_and_full")

    #input_video = tiff.imread(r"\\rivendell.physics.ox.ac.uk\user\students\2024\jesu4837\summer_internship\real_data\scaled\69_scaled.tif")

    model_loaded = keras.saving.load_model(r"C:\Users\jesu4837\Downloads\model_local_store\aug_17.keras",
                                           custom_objects = {"Conv2Plus1D": Conv2Plus1D} )

    model_loaded.summary()
    for layer in model_loaded.layers:
        print(layer.name, layer.__class__.__name__)

    for vid in input_dir.glob("*.tif"):
        vid_stem = vid.stem
        input_video = tiff.imread(vid)

        store_frame_labels = np.zeros((3000, 6))
        store_frame_labels[:, 0] = np.arange(0, 3000, 1)
        store_frame_labels_df = pd.DataFrame(store_frame_labels)


        for i in range(5):

            print(i)

            windowed_video = input_video[i:]

            results_df, store_frame_labels_df = predict_states(windowed_video, model_loaded, store_frame_labels_df, i)
            print(results_df , f"for i = {i}")

            # captioned_video = build_captioned_video(input_video, results_df)
            # tiff.imwrite(rf"\\rivendell.physics.ox.ac.uk\user\students\2024\jesu4837\summer_internship\run_model\labelled_videos\{vid_stem}_labelled_video.tif", captioned_video, photometric="rgb")

            name = vid_stem.split("_")[0]

           # colour_khymograph_sliding_window(name, results_df, i)

            store_frame_labels_df.to_csv(rf"H:\summer_internship\run_model\sliding_window_labels\{name}_per_frame_label.csv")

        print(results_df)



