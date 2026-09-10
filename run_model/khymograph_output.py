# from tkinter import Image
from PIL import Image

import tifffile as tiff
import numpy as np
import matplotlib.pyplot as plt


def colour_khymograph(video_number, results_df, alpha=0.2):

    input_path = (rf"\\rivendell.physics.ox.ac.uk\user\students\2024\jesu4837\summer_internship\run_model\khymographs\{video_number}_khymograph.tif")

    output_path = (rf"\\rivendell.physics.ox.ac.uk\user\students\2024\jesu4837\summer_internship\run_model\labelled_khymographs\{video_number}_labelled_khymograph.tif")

    # read image
    arr = tiff.imread(input_path)
    arr = np.squeeze(arr)

    # convert everything to 16-bit

    arr_float = arr.astype(np.float32)

    image_min = arr_float.min()
    image_max = arr_float.max()

    if arr.dtype == np.uint16: # if it is already 16 bit
        arr_scaled = arr_float

    else:
        # rescale arbitrary float/int image to 16 bit
        arr_scaled = ((arr_float - image_min) / (image_max - image_min) * 65535 )

    # converting greyscale to rgb
    h, w = arr_scaled.shape

    grey_rgb = np.repeat(arr_scaled[:, :, None], 3, axis=-1)

    out = grey_rgb.copy()

    # state colours mapped to eacg rgb channel
    state_colours = {
        "free": np.array( [0, 0, 1], dtype=np.float32), # blue
        "bound": np.array( [1, 0, 0] , dtype=np.float32), # red
        "confined": np.array( [0, 1, 0] , dtype=np.float32) # green
    }


    for i in range(w): # colouring each frame / column

        frame_chunk = i // 5
        state = results_df.iloc[frame_chunk]["predicted_label"]
        colour = state_colours[state]

        # Shape (height, 1)
        intensity = arr_scaled[:, i, None]

        # Colour has shape (3,), so result is (height, 3)
        coloured = intensity * colour

        # blending the original with the tint
        out[:, i, :] = ((1 - alpha) * grey_rgb[:, i, :] + alpha * coloured )


    out = np.clip(out, 0, 65535).astype(np.uint16) # Save as uintRGB


    tiff.imwrite(output_path, out, photometric="rgb")


    return out


def colour_khymograph_sliding_window(video_number, results_df, num ,alpha=0.2):

    input_path = (rf"\\rivendell.physics.ox.ac.uk\user\students\2024\jesu4837\summer_internship\run_model\khymographs\{video_number}_khymograph.tif")

    output_path = (rf"\\rivendell.physics.ox.ac.uk\user\students\2024\jesu4837\summer_internship\run_model\labelled_khymographs\{video_number}_{num + 1}_of_5_labelled_khymograph_SW.tif")

    # read image
    arr = tiff.imread(input_path)
    arr = np.squeeze(arr)

    # convert everything to 16-bit

    arr_float = arr.astype(np.float32)

    image_min = arr_float.min()
    image_max = arr_float.max()

    if arr.dtype == np.uint16: # if it is already 16 bit
        arr_scaled = arr_float

    else:
        # rescale arbitrary float/int image to 16 bit
        arr_scaled = ((arr_float - image_min) / (image_max - image_min) * 65535 )

    # converting greyscale to rgb
    h, w = arr_scaled.shape

    grey_rgb = np.repeat(arr_scaled[:, :, None], 3, axis=-1)

    out = grey_rgb.copy()

    # state colours mapped to eacg rgb channel
    state_colours = {
        "free": np.array( [0, 0, 1], dtype=np.float32), # blue
        "bound": np.array( [1, 0, 0] , dtype=np.float32), # red
        "confined": np.array( [0, 1, 0] , dtype=np.float32) # green
    }

    n_chunks = len(results_df)
    last_valid_frame = num + n_chunks * 5

    for i in range(w): # colouring each frame / column

        if i < num or i >= last_valid_frame:
            out[:, i, :] = grey_rgb[:, i, :]
            continue

        frame_chunk = (i - num) // 5
        state = results_df.iloc[frame_chunk]["predicted_label"]
        colour = state_colours[state]

        # Shape (height, 1)
        intensity = arr_scaled[:, i, None]

        # Colour has shape (3,), so result is (height, 3)
        coloured = intensity * colour

        # blending the original with the tint
        out[:, i, :] = ((1 - alpha) * grey_rgb[:, i, :] + alpha * coloured )


    out = np.clip(out, 0, 65535).astype(np.uint16) # Save as uintRGB


    tiff.imwrite(output_path, out, photometric="rgb")

    return out

def colour_khymograph_with_bar_graph(video_number, results_df, alpha=0.2):

    input_path = (rf"\\rivendell.physics.ox.ac.uk\user\students\2024\jesu4837\summer_internship\run_model\khymographs\{video_number}_khymograph.tif")
    output_path = (rf"\\rivendell.physics.ox.ac.uk\user\students\2024\jesu4837\summer_internship\run_model\labelled_khymographs\{video_number}_labelled_khymograph_FT.png")

    # read image
    arr = tiff.imread(input_path)
    arr = np.squeeze(arr)

    # convert everything to 16-bit
    arr_float = arr.astype(np.float32)
    image_min = arr_float.min()
    image_max = arr_float.max()

    if arr.dtype == np.uint16:
        arr_scaled = arr_float
    else:
        arr_scaled = ((arr_float - image_min) / (image_max - image_min) * 65535)

    # converting greyscale to rgb
    h, w = arr_scaled.shape
    grey_rgb = np.repeat(arr_scaled[:, :, None], 3, axis=-1)
    out = grey_rgb.copy()

    state_colours = {
        "free": np.array([0, 0, 1], dtype=np.float32),
        "bound": np.array([1, 0, 0], dtype=np.float32),
        "confined": np.array([0, 1, 0], dtype=np.float32)
    }

    # per-column state and confidence, built once per column loop
    col_state = []
    col_confidence = []

    for i in range(w):
        frame_chunk = i // 5
        row = results_df.iloc[frame_chunk]
        state = row["predicted_label"]
        confidence = row["confidence"]
        colour = state_colours[state]

        intensity = arr_scaled[:, i, None]
        coloured = intensity * colour
        out[:, i, :] = ((1 - alpha) * grey_rgb[:, i, :] + alpha * coloured)

        col_state.append(state)
        col_confidence.append(confidence)

    out_uint8 = (out / 65535 * 255).clip(0, 255).astype(np.uint8)

    # --- build combined figure: kymograph on top, confidence bar chart below ---
    fig_w = max(6, w / 100)
    fig, (ax_img, ax_bar) = plt.subplots(
        2, 1, figsize=(fig_w, 6),
        gridspec_kw={"height_ratios": [3, 1]},
        sharex=True
    )

    ax_img.imshow(out_uint8, aspect="auto", extent=[0, w, h, 0])
    ax_img.set_ylabel("Position")
    ax_img.set_xlim(0, w)

    bar_colours = [state_colours[s] for s in col_state]
    ax_bar.bar(
        range(w), col_confidence,
        width=1.0, color=bar_colours, align="edge"
    )
    ax_bar.set_xlim(0, w)
    ax_bar.set_ylim(0, 1)
    ax_bar.set_ylabel("Confidence")
    ax_bar.set_xlabel("Frame")

    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close(fig)

    return 
