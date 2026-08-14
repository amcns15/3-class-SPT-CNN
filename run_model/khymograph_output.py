import tifffile as tiff
import numpy as np


def colour_khymograph(video_number, results_df, alpha=0.2):

    input_path = (rf"\\rivendell.physics.ox.ac.uk\user\students\2024\jesu4837\summer_internship\run_model\khymographs\{video_number}_khymograph.tif")

    output_path = (rf"\\rivendell.physics.ox.ac.uk\user\students\2024\jesu4837\summer_internship\run_model\labelled_khymographs\{video_number}_labelled_khymograph.tif")

    # read image
    arr = tiff.imread(input_path)
    arr = np.squeeze(arr)

    # input stats
    # print("Kymograph:")
    # print("  shape:", arr.shape)
    # print("  dtype:", arr.dtype)
    # print("  min:", arr.min())
    # print("  max:", arr.max())

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

    # Stats about format of output for debugging
    # print("Output:")
    # print("  shape:", out.shape)
    # print("  dtype:", out.dtype)
    # print("  min:", out.min())
    # print("  max:", out.max())

    tiff.imwrite(output_path, out, photometric="rgb")

    return out