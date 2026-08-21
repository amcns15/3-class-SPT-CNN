import numpy as np
from scipy.optimize import curve_fit
from scipy.ndimage import gaussian_filter
#import pandas as pd
import tifffile as tiff
#from PIL import Image, ImageDraw, ImageFont
from pathlib import Path
import pandas as pd
from itertools import groupby
import matplotlib.pyplot as plt
import os

def get_square_psf(coord, shape):
    OFFSETS = np.array([(-1,-1), (-1, 0), (-1,1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)])
    coord = np.array(coord)
    neighbors = coord + OFFSETS
    rows, cols = shape
    valid = (neighbors[:,0] >= 0) & (neighbors[:,0] < rows) & (neighbors[:,1] >= 0) & (neighbors[:,1] < cols)

    return neighbors[valid]

def estimate_std_from_fwhm(profile, peak_idx, background):
    peak_val = profile[peak_idx]
    half_max = background + (peak_val - background) / 2

    left = peak_idx
    while left > 0 and profile[left] > half_max:
        left -= 1

    right = peak_idx
    while right < len(profile) - 1 and profile[right] > half_max:
        right += 1
    

    fwhm = right-left
    sigma = fwhm / 2.355

    sigma_bound_lower = max(sigma, 0.5)

    return min(10, sigma_bound_lower)


def gaussian_2d(coords, amplitude, x0, y0, sigma_x, sigma_y, offset):
    x, y = coords
    exponent = -(((x - x0)**2) / (2 *  sigma_x**2) + ((y - y0)**2) / (2 * sigma_y**2))
    return offset + amplitude * np.exp(exponent)

def get_gaussian_psf(slice_2d):

    slice_2d = gaussian_filter(slice_2d, sigma=2)
    
    y_size, x_size = slice_2d.shape
    x = np.arange(x_size)
    y = np.arange(y_size)
    x_grid, y_grid = np.meshgrid(x,y)

    x_flat = x_grid.ravel()
    y_flat = y_grid.ravel()
    z_flat = slice_2d.ravel()

    y0_guess, x0_guess = np.unravel_index(np.argmax(slice_2d), slice_2d.shape)
    background =  slice_2d.min()

    row_profile = slice_2d[y0_guess, :] # horizontal profile
    col_profile = slice_2d[:, x0_guess] # verical profile

    sigma_x_guess = estimate_std_from_fwhm(row_profile, x0_guess, background)
    sigma_y_guess = estimate_std_from_fwhm(col_profile, y0_guess, background)

    initial_guess = (
        slice_2d.max(), # amplitude
        x0_guess, #x0
        y0_guess, #y0
        sigma_x_guess, #sigma_x
        sigma_y_guess, #sigma_y
        background  #offset
    )

    #print(initial_guess)

    try:
        popt, pcov = curve_fit(gaussian_2d, (x_flat, y_flat), z_flat, p0=initial_guess, bounds = ([0, 0, 0, 0.3, 0.3, 0], [65535, x_size, y_size, 10, 20, 65535]) )
    except RuntimeError:
        print("Gaussian fit failed")
        return None
    

    amplitude, x0, y0, sigma_x, sigma_y, offset = popt
    distance_in_std = np.sqrt(((x_grid - x0) / sigma_x)**2 + ((y_grid - y0) / sigma_y)**2)
    mask = distance_in_std <= 1
    #coords = np.argwhere(mask)

    return mask

def make_confinement_violin_plot(input_array, chunk_dict, images, vid_name, plot_dir, brightness_threshold=350):
    """
    For a given video, build one violin plot per label, where each violin
    shows the distribution of confinement (spot) sizes for that label across
    its chunks. Size = number of pixels in the integrated (summed then
    binarized) spot image, i.e. the footprint of the spot across the frames
    in that chunk. Chunks whose mean raw brightness (under the spot mask,
    across their frames) exceeds brightness_threshold are excluded.
    """
    label_sizes = {}   # label -> list of sizes
    excluded = 0

    for key, chunks in chunk_dict.items():
        for i, indices in enumerate(chunks):
            img_key = f"{key}_{i}_size_{len(indices)}"
            img = images.get(img_key)
            if img is None:
                continue

            mask = img > 0
            size = int(mask.sum())
            if size == 0:
                continue

            # brightness = mean raw intensity under the spot, over its frames
            region_frames = input_array[indices]  # shape (n_frames, H, W)
            brightness = region_frames[:, mask].mean()

            # if brightness > brightness_threshold:
            #     excluded += 1
            #     continue

            label_sizes.setdefault(key, []).append(size)

    if not label_sizes:
        print(f"No regions left for {vid_name} after excluding brightness > {brightness_threshold}")
        return

    labels = sorted(label_sizes.keys())
    data = [label_sizes[lab] for lab in labels]

    fig, ax = plt.subplots(figsize=(max(6, len(labels) * 0.8), 5))
    positions = np.arange(1, len(labels) + 1)
    parts = ax.violinplot(data, positions=positions, showmeans=True, showmedians=True)

    ax.set_xticks(positions)
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.set_ylabel("Confinement size (pixels)")
    ax.set_title(f"Confinement size distribution by label — {vid_name}\n"
                 f"(excluded {excluded} outliers, brightness > {brightness_threshold})")
    fig.tight_layout()
    fig.savefig(os.path.join(plot_dir, f"{vid_name}_confinement_violin.png"))
    plt.close(fig)

    return

def create_clean_array(input_array, per_frame_label, output_dir, vid_name, plot_dir):

    print(input_array.shape)

    output_array = np.zeros_like(input_array)

    for i, slice_2d in enumerate(input_array): # colour the ouput array with where the molecule is, a clean 3x3 square
        max_coords = np.unravel_index(np.argmax(slice_2d), slice_2d.shape)
        # output_array[i, max_coords[0], max_coords[1]] = 1
        mask = get_gaussian_psf(slice_2d)

        if mask is None:
            plt.imsave(rf"H:\summer_internship\analysis\failed_frames\frame{i}_failed.png", slice_2d, cmap = "gray")
            continue

        output_array[i, mask] = 1

    columns = per_frame_label.iloc[:, 1].tolist()

    result = []
    idx = 0
    for key, group in groupby(columns):
        group_list = list(group)
        indices = list(range(idx, idx + len(group_list)))
        result.append((key, indices))
        idx += len(group_list)

    chunk_dict = {} # find the indecies that all have the same label and are in a group together
    for key, indices in result:
        chunk_dict.setdefault(key, []).append(indices)

    # summing frames to create images
    images = {}

    for key, chunks in chunk_dict.items():
        for i, indices in enumerate(chunks):
            image = output_array[indices].sum(axis = 0)
            image[image != 0] = 1
            if np.sum(image) < 250:
                images[f"{key}_{i}_size_{len(indices)}"] = image

   # print(chunk_dict)

    y = []
    x = np.arange(len(images)) 
    labels = [] 
    base_labels = []

    vid_output_dir = os.path.join(output_dir, vid_name)
    os.makedirs(vid_output_dir, exist_ok=True)


    for key, img in images.items(): 
        y.append(np.sum(img)) # fixed: scalar pixel count, not sum(img) 
        labels.append(key) 
        base_labels.append(key.rsplit("_", 3)[0]) # strip off "_i_size_n" to get the raw label 
        plt.imsave(os.path.join(vid_output_dir, f"{key}.png"), img, cmap="gray") 
        print("saved to: ",vid_output_dir, f"{key}.png")

        # map each distinct base label to a colour 
    unique_labels = sorted(set(base_labels))
    label_colour_map = {"bound": "red", "free": "blue", "confined": "green"}
    colour_map = {lab: label_colour_map.get(lab, "grey") for lab in unique_labels}
    colours = [colour_map[lab] for lab in base_labels]

    fig, ax = plt.subplots(figsize=(max(6, len(x) * 0.4), 5)) 
    ax.bar(x, y, color=colours) 
    ax.set_xticks(x) 
    ax.set_xticklabels(labels, rotation=90, fontsize=7) 
    ax.set_ylabel("PSF pixel count") 
    ax.set_title(vid_name) # legend for the label colours 

    handles = [plt.Rectangle((0, 0), 1, 1, color=colour_map[lab]) for lab in unique_labels] 
    ax.legend(handles, unique_labels, title="label", loc="upper right", fontsize=8) 
    fig.tight_layout() 
    fig.savefig(os.path.join(plot_dir, f"{vid_name}_plot.png"))
    print("saved!")
    plt.close(fig)

    return output_array, chunk_dict, images


if __name__ == "__main__":

    input_dir = Path(r"\\rivendell.physics.ox.ac.uk\user\students\2024\jesu4837\summer_internship\real_data\cut_bleached")

    os.makedirs(r"H:\summer_internship\analysis\failed_frames", exist_ok=True)

    print("here_1")

    plot_dir = r"H:\summer_internship\analysis\area_plots"

    for vid in input_dir.glob("*.tif"):

        name = vid.stem.split("_")[0]
        print(name)
        per_frame_label = pd.read_csv(rf"h:\summer_internship\analysis\label_per_frame\{name}_per_frame_label.csv")

        output_dir = "H:/summer_internship/analysis/regions/"
        os.makedirs(output_dir, exist_ok=True)

        print("here")
        vid_stem = vid.stem
        input_video = tiff.imread(vid)

        per_frame_label = per_frame_label.iloc[:input_video.shape[0]]


        output_array, chunk_dict, images = create_clean_array(input_video, per_frame_label, output_dir, vid_stem, plot_dir)
        make_confinement_violin_plot(input_video, chunk_dict, images, vid_stem, r"H:\summer_internship\analysis\violin_plots", brightness_threshold=350)

    plt.show()



