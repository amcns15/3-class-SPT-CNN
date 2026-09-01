import tifffile
from pathlib import Path
import os

def split_tif(states_dict, output_dir, frames_per_segment=5):
    for state, path in states_dict.items():
        output_folder = os.path.join(output_dir, state)
        os.makedirs(output_folder, exist_ok=True)

        input_path = Path(path)

        for video in input_path.glob("*.tif"): # loop through directory for each class
            base_name = video.stem

            frames = tifffile.imread(video) # open with tiff
            total_frames = frames.shape[0]

            print(f"Found {total_frames} frame(s) in '{video}'.")

            segment_index = 1
            for start in range(0, total_frames - (total_frames % frames_per_segment), frames_per_segment): # discard frames at the end if there is not a multiple of 5
                segment_frames = frames[start:start + frames_per_segment]
                out_path = os.path.join(
                    output_folder, f"{base_name}_segment_{segment_index:03d}.tif"
                )

                tifffile.imwrite(out_path, segment_frames)  # store chunk

                print(f"Saved {segment_frames.shape[0]} frame(s) -> {out_path}")
                segment_index += 1

            print(f"Done. {segment_index - 1} segment file(s) written to '{output_folder}'.")


if __name__ == "__main__":

    states_dict = {
        "0_free": Path(r"H:\summer_internship\fine_tuning_pipeline\samples\free"),
       "1_bound": Path(r"H:\summer_internship\fine_tuning_pipeline\samples\bound"),
       "2_confined" : Path(r"H:\summer_internship\fine_tuning_pipeline\samples\confined")
    }
    
    output_dir = "H:/summer_internship/fine_tuning_pipeline/tuning_data"

    print("here")

    split_tif(states_dict, output_dir)


