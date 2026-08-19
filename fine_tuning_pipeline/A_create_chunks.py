import os
from PIL import Image, ImageSequence
from pathlib import Path


def split_tif(states_dict, output_dir, frames_per_segment=5):
    # if not os.path.isfile(input_path):
    #     raise FileNotFoundError(f"Input file not found: {input_path}")

    print("here!")

    for state, path in states_dict.items():

        output_folder = output_dir + f"/{state}"
        os.makedirs(output_folder, exist_ok=True)

        input_path = Path(path)

        for video in input_path.glob("*.tif"):

            base_name = video.stem

            with Image.open(video) as img:
                frames = [frame.copy() for frame in ImageSequence.Iterator(img)]

            total_frames = len(frames)
            # if total_frames == 0:
            #     print("No frames found in the TIFF file.")
            #     return

            print(f"Found {total_frames} frame(s) in '{input_path}'.")

            segment_index = 1
            for start in range(0, total_frames - (total_frames % frames_per_segment), frames_per_segment):  #modulo makes sure we get a full x frame video
                segment_frames = frames[start:start + frames_per_segment]
                out_path = os.path.join(
                    output_folder, f"{base_name}_segment_{segment_index:03d}.tif"
                )

                if len(segment_frames) == 1:
                    segment_frames[0].save(out_path)
                else:
                    segment_frames[0].save(
                        out_path,
                        save_all=True,
                        append_images=segment_frames[1:],
                    )

                print(f"Saved {len(segment_frames)} frame(s) -> {out_path}")
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


