import numpy as np
import pandas as pd
import tifffile as tiff
import tensorflow as tf
from PIL import Image, ImageDraw, ImageFont

def add_caption(frame: np.ndarray, text:str):

   # convert the frame back from 0-1 
    f= frame.astype("float32")
    f = f - f.min()
    f = f / f.max()
    f = (f * 255).astype("uint8")

    #sprint(f"frame shape is {f.shape}")


    img = Image.fromarray(f).convert("RGB") # converting to RGB PIL

    draw = ImageDraw.Draw(img)
    font = ImageFont.load_default()

    # adding outline for text so we can read it easily
    x = 5
    y = 5
    outline_range = 2
    for dx in range(-outline_range, outline_range + 1):
        for dy in range(-outline_range, outline_range + 1):
            draw.text((x + dx, y + dy), text, font = font, fill = "black")
    draw.text((x,y), text, font=font, fill = "white")

    return np.array(img)

def build_captioned_video(video: np.ndarray, df: pd.DataFrame):
    captioned_frames = []
    for _, row in df.iterrows():
        start, end, label = row["start_frame"], row["end_frame"], row["predicted_label"]
        for f_idx in range(start, end + 1): # apply label for each frAME
            captioned_frames.append(add_caption(video[f_idx], str(label)[0]))
    return np.stack(captioned_frames, axis = 0)


def tint_frame(frame: np.ndarray, colour, alpha: float = 0.2): 

    # convert the frame back from 0-1 
    f = frame.astype("float32") 
    f = f - f.min() 
    f = f / f.max() 
    f = (f * 255).astype("uint8") 

    img = Image.fromarray(f).convert("RGB") # converting to RGB PIL 
    arr = np.array(img).astype("float32") 
    colour = np.array(colour, dtype="float32") * 255 # scale colour to 0-255 range 

    # blend the original frame with the tint colour 
    tinted = (1 - alpha) * arr + alpha * colour 
    tinted = np.clip(tinted, 0, 255).astype("uint8") 

    return tinted 

state_colours = { "free": (0, 0, 1), # blue 
                "bound": (1, 0, 0), # red 
                "confined": (0, 1, 0), # green 
            } 

def build_tinted_video(video: np.ndarray, df: pd.DataFrame, alpha: float = 0.2): 
    tinted_frames = [] 
    for _, row in df.iterrows(): 
        start, end, label = row["start_frame"], row["end_frame"], row["predicted_label"] 
        colour = state_colours[label] 
        for f_idx in range(start, end + 1): 
            tinted_frames.append(tint_frame(video[f_idx], colour, alpha=alpha)) 

    return np.stack(tinted_frames, axis=0) 

