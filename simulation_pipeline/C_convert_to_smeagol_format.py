import pandas as pd
import numpy as np
from pathlib import Path
from os import name
import os
import tqdm as tqdm

# only works for single molecules

def convert_smoldyn_to_smeagol(dir_in: str, dir_out: str):

    input_path = Path(dir_in)

    for item in tqdm.tqdm(input_path.glob("*.csv")):

            state = item.stem.split("_")[0]
            number = item.stem.split("_")[2]    

            directory = Path(dir_out) / f"{state}_smeagol_input"
            directory.mkdir(parents=True, exist_ok = True)

            # Read whitespace-separated columns directly.
            df = pd.read_csv(
                item,
                sep=r"\s+",
                header=None,
                names=["time", "x", "y", "z"],
            )

            # Convert numeric columns and raise an error for invalid values.
            numeric_columns = ["time", "x", "y", "z"]
            df[numeric_columns] = df[numeric_columns].apply(
                pd.to_numeric,
                errors="raise",
            )

            df.insert(1, "molecule_id", 1)
            df.insert(2, "species", state)

            # df = pd.read_csv(item, names = ["text"])
            # df["text"].str.split(" ", expand = True).rename(columns = {0: "time", 1: "x", 2: "y", 3: "z"})

            # print(df.head())

            # # insert columns for molecule id and species
            # df.insert(1, column = "molecule_id", value = [1 for n in  range(df.shape[0])])
            # df.insert(2, column="species", value = [state for n in range(df.shape[0])])
            
            time_col = "time"
            id_col = "molecule_id"
            species_col = "species"
            x_col = "x"
            y_col = "y"
            z_col = "z"

            # # reorder columns to fit SMeagol
            # df = df[[time_col, id_col, species_col, x_col, y_col, z_col]].copy()
            # df[time_col] = pd.to_numeric(df[time_col], errors="raise")

            txt_out = directory / f"{state}_smeagol_input_{number}.txt"

            # convert to text format
            with open(txt_out, "w") as f:
                for t, group in df.groupby(time_col, sort=True):
                    parts = [f"{t:.12g}"]
                    for _, row in group.iterrows():
                        parts.extend([
                            str(row[id_col]),
                            str(row[species_col]),
                            f"{row[x_col]:.12g}",
                            f"{row[y_col]:.12g}",
                            f"{row[z_col]:.12g}",
                        ])
                    f.write(" ".join(parts) + "\n")

    return


if __name__ == "__main__":

    #dir_in = r"\\rivendell.physics.ox.ac.uk\user\students\2024\jesu4837\summer_internship\simulation_pipeline\smoldyn_output"
    dir_in = r"\\rivendell.physics.ox.ac.uk\user\students\2024\jesu4837\summer_internship\simulation_pipeline\smoldyn_output\confined_output"

    dir_out = r"\\rivendell.physics.ox.ac.uk\user\students\2024\jesu4837\summer_internship\simulation_pipeline\smeagol_input"


    convert_smoldyn_to_smeagol(dir_in, dir_out)