# run smoldyn
        # convert to smeagol
        # run smeagol
        # chop up states
        # train model

import smoldyn
import os
import csv
from os import name
from pathlib import Path
from tqdm import tqdm
import random


def run_simulations(no_batches, states):

    output_dir = Path(r"\\rivendell.physics.ox.ac.uk\user\students\2024\jesu4837\summer_internship\simulation_pipeline\smoldyn_output\confined_output")
    #mkdir if it doesn't exist

    for state, difc in states.items():
        for num in tqdm(range(no_batches)):

            x_origin = random.uniform(-0.8, 0.8)
            y_origin = random.uniform(-0.2, 0.2)
            z_origin = random.uniform(-0.2, 0.2)

            x_min = x_origin - 0.1
            x_max = x_origin + 0.1
            y_min = y_origin - 0.1
            y_max = y_origin + 0.1
            z_min = z_origin - 0.1
            z_max = z_origin + 0.1

        # creating a separate simulation for each run
            sim = smoldyn.Simulation(
                low = [x_min, y_min, z_min],
                high = [x_max, y_max, z_max],
            )

            sim.setGraphics("none")


            fluorophore = sim.addSpecies(
                name = "fluorophore",
                difc = difc,
                color = "blue"
            )

            fluorophore.addToSolution(
                1,
                lowpos = [x_min, y_min, z_min],
                highpos = [x_max, y_max, z_max]
            )

            output_file = output_dir / f"{state}_simulations_{num+1}.csv"

            sim.setOutputFile(str(output_file), append = False)

            sim.addCommand(
                cmd = f"molpos fluorophore {output_file.name}",
                cmd_type = "E",
            )

            sim.run(
                stop = 100, 
                dt = 0.01, 
                display = False,
                overwrite = True
            )

            print(f"Simulation for {state} state, batch {num+1} completed. Output saved to {output_file}")
    


if __name__ == "__main__":

    no_batches = 100

    states = {
        "confined": 1.5,
    }

    run_simulations(no_batches, states)