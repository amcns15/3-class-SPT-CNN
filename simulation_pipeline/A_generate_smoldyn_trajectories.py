# run smoldyn
        # convert to smeagol
        # run smeagol
        # chop up states
        # train model

import smoldyn
import os
import csv
import numpy as np
from os import name
from pathlib import Path
from tqdm import tqdm


def run_simulations(no_batches, state_ranges):

    output_dir = Path("//rivendell.physics.ox.ac.uk/user/students/2024/jesu4837/summer_internship/simulation_pipeline/smoldyn_output/")
    output_dir.mkdir(parents=True, exist_ok=True)

    for state, (low, high) in state_ranges.items():

        # 100 poisson-distributed coefficients across the given range
        difc_values = np.random.poisson(1.16, no_batches) # Generate Poisson-distributed values
        difc_values = np.clip(difc_values, low, high) # Ensure values are within the specified range

        for num, difc in enumerate(tqdm(difc_values)):

            # creating a separate simulation for each run
            sim = smoldyn.Simulation(
                low = [-1.0, -0.4, -0.4],
                high = [1.0, 0.4, 0.4],
            )

            sim.setGraphics("none")

            fluorophore = sim.addSpecies(
                name = "fluorophore",
                difc = difc,
                color = "blue"
            )

            fluorophore.addToSolution(
                1,
                lowpos = [-0.8, -0.3, -0.3],
                highpos = [0.8, 0.3, 0.3]
            )

            output_file = output_dir / f"{state}_simulations_{num+1}.csv"

            sim.setOutputFile(str(output_file), append = False)

            # sim.addCommand(
            #     cmd = "set output format csv",
            #     cmd_type  ="B"
            # )

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

            print(f"Simulation for {state} state (difc={difc:.4f}), batch {num+1} completed. Output saved to {output_file}")


if __name__ == "__main__":

    no_batches = 100

    # (low, high) diffusion coefficient ranges for each state
    state_ranges = {
        "free": (2.5, 5),
  #      "bound": (0.00.19),
    }

    run_simulations(no_batches, state_ranges)