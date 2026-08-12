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


def run_simulations(no_batches, states):

    output_dir = Path("//rivendell.physics.ox.ac.uk/user/students/2024/jesu4837/summer_internship/simulation_pipeline/smoldyn_output/")
    #mkdir if it doesn't exist

    for state, difc in states.items():
        for num in tqdm(range(no_batches)):

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

            print(f"Simulation for {state} state, batch {num+1} completed. Output saved to {output_file}")
    


if __name__ == "__main__":

    no_batches = 100

    states = {
        "free": 2,
       "bound": 0.0001
    }

    run_simulations(no_batches, states)