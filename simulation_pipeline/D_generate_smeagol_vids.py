# RUNS ON MATLAB 2024a  !!!!!!

from pathlib import Path
import matlab.engine
import tqdm as tqdm


def run_smeagol_simulations(states: dict):

    engine = matlab.engine.start_matlab()
    print("Starting Matlab engine...")

    try:
        fid = engine.fopen(r"\\rivendell.physics.ox.ac.uk\user\students\2024\jesu4837\summer_internship\simulation_pipeline\smeagol_input\free_smeagol_input\free_smeagol_input_1.txt")
        print(fid)
    except Exception as e:
        print("Exception", repr(e))

    SMEAGOL_SETUP = Path(r"C:\Users\jesu4837\Downloads\SMeagol 1.0.2\SMeagol_setup.m")

    REACTION_FILE = Path(r"\\rivendell.physics.ox.ac.uk\user\students\2024\jesu4837\summer_internship\simulation_pipeline\no_reactions.txt")

    MATLAB_FUNCTION_FOLDER = Path(r"\\rivendell.physics.ox.ac.uk\user\students\2024\jesu4837\summer_internship\simulation_pipeline")

    TEMPLATE_RUNINPUT = Path(r"\\rivendell.physics.ox.ac.uk\user\students\2024\jesu4837\summer_internship\simulation_pipeline\parameters_for_simulation_realistic.m")

    engine.addpath(str(SMEAGOL_SETUP.parent), nargout = 0)
    
    engine.addpath(str(MATLAB_FUNCTION_FOLDER), nargout=0)

    runinput_root = TEMPLATE_RUNINPUT.parent
    
    try:

        for state_difc, input_dir in states.items():

           # state = state_difc.split('_')[0]
            state = state_difc
           # difc = state_difc.split('_')[1]

            # defining paths for each state
            INPUT_DIR = Path(input_dir)

            #print("input directory is: ", input_dir)

            OUTPUT_DIR = Path(r"\\rivendell.physics.ox.ac.uk\user\students\2024\jesu4837\summer_internship\simulation_pipeline\smeagol_full_videos" + f"/{state}_smeagol_output")

            OUTPUT_DIR.mkdir(parents = True, exist_ok = True)

            trajectory_files = sorted(INPUT_DIR.glob("*.txt"))

            for index, trajectory_file in enumerate(trajectory_files, start=1):

                # traj_rel = trajectory_file.relative_to(runinput_root)
                # react_rel = REACTION_FILE.relative_to(runinput_root)


                output_mat = OUTPUT_DIR / f"{state}_smeagol_output_{index}.mat"

                print(f"[{index}/{len(trajectory_files)}] Running SMeagol simulation for state '{state}' with trajectory file: {trajectory_file.name}")

                print("trajectory exists:", trajectory_file.exists())

                try: 
                    engine.run_one_smeagol(
                        str(SMEAGOL_SETUP),
                        str(TEMPLATE_RUNINPUT), 
                        str(trajectory_file),
                        str(REACTION_FILE), 
                        str(output_mat),
                        state,
                       # difc, 
                        nargout=0
                    )

                except matlab.engine.MatlabExecutionError as error:
                    print(f"Error occurred while running SMeagol for trajectory file '{trajectory_file.name}': {error}")
                    continue


    finally:
        engine.quit()
        print("Matlab engine closed.")

    return

if __name__ == "__main__":

    states = {
        "free": Path(r"\\rivendell.physics.ox.ac.uk\user\students\2024\jesu4837\summer_internship\simulation_pipeline\smeagol_input\free_smeagol_input"),
        "bound": Path(r"\\rivendell.physics.ox.ac.uk\user\students\2024\jesu4837\summer_internship\simulation_pipeline\smeagol_input\bound_smeagol_input"),
        "confined": Path(r"\\rivendell.physics.ox.ac.uk\user\students\2024\jesu4837\summer_internship\simulation_pipeline\smeagol_input\confined_smeagol_input")
    }

    run_smeagol_simulations(states)