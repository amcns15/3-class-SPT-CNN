from pathlib import Path


from A_generate_smoldyn_trajectories import run_simulations
from B_generate_confined_trajectories import run_simulations as run_simulations_confined
from C_convert_to_smeagol_format import convert_smoldyn_to_smeagol
from D_generate_smeagol_vids import run_smeagol_simulations
from E_chop_up_tiffs import split_tif

if __name__ == "__main__":


# STEP ONE
    no_batches = 2

    states = {
        "free": 2,
       "bound": 0.0001
    }
    run_simulations(no_batches, states)

    states = {
        "confined": 1.5,
    }

    run_simulations_confined(no_batches, states)

# STEP TWO
    dir_out = r"\\rivendell.physics.ox.ac.uk\user\students\2024\jesu4837\summer_internship\simulation_pipeline\smeagol_input"

    dir_in = r"\\rivendell.physics.ox.ac.uk\user\students\2024\jesu4837\summer_internship\simulation_pipeline\smoldyn_output"
    convert_smoldyn_to_smeagol(dir_in, dir_out)

    dir_in = r"\\rivendell.physics.ox.ac.uk\user\students\2024\jesu4837\summer_internship\simulation_pipeline\smoldyn_output\confined_output"
    convert_smoldyn_to_smeagol(dir_in, dir_out)

# STEP THREE
    states = {
        "free": Path(r"\\rivendell.physics.ox.ac.uk\user\students\2024\jesu4837\summer_internship\simulation_pipeline\smeagol_input\free_smeagol_input"),
        "bound": Path(r"\\rivendell.physics.ox.ac.uk\user\students\2024\jesu4837\summer_internship\simulation_pipeline\smeagol_input\bound_smeagol_input"),#
        "confined": Path(r"\\rivendell.physics.ox.ac.uk\user\students\2024\jesu4837\summer_internship\simulation_pipeline\smeagol_input\confined_smeagol_input")
    }

    run_smeagol_simulations(states)

# STEP 4
    states_dict = {
        "0_free": Path(r"\\rivendell.physics.ox.ac.uk\user\students\2024\jesu4837\summer_internship\simulation_pipeline\smeagol_full_videos\free_smeagol_output"),
       "1_bound": Path(r"\\rivendell.physics.ox.ac.uk\user\students\2024\jesu4837\summer_internship\simulation_pipeline\smeagol_full_videos\bound_smeagol_output"),
       "2_confined" : Path(r"\\rivendell.physics.ox.ac.uk\user\students\2024\jesu4837\summer_internship\simulation_pipeline\smeagol_full_videos\confined_smeagol_output")
    }
    
    output_dir = "//rivendell.physics.ox.ac.uk/user/students/2024/jesu4837/summer_internship/raw_training_data"

    split_tif(states_dict, output_dir)
    
   


    
