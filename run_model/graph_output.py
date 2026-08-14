import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

def plot_states(results_df):
    states = {"free" : 0, 
              "bound" : 1,
              "confined" : 2
            }
    
    results_df["label_numeric"] = results_df["predicted_label"].map(states)

    x = results_df["start_frame"]
    y = results_df["label_numeric"]

    # Adding line between each state and coloured circles corresponding to the state
    plt.plot(x, y, "k--", linewidth=1)
    colours = { # dictionary with colours for each state
        "free": "blue",
        "bound": "red",
        "confined": "green"
    }
    for state, colour in colours.items():
        mask = results_df["predicted_label"] == state
        plt.scatter(
            results_df.loc[mask, "start_frame"],
            results_df.loc[mask, "label_numeric"],
            color=colour,
            marker="o",
            label=state
        )

    # axis labels and ticks
    plt.xlabel("Frame")
    plt.ylabel("State")
    plt.yticks([0, 1, 2], ["Free", "Bound", "Confined"])

    # Title and legend
    plt.title("State Classified by Model Against Frame")
    plt.legend()

    plt.tight_layout()
    #plt.show()

    return