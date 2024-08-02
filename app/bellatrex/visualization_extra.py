import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde
from matplotlib.patches import FancyArrowPatch
from .utilities import frmt_pretty_print

def _input_validation(rules, preds, baselines, weights):
    """
    Validates the input parameters for consistency and computes the maximum rule length.

    Raises:
    - AssertionError: If the lengths of `rules`, `preds`, `baselines`, and `weights` are not equal.
    """
    assert len(rules) == len(preds) == len(baselines) == len(weights), "All input lists must have the same length."

    return None


def max_rulelength_visual(rules, max_rulelen=None):

    # Calculate the maximum rule length from the list of rules.
    max_rulelen_calculated = max(len(rule) for rule in rules)

    # Use the provided max_rulelen if it is not None and less than the calculated value.
    if max_rulelen is not None:
        max_rulelen = min(max_rulelen_calculated, max_rulelen)
    else:
        max_rulelen = max_rulelen_calculated

    return max_rulelen

def define_relative_position(n_cols):

    assert isinstance(n_cols, int), f'n_cols must be int, found {type(n_cols)} instead'

    if n_cols == 1:
        pos_list = ['center']
    elif n_cols == 2:
        pos_list = ['left', 'right']
    elif n_cols > 2:
        pos_list = n_cols*['center']
        pos_list[0] = 'left'
        pos_list[-1] = 'right'
    else:
        raise ValueError(f'n_cols but be positive integer, got {n_cols} instead')
    return pos_list

def convert_to_data_coords(ax, point):
    """
    Converts a point from axes fraction coordinates to data coordinates.
    """
    x_range = ax.get_xlim()
    y_range = ax.get_ylim()

    x_data = x_range[0] + point[0] * (x_range[1] - x_range[0])
    y_data = y_range[0] + point[1] * (y_range[1] - y_range[0])

    return (x_data, y_data)


def plot_arrow(ax, pos, weight, pred_out, fontsize, tot_digits):

    #Define start position of the arrow based on predicted value (min-max transformed)
    x_min, x_max = ax.get_xlim()
    x_start =  (pred_out-x_min)/(x_max-x_min)
    ax.axis('off')
    y_start = 0.98  # Starting height of the arrow

    if pos in ['c', 'center']:
        x_end, y_end = 0.5, 0.05

    elif pos in ['r', 'right']:
        x_end, y_end = -0.02, 0.05

    elif pos in ['l', 'left']:
        x_end, y_end = 1.02 , 0.05

    y_med = 0.4

    # Define points for the arrow segments
    points = [
        (x_start, y_start),  # Start
        (x_start, y_med),  # Turn point 1
        (x_end, y_med),  # Turn point 2
        (x_end, y_end),  # End (without arrowhead)
        (x_end, 0)  # End (with arrowhead)
    ]

    linestyles = [':', '-', '-', '-', '-'] #Differing line patterns
    alphas = [0.55, 0.8, 0.8, 1, 1] # some transparency in the first segment
    zorders = [0, 1, 1, 1, 1]
    weight_width = 1 + 0.7*(weight > 0.2) + 0.7*(weight > 0.35) + 0.7*(weight > 0.5)

    assert len(alphas) == len(points) == len(linestyles) == len(zorders)

    # Draw segments (all except final arrow head)
    for i in range(len(points) - 2):

        posA = convert_to_data_coords(ax, points[i])
        posB = convert_to_data_coords(ax, points[i + 1])

        # TODO drawn segments are separated by a small ( annoying) blank space. Fix

        arrow = FancyArrowPatch(
            posA=posA,
            posB=posB,
            arrowstyle='-',
            linewidth=weight_width,
            linestyle=linestyles[i],
            color='black',
            alpha=alphas[i],
            zorder=zorders[i]
        )
        arrow.set_clip_on(False)  # Disable clipping
        ax.add_patch(arrow)


    # Now all segemts are drawn, place text conveniently

    movesRight = 2*((x_end > x_start)-0.5)
    # traslate the text to the left from the end if x_end > x_start,
    # otherwise traslate to the right
    x_text = x_end - 0.23*movesRight

    y_text = 0.25*y_med+0.75*y_end

    ax.annotate(
        text=rf"{frmt_pretty_print(pred_out, tot_digits)}$\times${weight:.2f}",  # Displaying the product of prediction and weight
        xytext=(x_text, y_text),
        xy=(x_end, y_end + 0.25),
        xycoords="axes fraction",
        textcoords="axes fraction",
        fontsize=fontsize,
        ha='center', va='bottom')

    # Final arrowhead
    ax.annotate(
        text="",
        xytext=points[-2],
        xy=points[-1],
        xycoords="axes fraction",
        textcoords="axes fraction",
        arrowprops=dict(
            arrowstyle="-|>",
            linewidth=weight_width,
            mutation_scale=25,
            color='black')
    )

    return ax

if __name__ == "__main__":

    # fig, ax = plt.subplots()
    # ax.set_xlim(0, 0.6)
    # ax = plot_arrow(ax, 'center', 0.2, 0.35, 1.5, fontsize=12)  # Example with 'weight' as 'w', 'pred_out' as 'p', and 'color' as 'blue'
    # plt.show()

    fig, ax = plt.subplots()
    ax.set_xlim(0, 0.70)
    ax = plot_arrow(ax, 'c', 0.35, 0.65, fontsize=12)  # Example with 'weight' as 'w', 'pred_out' as 'p', and 'color' as 'blue'
    plt.show()





