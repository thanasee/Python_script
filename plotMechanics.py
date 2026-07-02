#!/usr/bin/env python

from sys import argv, exit
import readline
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# Set Times New Roman (with fallback)
matplotlib.rcParams['font.family'] = 'serif'
matplotlib.rcParams['font.serif'] = ['Times New Roman', 'Times', 'DejaVu Serif']
matplotlib.rcParams['mathtext.fontset'] = 'stix'   # for math consistency

COLORS = ['red', 'green', 'blue', 'cyan', 'magenta', 'orange']

PROPERTY_MAP = {
    "1": {"data_type": "Young's Modulus",  "unit": "N/m", "decimal": 0, "save_file": "Young.png"},
    "2": {"data_type": "Poisson's Ratio",  "unit": "",    "decimal": 2, "save_file": "Poisson.png"},
    "3": {"data_type": "Shear Modulus",    "unit": "N/m", "decimal": 0, "save_file": "Shear.png"},
}


def usage():
    """Print usage information and exit."""
    print("""
Usage: plotMechanics <input file> [input file2 ... input file6]

This script plot Young's modulus and Poisson's Ratio as functions of crystal orientation.
Title can choose by:
1 - Young's Modulus
2 - Poisson's Ratio
3 - Shear Modulus

This script was inspired by Klichchupong Dabsamut
and developed by Thanasee Thanasarnsurapong.
""")
    exit(0)


def ask_property():
    """
    Prompt the user to select a mechanical property type.
 
    Presents a numbered menu (Young's Modulus, Poisson's Ratio,
    Shear Modulus) and loops until a valid choice is entered.
 
    Returns
    -------
    dict
        Metadata for the chosen property, containing:
        - data_type (str)  : display name of the property
        - unit      (str)  : physical unit string, empty if dimensionless
        - decimal   (int)  : number of decimal places for formatting
        - save_file (str)  : output PNG filename
        - title     (str)  : axis label combining data_type and unit
    """
    print("""Enter type of your data
1) Young's Modulus
2) Poisson's Ratio
3) Shear Modulus""")
    while True:
        choice = input()
        if choice in PROPERTY_MAP:
            meta = PROPERTY_MAP[choice]
            data_type = meta["data_type"]
            unit = meta["unit"]
            title = f"{data_type} ({unit})" if unit else data_type
            return {**meta, "title": title}
        print("Choose again!")
 
 
def ask_material_label(index):
    """
    Prompt the user for a non-empty name for a material.
 
    Loops until a non-empty string is entered.
 
    Parameters
    ----------
    index : int
        1-based position of the material in the input file list,
        used in the prompt message.
 
    Returns
    -------
    str
        The material label entered by the user.
    """
    while True:
        label = input(f"Enter name of {index} material: ")
        if label:
            return label
 
 
def ask_positive_float(prompt):
    """
    Prompt the user for a positive floating-point number.
 
    Loops until a valid positive float is entered, handling both
    non-numeric input (ValueError) and non-positive values.
 
    Parameters
    ----------
    prompt : str
        The message displayed to the user before input.
 
    Returns
    -------
    float
        A positive float value entered by the user.
    """
    while True:
        try:
            value = float(input(prompt))
            if value > 0:
                return value
            print("Must be a positive number.")
        except ValueError:
            print("Invalid input. Enter a valid number.")
 
 
def ask_step(ymax):
    """
    Suggest valid step sizes then prompt the user to choose one.
 
    Computes integer factors of ymax as step size candidates and
    prints them before prompting. Loops until a valid step is entered
    (positive and strictly less than ymax).
 
    Parameters
    ----------
    ymax : float
        The maximum radial value, used both for factor suggestion
        and as the upper bound for step validation.
 
    Returns
    -------
    float
        A valid step size entered by the user.
    """
    factors = get_factors(ymax)
    print(f"Suggested step sizes (factors of {int(ymax)}): {factors}")
    while True:
        try:
            step = float(input("Enter your step size: "))
            if 0 < step < ymax:
                return step
            print(f"Step must be positive and less than {ymax}.")
        except ValueError:
            print("Invalid input. Enter a valid number.")
 
 
def load_data(filepath):
    """
    Read orientation angle and property value columns from a data file.
 
    Skips blank lines and lines beginning with '#'. Expects two
    whitespace-separated columns: angle (degrees) and property value.
 
    Parameters
    ----------
    filepath : str
        Path to the input data file.
 
    Returns
    -------
    degrees : numpy.ndarray, shape (N,)
        Orientation angles in degrees.
    radians : numpy.ndarray, shape (N,)
        The same angles converted to radians.
    values : numpy.ndarray, shape (N,)
        Mechanical property values at each orientation angle.
    """
    with open(filepath, 'r') as f:
        lines = np.array(
            [line.split() for line in f
             if line.strip() and not line.strip().startswith('#')],
            dtype=float
        )
    degrees = lines[:, 0]
    radians = np.radians(degrees)
    values  = lines[:, 1]
 
    return degrees, radians, values
 
 
def get_factors(n):
    """
    Return all positive integer factors of n in ascending order.
 
    Uses trial division up to sqrt(n) and collects both the divisor
    and its complement for efficiency.
 
    Parameters
    ----------
    n : float or int
        The value to factorize. Rounded to the nearest integer
        before computation.
 
    Returns
    -------
    list of int
        Sorted list of all integer factors of round(n).
    """
    n = int(round(n))
    factors = sorted(set(
        f for i in range(1, int(n**0.5) + 1) if n % i == 0
        for f in (i, n // i)
    ))
 
    return factors
 
 
def build_tick_labels(ymax, step, decimal):
    """
    Build a symmetric array of tick positions from -ymax to +ymax.
 
    Generates evenly spaced ticks at multiples of step, mirrored
    about zero and including zero itself.
 
    Parameters
    ----------
    ymax : float
        Maximum absolute value on the axis.
    step : float
        Spacing between consecutive ticks.
    decimal : int
        Number of decimal places for rounding tick values.
 
    Returns
    -------
    numpy.ndarray
        Sorted array of tick positions:
        [-N*step, ..., -step, 0, step, ..., N*step]
        where N = floor(ymax / step).
    """
    number = int(np.floor(ymax / step))
    tick = np.arange(1, number + 1) * step
    labels = np.round(np.concatenate((-tick[::-1], [0.0], tick)), decimals=decimal)
 
    return labels
 
 
def setup_figure():
    """
    Create the matplotlib figure with a Cartesian and a polar axis.
 
    Both axes share the same subplot position (111). The Cartesian
    axis (ax) is used only for the y-axis label and tick labels on
    the left side; the polar axis (axp) carries the actual data.
 
    Returns
    -------
    fig : matplotlib.figure.Figure
    ax  : matplotlib.axes.Axes
        Background Cartesian axis for y-axis decoration.
    axp : matplotlib.axes.Axes (polar)
        Foreground polar axis for data curves.
    """
    fig = plt.figure(dpi=300)
    ax  = fig.add_subplot(111)
    axp = fig.add_subplot(111, polar=True)
 
    return fig, ax, axp
 
 
def plot_material(axp, radians, values, color, label, neg_envelope_labeled):
    """
    Plot one material's polar curve, with an optional negative envelope.
 
    If any value is negative, the absolute-value envelope is drawn as
    a dashed line in the same color. The envelope receives a legend
    entry ("|value| envelope") only for the first material that
    triggers it; subsequent materials use "_nolegend_" to avoid
    duplicate legend entries.
 
    Parameters
    ----------
    axp : matplotlib.axes.Axes (polar)
        The polar axis to draw on.
    radians : numpy.ndarray
        Orientation angles in radians.
    values : numpy.ndarray
        Mechanical property values (may include negatives).
    color : str
        Line color for this material.
    label : str
        Legend label for this material.
    neg_envelope_labeled : bool
        Whether the envelope legend entry has already been created
        by a previous material.
 
    Returns
    -------
    bool
        Updated neg_envelope_labeled flag (True once an envelope
        legend entry has been added).
    """
    if (values < 0.0).any():
        neg_label = "|value| envelope" if not neg_envelope_labeled else "_nolegend_"
        axp.plot(radians, np.abs(values), linestyle='dashed', linewidth=1.5,
                 color=color, label=neg_label)
        neg_envelope_labeled = True
 
    axp.plot(radians, values, linestyle='solid', linewidth=2, color=color, label=label)
 
    return neg_envelope_labeled
 
 
def configure_cartesian_axis(ax, sub_y_labels, ymax, title, decimal):
    """
    Style the background Cartesian axis for y-axis decoration.
 
    Sets tick positions, formats tick labels as absolute values
    (to match polar plot convention), hides the three non-left spines,
    and adds the property name as the y-axis label.
 
    Parameters
    ----------
    ax : matplotlib.axes.Axes
        The Cartesian axis to configure.
    sub_y_labels : numpy.ndarray
        Symmetric tick positions from build_tick_labels().
    ymax : float
        Axis limit in both positive and negative directions.
    title : str
        Y-axis label text (property name with unit).
    decimal : int
        Number of decimal places for tick label formatting.
    """
    formatted = [f"{abs(y):.{decimal}f}" for y in sub_y_labels]
    ax.set_ylim(-ymax, ymax)
    ax.set_yticks(sub_y_labels)
    ax.set_yticklabels(formatted, fontsize=12)
    ax.spines[['right', 'top', 'bottom']].set_visible(False)
    ax.tick_params(bottom=False)
    ax.set_xticklabels([])
    ax.set_ylabel(title, fontsize=18)
 
 
def configure_polar_axis(axp, sub_y_labels, ymax):
    """
    Style the polar axis for angle ticks, radial ticks, grid, and legend.
 
    Angle ticks are placed every 30 degrees (0-330). Radial ticks use the
    positive portion of sub_y_labels (excluding zero to avoid pole
    overlap). The legend is placed outside the plot to the upper-left.
 
    Parameters
    ----------
    axp : matplotlib.axes.Axes (polar)
        The polar axis to configure.
    sub_y_labels : numpy.ndarray
        Symmetric tick array; only values > 0 are used for radial ticks.
    ymax : float
        Maximum radial limit of the polar plot.
    """
    sub_degree_labels = np.arange(0, 360, 30)
    sub_r_labels = np.delete(sub_y_labels, np.where(sub_y_labels <= 0))
 
    axp.set_xticks(np.radians(sub_degree_labels))
    axp.set_xticklabels(
        [str(d) + '\u00B0' for d in sub_degree_labels],
        fontsize=12, color='black'
    )
    axp.patch.set_alpha(0)
    axp.set_rlim(0, ymax)
    axp.set_rticks(sub_r_labels)
    axp.set_yticklabels([])
    axp.set_title('')
    axp.grid(True)
 
    axp.legend(loc='upper left', bbox_to_anchor=(1.05, 1), fontsize=12)
    leg = axp.get_legend()
    leg.get_frame().set_edgecolor('black')
    leg.get_frame().set_linewidth(1.5)
 
 
def main():
    """
    Parse arguments, ask data types, ask configuration of figure, plot data values, and save output figure.
    """
    if '-h' in argv or len(argv) < 2 or len(argv) > 7:
        usage()
 
    input_files = argv[1:]
    meta = ask_property()
 
    data_type = meta["data_type"]
    unit      = meta["unit"]
    decimal   = meta["decimal"]
    title     = meta["title"]
    save_file = meta["save_file"]
 
    fig, ax, axp = setup_figure()
 
    neg_envelope_labeled = False
    for i, filepath in enumerate(input_files):
        degrees, radians, values = load_data(filepath)
        label = ask_material_label(i + 1)
 
        idx_largest = np.argmax(np.abs(values))
        print(f"Your largest {data_type} is {values[idx_largest]:>6.{decimal + 2}f} {unit} "
              f"at {degrees[idx_largest]:>5.1f}\u00B0")
 
        color = COLORS[i % len(COLORS)]
        neg_envelope_labeled = plot_material(axp, radians, values, color, label,
                                             neg_envelope_labeled)
 
    ymax = ask_positive_float("Enter your maximum size: ")
    step = ask_step(ymax)
 
    sub_y_labels = build_tick_labels(ymax, step, decimal)
    configure_cartesian_axis(ax, sub_y_labels, ymax, title, decimal)
    configure_polar_axis(axp, sub_y_labels, ymax)
 
    plt.savefig(save_file, dpi=300, bbox_inches='tight', format='png')
 
 
if __name__ == '__main__':
    main()
 
