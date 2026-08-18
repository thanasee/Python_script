#!/usr/bin/env python

from sys import argv, exit
import os
import readline
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# Set Times New Roman (with fallback)
matplotlib.rcParams['font.family'] = 'serif'
matplotlib.rcParams['font.serif'] = ['Times New Roman', 'Times', 'DejaVu Serif']
matplotlib.rcParams['mathtext.fontset'] = 'stix'   # for math consistency

# Edit this to match the quantity actually stored in the CHGLAVG.grd file
CHGLAVG_LABEL = r"$\rho$ (e/$\mathrm{\AA}^3$)"
COLORMAP      = "viridis"
N_LEVELS      = 100


def usage():
    """Print usage information and exit."""
    print("""
Usage: plotGrid.py <X.grd> <Y.grd> <CHGLAVG.grd> [output prefix]

This script reads a VASPKIT-style 2D grid (X.grd, Y.grd, CHGLAVG.grd, e.g.
CHGLAVG.grd) and:
  1) plots a filled-contour heatmap as <prefix>.png
  2) writes <prefix>_matrix.dat, an Origin-ready matrix ASCII file
     (row 1 = X values, column 1 = Y values, interior = CHGLAVG), importable
     via Origin's "Import Matrix > Simple ASCII"
  3) writes <prefix>_xyz.dat, a plain X Y CHGLAVG column file for Origin's
     Data > Convert to Matrix > XYZ Gridding (or any generic plotting
     program)

If [output prefix] is omitted, the CHGLAVG.grd filename (without extension)
is used.

This script was developed by Thanasee Thanasarnsurapong.
""")
    exit(0)


def read_grd(filepath):
    """
    Read a whitespace-delimited 2D grid file (VASPKIT .grd format).

    Parameters
    ----------
    filepath : str
        Path to the .grd file.

    Returns
    -------
    grid : numpy.ndarray, shape (n_rows, n_cols)
        The parsed 2D array.
    """
    if not os.path.isfile(filepath):
        raise FileNotFoundError(f"Input file not found: {filepath}")

    grid = np.loadtxt(filepath)

    if grid.ndim != 2:
        raise ValueError(f"{filepath} did not parse as a 2D grid "
                          f"(got shape {grid.shape}).")

    return grid


def validate_grids(X, Y, CHGLAVG, atol=2e-4):
    """
    Check that X, Y, CHGLAVG grids share the same shape, and warn if X, Y
    do not form a regular (axis-aligned) mesh, i.e. X varies only along
    columns and Y varies only along rows.

    The shape check is a hard requirement: write_origin_column zips the
    three flattened arrays together, so a shape mismatch would silently
    truncate to the shortest array rather than error.

    The axis-alignment check is informational only. It matters only if
    a rectilinear-grid consumer (e.g. an Origin matrix export) is added
    back later; contourf and the plain XYZ column writer both handle a
    non-rectilinear grid correctly. The tolerance defaults to 2e-4, twice
    the last-digit resolution of VASPKIT's 4-decimal-place .grd output,
    so harmless rounding noise doesn't trigger a warning; a genuinely
    sheared (non-orthogonal) grid deviates by amounts tracking the cell
    size, several orders of magnitude larger than this.

    Parameters
    ----------
    X, Y, CHGLAVG : numpy.ndarray, shape (n_rows, n_cols)
        Coordinate and value grids read from X.grd, Y.grd, CHGLAVG.grd.
    atol : float, optional
        Absolute tolerance for the axis-alignment warning (default 2e-4).

    Raises
    ------
    ValueError
        If shapes mismatch.
    """
    if not (X.shape == Y.shape == CHGLAVG.shape):
        raise ValueError(f"Shape mismatch: X{X.shape}, Y{Y.shape}, CHGLAVG{CHGLAVG.shape} "
                          f"must all be identical.")

    x_dev = np.max(np.abs(X - X[0:1, :]))
    if x_dev > atol:
        print(f"Warning: X.grd is not constant along rows (max deviation "
              f"{x_dev:.6f} > atol={atol}); grid may be sheared "
              f"(non-orthogonal cell) or mismatched. Continuing anyway.")

    y_dev = np.max(np.abs(Y - Y[:, 0:1]))
    if y_dev > atol:
        print(f"Warning: Y.grd is not constant along columns (max deviation "
              f"{y_dev:.6f} > atol={atol}); grid may be sheared "
              f"(non-orthogonal cell) or mismatched. Continuing anyway.")


def prompt_axis_labels():
    """
    Interactively prompt for which axis identifier (X, Y, or Z) each
    plot axis represents. The unit is always appended as (Å); only the
    identifier letter is asked for. Input is required and restricted
    to X, Y, or Z; invalid entries re-prompt.

    Returns
    -------
    x_label, y_label : str
        Axis label strings, e.g. "X ($\\mathrm{\\AA}$)".
    """
    valid = ('X', 'Y', 'Z')

    x_axis = input("Enter axis identifier for the plot's X-axis (X/Y/Z): ").strip().upper()
    while x_axis not in valid:
        x_axis = input("Invalid entry, must be X, Y, or Z. Enter axis identifier "
                        "for the plot's X-axis (X/Y/Z): ").strip().upper()

    y_axis = input("Enter axis identifier for the plot's Y-axis (X/Y/Z): ").strip().upper()
    while y_axis not in valid:
        y_axis = input("Invalid entry, must be X, Y, or Z. Enter axis identifier "
                        "for the plot's Y-axis (X/Y/Z): ").strip().upper()

    x_label = rf"{x_axis} ($\mathrm{{\AA}}$)"
    y_label = rf"{y_axis} ($\mathrm{{\AA}}$)"

    return x_label, y_label


def plot_grid(X, Y, CHGLAVG, output_png):
    """
    Plot a filled-contour heatmap of CHGLAVG(X, Y) and save as PNG.

    Always interactively prompts for the X- and Y-axis labels via
    prompt_axis_labels() before plotting.

    Parameters
    ----------
    X, Y, CHGLAVG : numpy.ndarray, shape (n_rows, n_cols)
        Coordinate and value grids.
    output_png : str
        Destination PNG path.
    """
    x_label, y_label = prompt_axis_labels()

    fig, ax = plt.subplots(figsize=(6, 6 * X.max() / max(X.max(), Y.max()) or 6),
                            dpi=300)

    cf = ax.contourf(X, Y, CHGLAVG, levels=N_LEVELS, cmap=COLORMAP)
    ax.set_xlabel(x_label, fontsize=14)
    ax.set_ylabel(y_label, fontsize=14)
    ax.set_aspect('equal')
    ax.tick_params(labelsize=11)

    cbar = fig.colorbar(cf, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label(CHGLAVG_LABEL, fontsize=13)
    cbar.ax.tick_params(labelsize=10)

    plt.savefig(output_png, dpi=300, bbox_inches='tight', format='png')
    plt.close(fig)


# def write_origin_matrix(X, Y, CHGLAVG, output_path):
#     """
#     Write an Origin-ready matrix ASCII file.

#     Layout: row 1 holds the unique X values (with a leading blank
#     cell), column 1 holds the unique Y values, and the interior holds
#     CHGLAVG. In Origin: File > Import > Matrix > Simple ASCII (or drag the
#     file onto a Matrix book).

#     Parameters
#     ----------
#     X, Y, CHGLAVG : numpy.ndarray, shape (n_rows, n_cols)
#         Coordinate and value grids (X constant along rows, Y constant
#         along columns, as validated by ``validate_grids``).
#     output_path : str
#         Destination file path.
#     """
#     x_vals = X[0, :]
#     y_vals = Y[:, 0]

#     header = "\t".join([""] + [f"{x:.6f}" for x in x_vals])
#     rows = [header]
#     for i, y in enumerate(y_vals):
#         row = "\t".join([f"{y:.6f}"] + [f"{v:.6f}" for v in CHGLAVG[i, :]])
#         rows.append(row)

#     with open(output_path, 'w') as f:
#         f.write("\n".join(rows) + "\n")


def write_origin_column(X, Y, CHGLAVG, output_path):
    """
    Write a plain three-column X Y CHGLAVG ASCII file.

    Suitable for Origin's Data > Convert to Matrix > XY-CHGLAVG Gridding, or
    for import into any generic contour/3D plotting tool.

    Parameters
    ----------
    X, Y, CHGLAVG : numpy.ndarray, shape (n_rows, n_cols)
        Coordinate and value grids.
    output_path : str
        Destination file path.
    """
    with open(output_path, 'w') as f:
        f.write("X               Y               CHGLAVG\n")
        for x, y, chglavg in zip(X.ravel(), Y.ravel(), CHGLAVG.ravel()):
            f.write(f"{x:.6f}        {y:.6f}        {chglavg:.6f}\n")


def main():
    """
    Parse arguments, read X/Y/CHGLAVG grids, plot a contour heatmap, and
    write Origin-ready matrix and XYZ ASCII files.
    """
    if '-h' in argv or len(argv) not in (4, 5):
        usage()

    x_file, y_file, chglavg_file = argv[1], argv[2], argv[3]
    prefix = argv[4] if len(argv) == 5 else os.path.splitext(os.path.basename(chglavg_file))[0]

    X = read_grd(x_file)
    Y = read_grd(y_file)
    CHGLAVG = read_grd(chglavg_file)

    validate_grids(X, Y, CHGLAVG)

    png_path    = f"{prefix}.png"
    # matrix_path = f"{prefix}_matrix.dat"
    xy_CHGLAVG_path    = f"{prefix}.dat"

    plot_grid(X, Y, CHGLAVG, png_path)
    # write_origin_matrix(X, Y, CHGLAVG, matrix_path)
    write_origin_column(X, Y, CHGLAVG, xy_CHGLAVG_path)

    print(f"Grid shape: {CHGLAVG.shape[0]} x {CHGLAVG.shape[1]}")
    print(f"CHGLAVG range: {CHGLAVG.min():.4f} to {CHGLAVG.max():.4f}")
    # print(f"Wrote {png_path}, {matrix_path}, {xy_CHGLAVG_path}")
    print(f"Wrote {png_path}, {xy_CHGLAVG_path}")


if __name__ == '__main__':
    main()
