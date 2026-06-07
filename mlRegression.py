#!/usr/bin/env python

from sys import argv, exit
import os
import numpy as np


def usage():
    """Print usage information and exit."""
    print("""
Usage: mlRegression.py <ML_REG input>

This script extract energies, forces, and stress from ML_REG file.
Output files can plot by xmgrace.

This script was developed by Thanasee Thanasarnsurapong.
""")
    exit(0)


def read_files(input_file):
    """Read all lines from the ML_REG file.

    Parameters
    ----------
    input_file : str
        Path to the ML_REG file.

    Returns
    -------
    list of str
        All lines in the file.
    """
    if not os.path.exists(input_file):
        print(f"ERROR!\nFile: {input_file} does not exist.")
        exit(1)
    with open(input_file, 'r') as f:
        return f.readlines()


def find_section_index(lines):
    """Locate the line indices of the three data sections in the ML_REG file.

    Searches for the header lines of the energy, force, and stress sections
    and returns their positions. Exits with an error message if any section
    is not found.

    Parameters
    ----------
    lines : list of str
        All lines read from the ML_REG file.

    Returns
    -------
    energy_index : int
        Line index of the 'Total energies (eV)' header.
    force_index : int
        Line index of the 'Forces (eV ang.^-1)' header.
    stress_index : int
        Line index of the 'Stress (kbar)' header.
    """
    energy_index = force_index = stress_index = None
    for i, line in enumerate(lines):
        if 'Total energies (eV)' in line:
            energy_index = i
        if 'Forces (eV ang.^-1)' in line:
            force_index = i
        if 'Stress (kbar)' in line:
            stress_index = i
            break
    if energy_index is None:
        print("The 'Total energies (eV)' section was not found in the ML_REG file.")
        exit(1)
    if force_index is None:
        print("The 'Forces (eV ang.^-1)' section was not found in the ML_REG file.")
        exit(1)
    if stress_index is None:
        print("The 'Stress (kbar)' section was not found in the ML_REG file.")
        exit(1)
    return energy_index, force_index, stress_index


def parse_block(line_slice):
    """Parse a slice of lines into a 2D float array, skipping blank lines."""
    return np.array([[float(x) for x in line.split()]
                     for line in line_slice if line.strip()])


def extract_arrays(lines, energy_index, force_index, stress_index):
    """Extract energy, force, and stress arrays from the ML_REG file.

    Each section is sliced using the section indices and parsed into a
    2D NumPy array with shape (N, 2), where column 0 is the DFT reference
    and column 1 is the MLFF prediction. Blank lines are skipped.

    Parameters
    ----------
    lines : list of str
        All lines read from the ML_REG file.
    energy_index : int
        Line index of the energy section header.
    force_index : int
        Line index of the force section header.
    stress_index : int
        Line index of the stress section header.

    Returns
    -------
    energy : ndarray, shape (N_frames, 2)
        Total energies (eV) — DFT and MLFF columns.
    force : ndarray, shape (N_frames * N_atoms * 3, 2)
        Forces (eV/Å) — DFT and MLFF columns.
    stress : ndarray, shape (N_frames * 6, 2)
        Stress tensor components (kbar) — DFT and MLFF columns.
    """

    energy = parse_block(lines[energy_index + 2 : force_index - 1])
    force  = parse_block(lines[force_index  + 2 : stress_index - 1])
    stress = parse_block(lines[stress_index + 2 :])
    return energy, force, stress


def validate_dimensions(energy_count, force_count, stress_count):
    """Verify that force and stress counts are consistent with energy count.

    Force rows must equal 3 * N_atoms * N_frames, and stress rows must
    equal 6 * N_frames. Exits with an error message if either check fails.

    Parameters
    ----------
    energy_count : int
        Number of frames (rows in the energy array).
    force_count : int
        Total number of force rows (N_frames * N_atoms * 3).
    stress_count : int
        Total number of stress rows (N_frames * 6).
    """
    if force_count % (3 * energy_count) != 0:
        print("ERROR! Force count is not divisible by 3 * energy_count. Check ML_REG structure.")
        exit(1)
    if stress_count % (6 * energy_count) != 0:
        print("ERROR! Stress count does not match 6 * energy_count. Check ML_REG structure.")
        exit(1)


def compute_rmse(dft, mlff):
    """Compute the Root Mean Square Error (RMSE) between DFT and MLFF arrays.

    Parameters
    ----------
    dft : ndarray
        DFT reference values.
    mlff : ndarray
        MLFF predicted values.

    Returns
    -------
    float
        RMSE value in the same units as the input arrays.
    """
    return np.sqrt(np.mean((dft - mlff) ** 2))


def compute_mae(dft, mlff):
    """Compute the Mean Absolute Error (MAE) between DFT and MLFF arrays.

    Parameters
    ----------
    dft : ndarray
        DFT reference values.
    mlff : ndarray
        MLFF predicted values.

    Returns
    -------
    float
        MAE value in the same units as the input arrays.
    """
    return np.mean(np.abs(dft - mlff))


def compute_r2(dft, mlff):
    """Compute the R-squared (coefficient of determination) score.

    R² = 1 - SS_res / SS_tot, where SS_res is the residual sum of squares
    and SS_tot is the total sum of squares relative to the DFT mean.
    A value of 1.0 indicates a perfect fit.

    Parameters
    ----------
    dft : ndarray
        DFT reference values.
    mlff : ndarray
        MLFF predicted values.

    Returns
    -------
    float
        R² score (dimensionless). Ranges from -inf to 1.0.
    """
    ss_res = np.sum((dft - mlff) ** 2)
    ss_tot = np.sum((dft - np.mean(dft)) ** 2)
    return 1 - ss_res / ss_tot


def compute_metrics(energy_per_atom, force, stress):
    """Compute RMSE, MAE, and R² for energy, force, and stress.

    Energy metrics are converted to meV atom^-1 by multiplying by 1e3.
    Force and stress metrics are kept in their native units (eV/Å and kbar).

    Parameters
    ----------
    energy_per_atom : ndarray, shape (N_frames, 2)
        Per-atom energies (eV/atom) — DFT and MLFF columns.
    force : ndarray, shape (N_frames * N_atoms * 3, 2)
        Forces (eV/Å) — DFT and MLFF columns.
    stress : ndarray, shape (N_frames * 6, 2)
        Stress components (kbar) — DFT and MLFF columns.

    Returns
    -------
    dict
        Dictionary containing nine metric values:
        'rmse_energy' (meV/atom), 'rmse_force' (eV/Å), 'rmse_stress' (kbar),
        'mae_energy'  (meV/atom), 'mae_force'  (eV/Å), 'mae_stress'  (kbar),
        'r2_energy', 'r2_force', 'r2_stress' (dimensionless).
    """
    metrics = {
        'rmse_energy': compute_rmse(energy_per_atom[:, 0], energy_per_atom[:, 1]) * 1e3,
        'rmse_force' : compute_rmse(force[:, 0], force[:, 1]),
        'rmse_stress': compute_rmse(stress[:, 0], stress[:, 1]),
        'mae_energy' : compute_mae(energy_per_atom[:, 0], energy_per_atom[:, 1]) * 1e3,
        'mae_force'  : compute_mae(force[:, 0], force[:, 1]),
        'mae_stress' : compute_mae(stress[:, 0], stress[:, 1]),
        'r2_energy'  : compute_r2(energy_per_atom[:, 0], energy_per_atom[:, 1]),
        'r2_force'   : compute_r2(force[:, 0], force[:, 1]),
        'r2_stress'  : compute_r2(stress[:, 0], stress[:, 1]),
    }
    return metrics


def write_energy(energy_per_atom, filename='Energy.dat'):
    """Write per-atom energy parity data to a file for xmgrace plotting.

    Output format: two columns — DFT (eV/atom) and MLFF (eV/atom).

    Parameters
    ----------
    energy_per_atom : ndarray, shape (N_frames, 2)
        Per-atom energies — DFT and MLFF columns.
    filename : str, optional
        Output filename (default: 'Energy.dat').
    """
    with open(filename, 'w') as o:
        o.write("# Total energies per atom(eV atom^-1)\n")
        o.write("#  DFT              MLFF\n")
        for row in energy_per_atom:
            o.write(f" {row[0]:>14.6f}   {row[1]:>14.6f}\n")


def write_force(force_reshape, filename='Force.dat'):
    """Write force parity data along X, Y, Z directions to a file.

    Each direction is written as a separate block separated by a blank line,
    suitable for multi-dataset xmgrace plots.

    Parameters
    ----------
    force_reshape : ndarray, shape (N_frames * N_atoms, 3, 2)
        Reshaped force array with axes [atom-frame, direction, DFT/MLFF].
    filename : str, optional
        Output filename (default: 'Force.dat').
    """
    labels = ['X', 'Y', 'Z']
    with open(filename, 'w') as o:
        for i, label in enumerate(labels):
            if i > 0:
                o.write("\n")
            o.write(f"# Forces (eV ang.^-1) along {label} direction\n")
            o.write("#  DFT              MLFF\n")
            for row in force_reshape[:, i, :]:
                o.write(f" {row[0]:>14.6E}   {row[1]:>14.6E}\n")


def write_stress(stress_reshape, filename='Stress.dat'):
    """Write stress parity data for all 6 Voigt components to a file.

    Components are written in order: XX, YY, ZZ, XY, XZ, YZ, each as a
    separate block separated by a blank line for xmgrace multi-dataset plots.

    Parameters
    ----------
    stress_reshape : ndarray, shape (N_frames, 6, 2)
        Reshaped stress array with axes [frame, Voigt component, DFT/MLFF].
    filename : str, optional
        Output filename (default: 'Stress.dat').
    """
    labels = ['XX', 'YY', 'ZZ', 'XY', 'XZ', 'YZ']
    with open(filename, 'w') as o:
        for i, label in enumerate(labels):
            if i > 0:
                o.write("\n")
            o.write(f"# Stress (kbar) along {label} component\n")
            o.write("#  DFT              MLFF\n")
            for row in stress_reshape[:, i, :]:
                o.write(f" {row[0]:>14.6E}   {row[1]:>14.6E}\n")


def write_errors(metrics, filename='ERROR.dat'):
    """Write all error metrics to a file and print them to the terminal.

    Writes and displays RMSE, MAE, and R² for energy, force, and stress.
    Both the file and terminal output share the same formatted lines.

    Parameters
    ----------
    metrics : dict
        Dictionary of metric values from compute_metrics().
    filename : str, optional
        Output filename (default: 'ERROR.dat').
    """
    lines = [
        f"RMSE of energy per atom (meV atom^-1) : {metrics['rmse_energy']:>7.3f}",
        f"RMSE of force (eV ang.^-1)            : {metrics['rmse_force']:>7.3f}",
        f"RMSE of stress (kbar)                 : {metrics['rmse_stress']:>7.3f}",
        f"MAE of energy per atom (meV atom^-1)  : {metrics['mae_energy']:>7.3f}",
        f"MAE of force (eV ang.^-1)             : {metrics['mae_force']:>7.3f}",
        f"MAE of stress (kbar)                  : {metrics['mae_stress']:>7.3f}",
        f"R-square score of energy per atom     : {metrics['r2_energy']:>7.4f}",
        f"R-square score of force               : {metrics['r2_force']:>7.4f}",
        f"R-square score of stress              : {metrics['r2_stress']:>7.4f}",
    ]
    with open(filename, 'w') as o:
        o.write('\n'.join(lines) + '\n')
    for line in lines:
        print(line)


def main():
    """Parse arguments, collect data types, compute statistic variables, and write all outputs."""
    if '-h' in argv or len(argv) != 2:
        usage()

    input_file = argv[1]

    # Parse
    lines = read_files(input_file)
    energy_index, force_index, stress_index = find_section_index(lines)
    energy, force, stress = extract_arrays(lines, energy_index, force_index, stress_index)

    # Counts and derived quantities
    energy_count = len(energy)
    force_count  = len(force)
    stress_count = len(stress)
    atom_count   = force_count // (3 * energy_count)

    # Validate
    validate_dimensions(energy_count, force_count, stress_count)

    # Transform
    energy_per_atom = energy / atom_count
    force_reshape   = force.reshape((force_count // 3, 3, 2))
    stress_reshape  = stress.reshape((energy_count, 6, 2))

    # Compute metrics
    metrics = compute_metrics(energy_per_atom, force, stress)

    # Write outputs
    write_energy(energy_per_atom)
    write_force(force_reshape)
    write_stress(stress_reshape)
    write_errors(metrics)


if __name__ == '__main__':
    main()
