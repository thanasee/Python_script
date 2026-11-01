#!/usr/bin/env python

from sys import argv, exit
import os
import numpy as np


def usage():
    """Print usage information and exit."""
    print("""
Usage: mlError.py <ML_LOGFILE input>

This script extract the Bayesian error estimation and the Root mean square error from ML_LOGFILE file.
Output files can plot by xmgrace.

This script was developed by Thanasee Thanasarnsurapong.
""")
    exit(0)


def read_files(input_file):
    """
    Check if the input file exists and read all lines.

    Parameters
    ----------
    input_file : str
        Path to the ML_LOGFILE.

    Returns
    -------
    list of str
        All lines read from the input file.
    """
    if not os.path.exists(input_file):
        print(f"ERROR!\nFile: {input_file} does not exist.")
        exit(1)
    with open(input_file, 'r') as f:
        return f.readlines()


def parse_lines(log_lines, keyword):
    """
    Extract data rows from log lines where the first token matches the given keyword.

    Parameters
    ----------
    log_lines : list of str
        All lines read from the ML_LOGFILE.
    keyword : str
        The keyword to match at the start of each line (e.g. 'BEEF' or 'ERR').

    Returns
    -------
    numpy.ndarray
        2D array of string values extracted from matched lines, excluding the keyword token.
    """
    data = np.array([
        line.split()[1:] for line in log_lines
        if line.split() and line.split()[0] == keyword
    ])
    if len(data) == 0:
        print(f"ERROR! No {keyword} data found. Is this a valid ML_LOGFILE?")
        exit(1)
    return data


def write_beef(beef_lines, filename="BEEF.dat"):
    """
    Write Bayesian Error Estimation in Forces (BEEF) data to an output file.
    Columns: nstep, bee_energy, bee_max_force, bee_ave_force,
             threshold, bee_max_stress, bee_ave_stress.

    Parameters
    ----------
    beef_lines : numpy.ndarray
        Parsed BEEF data rows from parse_lines().
    filename : str, optional
        Output filename (default: 'BEEF.dat').
    """
    with open(filename, 'w') as o:
        o.write("# The Bayesian error estimations\n")
        o.write("# bee_energy (eV atom^-1)\n")
        o.write("# bee_max_force (eV Angst^-1)\n")
        o.write("# bee_ave_force (eV Angst^-1)\n")
        o.write("# threshold (eV Angst^-1)\n")
        o.write("# bee_max_stress (kB)\n")
        o.write("# bee_ave_stress (kB)\n")
        o.write("\n")
        o.write("# nstep   bee_energy       bee_max_force    bee_ave_force    threshold        bee_max_stress   bee_ave_stress\n")
        for line in beef_lines:
            o.write(f"  {int(line[0]):>5.0f}   {float(line[1]):>14.8E}   {float(line[2]):>14.8E}   {float(line[3]):>14.8E}   "
                    f"{float(line[4]):>14.8E}   {float(line[5]):>14.8E}   {float(line[6]):>14.8E}\n")


def write_err(err_lines, filename="ERR.dat"):
    """
    Write Root Mean Square Error (RMSE) data to an output file.
    Columns: nstep, rmse_energy, rmse_force, rmse_stress.

    Parameters
    ----------
    err_lines : numpy.ndarray
        Parsed ERR data rows from parse_lines().
    filename : str, optional
        Output filename (default: 'ERR.dat').
    """
    with open(filename, 'w') as o:
        o.write("# The Root mean square errors\n")
        o.write("# rmse_energy (eV atom^-1)\n")
        o.write("# rmse_force (eV Angst^-1)\n")
        o.write("# rmse_stress (kB)\n")
        o.write("\n")
        o.write("# nstep   rmse_energy      rmse_force       rmse_stress\n")
        for line in err_lines:
            o.write(f"  {int(line[0]):>5.0f}   {float(line[1]):>14.8E}   {float(line[2]):>14.8E}   {float(line[3]):>14.8E}\n")


def main():
    """
    Parses arguments, reads the ML_LOGFILE, extracts BEEF and ERR data, and writes output files.
    """
    if '-h' in argv or '--help' in argv or len(argv) != 2:
        usage()

    input_file = argv[1]
    lines      = read_files(input_file)
    beef_lines = parse_lines(lines, 'BEEF')
    err_lines  = parse_lines(lines, 'ERR')
    write_beef(beef_lines)
    write_err(err_lines)
    print("Done! Output written to BEEF.dat and ERR.dat")


if __name__ == '__main__':
    main()
