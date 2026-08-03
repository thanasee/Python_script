#!/usr/bin/env python

from sys import argv, exit
import os
import numpy as np

def usage():
    """Print usage information and exit."""
    print("""
Usage: getQPATH.py <input band.dat>

This script read second line in band.dat file
which generated from phonopy-bandplot --gnuplot command
and write QLINES.dat in same format with KLINES.dat from VASPKIT

This script was developed by Thanasee Thanasarnsurapong.
""")
    exit(0)


def read_band_dat(filepath):
    """
    Read q-point path positions and frequency data from a phonopy band.dat file.
 
    Parameters
    ----------
    filepath : str
        Path to the band.dat file produced by ``phonopy-bandplot --gnuplot``.
 
    Returns
    -------
    q_points : numpy.ndarray, shape (N,)
        Q-path distances (1/Angstrom) extracted from the second line of the file.
    fmin : float
        Minimum frequency (THz) found in the data columns, rounded down
        to the nearest multiple of 5.
    fmax : float
        Maximum frequency (THz) found in the data columns, rounded up
        to the nearest multiple of 5.
    """
    with open(filepath, 'r') as f:
        lines = f.readlines()
 
    if len(lines) < 2:
        raise ValueError(
            f"{filepath}: file has {len(lines)} line(s), expected a header "
            "and a q-point line."
        )
 
    if not lines[1].lstrip().startswith('#'):
        raise ValueError(
            "Line 2 does not start with '#' — not a phonopy --gnuplot "
            "band.dat file."
        )
 
    q_points = np.array([float(x) for x in lines[1].split()[1:]])
 
    freqs = []
    for num, line in enumerate(lines[2:], start=3):
        parts = line.strip().split()
        if not parts:
            continue
        if len(parts) != 2:
            raise ValueError(
                f"Line {num}: expected 2 columns (distance, frequency), "
                f"got {len(parts)}: {line.strip()!r}"
            )
        try:
            freqs.append(float(parts[1]))
        except ValueError:
            raise ValueError(
                f"Line {num}: could not parse frequency from {parts[1]!r}"
            )
 
    if not freqs:
        raise ValueError("No frequency data found in input file (2nd column).")
 
    # Round outward to the nearest multiple of 5 so the last digit is 0 or 5
    fmin = 5 * np.floor(min(freqs) / 5)
    fmax = 5 * np.ceil(max(freqs) / 5)
 
    return q_points, fmin, fmax


def write_QLINES(q_points, fmin, fmax, output_path="QLINES.dat"):
    """
    Write the q-path boundary file QLINES.dat.
 
    For each interior high-symmetry q-point, three lines are written that
    trace a vertical tick from ``fmin`` up to ``fmax`` and back down.
    The outer box and the zero-frequency axis are appended at the end.
 
    Parameters
    ----------
    q_points : numpy.ndarray, shape (N,)
        Q-path distances (1/Angstrom) of the high-symmetry points.
    fmin : float
        Lower frequency boundary (THz).
    fmax : float
        Upper frequency boundary (THz).
    output_path : str, optional
        Destination file path (default: ``"QLINES.dat"``).
    """
    with open(output_path, 'w') as o:
        o.write("#Q-Path(1/A) Frequency-Window(THz)\n")
 
        # Left edge baseline
        o.write(f"{q_points[0]:12.8f}{fmin:13.6f}\n")
 
        # Vertical ticks at interior high-symmetry points
        for x in q_points[1:-1]:
            o.write(f"{x:12.8f}{fmin:13.6f}\n")
            o.write(f"{x:12.8f}{fmax:13.6f}\n")
            o.write(f"{x:12.8f}{fmin:13.6f}\n")
 
        # Outer box: right edge then top and back to origin
        o.write(f"{q_points[-1]:12.8f}{fmin:13.6f}\n")
        o.write(f"{q_points[-1]:12.8f}{fmax:13.6f}\n")
        o.write(f"{q_points[0]:12.8f}{fmax:13.6f}\n")
        o.write(f"{q_points[0]:12.8f}{fmin:13.6f}\n")
 
        # Zero-frequency axis
        o.write(f"{q_points[0]:12.8f}{0.0:13.6f}\n")
        o.write(f"{q_points[-1]:12.8f}{0.0:13.6f}\n")


def main():
    """Parse arguments, read band.dat, write QLINES.dat."""
    if '-h' in argv or len(argv) != 2:
        usage()
 
    input_file = argv[1]
    if not os.path.exists(input_file):
        print(f"ERROR!\nFile: {input_file} does not exist.")
        exit(1)
 
    try:
        q_points, fmin, fmax = read_band_dat(input_file)
    except ValueError as e:
        print(f"ERROR!\n{e}")
        exit(1)
 
    write_QLINES(q_points, fmin, fmax)
 
 
if __name__ == '__main__':
    main()
