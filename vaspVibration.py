#!/usr/bin/env python

from sys import argv, exit
import os
import readline
import numpy as np
from ase.io import read


def usage():
    """Print usage information and exit."""
    print("""
Usage: vaspVibration.py <POSCAR input> <input file> [scaling factor]

This script extracts vibrational modes (at Γ for OUTCAR, at a chosen q-point for YAML) and writes them in XSF format.
Supports both VASP (OUTCAR) and Phonopy (band.yaml/mesh.yaml/qpoints.yaml) outputs with write eigenvectors.

This script was developed by Thanasee Thanasarnsurapong.
""")
    exit(0)


def read_structure(poscar_file):
    """Read atomic structure from a POSCAR file using ASE.

    Parameters
    ----------
    poscar_file : str
        Path to the POSCAR file.

    Returns
    -------
    structure : ase.Atoms
        ASE Atoms object containing the crystal structure.
    """

    if not os.path.exists(poscar_file):
        print(f"ERROR!\nFile: {poscar_file} does not exist.")
        exit(1)

    return read(poscar_file)


def read_modes_outcar(outcar_file, structure):
    """Read vibrational frequencies and eigenvectors from a VASP OUTCAR file.

    Locates the 'Eigenvectors and eigenvalues of the dynamical matrix'
    section, extracts all normal mode frequencies and mass-unweighted
    Cartesian displacement vectors. The 'Eigenvectors after division by
    SQRT(mass)' block (NWRITE = 3) is not read.

    Parameters
    ----------
    outcar_file : str
        Path to the OUTCAR file.
    structure : ase.Atoms
        ASE Atoms object, used to obtain atomic masses for mass-unweighting.

    Returns
    -------
    frequency : np.ndarray, shape (n_modes,)
        Vibrational frequencies in THz (2PiTHz convention from OUTCAR).
    modes : np.ndarray, shape (n_modes, n_atoms, 3)
        Mass-unweighted Cartesian displacement vectors for each normal mode.
    """

    if not os.path.exists(outcar_file):
        print(f"ERROR!\nFile: {outcar_file} does not exist.")
        exit(1)

    with open(outcar_file, 'r') as f:
        outcar_lines = f.readlines()

    total_ions = None
    for line in outcar_lines:
        if 'NIONS =' in line:
            total_ions = int(line.split()[-1])
            break

    if total_ions is None:
        print("ERROR!\n'NIONS' not found in OUTCAR.")
        exit(1)

    index_start = None
    frequency_index = []
    for i, line in enumerate(outcar_lines):
        if 'Eigenvectors and eigenvalues of the dynamical matrix' in line:
            index_start = i + 2
            frequency_index = []
        elif 'Eigenvectors after division by SQRT(mass)' in line:
            break
        elif '2PiTHz' in line:
            frequency_index.append(i)

    if index_start is None or len(frequency_index) == 0:
        print("ERROR!\nEigenvector block not found in OUTCAR.")
        exit(1)

    index_stop = frequency_index[-1] + total_ions + 2

    frequency = np.array(
        [line.split()[-8] for line in outcar_lines[index_start:index_stop] if '2PiTHz' in line],
        dtype=float
    )
    modes = [line.split()[3:6] for line in outcar_lines[index_start:index_stop]
             if ('dx' not in line) and ('2PiTHz' not in line)]
    modes = np.array([m for m in modes if len(m) > 0], dtype=float)
    modes = modes.reshape((-1, total_ions, 3))

    if total_ions != len(structure):
        print("ERROR!\nShape mismatch between eigenvectors and atomic positions.")
        exit(1)

    # Mass-unweight: OUTCAR stores mass-weighted eigenvectors; divide by sqrt(m)
    modes /= np.sqrt(structure.get_masses()[None, :, None])

    return frequency, modes


def read_modes_phonopy(yaml_file, structure):
    """Read vibrational frequencies and eigenvectors from a Phonopy YAML file.

    Parses band.yaml, mesh.yaml, or qpoints.yaml by string splitting.
    Candidate q-points are the segment end points for band.yaml (with
    'segment_nqpoint') or all q-points otherwise, with duplicated
    q-positions removed. If more than one candidate remains, the user is
    prompted to choose one. The Bloch phase exp(2πi q·r) is applied and
    the global phase of each mode is removed before the real part is
    kept; a warning is printed if the imaginary part is not negligible.

    Parameters
    ----------
    yaml_file : str
        Path to the Phonopy YAML file.
    structure : ase.Atoms
        ASE Atoms object, used to obtain atomic masses for mass-unweighting.

    Returns
    -------
    frequency : np.ndarray, shape (n_bands,)
        Phonon frequencies at the chosen q-point (THz).
    modes : np.ndarray, shape (n_bands, n_atoms, 3)
        Mass-unweighted real part of eigenvectors at the chosen q-point.
    """

    if not os.path.exists(yaml_file):
        print(f"ERROR!\nFile: {yaml_file} does not exist.")
        exit(1)

    with open(yaml_file, 'r') as f:
        content = f.read()

    if 'phonon:' not in content:
        print("ERROR!\nNo q-point data found in YAML file.")
        exit(1)

    header, body = content.split('phonon:', 1)
    q_points = body.split('q-position:')[1:]

    if len(q_points) == 0:
        print("ERROR!\nNo q-point data found in YAML file.")
        exit(1)

    # q-position and optional label of each q-point
    positions = []
    labels = []
    for qp in q_points:
        positions.append(np.array([float(x) for x in qp.split(']')[0].strip(' [').split(',')]))
        label = ''
        for line in qp.split('band:')[0].splitlines():
            if line.strip().startswith('label:'):
                label = line.split(':', 1)[1].strip().strip('\'"')
        labels.append(label)

    # Segment end points of band.yaml are the high-symmetry q-points
    segments = []
    if 'segment_nqpoint:' in header:
        for line in header.split('segment_nqpoint:')[1].splitlines()[1:]:
            if not line.strip().startswith('-'):
                break
            segments.append(int(line.split('-')[1]))
    is_band = len(segments) > 0 and sum(segments) == len(q_points)
    if is_band:
        ends = np.cumsum(segments)
        candidates = sorted({i for n, e in zip(segments, ends) for i in (e - n, e - 1)})
        # Phonopy 4.x writes labels only in the header list, one [start, end] pair per segment
        segment_labels = []
        if 'labels:' in header:
            for line in header.split('labels:')[1].splitlines()[1:]:
                if not line.strip().startswith('-'):
                    break
                segment_labels.append([s.strip().strip('\'"') for s in line.strip()[1:].strip(' []').split(',')])
        if len(segment_labels) == len(segments):
            for (n, e), pair in zip(zip(segments, ends), segment_labels):
                labels[e - n] = labels[e - n] or pair[0]
                labels[e - 1] = labels[e - 1] or pair[-1]
    else:
        candidates = list(range(len(q_points)))

    unique = []
    for i in candidates:
        if not any(np.allclose(positions[i], positions[j], atol=1e-6) for j in unique):
            unique.append(i)

    if len(unique) == 1:
        iq = unique[0]
    else:
        kind = "high-symmetry q-points" if is_band else "q-points"
        print(f"\nFound {len(unique)} {kind} in {yaml_file}:")
        for n, i in enumerate(unique, start=1):
            q = positions[i]
            print(f"  {n:>3d}) {labels[i]:<8s} [{q[0]:10.6f}{q[1]:10.6f}{q[2]:10.6f} ]")
        while True:
            try:
                choice = int(input(f"Enter the q-point index (1 to {len(unique)}): ")) - 1
                if 0 <= choice < len(unique):
                    iq = unique[choice]
                    break
                print("ERROR! Index out of range. Try again.")
            except ValueError:
                print("ERROR! Must enter a number. Try again.")

    bands = q_points[iq].split('frequency:')[1:]
    nbands = len(bands)
    natoms = len(bands[0].split('atom')[1:]) if nbands > 0 else 0

    if natoms == 0:
        print("ERROR!\nNo eigenvectors found in YAML file. Rerun Phonopy with --eigenvectors.")
        exit(1)

    if natoms != len(structure):
        print("ERROR!\nShape mismatch between eigenvectors and atomic positions.")
        exit(1)

    # Extract frequencies and eigenvector components (real, imaginary) at the chosen q-point
    frequency = np.array(
        [float(bands[b].split('eigenvector')[0]) for b in range(nbands)],
        dtype=float
    )

    eigenvectors = np.zeros((nbands, natoms, 3), dtype=complex)
    for b in range(nbands):
        atom_blocks = bands[b].split('atom')[1:]
        for a in range(natoms):
            for d in range(3):
                component = atom_blocks[a].split('[')[d + 1].split(']')[0].split(',')
                eigenvectors[b, a, d] = complex(float(component[0]), float(component[1]))

    # Bloch phase exp(2πi q·r) of each atom (not included in Phonopy eigenvectors),
    # then remove the arbitrary global phase of each mode (sign kept, so Γ is unchanged)
    eigenvectors *= np.exp(2j * np.pi * structure.get_scaled_positions() @ positions[iq])[None, :, None]
    for b in range(nbands):
        k = np.abs(eigenvectors[b]).argmax()
        theta = (np.angle(eigenvectors[b].flat[k]) + np.pi / 2) % np.pi - np.pi / 2
        eigenvectors[b] *= np.exp(-1j * theta)

    if np.abs(eigenvectors.imag).max() > 1e-6:
        print("Warning! Eigenvectors at this q-point are complex. Only the real part is written.")

    # Mass-unweight: Phonopy stores mass-weighted eigenvectors; divide by sqrt(m)
    modes = eigenvectors.real / np.sqrt(structure.get_masses()[None, :, None])

    return frequency, modes


def write_xsf_modes(structure, modes, scale, is_phonopy):
    """Write each vibrational mode to a separate XSF file.

    Each output file mode_N.xsf contains the crystal lattice vectors
    and atomic positions augmented with displacement vectors, suitable
    for visualization in VESTA or XCrySDen. Mode numbering follows
    Phonopy order (ascending) for Phonopy input and VASP order
    (descending, highest frequency first) for OUTCAR input.

    Parameters
    ----------
    structure : ase.Atoms
        ASE Atoms object containing the crystal structure.
    modes : np.ndarray, shape (n_modes, n_atoms, 3)
        Displacement vectors for each mode (mass-unweighted).
    scale : float
        Scaling factor applied to displacement vectors for visualization.
    is_phonopy : bool
        If True, number modes in ascending order (Phonopy convention).
        If False, number modes in descending order (VASP convention).
    """

    total_modes = modes.shape[0]
    total_atoms = len(structure)
    symbols     = structure.get_chemical_symbols()
    place       = len(str(total_modes))
    if modes.shape[1] != total_atoms:
        print("ERROR!\nShape mismatch between eigenvectors and atomic positions.")
        exit(1)

    for j in range(total_modes):
        vector = modes[j] * scale
        positions_vector = np.hstack((structure.positions, vector))

        mode_index  = j + 1 if is_phonopy else total_modes - j
        output_name = f"mode_{str(mode_index).zfill(place)}.xsf"

        with open(output_name, 'w') as o:
            o.write("CRYSTAL\n")
            o.write("PRIMVEC\n")
            o.write("\n".join(
                ' '.join(f'{a:20.16f}' for a in lattice_row)
                for lattice_row in structure.cell
            ))
            o.write("\nPRIMCOORD\n")
            o.write(f"{total_atoms:3d} 1\n")
            o.write("\n".join(
                f'{symbols[k]:3s}' + ' '.join(f'{a:20.16f}' for a in positions_vector[k])
                for k in range(total_atoms)
            ))


def main():
    """Parse arguments, read inputs, and write vibrational mode XSF files."""

    if '-h' in argv or '--help' in argv or not (3 <= len(argv) <= 4):
        usage()

    poscar_file = argv[1]
    input_file  = argv[2]
    scale       = float(argv[3]) if len(argv) == 4 else 1.0

    structure   = read_structure(poscar_file)
    is_phonopy  = input_file.endswith(".yaml")

    if is_phonopy:
        _, modes = read_modes_phonopy(input_file, structure)
    else:
        _, modes = read_modes_outcar(input_file, structure)

    write_xsf_modes(structure, modes, scale, is_phonopy)


if __name__ == '__main__':
    main()
