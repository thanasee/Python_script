#!/usr/bin/env python

from sys import argv, exit
import os
import readline
import numpy as np


def usage():
    """Print usage information and exit."""
    print("""
Usage: calRMS.py <input POSCAR> <input FORCE_CONSTANTS>

This script calculate RMS of 2nd order of IFCs
and compare with distance between atoms from POSCAR/CONTCAR files

This script was developed by Thanasee Thanasarnsurapong.
""")
    exit(0)


_ELEMENT_SYMBOLS = [
    "H",  "He", "Li", "Be", "B",  "C",  "N",  "O",
    "F",  "Ne", "Na", "Mg", "Al", "Si", "P",  "S",
    "Cl", "Ar", "K",  "Ca", "Sc", "Ti", "V",  "Cr",
    "Mn", "Fe", "Co", "Ni", "Cu", "Zn", "Ga", "Ge",
    "As", "Se", "Br", "Kr", "Rb", "Sr", "Y",  "Zr",
    "Nb", "Mo", "Tc", "Ru", "Rh", "Pd", "Ag", "Cd",
    "In", "Sn", "Sb", "Te", "I",  "Xe", "Cs", "Ba",
    "La", "Ce", "Pr", "Nd", "Pm", "Sm", "Eu", "Gd",
    "Tb", "Dy", "Ho", "Er", "Tm", "Yb", "Lu", "Hf",
    "Ta", "W",  "Re", "Os", "Ir", "Pt", "Au", "Hg",
    "Tl", "Pb", "Bi", "Po", "At", "Rn", "Fr", "Ra",
    "Ac", "Th", "Pa", "U",  "Np", "Pu", "Am", "Cm",
    "Bk", "Cf", "Es", "Fm", "Md", "No", "Lr", "Rf",
    "Db", "Sg", "Bh", "Hs", "Mt", "Ds", "Rg", "Cn",
    "Nh", "Fl", "Mc", "Lv", "Ts", "Og"
]


def read_POSCAR(filepath):
    """Read a VASP POSCAR file and return its contents as a dictionary.

    Supports both VASP4 (no element line) and VASP5 (with element line) formats,
    scalar and negative (volume-based) scaling factors, a 3-component scaling
    vector, Selective Dynamics, and both Direct and Cartesian coordinate modes.

    Parameters
    ----------
    filepath : str
        Path to the POSCAR file to read.

    Returns
    -------
    dict with keys:
        lattice_matrix      : np.ndarray, shape (3, 3)  — lattice vectors in Å
        elements            : list[str]                 — element symbols
        atom_counts         : list[int]                 — number of atoms per element
        total_atoms         : int                       — total number of atoms
        is_direct           : bool                      — whether Direct coordinates are used
        positions_cartesian : np.ndarray, shape (N, 3)  — Cartesian coordinates in Å
        positions_direct    : np.ndarray, shape (N, 3)  — fractional coordinates
        species             : list[str]                 — element symbol per atom
        selective_dynamics  : bool                      — whether Selective Dynamics is present
        flags               : np.ndarray or None        — T/F flags per atom, or None
    """
    if not os.path.exists(filepath):
        print(f"ERROR!\nFile: {filepath} does not exist.")
        exit(1)

    with open(filepath, 'r') as poscar:
        lines = poscar.readlines()

    # Parse the scaling factor (line 2):
    # - 1 value  : uniform scalar; negative means target volume in Å**3
    # - 3 values : per-axis scale applied row-wise to the lattice matrix
    if len(lines[1].split()) == 1:
        raw_scale = float(lines[1])
        raw_lattice_matrix = np.array([[float(x) for x in line.split()]
                                       for line in lines[2:5]])
        if raw_scale < 0:
            volume = np.abs(np.linalg.det(raw_lattice_matrix))
            scale = np.cbrt(np.abs(raw_scale) / volume)
        elif raw_scale == 0:
            print("ERROR! The scaling factor must be not zero.")
            exit(1)
        else:
            scale = raw_scale
        lattice_matrix = raw_lattice_matrix * scale
    elif len(lines[1].split()) == 3:
        scale = np.array(list(map(float, lines[1].split())))
        lattice_matrix = np.array([[float(x) * scale[i] for i, x in enumerate(line.split())]
                                   for line in lines[2:5]])
    else:
        print("ERROR! The scaling factor must be 1 or 3 components.")
        exit(1)

    # Detect VASP4 vs VASP5 format by checking whether line 6 starts with a number.
    # VASP4 has no element-symbol line, so the user is prompted for species names.
    is_number = lines[5].split()[0].isdecimal()
    if is_number:
        # VASP4 format: no element line -> try POTCAR, else prompt user
        atom_counts = [int(x) for x in lines[5].split()]
        potcar_path = os.path.join(os.path.dirname(os.path.abspath(filepath)), "POTCAR")
        elements = read_POTCAR(potcar_path)
        if elements is None or len(elements) != len(atom_counts):
            elements = [None] * len(atom_counts)
            while None in elements:
                missing = [i for i, e in enumerate(elements) if e is None]
                names = input(f"Enter the name of species No. {missing[0] + 1:>3}: ").strip().split()
                for name, idx in zip(names, missing):
                    if name in _ELEMENT_SYMBOLS:
                        elements[idx] = name
                    else:
                        print("The name of species must be a valid element symbol.")
        atom_counts = [int(x) for x in lines[5].split()]
        selective_dynamics = lines[6].lower().startswith('s')
        position_start = 8 if selective_dynamics else 7
    else:
        # VASP5 format: element symbols present.
        # Strip potential PAW/GGA suffixes such as '_pv' or '/GGA'.
        names = lines[5].split()
        elements = [name.split('/')[0].split('_')[0] for name in names]
        atom_counts = [int(x) for x in lines[6].split()]
        selective_dynamics = lines[7].lower().startswith('s')
        position_start = 9 if selective_dynamics else 8

    # Read atomic positions
    total_atoms = sum(atom_counts)
    position_stop = position_start + total_atoms
    positions = np.array([[float(x) for x in lines[i].split()[:3]]
                          for i in range(position_start, position_stop)])

    # Build a per-atom species list (e.g. ['Mo', 'Mo', 'S', 'S', 'S'])
    species = [x for i, x in enumerate(elements)
               for _ in range(atom_counts[i])]

    # Read Selective Dynamics T/F flags if present
    flags = None
    if selective_dynamics:
        flags = np.array([[x for x in lines[i].split()[3:6]]
                          for i in range(position_start, position_stop)])

    # Convert coordinates to both Direct and Cartesian representations
    is_direct = lines[position_start - 1].strip().lower().startswith('d')
    if is_direct:
        positions_direct = positions % 1.0
        positions_cartesian = direct_to_cartesian(lattice_matrix, positions_direct)
    else:
        positions_cartesian = positions * scale
        positions_direct = cartesian_to_direct(lattice_matrix, positions_cartesian)

    return {"lattice_matrix":      lattice_matrix,
            "elements":            elements,
            "atom_counts":         atom_counts,
            "total_atoms":         total_atoms,
            "is_direct":           is_direct,
            "positions_cartesian": positions_cartesian,
            "positions_direct":    positions_direct,
            "species":             species,
            "selective_dynamics":  selective_dynamics,
            "flags":               flags if selective_dynamics else None}


def read_POTCAR(filepath):
    """Read element symbols from a POTCAR file's TITEL lines.

    Parameters
    ----------
    filepath : str
        Path to the POTCAR file.

    Returns
    -------
    elements : list[str] or None
        Element symbols in POTCAR order (PAW suffixes stripped), or None
        if the POTCAR file does not exist.
    """
    if not os.path.exists(filepath):
        elements = None
        return elements

    with open(filepath, 'r') as f:
        names = [line.split()[3] for line in f if line.strip().startswith('TITEL')]
    elements = [name.split('/')[0].split('_')[0] for name in names]
    return elements


def direct_to_cartesian(lattice_matrix, positions_direct):
    """Convert fractional (direct) coordinates to Cartesian coordinates.

    Parameters
    ----------
    lattice_matrix      : (3, 3) ndarray, Cartesian lattice vectors (Angstrom)
    positions_direct    : (N, 3) ndarray, fractional atomic positions

    Returns
    -------
    positions_cartesian : (N, 3) ndarray, Cartesian atomic positions (Angstrom)
    """

    positions = positions_direct % 1.0
    positions_cartesian = positions @ lattice_matrix

    return positions_cartesian


def cartesian_to_direct(lattice_matrix, positions_cartesian):
    """Convert Cartesian coordinates to fractional (direct) coordinates.

    Parameters
    ----------
    lattice_matrix      : (3, 3) ndarray, Cartesian lattice vectors (Angstrom)
    positions_cartesian : (N, 3) ndarray, Cartesian atomic positions (Angstrom)

    Returns
    -------
    positions_direct    : (N, 3) ndarray, fractional atomic positions in [0, 1)
    """

    positions_direct = (positions_cartesian @ np.linalg.inv(lattice_matrix)) % 1.0

    return positions_direct


def read_FORCE_CONSTANTS(filepath, total_atoms):
    """Read and parse a phonopy/phono3py FORCE_CONSTANTS file.

    The file format is:
      Line 0  : <total_symmetry> <total_atoms>
      Per block: one header line with atom pair indices,
                 followed by a 3x3 IFC matrix (3 lines of 3 values each).
    The RMS is computed per block as sqrt(mean(Phi_ij^2)) over all 9 elements.

    Parameters
    ----------
    filepath    : str, path to the FORCE_CONSTANTS file
    total_atoms : int, expected number of atoms (must match file header)

    Returns
    -------
    dict with keys:
        total_symmetry : int, number of symmetry-inequivalent atom pairs
        pair_list      : list of list of str, atom pair indices per block
        rms            : list of float, RMS of each 3x3 IFC block in eV/Angstrom^2
    """

    with open(filepath, 'r') as f:
        force_lines = f.readlines()

    total_symmetry, force_total_atoms = map(int, force_lines[0].split())

    if force_total_atoms != total_atoms:
        print("ERROR! Total atoms not match.")
        exit(1)

    pair_list = []
    rms = []
    line_index = 1

    for _ in range(total_symmetry):
        for _ in range(force_total_atoms):
            pair_list.append(force_lines[line_index].split())
            line_index += 1

            rms.append(np.sqrt(np.mean([float(x) ** 2
                                        for i in range(3)
                                        for x in force_lines[line_index + i].split()])))
            line_index += 3

    return {"total_symmetry": total_symmetry,
            "pair_list": pair_list,
            "rms": rms}


def compute_image_offsets(lattice_matrix):
    """Compute Cartesian offset vectors for all 27 periodic images.

    Generates all combinations of (k, l, m) in {-1, 0, 1}^3 and converts
    them to Cartesian coordinates using the lattice matrix. These offsets
    are used for minimum image distance calculations under PBC.

    Parameters
    ----------
    lattice_matrix : np.ndarray, shape (3, 3), Cartesian lattice vectors in Angstrom

    Returns
    -------
    image_offsets : np.ndarray, shape (27, 3), Cartesian offset vectors in Angstrom
    """

    klm = np.array([[k, l, m] for k in range(-1, 2)
                               for l in range(-1, 2)
                               for m in range(-1, 2)])   # (27, 3)

    return klm @ lattice_matrix


def calculate_distance_rms(lattice_matrix, total_atoms, positions_cartesian, image_offsets,
                           pair_list, rms, species):
    """Calculate minimum image distances and pair them with IFC RMS values.

    For each symmetry-inequivalent atom (select atom), computes the minimum
    periodic image distance to every other atom using vectorised NumPy
    broadcasting over all 27 image offsets. Results are sorted by distance.

    Parameters
    ----------
    lattice_matrix      : np.ndarray, shape (3, 3), Cartesian lattice vectors in Angstrom
    total_atoms         : int, total number of atoms
    positions_cartesian : np.ndarray, shape (N, 3), Cartesian coordinates in Angstrom
    image_offsets       : np.ndarray, shape (27, 3), PBC image offset vectors in Angstrom
    pair_list           : list of list of str, atom pair indices from FORCE_CONSTANTS
    rms                 : list of float, RMS of each 3x3 IFC block in eV/Angstrom^2
    species             : list of str, element symbol per atom in POSCAR order
                          (the same order as the FORCE_CONSTANTS indices)

    Returns
    -------
    distance_rms : list of tuple (species_i, species_j, distance, rms),
                   sorted by distance in Angstrom
    """

    distance_rms = []
    for index, s in enumerate(range(0, len(pair_list), total_atoms)):
        select = int(pair_list[s][0]) - 1
        reference_position_cartesian = positions_cartesian[select]

        # Mask out the selected atom
        mask = np.arange(total_atoms) != select
        other_positions_cartesian = positions_cartesian[mask]           # (N-1, 3)
        other_species = [species[i] for i in range(total_atoms) if i != select]
        other_rms = [rms[i + index * total_atoms] for i in range(total_atoms) if i != select]

        # Displacement vectors: (N-1, 3)
        diff = other_positions_cartesian - reference_position_cartesian

        # Add all image offsets: (N-1, 1, 3) + (1, 27, 3) → (N-1, 27, 3)
        diff_images = diff[:, np.newaxis, :] + image_offsets[np.newaxis, :, :]

        # Minimum image distances: (N-1,)
        min_distances = np.linalg.norm(diff_images, axis=2).min(axis=1)

        for symbol, distance, r in zip(other_species, min_distances, other_rms):
            distance_rms.append((species[select], symbol, distance, r))

    # Sort by distance
    distance_rms.sort(key=lambda x: x[2])

    return distance_rms


def write_output(elements, distance_rms):
    """Write distance vs RMS data to element-pair output files.

    Creates one output file per unique element pair named RMS_A-B.dat,
    containing two columns: interatomic distance (Angstrom) and
    RMS of the 3x3 IFC block (eV/Angstrom^2), sorted by distance.

    Parameters
    ----------
    elements     : list of str, unique element symbols in the structure
    distance_rms : list of tuple (species_i, species_j, distance, rms),
                   sorted by distance
    """

    element_pairs = [(elements[i], elements[j])
                     for i in range(len(elements))
                     for j in range(i, len(elements))]

    for pair in element_pairs:
        filename = f"RMS_{pair[0]}-{pair[1]}.dat"
        with open(filename, 'w') as o:
            o.write("# Distance vs RMS of 2nd IFCs\n")
            o.write("#   Distance      RMS\n")
            for item in distance_rms:
                if ((item[0] == pair[0] and item[1] == pair[1]) or
                        (item[0] == pair[1] and item[1] == pair[0])):
                    o.write(f"  {item[2]:>12.8f}  {item[3]:>12.8f}\n")


def main():
    """Parse arguments, calculate distance and IFC RMS, and write outputs."""

    if '-h' in argv or '--help' in argv or len(argv) != 3:
        usage()

    # Keep the original POSCAR order: FORCE_CONSTANTS indices refer to it
    poscar = read_POSCAR(argv[1])
    force_constants = read_FORCE_CONSTANTS(argv[2], poscar["total_atoms"])
    image_offsets = compute_image_offsets(poscar["lattice_matrix"])
    distance_rms = calculate_distance_rms(poscar["lattice_matrix"], poscar["total_atoms"], poscar["positions_cartesian"],
                                          image_offsets, force_constants["pair_list"], force_constants["rms"], poscar["species"])
    write_output(list(dict.fromkeys(poscar["elements"])), distance_rms)


if __name__ == "__main__":
    main()
