#!/usr/bin/env python

from sys import argv, exit
import os
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
    elements = []
    is_number = lines[5].split()[0].isdecimal()
    if is_number:
        # VASP4 format: no element line -> prompt user
        for i in range(len(lines[5].split())):
            while True:
                name = input(f"Enter the name of species No. {i + 1:>3}: ").strip()
                if name in _ELEMENT_SYMBOLS:
                    break
                else:
                    print("The name of species must be a valid element symbol.")
            elements.append(name)
        atom_counts = [int(x) for x in lines[5].split()]
        selective_dynamics = lines[6].lower().startswith('s')
        position_start = 8 if selective_dynamics else 7
    else:
        # VASP5 format: element symbols present.
        # Strip potential PAW/GGA suffixes such as '_pv' or '/GGA'.
        raw_elements = lines[5].split()
        for name in raw_elements:
            elements.append(name.split('/')[0].split('_')[0])
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


def check_elements(elements):
    """Check for duplicate element symbols and prompt the user for a canonical order.

    If duplicate symbols are found (e.g. ['Mo', 'S', 'Mo']), the user is asked
    to specify the desired ordering of the unique species. An empty input accepts
    the default order (first-occurrence order).

    Parameters
    ----------
    elements : list[str] — element symbols as parsed from the POSCAR

    Returns
    -------
    list[str] or None
        The user-specified element order if duplicates were found, else None.
    """

    unique_elements = list(dict.fromkeys(elements))

    if len(elements) != len(unique_elements):
        print("\nFound duplicated elements in POSCAR!")
        print("Unique elements: [" + " ".join(unique_elements) + "]")
        while True:
            sort_elements = input("Enter the desired element order (separate by space): ").split()
            if len(sort_elements) == 0:
                print("Warning! Empty input — using default unique element order.")
                return unique_elements.copy()
            if (len(sort_elements) == len(unique_elements) and
                    set(sort_elements) == set(unique_elements)):
                return sort_elements
            print("ERROR! The species do not match the unique elements. Try again.")
    else:
        return None


def mapping_elements(elements, atom_counts, positions_cartesian, positions_direct,
                     species, selective_dynamics, flags, sort_elements=None):
    """Re-order atoms so that each element block is contiguous and sorted canonically.

    Groups atomic positions by element symbol, resolves any duplicate element
    entries via check_elements(), and returns arrays sorted according to the
    specified (or user-supplied) element order. This is required because some
    POSCARs interleave atoms of the same species across multiple blocks.

    Parameters
    ----------
    elements            : list[str]            — element symbols from POSCAR
    atom_counts         : list[int]            — atoms per element block
    positions_cartesian : np.ndarray (N, 3)    — Cartesian coordinates in Å
    positions_direct    : np.ndarray (N, 3)    — fractional coordinates
    species             : list[str]            — per-atom element labels
    selective_dynamics  : bool                 — whether Selective Dynamics is used
    flags               : np.ndarray or None   — per-atom T/F flags
    sort_elements       : list[str] or None    — explicit element order (optional)

    Returns
    -------
    dict with keys:
        elements            : list[str]
        atom_counts         : list[int]
        positions_cartesian : np.ndarray (N, 3)
        positions_direct    : np.ndarray (N, 3)
        species             : list[str]
        flags               : np.ndarray or None
    """

    new_elements = elements.copy()
    new_atom_counts = atom_counts.copy()
    new_positions_cartesian = positions_cartesian.copy()
    new_positions_direct = positions_direct.copy()
    new_species = species.copy()
    new_flags = flags.copy() if selective_dynamics else None

    # Group positions and flags by element symbol
    elements_positions_cartesian = {}
    elements_positions_direct = {}
    elements_species = {}
    elements_flags = {} if selective_dynamics else None
    for idx in range(len(new_species)):
        element = new_species[idx]
        elements_positions_cartesian.setdefault(element, []).append(
            new_positions_cartesian[idx])
        elements_positions_direct.setdefault(element, []).append(
            new_positions_direct[idx])
        elements_species.setdefault(element, []).append(new_species[idx])
        if selective_dynamics and new_flags is not None:
            elements_flags.setdefault(element, []).append(new_flags[idx])

    # Resolve canonical element order (prompts user if duplicates exist)
    if sort_elements is None:
        sort_elements = check_elements(elements)

    # Rebuild arrays in the resolved order
    if sort_elements is not None:
        sort_positions_cartesian = []
        sort_positions_direct = []
        sort_species = []
        sort_flags = [] if selective_dynamics else None
        sort_atom_counts = []
        for element in sort_elements:
            sort_positions_cartesian.extend(elements_positions_cartesian[element])
            sort_positions_direct.extend(elements_positions_direct[element])
            sort_species.extend(elements_species[element])
            if selective_dynamics:
                sort_flags.extend(elements_flags[element])
            sort_atom_counts.append(len(elements_positions_direct[element]))

        new_positions_cartesian = np.array(sort_positions_cartesian, dtype=float)
        new_positions_direct = np.array(sort_positions_direct, dtype=float)
        new_species = list(sort_species)
        if selective_dynamics:
            new_flags = np.array(sort_flags)
        new_atom_counts = sort_atom_counts
        new_elements = sort_elements

    return {"elements":           new_elements,
            "atom_counts":        new_atom_counts,
            "positions_cartesian": new_positions_cartesian,
            "positions_direct":   new_positions_direct,
            "species":            new_species,
            "flags":              new_flags if selective_dynamics else None}


def define_labels(elements, atom_counts):
    """Generate atom labels in the form 'Fe001', 'Fe002', ..., 'C001', etc.

    The numeric suffix is zero-padded to the width of the largest atom count
    plus one (e.g. if max count is 96, suffix width is 3: '001' to '096').

    Parameters
    ----------
    elements    : list of str, element symbols
    atom_counts : list of int, number of atoms per element

    Returns
    -------
    labels : list of str, one label per atom in POSCAR order
    """

    digits = len(str(max(atom_counts))) + 1
    labels = [f"{symbol}{str(counter).zfill(digits)}" for symbol, number in zip(elements, atom_counts)
              for counter in range(1, number + 1)]
    
    return labels


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
                           pair_list, rms, labels):
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
    labels              : list of str, atom labels in the same order as positions

    Returns
    -------
    distance_rms : list of tuple (label_i, label_j, distance, rms),
                   sorted by distance in Angstrom
    """

    distance_rms = []
    for index, s in enumerate(range(0, len(pair_list), total_atoms)):
        select = int(pair_list[s][0]) - 1
        reference_position_cartesian = positions_cartesian[select]

        # Mask out the selected atom
        mask = np.arange(total_atoms) != select
        other_positions_cartesian = positions_cartesian[mask]           # (N-1, 3)
        other_labels = [labels[i] for i in range(total_atoms) if i != select]
        other_rms = [rms[i + index * total_atoms] for i in range(total_atoms) if i != select]

        # Displacement vectors: (N-1, 3)
        diff = other_positions_cartesian - reference_position_cartesian

        # Add all image offsets: (N-1, 1, 3) + (1, 27, 3) → (N-1, 27, 3)
        diff_images = diff[:, np.newaxis, :] + image_offsets[np.newaxis, :, :]

        # Minimum image distances: (N-1,)
        min_distances = np.linalg.norm(diff_images, axis=2).min(axis=1)

        for label, distance, r in zip(other_labels, min_distances, other_rms):
            distance_rms.append((labels[select], label, distance, r))

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
    distance_rms : list of tuple (label_i, label_j, distance, rms),
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
                if ((pair[0] in item[0] and pair[1] in item[1]) or
                        (pair[1] in item[0] and pair[0] in item[1])):
                    o.write(f"  {item[2]:>12.8f}  {item[3]:>12.8f}\n")


def main():
    """Parse arguments, calculate distance and IFC RMS, and write outputs."""

    if '-h' in argv or len(argv) != 3:
        usage()
    
    poscar = read_POSCAR(argv[1])
    mapping = mapping_elements(poscar["elements"], poscar["atom_counts"], poscar["positions_cartesian"],
                               poscar["positions_direct"], poscar["species"],
                               poscar["selective_dynamics"], poscar["flags"])
    force_constants = read_FORCE_CONSTANTS(argv[2], poscar["total_atoms"])
    labels = define_labels(mapping["elements"], mapping["atom_counts"])
    image_offsets = compute_image_offsets(poscar["lattice_matrix"])
    distance_rms = calculate_distance_rms(poscar["lattice_matrix"], poscar["total_atoms"], mapping["positions_cartesian"],
                                          image_offsets, force_constants["pair_list"], force_constants["rms"], labels)
    write_output(mapping["elements"], distance_rms)


if __name__ == "__main__":
    main()
