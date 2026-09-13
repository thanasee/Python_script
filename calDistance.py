#!/usr/bin/env python

from sys import argv, exit
import os
import readline
import numpy as np


def usage():
    """Print usage information and exit."""
    print("""
Usage: calDistance.py <input>

This script calculate distance between atoms from POSCAR/CONTCAR files.

This script was inspired by Jiraroj T-Thienprasert
and developed by Thanasee Thanasarnsurapong.
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


def compute_image_offsets(lattice_matrix):
    """Pre-compute all 27 periodic image translation vectors.

    Generates vectors k*a + l*b + m*c for k, l, m in {-1, 0, 1},
    covering the origin cell and all 26 neighbouring cells.
    Used to find the minimum-image distance under periodic boundary conditions.

    Parameters
    ----------
    lattice_matrix : (3, 3) ndarray, Cartesian lattice vectors (Angstrom)

    Returns
    -------
    image_offsets : (27, 3) ndarray, translation vectors in Cartesian coordinates
    """

    klm = np.array([[k, l, m] for k in range(-1, 2)
                               for l in range(-1, 2)
                               for m in range(-1, 2)])
    
    return klm @ lattice_matrix


def min_image_distance(position_i, position_j, image_offsets):
    """Compute the minimum-image distance between two atomic positions.

    Adds all 27 image offsets to the displacement vector and returns
    the shortest distance, accounting for periodic boundary conditions.

    Parameters
    ----------
    position_i    : (3,) ndarray, Cartesian position of atom i (Angstrom)
    position_j    : (3,) ndarray, Cartesian position of atom j (Angstrom)
    image_offsets : (27, 3) ndarray, periodic image translation vectors

    Returns
    -------
    float : minimum-image distance in Angstrom
    """

    diff = position_j - position_i                   # (3,)
    diff_offset = diff[np.newaxis, :] + image_offsets # (27, 3)
    
    return np.linalg.norm(diff_offset, axis=1).min()


def min_image_distances(position_reference, positions_others, image_offsets):
    """Compute minimum-image distances from one reference atom to many others.

    Vectorised over N atoms using broadcasting:
      diff         : (N, 3)
      diff_offset  : (N, 27, 3)  via (N, 1, 3) + (1, 27, 3)
      distances    : (N, 27)
      result       : (N,) minimum over 27 images per atom

    Parameters
    ----------
    position_reference : (3,) ndarray, Cartesian position of the reference atom
    positions_others   : (N, 3) ndarray, Cartesian positions of N other atoms
    image_offsets      : (27, 3) ndarray, periodic image translation vectors

    Returns
    -------
    (N,) ndarray : minimum-image distances in Angstrom
    """

    diff = positions_others - position_reference                         # (N, 3)
    diff_offset = diff[:, np.newaxis, :] + image_offsets[np.newaxis, :, :] # (N, 27, 3)
    
    return np.linalg.norm(diff_offset, axis=2).min(axis=1)              # (N,)


def select_index(total_atoms, species, prompt=None, allow_all=True):
    """Prompt the user to select atoms by element symbol and/or atom index.
    
    Accepts free-format input mixing element symbols (e.g. 'C'), single indexes
    (e.g. '3'), ranges (e.g. '1-4'), and the keyword 'all'. Repeats the prompt
    on invalid input. The selected_atoms list is reset on each retry to prevent
    duplicates accumulating across bad inputs.

    Parameters
    ----------
    total_atoms : int       — total number of atoms (used for bounds checking)
    species     : list[str] — per-atom element label (used for symbol selection)
    prompt      : srt       — message printed before the input prompt
    allow_all   : bool      — whether the keyword 'all' is permitted (default True)

    Returns
    -------
    selected_atoms : list[int] — 0-based indexes of the chosen atoms
    """

    if prompt != None:
        print(prompt)
    else:
        print(f"""
Input element-symbol and/or atom-indexes to choose ({1:>3} to {total_atoms:>3})
(Free-format input, e.g., 1 3 1-4 C H all)""")
    while True:
        selected_atoms = []
        input_select = input().split()
        valid = True
        
        for select in input_select:
            if 'all' in select:
                if not allow_all:
                    print("Cannot use 'all' in this session. TRY AGAIN!")
                    valid = False
                    break
                selected_atoms.extend(range(total_atoms))
                break
            if select.isnumeric() or '-' in select:
                if '-' in select:
                    start, end = map(int, select.split('-'))
                    selected_atoms.extend(range(start - 1, end))
                else:
                    selected_atoms.append(int(select) - 1)
            else:
                selected_atoms.extend([i for i, label in enumerate(species) if label == select])

        if not valid:
            continue

        if len(selected_atoms) > total_atoms or not all(0 <= idx < total_atoms for idx in selected_atoms):
            print("Wrong input atom-indexes! TRY AGAIN!")
        else:
            break

    return selected_atoms


def one_to_all(total_atoms, positions_cartesian, labels, image_offsets):
    """Method 1: compute distances from one selected atom to all other atoms.

    Writes two output files:
    - distance-unsorted.dat : distances in POSCAR atom order
    - distance-sorted.dat   : distances sorted from shortest to longest

    Parameters
    ----------
    total_atoms         : int, total number of atoms
    positions_cartesian : (N, 3) ndarray, Cartesian atomic positions (Angstrom)
    labels              : list of str, atom labels (e.g. 'Fe001')
    image_offsets       : (27, 3) ndarray, periodic image translation vectors
    """

    while True:
        try:
            select = [int(s) for s in input(f"Choose the selected atom (  1 to {total_atoms:>3}): ").split()]
            if len(select) == 1:
                index_select = select[0] - 1
                if 0 <= index_select < total_atoms:
                    break
                print('WRONG No. of the selected atom')
                continue
            print("  Invalid input! Range of input must be 1.")
            continue
        except ValueError:
            print("  Invalid input! Please enter a number.")
            continue

    mask = np.arange(total_atoms) != index_select
    other_positions = positions_cartesian[mask]
    other_labels = [labels[i] for i in range(total_atoms) if i != index_select]
 
    min_distances = min_image_distances(positions_cartesian[index_select], other_positions, image_offsets)
    pair = [(labels[index_select], lbl) for lbl in other_labels]

    with open('distance-unsorted.dat', 'w') as o:
        o.write(f"# Distance between {labels[index_select]} and all other atoms\n")
        o.write("#   Atom1  Atom2     Distance\n")
        for (a1, a2), d in zip(pair, min_distances):
            o.write(f"  {a1:>5s}  {a2:>5s}  {d:>12.8f}\n")
        o.write(f"      Average   {np.mean(min_distances):>12.8f}\n")

    order = np.argsort(min_distances)
    with open('distance-sorted.dat', 'w') as o:
        o.write(f"# Distance between {labels[index_select]} and all other atoms\n")
        o.write("#   Atom1  Atom2     Distance\n")
        for i in order:
            a1, a2 = pair[i]
            o.write(f"  {a1:>5s}  {a2:>5s}  {min_distances[i]:>12.8f}\n")
        o.write(f"      Average   {np.mean(min_distances):>12.8f}\n")


def atom_pairs(total_atoms, positions_cartesian, labels, image_offsets):
    """Method 2: compute distances between user-specified atom pairs.

    Prompts for the number of pairs, then for each pair prompts for
    the 1st and 2nd atom indexes. Prints results to stdout and writes
    to distance-atom-atom.dat.

    Parameters
    ----------
    total_atoms         : int, total number of atoms
    positions_cartesian : (N, 3) ndarray, Cartesian atomic positions (Angstrom)
    labels              : list of str, atom labels (e.g. 'Fe001')
    image_offsets       : (27, 3) ndarray, periodic image translation vectors
    """

    while True:
        try:
            input_pair = [int(p) for p in input("Enter number of pair atoms: ").split()]
            if len(input_pair) == 1:
                number_pair = input_pair[0]
                if number_pair > 0:
                    break
                print("Number of pair atoms must be a positive integer.")
                continue
            print("ERROR! Range of number of pair atoms must be 1.")
            continue
        except ValueError:
            print("ERROR! Please enter a number.")
            continue
 
    distances, pair = [], []
    for i in range(number_pair):
        j = i + 1
        print(f"\nFor pair {j:>3}")

        while True:
            try:
                s1 = [int(s) for s in input(f"Choose the 1st selected atom of pair {j:>3} (  1 to {total_atoms:>3}): ").split()]
                if len(s1) == 1:
                    idx1 = s1[0] - 1
                    if 0 <= idx1 < total_atoms:
                        break
                    print('WRONG No. of the 1st selected atom')
                    continue
                print("  Invalid input! Range of input must be 1.")
                continue
            except ValueError:
                print("  Invalid input! Please enter a number.")
                continue

        while True:
            try:
                s2 = [int(s) for s in input(f"Choose the 2nd selected atom of pair {j:>3} (  1 to {total_atoms:>3}): ").split()]
                if len(s2) == 1:
                    idx2 = s2[0] - 1
                    if 0 <= idx2 < total_atoms:
                        break
                    print('WRONG No. of the 2nd selected atom')
                    continue
                print("  Invalid input! Range of input must be 1.")
                continue
            except ValueError:
                print("  Invalid input! Please enter a number.")
                continue
 
        min_distance = min_image_distance(positions_cartesian[idx1], positions_cartesian[idx2], image_offsets)
        pair.append((labels[idx1], labels[idx2]))
        distances.append(min_distance)
 
    print("# Distance between 2 atoms")
    print("# Atom1  Atom2    Distance")
    for (a1, a2), min_distance in zip(pair, distances):
        print(f"  {a1:>5s}  {a2:>5s}  {min_distance:>12.8f}")
    print(f"      Average   {np.mean(distances):>12.8f}")
 
    with open('distance-atom-atom.dat', 'w') as o:
        o.write("# Distance between 2 atoms\n")
        o.write("# Atom1  Atom2    Distance\n")
        for (a1, a2), min_distance in zip(pair, distances):
            o.write(f"  {a1:>5s}  {a2:>5s}  {min_distance:>12.8f}\n")
        o.write(f"      Average   {np.mean(distances):>12.8f}\n")


def atom_molecule(total_atoms, positions_cartesian, species, labels, image_offsets):
    """Method 3: compute distances from selected atoms to molecule centroids.

    For each pair, prompts for a reference atom and a group of atoms
    defining the molecule. The distance is measured from the reference atom
    to the centroid (geometric center) of the selected molecule atoms.
    Prints results to stdout and writes to distance-atom-molecule.dat.

    Parameters
    ----------
    total_atoms         : int, total number of atoms
    positions_cartesian : (N, 3) ndarray, Cartesian atomic positions (Angstrom)
    species             : list of str, element symbol for each atom
    labels              : list of str, atom labels (e.g. 'Fe001')
    image_offsets       : (27, 3) ndarray, periodic image translation vectors
    """

    digits = len(str(total_atoms)) + 1

    while True:
        try:
            input_pair = [int(p) for p in input("Enter number of pair atom-molecule: ").split()]
            if len(input_pair) == 1:
                number_pair = input_pair[0]
                if number_pair > 0:
                    break
                print("Number of pair atom-molecule must be a positive integer.")
                continue
            print("ERROR! Range of number of pair atom-molecule must be 1.")
            continue
        except ValueError:
            print("ERROR! Please enter a number.")
            continue
 
    distances, pair = [], []
    for i in range(number_pair):
        j = i + 1
        print(f"\nFor pair {j:>3}")

        while True:
            try:
                select = [int(s) for s in input(f"Choose the selected atom of pair {j:>3} (  1 to {total_atoms:>3}): ").split()]
                if len(select) == 1:
                    index_select = select[0] - 1
                    if 0 <= index_select < total_atoms:
                        break
                    print('WRONG No. of the selected atom')
                    continue
                print("  Invalid input! Range of input must be 1.")
                continue
            except ValueError:
                print("  Invalid input! Please enter a number.")
                continue
 
        targets = select_index(total_atoms, species, f"\nInput element-symbol and/or atom-indexes to choose ({1:>3} to {total_atoms:>3})\n"
"(Free-format input, e.g., 1 3 1-4 C H all)")
 
        target_site = np.mean(positions_cartesian[targets], axis=0)  # centroid (3,)
        min_distance = min_image_distance(positions_cartesian[index_select], target_site, image_offsets)
        pair.append((labels[index_select], str(i + 1).zfill(digits)))
        distances.append(min_distance)
 
    print("# Distance between selected atom and molecule")
    print("# Atom     Molecule  Distance")
    for (atom, mol), min_distance in zip(pair, distances):
        print(f"  {atom:>5s}  {mol:>5s}    {min_distance:>12.8f}")
    print(f"        Average   {np.mean(distances):>12.8f}")
 
    with open('distance-atom-molecule.dat', 'w') as o:
        o.write("# Distance between selected atom and molecule\n")
        o.write("# Atom     Molecule  Distance\n")
        for (atom, mol), min_distance in zip(pair, distances):
            o.write(f"  {atom:>5s}  {mol:>5s}    {min_distance:>12.8f}\n")
        o.write(f"        Average   {np.mean(distances):>12.8f}\n")


def z_distance(total_atoms, positions, species):
    """Method 4: compute the z-axis distance between substrate top and adsorbent bottom.

    Prompts separately for the substrate atom group and the adsorbent atom group.
    Finds the atom with the highest z-coordinate in the substrate and the atom
    with the lowest z-coordinate in the adsorbent, then reports their separation
    along the z-axis. Useful for measuring adsorption height or slab thickness.
    Results are printed to stdout only.

    Parameters
    ----------
    total_atoms : int, total number of atoms
    positions   : (N, 3) ndarray, Cartesian atomic positions (Angstrom)
    species     : list of str, element symbol for each atom
    """

    print("Tip: this method can measure the thickness of your system.")
 
    # Substrate: find the atom with the maximum z-coordinate
    substrate_index = select_index(total_atoms, species, f"\nSubstrate — input element-symbol and/or atom-indexes ({1:>3} to {total_atoms:>3})\n"
"(Free-format input, e.g., 1 3 1-4 C H  — 'all' not allowed)", False)
 
    if len(substrate_index) == 1:
        highest_substrate = positions[substrate_index[0]]
    else:
        z_sub = positions[substrate_index, 2]
        top_candidates = [substrate_index[j] for j, z in enumerate(z_sub) if z == z_sub.max()]
        if len(top_candidates) == 1:
            highest_substrate = positions[top_candidates[0]]
        else:
            print(f"  The highest atoms in substrate : {[i + 1 for i in top_candidates]}")
            while True:
                try:
                    select = [int(s) for s in input(f"  Select atom in substrate (  1 to {total_atoms:>3}): ").split()]
                    if len(select) == 1:
                        index_select = select[0] - 1
                        if index_select in top_candidates:
                            highest_substrate = positions[index_select]
                            break
                        print('WRONG! No. of atom in substrate')
                        continue
                    print("  Invalid input! Range of input must be 1.")
                    continue
                except ValueError:
                    print("  Invalid input! Please enter a number.")
                    continue
 
    # Adsorbent: find the atom with the minimum z-coordinate
    adsorbent_index = select_index(total_atoms, species, f"\nAdsorbent — input element-symbol and/or atom-indexes ({1:>3} to {total_atoms:>3})\n"
"(Free-format input, e.g., 1 3 1-4 C H  — 'all' not allowed)", False)
 
    if len(adsorbent_index) == 1:
        lowest_adsorbent = positions[adsorbent_index[0]]
    else:
        z_ads = positions[adsorbent_index, 2]
        bot_candidates = [adsorbent_index[j] for j, z in enumerate(z_ads) if z == z_ads.min()]
        if len(bot_candidates) == 1:
            lowest_adsorbent = positions[bot_candidates[0]]
        else:
            print(f"  The lowest atoms in adsorbent : {[i + 1 for i in bot_candidates]}")
            while True:
                try:
                    select = [int(s) for s in input(f"  Select atom in adsorbent (  1 to {total_atoms:>3}): ").split()]
                    if len(select) == 1:
                        index_select = select[0] - 1
                        if index_select in bot_candidates:
                            lowest_adsorbent = positions[index_select]
                            break
                        print('WRONG! No. of atom in adsorbent')
                        continue
                    print("  Invalid input! Range of input must be 1.")
                    continue
                except ValueError:
                    print("  Invalid input! Please enter a number.")
                    continue
 
    distance = np.abs(lowest_adsorbent[2] - highest_substrate[2])
    print(f"Distance along z-axis is {distance:>12.8f} Angstrom.")


def main():
    """Parse arguments, read POSCAR, calculate distance by selected method, write output files."""

    if '-h' in argv or len(argv) != 2:
        usage()
 
    poscar = read_POSCAR(argv[1])
    mapping = mapping_elements(poscar["elements"], poscar["atom_counts"], poscar["positions_cartesian"],
                               poscar["positions_direct"], poscar["species"],
                               poscar["selective_dynamics"], poscar["flags"])
    labels = define_labels(mapping["elements"], mapping["atom_counts"])
    image_offsets = compute_image_offsets(poscar["lattice_matrix"])
 
    print("""
Choices of calculating distance
 1) Between selected atom and all other atoms
 2) Between 2 selected atoms
 3) Between selected atom and molecule
 4) Between highest atom in substrate and lowest atom in molecule (along z-axis only)""")
 
    while True:
        method = input("Enter choice : ")
        if method == '1':
            one_to_all(poscar["total_atoms"], mapping["positions_cartesian"], labels, image_offsets)
            break
        elif method == '2':
            atom_pairs(poscar["total_atoms"], mapping["positions_cartesian"], labels, image_offsets)
            break
        elif method == '3':
            atom_molecule(poscar["total_atoms"], mapping["positions_cartesian"], mapping["species"], labels, image_offsets)
            break
        elif method == '4':
            z_distance(poscar["total_atoms"], mapping["positions_cartesian"], mapping["species"])
            break
        else:
            print("ERROR! Wrong choice")


if __name__ == "__main__":
    main()
