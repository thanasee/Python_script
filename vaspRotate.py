#!/usr/bin/env python

from sys import argv, exit
import os
import readline
import numpy as np


def usage():
    """Print usage information and exit."""
    print("""
Usage: vaspRotate.py <input> <output>

This script rotate atoms in POSCAR/CONTCAR file.

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

# Standard atomic weights (amu); most stable isotope for elements without one
_ATOMIC_MASS = [
    1.008,     4.0026,    6.94,      9.0122,    10.81,     12.011,    14.007,    15.999,
    18.9984,   20.1797,   22.9898,   24.305,    26.9815,   28.085,    30.9738,   32.06,
    35.45,     39.948,    39.0983,   40.078,    44.9559,   47.867,    50.9415,   51.9961,
    54.938,    55.845,    58.9332,   58.6934,   63.546,    65.38,     69.723,    72.63,
    74.9216,   78.971,    79.904,    83.798,    85.4678,   87.62,     88.9058,   91.224,
    92.9064,   95.95,     97.9072,   101.07,    102.9055,  106.42,    107.8682,  112.414,
    114.818,   118.71,    121.76,    127.6,     126.9045,  131.293,   132.9055,  137.327,
    138.9055,  140.116,   140.9077,  144.242,   144.9128,  150.36,    151.964,   157.25,
    158.9254,  162.5,     164.9303,  167.259,   168.9342,  173.054,   174.9668,  178.49,
    180.9479,  183.84,    186.207,   190.23,    192.217,   195.084,   196.9666,  200.592,
    204.38,    207.2,     208.9804,  208.9824,  209.9872,  222.0176,  223.0197,  226.0254,
    227.0277,  232.0377,  231.0359,  238.0289,  237.0482,  244.0642,  243.0614,  247.0703,
    247.0703,  251.0796,  252.083,   257.0951,  258.0984,  259.101,   262.11,    267.122,
    268.126,   271.134,   270.133,   269.1338,  278.156,   281.165,   281.166,   285.177,
    286.182,   289.19,    289.194,   293.204,   293.208,   294.214
]

_MASS = dict(zip(_ELEMENT_SYMBOLS, _ATOMIC_MASS))


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
    """Convert fractional (Direct) coordinates to Cartesian coordinates.

    Uses the relation:  r_cart = r_direct @ lattice_matrix

    Parameters
    ----------
    lattice_matrix    : np.ndarray, shape (3, 3) — row vectors of the lattice in Å
    positions_direct  : np.ndarray, shape (N, 3) — fractional coordinates

    Returns
    -------
    positions_cartesian : np.ndarray, shape (N, 3) — Cartesian coordinates in Å
    """

    positions = positions_direct % 1.0
    positions_cartesian = positions @ lattice_matrix

    return positions_cartesian


def cartesian_to_direct(lattice_matrix, positions_cartesian):
    """Convert Cartesian coordinates to fractional (Direct) coordinates.

    Uses the relation:  r_direct = r_cart @ lattice_matrix⁻¹

    Parameters
    ----------
    lattice_matrix      : np.ndarray, shape (3, 3) — row vectors of the lattice in Å
    positions_cartesian : np.ndarray, shape (N, 3) — Cartesian coordinates in Å

    Returns
    -------
    positions_direct : np.ndarray, shape (N, 3) — fractional coordinates in [0, 1)
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
    """Generate per-atom labels used as comments in the POSCAR position block.

    Labels take the form '<Symbol><index>' with the index zero-padded to the
    width of the largest atom count plus one (e.g. 'Mo01', 'S003').

    Parameters
    ----------
    elements    : list[str]  — element symbols in canonical order
    atom_counts : list[int]  — number of atoms per element

    Returns
    -------
    labels : list[str] — one label per atom in the same order as the position arrays
    """

    digits = len(str(max(atom_counts))) + 1
    labels = [f"{symbol}{str(counter).zfill(digits)}"
              for symbol, number in zip(elements, atom_counts)
              for counter in range(1, number + 1)]

    return labels


def write_POSCAR(filepath, lattice_matrix, elements, atom_counts, positions_cartesian,
                 positions_direct, selective_dynamics, flags, labels, direct=True):
    """Write a VASP5-format POSCAR file with Direct coordinates.

    The scale factor is always written as 1.0 because lattice vectors are
    already stored in absolute Å units. Atom labels are appended as inline
    comments after each position line for readability.

    Parameters
    ----------
    filepath           : str
    lattice_matrix     : np.ndarray (3, 3)  — lattice vectors in Å
    elements           : list[str]          — element symbols in canonical order
    atom_counts        : list[int]          — atoms per element
    positions_cartesian: np.ndarray (N, 3)  — Cartesian coordinates in Å
    positions_direct   : np.ndarray (N, 3)  — fractional coordinates
    selective_dynamics : bool
    flags              : np.ndarray or None  — per-atom T/F flags
    labels             : list[str]          — per-atom comment labels
    direct             : bool               — True for Direct coordinates, False for Cartesian
    """

    with open(filepath, 'w') as o:
        o.write("Generated by vaspRotate.py\n")
        o.write(f"   {1.0:.14f}\n")
        for lattice in lattice_matrix:
            o.write(f"   {lattice[0]:20.16f}  {lattice[1]:20.16f}  {lattice[2]:20.16f}\n")
        o.write("   " + "    ".join(elements) + " \n")
        o.write("     " + "    ".join(map(str, atom_counts)) + "\n")
        if selective_dynamics:
            o.write("Selective dynamics\n")
        o.write("Direct\n" if direct else "Cartesian\n")
        positions = positions_direct.copy() if direct else positions_cartesian.copy()
        for i, position in enumerate(positions):
            o.write(f"{position[0]:20.16f}{position[1]:20.16f}{position[2]:20.16f}")
            if selective_dynamics:
                o.write(f"   {flags[i, 0]:s}   {flags[i, 1]:s}   {flags[i, 2]:s}")
            o.write(f"   {labels[i]:>6s}\n")


def rotation_matrix():
    """Construct a 3×3 rotation matrix for rotation about an arbitrary axis.

    Uses the Rodrigues rotation formula:
        R = cos θ · I + sin θ · (u×) + (1 − cos θ) · u⊗u

    Returns
    -------
    rotate : np.ndarray (3, 3) — rotation matrix
    """
    
    text = """
Choices of rotation axis
1) X Axis
2) Y Axis
3) Z Axis
4) An Axis Passing Through Specified Vector"""
    print(text)
    
    while True:
        try:
            option_axis = [int(a) for a in input("Enter axis: ").split()]
            if len(option_axis) == 1:
                if 0 < option_axis[0] < 4:
                    axis = option_axis[0] - 1
                    u = np.array([1. if i == axis else 0. for i in range(3)])
                    break
                elif option_axis[0] == 4:
                    print("ex. 1 0 0 means the rotation axis is x axis")
                    while True:
                        try:
                            v = np.array([float(vi) for vi in input("Enter the vector: ").split()])
                            u = v / np.linalg.norm(v)
                            break
                        except ValueError:
                            print("ERROR! Wrong input vector")
                            continue
                    break
                else:
                    print("ERROR! Choose again")
                    continue
            print("  Invalid input! Range of input must be 1.")
            continue
        except ValueError:
            print("  Invalid input! Please enter a number.")
            continue

    # Choose the rotation degree
    while True:
        try:
            input_degree = [float(d) for d in input("Input rotation degree: ").split()]
            if len(input_degree) == 1:
                degree = np.radians(input_degree[0])
                break
            print("ERROR! Wrong input degree")
            continue
        except ValueError:
            print("  Invalid input! Please enter a number.")
            continue

    # define trigonometry functions
    sin = np.sin(degree)
    cos = np.cos(degree)

    # Matrix of rotation
    rotate = cos * np.eye(3) + sin * np.cross(np.eye(3), u) + (1 - cos) * np.outer(u, u)
    
    return rotate


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
                    if start < end:
                        selected_atoms.extend(range(start - 1, end))
                    else:
                        selected_atoms.extend(range(end - 1, start))
                else:
                    selected_atoms.append(int(select) - 1)
            else:
                if select not in species:
                    print(f"  '{select}' is not a species in this structure. TRY AGAIN!")
                    valid = False
                    break
                if select not in _ELEMENT_SYMBOLS:
                    print(f"  '{select}' is not a valid element name. TRY AGAIN!")
                    valid = False
                    break
                selected_atoms.extend([i for i, label in enumerate(species) if label == select])

        if not valid:
            continue

        if len(selected_atoms) > total_atoms or not all(0 <= idx < total_atoms for idx in selected_atoms):
            print("Wrong input atom-indexes! TRY AGAIN!")
        else:
            break

    return selected_atoms


def select_pivot(lattice_matrix, total_atoms, positions_direct, species):
    """Determine the rotated atoms and pivot point interactively.

    The rotated atoms are unwrapped across periodic boundaries (minimum image
    relative to the first one), so a group split across the cell edge stays intact.
    Molecules (type 1): all atoms rotate about the geometric center, the center of
    mass, a specific atom, or a custom fractional point.
    2D/3D materials (type 2): only the selected atoms rotate, about their own
    geometric center or center of mass.

    Parameters
    ----------
    lattice_matrix   : np.ndarray (3, 3) — lattice vectors in Å
    total_atoms      : int               — total number of atoms
    positions_direct : np.ndarray (N, 3) — fractional coordinates
    species          : list[str]         — per-atom element labels

    Returns
    -------
    selected_atoms : list[int]         — 0-based indexes of the rotated atoms
    group          : np.ndarray (M, 3) — unwrapped Cartesian coordinates of the rotated atoms in Å
    ref_point      : np.ndarray (3,)   — pivot point in Cartesian coordinates
    """

    print("""
Choices of type of material:
1) molecules
2) 2D/3D materials""")

    while True:
        input_type = input("Enter choice: ")
        if input_type in ['1', '2']:
            break
        print("ERROR! Wrong input")

    if input_type == '1':
        selected_atoms = list(range(total_atoms))
        print("""
Method for selecting the pivot point of molecule
1) geometric center of molecule
2) center of mass of molecule
3) position of atom in molecule
4) Custom""")
    else:
        selected_atoms = list(dict.fromkeys(select_index(total_atoms, species)))
        print("""
Method for selecting the pivot point of selected atoms
1) geometric center of selected atoms
2) center of mass of selected atoms""")

    # Unwrap the rotated atoms relative to the first one (minimum image)
    group = positions_direct[selected_atoms]
    group = group[0] + (group - group[0]) - np.round(group - group[0])
    group_species = [species[i] for i in selected_atoms]

    options = ['1', '2', '3', '4'] if input_type == '1' else ['1', '2']
    while True:
        option = input("Enter method: ")
        if option not in options:
            print("ERROR!! Choose method again")
            continue
        if option == '1':
            ref_point = np.mean(group, axis=0)
        elif option == '2':
            unknown = sorted(set(s for s in group_species if s not in _MASS))
            if unknown:
                print(f"ERROR! No atomic mass for {' '.join(unknown)}. Choose method again")
                continue
            ref_point = np.average(group, axis=0, weights=[_MASS[s] for s in group_species])
        elif option == '3':
            for j in range(total_atoms):
                print(f"{species[j]} atom : {j + 1:>3.0f}")
            while True:
                try:
                    select = [int(s) for s in input(f"Select the atom as the pivot point (  1 to {total_atoms:>3}): ").split()]
                    if len(select) == 1:
                        select_atom = select[0] - 1
                        if 0 <= select_atom < total_atoms:
                            break
                        print('WRONG No. of atom')
                        continue
                    print("  Invalid input! Range of input must be 1.")
                    continue
                except ValueError:
                    print("  Invalid input! Please enter a number.")
                    continue
            ref_point = group[select_atom]
        else:
            point = []
            for i in ('a', 'b', 'c'):
                while True:
                    try:
                        p = float(input(f"Enter position in {i} direction (direct): "))
                        break
                    except ValueError:
                        print("Wrong input! Please enter a number.")
                point.append(p)
            ref_point = np.array(point)
            # Move the molecule to the periodic image nearest the custom point
            group = group - np.round(np.mean(group, axis=0) - ref_point)
        break

    return selected_atoms, group @ lattice_matrix, ref_point @ lattice_matrix


def rotate_atoms(lattice_matrix, total_atoms, positions_cartesian, positions_direct, species, rotate_matrix):
    """Apply a rotation matrix to atoms and return new Cartesian coordinates.

    Delegates atom selection and pivot point to select_pivot(). Only the selected
    atoms (all atoms for molecules) are rotated; the rest remain fixed.

    Parameters
    ----------
    lattice_matrix      : np.ndarray (3, 3) — lattice vectors in Å
    total_atoms         : int               — total number of atoms
    positions_cartesian : np.ndarray (N, 3) — Cartesian coordinates in Å
    positions_direct    : np.ndarray (N, 3) — fractional coordinates
    species             : list[str]         — per-atom element labels
    rotate_matrix       : np.ndarray (3, 3) — rotation matrix from rotation_matrix()

    Returns
    -------
    new_positions_cartesian : np.ndarray (N, 3) — rotated Cartesian coordinates in Å
    """

    selected_atoms, group, ref_point = select_pivot(lattice_matrix, total_atoms, positions_direct, species)

    new_positions_cartesian = np.copy(positions_cartesian)
    new_positions_cartesian[selected_atoms] = (group - ref_point) @ rotate_matrix.T + ref_point

    return new_positions_cartesian


def main():
    """Parse arguments, build rotation matrix from specified axis and angle,
    rotate selected atom, and write outputs.
    """
    if '-h' in argv or '--help' in argv or len(argv) != 3:
        usage()
    
    unrotate = read_POSCAR(argv[1])
    rotate = rotation_matrix()
    rotate_positions_cartesian = rotate_atoms(unrotate["lattice_matrix"], unrotate["total_atoms"], unrotate["positions_cartesian"],
                                              unrotate["positions_direct"], unrotate["species"], rotate)
    rotate_positions_direct = cartesian_to_direct(unrotate["lattice_matrix"], rotate_positions_cartesian)
    mapping = mapping_elements(unrotate["elements"], unrotate["atom_counts"], rotate_positions_cartesian,
                               rotate_positions_direct, unrotate["species"], unrotate["selective_dynamics"],
                               unrotate["flags"])
    labels = define_labels(mapping["elements"], mapping["atom_counts"])
    write_POSCAR(argv[2], unrotate["lattice_matrix"], mapping["elements"], mapping["atom_counts"],
                 mapping["positions_cartesian"], mapping["positions_direct"], unrotate["selective_dynamics"],
                 mapping["flags"], labels)
    
    print("")


if __name__ == "__main__":
    main()
