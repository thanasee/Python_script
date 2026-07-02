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
        option_axis = input("Enter axis: ")
        if option_axis.isdecimal() and option_axis != '0':
            if option_axis in ['1', '2', '3']:
                axis = int(option_axis) - 1
                u = np.array([1. if i == axis else 0. for i in range(3)])
                break
            elif option_axis == '4':
                print("ex. 1 0 0 means the rotation axis is x axis")
                while True:
                    v = input("Enter the vector: ")
                    if all(vi.lstrip('-').replace('.', '').isdigit() for vi in v.split()):
                        break
                    else:
                        print("ERROR! Wrong input vector")
                u = np.array([float(vi) for vi in v.split()])
                u /= np.linalg.norm(u)
                break
            else:
                print("ERROR!! Choose again")

    # Choose the rotation degree
    while True:
        input_degree = input("Input rotation degree: ")
        if input_degree.lstrip('-').replace('.', '').isdigit():
            break
        else:
            print("ERROR! Wrong input degree")
    degree = np.radians(float(input_degree))

    # define trigonometry functions
    sin = np.sin(degree)
    cos = np.cos(degree)

    # Matrix of rotation
    rotate = cos * np.eye(3) + sin * np.cross(np.eye(3), u) + (1 - cos) * np.outer(u, u)
    
    return rotate


def select_index(total_atoms, species):
    """Parse free-format atom selection input and return a list of 0-based indexes.

    Accepts element symbols, integer indexes, hyphen-separated ranges, and 'all'.
    Loops until a valid, non-empty selection within bounds is entered.

    Parameters
    ----------
    total_atoms : int       — upper bound (exclusive) for valid atom indexes
    species     : list[str] — per-atom element labels for symbol-based selection

    Returns
    -------
    selected : list[int] — 0-based atom indexes
    """
    
    print(f"""
Input element-symbol and/or atom-indexes to choose ({1:>3} to {total_atoms:>3})
(Free-format input, e.g., 1 3 1-4 C H all)""")
    while True:
        selected_atoms = []
        input_select = input().split()
 
        for select in input_select:
            if 'all' in select:
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
 
        if len(selected_atoms) > total_atoms or not all(0 <= idx < total_atoms for idx in selected_atoms):
            print("Wrong input atom-indexes! TRY AGAIN!")
        else:
            break
 
    return selected_atoms


def select_pivot(lattice_matrix, total_atoms, positions_cartesian, species):
    """Determine the material type and pivot point for rotation interactively.

    For molecules (type 1), the pivot point is chosen from three methods:
    geometric center, a specific atom position, or a custom fractional coordinate.
    For 2D/3D materials (type 2), atoms are selected via select_index() and the
    pivot point is set to their geometric center.

    Parameters
    ----------
    lattice_matrix      : np.ndarray (3, 3) — lattice vectors in Å
    total_atoms         : int               — total number of atoms
    positions_cartesian : np.ndarray (N, 3) — Cartesian coordinates in Å
    species             : list[str]         — per-atom element labels

    Returns
    -------
    input_type     : str            — '1' for molecule, '2' for 2D/3D material
    selected_atoms : list[int] or None — 0-based atom indexes (None for type 1)
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
        print("""
Method for selecting the pivot point of molecule
1) center of molecule
2) position of atom in molecule
3) Custom""")

        while True:
            option = input("Enter method: ")
            if option == '1':
                return input_type, None, np.mean(positions_cartesian, axis=0)
            elif option == '2':
                for j in range(total_atoms):
                    print(f"{species[j]} atom : {j + 1:>3.0f}")
                while True:
                    try:
                        select_atom = int(input(f"Select the atom as the pivot point ({1:>3} to {total_atoms:>3}): "))
                        if 0 < select_atom <= total_atoms:
                            break
                    except ValueError:
                        print("Wrong No. of atom")
                return input_type, None, positions_cartesian[int(select_atom) - 1]
            elif option == '3':
                point = []
                for i in ('a', 'b', 'c'):
                    while True:
                        try:
                            p = float(input(f"Enter position in {i} direction (direct): "))
                            break
                        except ValueError:
                            print("Wrong input! Please enter a number.")
                    point.append(float(p))
                point = np.array(point)
                return input_type, None, point @ lattice_matrix
            else:
                print("ERROR!! Choose method again")

    else:
        selected_atoms = select_index(total_atoms, species)
        ref_point = np.mean(positions_cartesian[selected_atoms], axis=0)
        return input_type, selected_atoms, ref_point


def rotate_atoms(lattice_matrix, total_atoms, positions_cartesian, species, rotate_matrix):
    """Apply a rotation matrix to atoms and return new Cartesian coordinates.

    Delegates pivot point and atom selection to select_pivot(). For molecules
    (type 1), all atoms are rotated about the pivot. For 2D/3D materials (type 2),
    only the selected subset is rotated while the rest remain fixed.

    Parameters
    ----------
    lattice_matrix      : np.ndarray (3, 3) — lattice vectors in Å
    total_atoms         : int               — total number of atoms
    positions_cartesian : np.ndarray (N, 3) — Cartesian coordinates in Å
    species             : list[str]         — per-atom element labels
    rotate_matrix       : np.ndarray (3, 3) — rotation matrix from rotation_matrix()

    Returns
    -------
    new_positions_cartesian : np.ndarray (N, 3) — rotated Cartesian coordinates in Å
    """
    
    input_type, selected_atoms, ref_point = select_pivot(lattice_matrix, total_atoms, positions_cartesian, species)

    if input_type == '1':
        new_positions_cartesian = (positions_cartesian - ref_point) @ rotate_matrix.T + ref_point
    else:
        new_positions_cartesian = np.copy(positions_cartesian)
        new_positions_cartesian[selected_atoms] = (positions_cartesian[selected_atoms] - ref_point) @ rotate_matrix.T + ref_point

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
                                              unrotate["species"], rotate)
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
