#!/usr/bin/env python

from sys import argv, exit
import os
import numpy as np


def usage():
    """Print usage information and exit."""
    print("""
Usage: vaspSupercell.py <input> <output>
 
This script supports VASP5 structure file format (i.e. POSCAR)
for extending a structure file from a unit cell to a supercell.
 
The expansion matrix can be specified as:
  3 components  ->  diagonal matrix  S_xx S_yy S_zz
  9 components  ->  full 3×3 matrix  S_xx S_xy S_xz  S_yx S_yy S_yz  S_zx S_zy S_zz
 
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
        o.write("Generated by analyzeNEB.py\n")
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


def input_expansion():
    """Prompt the user to enter a supercell expansion matrix and validate it.

    Accepts either 3 components (diagonal) or 9 components (full 3×3 matrix).
    Validates that the determinant is a positive integer, which is required for
    the expansion matrix to define a valid supercell.

    Returns
    -------
    expansion_matrix : np.ndarray, shape (3, 3) — integer expansion matrix
    det_int          : int                       — determinant (number of unit-cell replicas)
    """
    
    text = """
Expansion matrix components:
  3 components  ->  S_xx S_yy S_zz  (diagonal)
  9 components  ->  S_xx S_xy S_xz  S_yx S_yy S_yz  S_zx S_zy S_zz  (full 3×3)
Enter expansion matrix (separate by space):"""
    print(text)

    while True:
        expansion = input()
        try:
            values = np.array(list(map(int, expansion.split())))
            if len(values) == 3:
                expansion_matrix = np.diag(values)
            elif len(values) == 9:
                expansion_matrix = values.reshape(3, 3)
            else:
                print("Input must be 3 or 9 compenents!")
                continue
        except ValueError:
            print("Invalid input. Please enter integer numbers separated by spaces.")
            continue
        det = np.linalg.det(expansion_matrix)
        det_int = int(round(det))
        if det > 1e-10 and abs(det - det_int) < 1e-6 and det_int > 0:
            break
        elif det <= 1e-10:
            print("Invalid expansion matrix: determinant must be a positive integer.")
        else:
            print("Invalid expansion matrix: determinant is not an integer. "
"Please enter integer components that yield a positive integer determinant.")
    
    return expansion_matrix, det_int


def build_supercell(expansion_matrix, replicas, lattice_matrix, atom_counts, total_atoms,
                    positions_cartesian, species, selective_dynamics, flags):
    """Construct a supercell from a unit cell using an integer expansion matrix.

    Generates the supercell lattice, enumerates all valid unit-cell translation
    vectors inside the supercell via an integer adjugate filter, and replicates
    all atomic positions, species, and Selective Dynamics flags accordingly.

    Parameters
    ----------
    expansion_matrix    : np.ndarray, shape (3, 3) — integer supercell expansion matrix
    replicas            : int                       — number of unit-cell replicas (= det of expansion_matrix)
    lattice_matrix      : np.ndarray, shape (3, 3)  — unit-cell lattice vectors in Å
    atom_counts         : list[int]                 — atoms per element in the unit cell
    total_atoms         : int                       — total atoms in the unit cell
    positions_cartesian : np.ndarray, shape (N, 3)  — unit-cell Cartesian coordinates in Å
    species             : list[str]                 — per-atom element label in the unit cell
    selective_dynamics  : bool                      — whether Selective Dynamics flags are present
    flags               : np.ndarray or None        — per-atom T/F flags, or None

    Returns
    -------
    dict with keys:
        lattice_matrix      : np.ndarray, shape (3, 3)      — supercell lattice vectors in Å
        atom_counts         : list[int]                     — atoms per element in the supercell
        total_atoms         : int                           — total atoms in the supercell
        positions_direct    : np.ndarray, shape (N·R, 3)    — supercell fractional coordinates
        positions_cartesian : np.ndarray, shape (N·R, 3)    — supercell Cartesian coordinates in Å
        species             : list[str]                     — per-atom element label in the supercell
        flags               : np.ndarray or None            — replicated T/F flags, or None
    """
    
    # Expansion of lattice matrix
    new_lattice_matrix = expansion_matrix @ lattice_matrix
    
    # Generate lattice grid points inside the supercell
    # Use the 8 corners of the unit supercell box to bound the search range
    corners = np.array([[i, j, k] for i in range(2)
                        for j in range(2) 
                        for k in range(2)])
    corner_transformed = corners @ expansion_matrix
    min_points = np.min(corner_transformed, axis=0).astype(int)
    max_points = np.max(corner_transformed, axis=0).astype(int) + 1

    # Generate all combinations of i, j, k within the given expansion matrix
    all_points = np.array([[i, j, k] for i in range(min_points[0], max_points[0])
                           for j in range(min_points[1], max_points[1])
                           for k in range(min_points[2], max_points[2])])

    # Keep only points whose fractional coordinates in the supercell are in [0, 1)
    adj = np.round(np.linalg.inv(expansion_matrix) * replicas).astype(int)
    frac_points = all_points @ adj
    mask = (np.all(frac_points >= 0, axis=1) &
            np.all(frac_points <  replicas, axis=1))
    grid_points = all_points[mask]
     
    if len(grid_points) != replicas:
        print(f"WARNING: Expected {replicas} grid points but found {len(grid_points)}. "
"Check your expansion matrix.")
    
    # Generate new atomic positions
    # positions are in Å; grid_points are in primitive-cell lattice coordinates
    # Cartesian displacement for each grid point: dot(grid_point, lattice_matrix)
    grid_cartesian = grid_points @ lattice_matrix  # shape: (n_replicas, 3)
     
    # Broadcast: (n_atoms, 1, 3) + (1, n_replicas, 3) → (n_atoms, n_replicas, 3)
    new_positions_cartesian = (positions_cartesian[:, np.newaxis, :] + grid_cartesian[np.newaxis, :, :]).reshape(-1, 3)
    new_species = np.repeat(species, len(grid_points)).tolist()
    
    if selective_dynamics:
        new_flags = np.tile(flags[:, np.newaxis, :], (1, len(grid_points), 1)).reshape(-1, 3)

    # New atom counts per element
    new_atom_counts = [count * replicas for count in atom_counts]
    new_total_atoms = total_atoms * replicas
    
    new_positions_direct = cartesian_to_direct(new_lattice_matrix, new_positions_cartesian)
    
    return {"lattice_matrix": new_lattice_matrix,
            "atom_counts": new_atom_counts,
            "total_atoms": new_total_atoms,
            "positions_direct": new_positions_direct,
            "positions_cartesian": new_positions_cartesian,
            "species": new_species,
            "flags": new_flags if selective_dynamics else None}


def main():
    """Parse argumetns, expand supercell, and write output"""
    if '-h' in argv or '--help' in argv or len(argv) != 3:
        usage()
    
    unitcell = read_POSCAR(argv[1])
    expansion_matrix, replicas = input_expansion()
    supercell = build_supercell(expansion_matrix, replicas, unitcell["lattice_matrix"], unitcell["atom_counts"], unitcell["total_atoms"],
                                unitcell["positions_cartesian"], unitcell["species"], unitcell["selective_dynamics"], unitcell["flags"])
    mapping = mapping_elements(unitcell["elements"], supercell["atom_counts"], supercell["positions_cartesian"], supercell["positions_direct"],
                               supercell["species"], unitcell["selective_dynamics"], supercell["flags"])
    labels = define_labels(mapping["elements"], mapping["atom_counts"])
    write_POSCAR(argv[2], supercell["lattice_matrix"], mapping["elements"], mapping["atom_counts"], 
                 mapping["positions_cartesian"], mapping["positions_direct"], unitcell["selective_dynamics"],
                 mapping["flags"], labels)
    
    # Summary
    print(f"\nSupercell written to: {argv[2]}")
    print(f"Expansion matrix determinant: {replicas}")
    print("-" * 39)
    print("  Element  |  Unit cell  |  Supercell")
    print("-" * 39)
    for element, orig, new in zip(mapping["elements"], unitcell["atom_counts"], mapping["atom_counts"]):
        print(f"  {element:<9}|  {orig:<11}|  {new}")
    print("-" * 39)
    print(f"  Total    |  {unitcell['total_atoms']:<11}|  {supercell['total_atoms']}")
    print("-" * 39 + "\n")


if __name__ == "__main__":
    main()
