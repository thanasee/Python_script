#!/usr/bin/env python

from sys import argv, exit
import os
import numpy as np


def usage():
    """Print usage information and exit."""
    print("""
Usage: vaspFix.py <input> <output>

This script support VASP5 Structure file format (i.e. POSCAR) 
for fix atom in Structure file.

This script was developed by Thanasee Thanasarnsurapong.
""")
    exit(0)


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
                if name.isalpha():
                    break
                else:
                    print("The name of species must be alphabetic characters only.")
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

    return {"lattice_matrix":     lattice_matrix,
            "elements":           elements,
            "atom_counts":        atom_counts,
            "total_atoms":        total_atoms,
            "positions_cartesian": positions_cartesian,
            "positions_direct":   positions_direct,
            "species":            species,
            "selective_dynamics": selective_dynamics,
            "flags":              flags if selective_dynamics else None}


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


def write_POSCAR(filepath, lattice_matrix, elements, atom_counts,
                 positions_direct, selective_dynamics, flags, labels):
    """Write a VASP5-format POSCAR file with Direct coordinates.

    The scale factor is always written as 1.0 because the lattice vectors
    are already stored in absolute Å units. Atom labels are appended as
    inline comments after each position line for readability.

    Parameters
    ----------
    filepath           : str
    lattice_matrix     : np.ndarray (3, 3)  — lattice vectors in Å
    elements           : list[str]          — element symbols in canonical order
    atom_counts        : list[int]          — atoms per element
    positions_direct   : np.ndarray (N, 3)  — fractional coordinates
    selective_dynamics : bool
    flags              : np.ndarray or None  — per-atom T/F flags
    labels             : list[str]          — per-atom comment labels
    """

    with open(filepath, 'w') as o:
        o.write("Generated by vaspFix.py code\n")
        o.write(f"   {1.0:.14f}\n")
        for lattice in lattice_matrix:
            o.write(f"   {lattice[0]:20.16f}  {lattice[1]:20.16f}  {lattice[2]:20.16f}\n")
        o.write("   " + "    ".join(elements) + " \n")
        o.write("     " + "    ".join(map(str, atom_counts)) + "\n")
        if selective_dynamics:
            o.write("Selective dynamics\n")
        o.write("Direct\n")
        if selective_dynamics:
            for position, flag, label in zip(positions_direct, flags, labels):
                o.write(f"{position[0]:20.16f}{position[1]:20.16f}{position[2]:20.16f}"
                        f"   {flag[0]:s}   {flag[1]:s}   {flag[2]:s}"
                        f"   {label:>6s}\n")
        else:
            for position, label in zip(positions_direct, labels):
                o.write(f"{position[0]:20.16f}{position[1]:20.16f}{position[2]:20.16f}"
                        f"   {label:>6s}\n")


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


def parse_group(prompt, total_atoms, species, allow_all=True):
    """Interactively parse a free-format atom selection from the user.

    Accepts a mix of:
    - Individual atom indexes     : e.g. '1 3 5'
    - Ranges of atom indexes      : e.g. '1-4'  (inclusive, 1-based)
    - Element symbols             : e.g. 'Fe C'  (selects all atoms of that species)
    - Keyword 'all'               : selects all atoms (only if allow_all=True)

    Keeps prompting until a valid, non-empty selection within [1, total_atoms] is given.

    Parameters
    ----------
    prompt      : str, message printed before the input prompt
    total_atoms : int, total number of atoms in the system
    species     : list of str, element symbol for each atom (length N)
    allow_all   : bool, whether the keyword 'all' is permitted (default True)

    Returns
    -------
    group : list of int, 0-based atom indexes of the selected atoms
    """

    print(prompt)
    while True:
        group = []
        raw = input().split()
        valid = True
        for token in raw:
            if token == 'all':
                if not allow_all:
                    print("  Cannot use 'all' in this method. TRY AGAIN!")
                    valid = False; break
                group.extend(range(total_atoms))
            elif '-' in token:
                start, end = map(int, token.split('-'))
                group.extend(range(start - 1, end))
            elif token.isdigit():
                group.append(int(token) - 1)
            else:
                group.extend([j for j, lbl in enumerate(species) if lbl == token])
        if not valid:
            continue
        if group and all(0 <= idx < total_atoms for idx in group):
            return group
        print("  Wrong input atom-indexes! TRY AGAIN!")


def refix(total_atoms, selective_dynamics, flags):
    """Initialise Selective Dynamics flags and ask how to proceed when they already exist.
 
    If Selective Dynamics is not present in the input, all flags are set to
    'T T T' (fully free) and the script proceeds normally to the fix-mode prompt.
    If Selective Dynamics is already present, the user chooses one of three actions:
      Y — reset every flag to T T T, then re-apply fixes from scratch
      A — keep existing flags and layer additional fixes on top
      N — keep existing flags unchanged and skip straight to output
 
    Parameters
    ----------
    total_atoms        : int                       — total number of atoms
    selective_dynamics : bool                      — whether input has Selective Dynamics
    flags              : np.ndarray (N, 3) or None — existing T/F flags, or None
 
    Returns
    -------
    flags : np.ndarray (N, 3) — initialised or updated flags array
    skip  : bool              — True if the user chose N (bypass fix-mode selection)
    """
    
    # Ensure selective dynamics flags are initialized
    skip = False
    if not selective_dynamics:
        flags = np.full((total_atoms, 3), 'T')
    else:
        while True:
            print("""
Selective Dynamics already present in input
  Y) Reset all flags to T T T, then re-apply fixes
  A) Keep existing flags and add more fixes on top
  N) Keep existing flags as-is and skip to output""")
            option_refix = input("Choice (Y/A/N): ").strip().upper()
            if option_refix == 'Y':
                flags = np.full((total_atoms, 3), 'T')
                break
            elif option_refix == 'A':
                break  # keep flags, proceed to fix mode selection
            elif option_refix == 'N':
                skip = True  # skip selection, jump straight to output
                break
            else:
                print("ERROR! Enter Y, A, or N.")
    return flags, skip


def select_index(total_atoms, species):
    """Prompt the user to select atoms by element symbol and/or atom index.
 
    Accepts free-format input mixing element symbols (e.g. 'C'), single indexes
    (e.g. '3'), ranges (e.g. '1-4'), and the keyword 'all'. Repeats the prompt
    on invalid input. The selected_atoms list is reset on each retry to prevent
    duplicates accumulating across bad inputs.
 
    Parameters
    ----------
    total_atoms : int       — total number of atoms (used for bounds checking)
    species     : list[str] — per-atom element label (used for symbol selection)
 
    Returns
    -------
    selected_atoms : list[int] — 0-based indexes of the chosen atoms
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


def select_radius(lattice_matrix, total_atoms, positions_cartesian, species):
    """Select atoms that lie outside a cutoff radius from a reference point.
 
    The reference point is the centroid of a user-specified set of atoms.
    Distances are minimised over all 27 periodic image cells to correctly
    handle atoms near cell boundaries.
 
    Parameters
    ----------
    lattice_matrix      : np.ndarray (3, 3) — lattice vectors in Å (used for PBC)
    total_atoms         : int               — total number of atoms
    positions_cartesian : np.ndarray (N, 3) — Cartesian coordinates in Å
 
    Returns
    -------
    selected_atoms : list[int] — 0-based indexes of atoms outside the cutoff radius
    """
    
    targets = parse_group(f"\nChoose reference point\nInput element-symbol and/or atom-indexes to choose ({1:>3} to {total_atoms:>3})\n"
                          f"(Free-format input, e.g., 1 3 1-4 C H all)", total_atoms, species, allow_all=True)
 
    reference_point = np.mean(positions_cartesian[targets], axis=0)
 
    while True:
        try:
            input_radius = float(input("Input cutoff radius in Angstrom (select atoms outside cutoff radius): "))
            if input_radius > 0:
                break
        except ValueError:
            print("ERROR! Input cutoff radius must be a positive number.")
 
    selected_atoms = []
    image_offsets = compute_image_offsets(lattice_matrix)
    for index, position in enumerate(positions_cartesian):
        min_distance = min_image_distance(positions_cartesian[index], reference_point, image_offsets)
 
        if min_distance > input_radius:
            selected_atoms.append(index)
 
    return selected_atoms


def select_file(total_atoms):
    """Read Selective Dynamics flags from a SELECTED_FIX_ATOMS_LIST file.
 
    Tries the default filename 'SELECTED_FIX_ATOMS_LIST' in the current
    directory first. On failure or atom-count mismatch, the user is prompted
    for the correct path and the check is retried.
 
    Parameters
    ----------
    total_atoms : int — expected number of atoms (used to validate the file)
 
    Returns
    -------
    flags : np.ndarray (N, 3) — T/F flag strings read from the file
    """
    
    select_file = 'SELECTED_FIX_ATOMS_LIST'
    while True:
        try:
            with open(select_file, 'r') as s:
                file_lines = s.readlines()
        except FileNotFoundError:
            select_file = input("Enter your SELECTED_FIX_ATOMS_LIST path: ")
            continue
 
        flags = np.array([[x for x in line.split()[2:]] for line in file_lines[3:]])
        if len(flags) == total_atoms:
            return flags
        else:
            print("ERROR! The numbers of atom in SELECTED_FIX_ATOMS_LIST and POSCAR files not match!")
            select_file = input("Enter your SELECTED_FIX_ATOMS_LIST path: ")


def select_direction():
    """Prompt the user to choose which Cartesian directions to fix.
 
    Accepts free-format input: single indexes (e.g. '1'), ranges (e.g. '1-3'),
    and the keyword 'all'. Repeats the prompt on empty or out-of-range input.
 
    Returns
    -------
    fix_coordinates : list[int] — 0-based direction indexes (0=x, 1=y, 2=z)
    """
    print("""
Input the direction index to fix (1 to 3):
1) x direction
2) y direction
3) z direction
(Free-format input, e.g., 1 3 1-3 all)""")
    while True:
        fix_coordinates = []
        input_coordinates = input().split()
 
        for coordinate in input_coordinates:
            if 'all' in coordinate.lower():
                fix_coordinates = [0, 1, 2]
                break
            elif '-' in coordinate:
                try:
                    start, end = map(int, coordinate.split('-'))
                    fix_coordinates.extend(range(start - 1, end))
                except ValueError:
                    print("Invalid range format! Try again.")
            elif coordinate.isnumeric():
                fix_coordinates.append(int(coordinate) - 1)
            else:
                print("Invalid direction format! Try again.")
 
        if fix_coordinates and all(0 <= idx < 3 for idx in fix_coordinates):
            return fix_coordinates
        else:
            print("ERROR! Directions must be between 1 and 3. Try again.")
    

def fix_mode(lattice_matrix, total_atoms, positions_cartesian, species, flags):
    """Prompt the user to choose a fixing method and return the updated flags array.
 
    Three selection methods are available:
      1 — by atom index or element symbol  (calls select_index)
      2 — by cutoff radius from a centroid (calls select_radius)
      3 — read flags directly from file    (calls select_file, returns immediately)
 
    For modes 1 and 2, the user is also asked which Cartesian directions to fix
    via select_direction(), and the chosen atoms are set to 'F' accordingly.
 
    Parameters
    ----------
    lattice_matrix      : np.ndarray (3, 3) — lattice vectors in Å
    total_atoms         : int               — total number of atoms
    positions_cartesian : np.ndarray (N, 3) — Cartesian coordinates in Å
    species             : list[str]         — per-atom element label
    flags               : np.ndarray (N, 3) — current T/F flags to be modified
 
    Returns
    -------
    flags : np.ndarray (N, 3) — updated flags with selected atoms/directions set to 'F'
    """
    print("""
Choices of fixing atoms method
1) Atomic Indexes/Labels
2) Cutoff Radius From Reference Point
3) Read From SELECTED_ATOMS_LIST File""")
 
    while True:
        fix_mode = input("Enter choice: ")
 
        if fix_mode == '1':
            selected_atoms = select_index(total_atoms, species)
            break
        elif fix_mode == '2':
            selected_atoms = select_radius(lattice_matrix, total_atoms, positions_cartesian, species)
            break
        elif fix_mode == '3':
            flags = select_file(total_atoms)
            return flags
        else:
            print("ERROR!! Must choose fixing atoms method")
 
    fix_coordinates = select_direction()
 
    for atom in selected_atoms:
        for direction in fix_coordinates:
            flags[atom][direction] = 'F'
 
    return flags


def write_selected(flags, labels):
    """Write the SELECTED_FIX_ATOMS_LIST file summarising the current fix status.
 
    The file lists every atom's 1-based index, label, and T/F flags for x, y, z.
    It can be used as input to Mode 3 of fix_mode() in a future run.
 
    Parameters
    ----------
    flags  : np.ndarray (N, 3) — T/F flag strings per atom
    labels : list[str]         — per-atom labels (e.g. 'Mo01', 'S003')
    """
    select_file = "SELECTED_FIX_ATOMS_LIST"
    with open(select_file, 'w') as o:
        o.write("#                Fix Status\n")
        o.write("#Index   LABEL   x  y  z\n")
        o.write("#--------------------------\n")
        for i in range(len(flags)):
            o.write(f"{i + 1:5}   {labels[i]}    {flags[i, 0]}  {flags[i, 1]}  {flags[i, 2]}\n")


def main():
    """Parse arguments, fix selected atoms, and write outputs."""
    
    if '-h' in argv or '--help' in argv or len(argv) != 3:
        usage()
 
    poscar = read_POSCAR(argv[1])
    flags, skip = refix(poscar["total_atoms"], poscar["selective_dynamics"], poscar["flags"])
    if not skip:
        flags = fix_mode(poscar["lattice_matrix"], poscar["total_atoms"], poscar["positions_cartesian"],
                         poscar["species"], flags)
    mapping = mapping_elements(poscar["elements"], poscar["atom_counts"], poscar["positions_cartesian"],
                               poscar["positions_direct"], poscar["species"], True, flags)
    labels = define_labels(mapping["elements"], mapping["atom_counts"])
    write_POSCAR(argv[2], poscar["lattice_matrix"], mapping["elements"], mapping["atom_counts"],
                 mapping["positions_direct"], True, mapping["flags"], labels)
    write_selected(mapping["flags"], labels)
 
    
if __name__ == "__main__":
    main()
