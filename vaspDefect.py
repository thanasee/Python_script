#!/usr/bin/env python

from sys import argv, exit
import os
import numpy as np


def usage():
    """Print usage information and exit."""
    print("""
Usage: vaspDefect.py <input> <output>

This script design to apply defects to the systems
which has 3 defect types available:
  1) vacancy
  2) substitution
  3) interstitial
This script support VASP5 position file format (i.e. POSCAR).

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

    return {"elements":            new_elements,
            "atom_counts":         new_atom_counts,
            "positions_cartesian": new_positions_cartesian,
            "positions_direct":    new_positions_direct,
            "species":             new_species,
            "flags":               new_flags if selective_dynamics else None}


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
        o.write("Generated by vaspDefect.py code\n")
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


def apply_vacancy(positions_cartesian, positions_direct, species, selective_dynamics, flags, selected_atoms):
    """Remove selected atoms to create vacancy defects.

    Parameters
    ----------
    positions_cartesian : np.ndarray (N, 3)  — Cartesian coordinates in Å
    positions_direct    : np.ndarray (N, 3)  — fractional coordinates
    species             : list[str]          — per-atom element labels
    selective_dynamics  : bool
    flags               : np.ndarray or None — per-atom T/F flags
    selected_atoms      : list[int]          — 0-based indexes of atoms to remove

    Returns
    -------
    dict with keys:
        positions_cartesian : np.ndarray
        positions_direct    : np.ndarray
        species             : list[str]
        flags               : np.ndarray or None
    """
    mask = [i for i in range(len(species)) if i not in selected_atoms]
    new_positions_cartesian = positions_cartesian[mask]
    new_positions_direct = positions_direct[mask]
    new_species = [species[i] for i in mask]
    new_flags = flags[mask] if selective_dynamics and flags is not None else None

    return {"positions_cartesian": new_positions_cartesian,
            "positions_direct":    new_positions_direct,
            "species":             new_species,
            "flags":               new_flags}


def apply_substitution(positions_cartesian, positions_direct, species, selective_dynamics, flags, selected_atoms):
    """Replace selected atoms with a new element and reorder the atom list.

    The substituted atom is placed after the last atom of the same species if the
    new element already exists, otherwise appended after the last existing species.

    Parameters
    ----------
    positions_cartesian : np.ndarray (N, 3)  — Cartesian coordinates in Å
    positions_direct    : np.ndarray (N, 3)  — fractional coordinates
    species             : list[str]          — per-atom element labels
    selective_dynamics  : bool
    flags               : np.ndarray or None — per-atom T/F flags
    selected_atoms      : list[int]          — 0-based indexes of atoms to substitute

    Returns
    -------
    dict with keys:
        positions_cartesian : np.ndarray
        positions_direct    : np.ndarray
        species             : list[str]
        flags               : np.ndarray or None
    """
    while True:
        new_element = input("Enter the new element symbol for substitution: ").strip()
        if new_element.isalpha():
            break
        print("ERROR! Element symbol must contain alphabetic characters only.")

    new_species = species.copy()
    for idx in selected_atoms:
        new_species[idx] = new_element

    # mapping_elements will re-group and reorder all species blocks
    return {"positions_cartesian": positions_cartesian.copy(),
            "positions_direct":    positions_direct.copy(),
            "species":             new_species,
            "flags":               flags.copy() if selective_dynamics and flags is not None else None}


def apply_interstitial(lattice_matrix, positions_cartesian, positions_direct,
                       species, selective_dynamics, flags):
    """Add one interstitial atom at a site defined by averaging selected atom positions
    or by manual fractional coordinate input.

    The interstitial site is the mean fractional coordinate of >= 2 selected atoms
    (minimum image convention applied per component before averaging) or a
    user-supplied direct coordinate. The new atom is inserted:
      - after the last atom of matching species, if the element already exists
      - after the last atom of all existing species, if the element is new

    Parameters
    ----------
    lattice_matrix      : np.ndarray (3, 3)  — lattice vectors in Å
    positions_cartesian : np.ndarray (N, 3)  — Cartesian coordinates in Å
    positions_direct    : np.ndarray (N, 3)  — fractional coordinates
    species             : list[str]          — per-atom element labels
    selective_dynamics  : bool
    flags               : np.ndarray or None — per-atom T/F flags

    Returns
    -------
    dict with keys:
        positions_cartesian : np.ndarray
        positions_direct    : np.ndarray
        species             : list[str]
        flags               : np.ndarray or None
    """
    total_atoms = len(species)
    print("""
Interstitial site definition:
1) Average position of selected atoms (select >= 2 atoms)
2) Manual fractional coordinate input""")

    while True:
        site_mode = input("Enter the site definition mode (1 or 2): ").strip()
        if site_mode in ('1', '2'):
            break
        print("Invalid input! Please enter 1 or 2.")

    if site_mode == '1':
        # Require at least 2 atoms to define a meaningful average
        while True:
            selected_atoms = select_index(total_atoms, species)
            if len(selected_atoms) >= 2:
                break
            print("ERROR! Select at least 2 atoms to define the interstitial site.")

        # Apply minimum image convention before averaging to handle atoms on
        # opposite sides of the cell (wrap relative to the first selected atom)
        ref = positions_direct[selected_atoms[0]]
        wrapped = []
        for idx in selected_atoms:
            delta = positions_direct[idx] - ref
            delta -= np.round(delta)        # minimum image in fractional coords
            wrapped.append(ref + delta)
        new_pos_direct = np.mean(wrapped, axis=0) % 1.0

    else:
        # Manual fractional coordinate input
        print("Enter the fractional coordinates of the interstitial site:")
        while True:
            try:
                coords = list(map(float, input("  a b c: ").split()))
                if len(coords) == 3:
                    new_pos_direct = np.array(coords) % 1.0
                    break
                print("ERROR! Enter exactly 3 values.")
            except ValueError:
                print("ERROR! Non-numeric input. Try again.")

    new_pos_cartesian = new_pos_direct @ lattice_matrix

    # Prompt for the interstitial element symbol
    while True:
        new_element = input("Enter the element symbol for the interstitial atom: ").strip()
        if new_element.isalpha():
            break
        print("ERROR! Element symbol must contain alphabetic characters only.")

    new_species = species.copy()
    new_pos_cart_list = list(positions_cartesian)
    new_pos_dir_list = list(positions_direct)
    new_flags_list = list(flags) if selective_dynamics and flags is not None else None

    # Default flag for new atom: all free (T T T)
    default_flag = np.array(['T', 'T', 'T'])

    if new_element in new_species:
        # Insert after the last occurrence of the matching species
        insert_idx = max(i for i, s in enumerate(new_species) if s == new_element) + 1
    else:
        # Append after all existing atoms
        insert_idx = total_atoms

    new_species.insert(insert_idx, new_element)
    new_pos_cart_list.insert(insert_idx, new_pos_cartesian)
    new_pos_dir_list.insert(insert_idx, new_pos_direct)
    if selective_dynamics and new_flags_list is not None:
        new_flags_list.insert(insert_idx, default_flag)

    new_positions_cartesian = np.array(new_pos_cart_list, dtype=float)
    new_positions_direct = np.array(new_pos_dir_list, dtype=float)
    new_flags = np.array(new_flags_list) if selective_dynamics and new_flags_list is not None else None

    return {"positions_cartesian": new_positions_cartesian,
            "positions_direct":    new_positions_direct,
            "species":             new_species,
            "flags":               new_flags}


def apply_defect(lattice_matrix, positions_cartesian, positions_direct,
                 species, selective_dynamics, flags):
    """Dispatch to the appropriate defect routine based on user selection.

    Parameters
    ----------
    lattice_matrix      : np.ndarray (3, 3)  — lattice vectors in Å
    positions_cartesian : np.ndarray (N, 3)  — Cartesian coordinates in Å
    positions_direct    : np.ndarray (N, 3)  — fractional coordinates
    species             : list[str]          — per-atom element labels
    selective_dynamics  : bool
    flags               : np.ndarray or None — per-atom T/F flags

    Returns
    -------
    dict with keys:
        positions_cartesian : np.ndarray
        positions_direct    : np.ndarray
        species             : list[str]
        flags               : np.ndarray or None
    """
    total_atoms = len(species)

    print("""
Choices of defect types:
1) Vacancy
2) Substitution
3) Interstitial""")

    while True:
        defect_type = input("Enter the defect type (1, 2, or 3): ").strip()
        if defect_type in ('1', '2', '3'):
            break
        print("Invalid defect type! Please enter 1, 2, or 3.")

    if defect_type == '1':
        print("\n--- Vacancy: select atom(s) to remove ---")
        selected_atoms = select_index(total_atoms, species)
        result = apply_vacancy(positions_cartesian, positions_direct,
                               species, selective_dynamics, flags, selected_atoms)

    elif defect_type == '2':
        print("\n--- Substitution: select atom(s) to replace ---")
        selected_atoms = select_index(total_atoms, species)
        result = apply_substitution(positions_cartesian, positions_direct,
                                    species, selective_dynamics, flags, selected_atoms)

    else:
        print("\n--- Interstitial: define insertion site ---")
        result = apply_interstitial(lattice_matrix, positions_cartesian, positions_direct,
                                    species, selective_dynamics, flags)

    return result


def main():
    """Parse command-line arguments, apply the chosen defect, and write output POSCAR."""
    if '-h' in argv or '--help' in argv or len(argv) != 3:
        usage()

    undoped = read_POSCAR(argv[1])

    # Reorder atoms into contiguous element blocks before applying the defect
    mapping = mapping_elements(
        undoped["elements"], undoped["atom_counts"],
        undoped["positions_cartesian"], undoped["positions_direct"],
        undoped["species"], undoped["selective_dynamics"], undoped["flags"])

    result = apply_defect(
        undoped["lattice_matrix"],
        mapping["positions_cartesian"], mapping["positions_direct"],
        mapping["species"], undoped["selective_dynamics"], mapping["flags"])

    # Re-map to canonical element blocks after the defect (handles new/removed species)
    final = mapping_elements(
        list(dict.fromkeys(result["species"])),
        [],                                    # atom_counts rebuilt inside mapping_elements
        result["positions_cartesian"], result["positions_direct"],
        result["species"], undoped["selective_dynamics"], result["flags"],
        sort_elements=list(dict.fromkeys(result["species"])))

    labels = define_labels(final["elements"], final["atom_counts"])

    write_POSCAR(argv[2], undoped["lattice_matrix"],
                 final["elements"], final["atom_counts"],
                 final["positions_direct"], undoped["selective_dynamics"],
                 final["flags"], labels)


if __name__ == "__main__":
    main()
