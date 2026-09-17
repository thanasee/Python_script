#!/usr/bin/env python

from sys import argv, exit
import os
import readline
import numpy as np


def usage():
    """Print usage information and exit."""
    print("""
Usage: vaspVacuum.py <input> <output>

This script adds or rebuilds vacuum space in a structure file along a
lattice direction (a, b, or c).

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
        o.write("Generated by vaspVacuum.py\n")
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


def unwrap(positions_direct):
    """Reconstruct a contiguous cluster by unwrapping periodic boundary conditions.

    Shifts all atoms into the minimum-image frame relative to atom[0], so that
    atoms split across a cell boundary are treated as geometrically contiguous.
    Interatomic distances are preserved exactly.

    Parameters
    ----------
    positions_direct : np.ndarray (N, 3) — fractional coordinates in [0, 1)

    Returns
    -------
    reference : np.ndarray (3,)   — fractional coordinate of atom[0]
    unwrapped : np.ndarray (N, 3) — unwrapped fractional coordinates
    """
    
    reference = np.copy(positions_direct[0])
    delta = positions_direct - reference
    delta -= np.round(delta)
    
    return reference, reference + delta


def shift_sheet(positions_direct, vacuum):
    """Shift a 2D sheet so the vacuum direction is centered at 0.5 and the
    periodic directions start at origin.

    Adapted from vaspShift.py's shift_sheet(): takes the vacuum-direction
    axis index directly as a parameter instead of prompting for it (that
    version calls get_direction() internally), since the caller here
    already knows which axis is being modified.

    Parameters
    ----------
    positions_direct : np.ndarray (N, 3) — fractional coordinates
    vacuum            : int — 0-based axis index of the vacuum direction

    Returns
    -------
    np.ndarray (N, 3) — shifted fractional coordinates in [0, 1)
    """

    reference, unwrapped = unwrap(positions_direct)
    center = np.mean(unwrapped, axis=0)
    periodic = [i for i in range(3) if i != vacuum]
    new = np.copy(unwrapped)
    new[:, periodic] = unwrapped[:, periodic] - reference[periodic]
    new[:, vacuum]   = unwrapped[:, vacuum] - center[vacuum] + 0.5

    return new % 1.0


def get_direction(prompt):
    """Prompt the user to select a lattice direction (a, b, or c).

    Parameters
    ----------
    prompt : str — label describing the direction role (e.g. 'extend', 'vacuum')

    Returns
    -------
    idx : int — 0-based axis index (0=a, 1=b, 2=c)
    """
    
    print(f"""
Input the direction index of {prompt} direction (1 to 3):
1) a direction
2) b direction
3) c direction""")
    while True:
        try:
            idx = int(input()) - 1
            if 0 <= idx < 3:
                return idx
            print("ERROR! Directions must be between 1 and 3. Try again.")
        except ValueError:
            print("ERROR! Must enter a number. Try again.")


def get_vacuum(prompt):
    """Prompt for a non-negative vacuum thickness, in Å.

    Parameters
    ----------
    prompt : str — text shown to the user

    Returns
    -------
    float — the entered thickness
    """
    while True:
        try:
            vacuum = float(input(prompt).strip())
            if vacuum < 0:
                print("ERROR! Vacuum thickness must not be negative.")
                continue
            return vacuum
        except ValueError:
            print("ERROR! Enter a numeric value.")


def add_vacuum(lattice_matrix, positions_direct):
    """Grow the cell along a lattice vector, adding vacuum on top of any
    that already exists.

    The chosen vector's own length grows by the entered thickness; its
    direction is preserved and the other two vectors are untouched. Valid
    for any cell, orthogonal or not. The structure is shifted with
    shift_sheet() before and after the lattice change, so a slab split
    across the periodic boundary is resolved and the vacuum ends up
    centered in the new cell.

    Parameters
    ----------
    lattice_matrix : np.ndarray, shape (3, 3)
        Lattice vectors in Å.
    positions_direct : np.ndarray, shape (N, 3)
        Fractional coordinates.

    Returns
    -------
    dict with keys:
        direction        : str                        — direction letter modified ('a', 'b', or 'c')
        mode             : str                        — always "add"
        vacuum           : float                      — thickness added, in Å
        lattice_matrix   : np.ndarray, shape (3, 3)   — new lattice matrix
        positions_direct : np.ndarray, shape (N, 3)   — new fractional coordinates
    """
    axis = get_direction("vacuum")
    direction = "abc"[axis]
    vacuum = get_vacuum("Vacuum to add (\u00c5): ")
    unit_vector = lattice_matrix[axis] / np.linalg.norm(lattice_matrix[axis])
    new_length = np.linalg.norm(lattice_matrix[axis]) + vacuum

    shift_positions_direct = shift_sheet(positions_direct, axis)
    shift_positions_cartesian = direct_to_cartesian(lattice_matrix, shift_positions_direct)

    new_lattice_matrix = lattice_matrix.copy()
    new_lattice_matrix[axis] = unit_vector * new_length

    added_positions_direct = cartesian_to_direct(new_lattice_matrix, shift_positions_cartesian)
    new_positions_direct = shift_sheet(added_positions_direct, axis)

    return {"direction":         direction,
            "mode":              "add",
            "vacuum":            vacuum,
            "lattice_matrix":    new_lattice_matrix,
            "positions_direct":  new_positions_direct}


def rebuild_vacuum(lattice_matrix, positions_direct):
    """Reset the vacuum along a lattice vector to a specific total thickness,
    discarding whatever vacuum already exists.

    Shifts first with shift_sheet() so the slab's true thickness can be
    measured cleanly under the OLD lattice (resolving any split across the
    periodic boundary), then sizes the vector to slab thickness plus the
    entered total vacuum.

    Parameters
    ----------
    lattice_matrix : np.ndarray, shape (3, 3)
        Lattice vectors in Å.
    positions_direct : np.ndarray, shape (N, 3)
        Fractional coordinates.

    Returns
    -------
    dict with keys:
        direction        : str                        — direction letter modified ('a', 'b', or 'c')
        mode             : str                        — always "rebuild"
        vacuum           : float                      — total vacuum entered, in Å
        lattice_matrix   : np.ndarray, shape (3, 3)   — new lattice matrix
        positions_direct : np.ndarray, shape (N, 3)   — new fractional coordinates
    """
    axis = get_direction("vacuum")
    direction = "abc"[axis]
    vacuum = get_vacuum("Total vacuum (\u00c5): ")
    unit_vector = lattice_matrix[axis] / np.linalg.norm(lattice_matrix[axis])

    shift_positions_direct = shift_sheet(positions_direct, axis)
    shift_positions_cartesian = direct_to_cartesian(lattice_matrix, shift_positions_direct)
    rebuilt_length = shift_positions_cartesian @ unit_vector
    new_length = (rebuilt_length.max() - rebuilt_length.min()) + vacuum

    new_lattice_matrix = lattice_matrix.copy()
    new_lattice_matrix[axis] = unit_vector * new_length

    rebuilt_positions_direct = cartesian_to_direct(new_lattice_matrix, shift_positions_cartesian)
    new_positions_direct = shift_sheet(rebuilt_positions_direct, axis)

    return {"direction":         direction,
            "mode":              "rebuild",
            "vacuum":            vacuum,
            "lattice_matrix":    new_lattice_matrix,
            "positions_direct":  new_positions_direct}


def modify_vacuum(lattice_matrix, positions_direct):
    """Prompt the user to select a vacuum-modification mode and apply it.

    Modes
    -----
    1 — Add vacuum     : add_vacuum()
    2 — Rebuild vacuum : rebuild_vacuum()

    Parameters
    ----------
    lattice_matrix   : np.ndarray, shape (3, 3) — lattice vectors in Å
    positions_direct : np.ndarray, shape (N, 3) — fractional coordinates

    Returns
    -------
    dict with keys "direction", "mode", "vacuum", "lattice_matrix", "positions_direct"
    """

    print("""
Choices of vacuum modification
  1) Add vacuum
  2) Rebuild vacuum""")

    dispatch = {
        '1': lambda: add_vacuum(lattice_matrix, positions_direct),
        '2': lambda: rebuild_vacuum(lattice_matrix, positions_direct),
    }

    while True:
        mode = input("Enter choice: ")
        if mode in dispatch:
            return dispatch[mode]()
        elif mode.isdigit():
            print("ERROR!! Must choose type of modification")
        else:
            print("ERROR!! Choose again")


def main():
    """Parses command-line arguments and writes a POSCAR with vacuum added.
    """
    if '-h' in argv or '--help' in argv or len(argv) != 3:
        usage()

    unformat = read_POSCAR(argv[1])
    vacuumed = modify_vacuum(unformat["lattice_matrix"], unformat["positions_direct"])
    positions_cartesian = direct_to_cartesian(vacuumed["lattice_matrix"], vacuumed["positions_direct"])
    mapping = mapping_elements(unformat["elements"], unformat["atom_counts"], positions_cartesian,
                               vacuumed["positions_direct"], unformat["species"], unformat["selective_dynamics"],
                               unformat["flags"])
    labels = define_labels(mapping["elements"], mapping["atom_counts"])
    write_POSCAR(argv[2], vacuumed["lattice_matrix"], mapping["elements"], mapping["atom_counts"],
                 mapping["positions_cartesian"], mapping["positions_direct"], unformat["selective_dynamics"],
                 mapping["flags"], labels)

    verb = "Added" if vacuumed["mode"] == "add" else "Rebuilt"
    print(f"\n{verb} vacuum of {vacuumed['vacuum']:.4f} \u00c5 along the {vacuumed['direction']}-direction.\n")


if __name__ == "__main__":
    main()
