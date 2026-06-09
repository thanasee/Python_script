#!/usr/bin/env python

from sys import argv, exit
import os
import numpy as np


def usage():
    """Print usage information and exit."""
    print("""
Usage: vaspShift.py <input> <output>

This script support VASP5 Structure file format (i.e. POSCAR) 
for shifting a structure file:
  molecule (0D)  ->  shift to center
  nanowire (1D)  ->  shift to origin in extend direction and center in other direction
  sheet (2D)     ->  shift to center in vacuum direction and center in other direction
  bulk (3D)      ->  shift to origin
  adsorbate      ->  shift to origin in XY

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
    

def get_direction(prompt):
    """Prompt the user to select a lattice direction (X, Y, or Z).

    Parameters
    ----------
    prompt : str — label describing the direction role (e.g. 'extend', 'vacuum')

    Returns
    -------
    idx : int — 0-based axis index (0=X, 1=Y, 2=Z)
    """
    
    print(f"""
Input the direction index of {prompt} direction (1 to 3):
1) x direction
2) y direction
3) z direction""")
    while True:
        try:
            idx = int(input()) - 1
            if 0 <= idx < 3:
                return idx
            print("ERROR! Directions must be between 1 and 3. Try again.")
        except ValueError:
            print("ERROR! Must enter a number. Try again.")


def get_adsorbent_atoms(total_atoms, species):
    """Prompt the user to select a set of adsorbate atoms by index, range, or element.

    Accepts free-format input combining single indices, ranges (e.g. 1-4),
    element symbols, and the keyword 'all'.

    Parameters
    ----------
    total_atoms : int       — total number of atoms in the system
    species     : list[str] — per-atom element labels

    Returns
    -------
    adsorbent_atoms : list[int] — 0-based indices of selected adsorbate atoms
    """
    
    print(f"""
Input element-symbol and/or atom-indexes of adsorbent ({1:>3} to {total_atoms:>3})
(Free-format input, e.g., 1 3 1-4 C H all)""")
    while True:
        adsorbent_atoms = []
        for adsorbent in input().split():
            if 'all' in adsorbent:
                adsorbent_atoms.extend(range(total_atoms))
                break
            if adsorbent.isnumeric() or '-' in adsorbent:
                if '-' in adsorbent:
                    start, end = map(int, adsorbent.split('-'))
                    adsorbent_atoms.extend(range(start - 1, end))
                else:
                    adsorbent_atoms.append(int(adsorbent) - 1)
            else:
                adsorbent_atoms.extend([i for i, label in enumerate(species) if label == adsorbent])
        if len(adsorbent_atoms) > total_atoms or not all(0 <= idx < total_atoms for idx in adsorbent_atoms):
            print("Wrong input atom-indexes !TRY AGAIN!")
        else:
            return adsorbent_atoms


def shift_molecule(positions_direct):
    """Shift all atoms so the structure centroid lands at the cell center (0.5, 0.5, 0.5).

    Intended for isolated molecules (0D systems) surrounded by vacuum on all sides.
    Atoms are first unwrapped across periodic boundaries to compute the correct
    centroid before shifting.

    Parameters
    ----------
    positions_direct : np.ndarray (N, 3) — fractional coordinates in [0, 1)

    Returns
    -------
    np.ndarray (N, 3) — shifted fractional coordinates in [0, 1)
    """
    
    _, unwrapped = unwrap(positions_direct)
    center = np.mean(unwrapped, axis=0)
    
    return (unwrapped - center + 0.5) % 1.0


def shift_wire(positions_direct):
    """Shift a nanowire so the extend direction starts at origin and the
    transverse directions are centered at 0.5.

    Intended for 1D periodic systems (nanowires) where the structure is
    periodic along one axis and surrounded by vacuum in the other two.
    The user is prompted to select the extend (periodic) direction.

    Parameters
    ----------
    positions_direct : np.ndarray (N, 3) — fractional coordinates in [0, 1)

    Returns
    -------
    np.ndarray (N, 3) — shifted fractional coordinates in [0, 1)
    """
    
    reference, unwrapped = unwrap(positions_direct)
    center = np.mean(unwrapped, axis=0)
    extend = get_direction("extend")
    periodic = [i for i in range(3) if i != extend]
    new = np.copy(unwrapped)
    new[:, extend]   = unwrapped[:, extend] - reference[extend]
    new[:, periodic] = unwrapped[:, periodic] - center[periodic] + 0.5
    
    return new % 1.0


def shift_sheet(positions_direct):
    """Shift a 2D sheet so the vacuum direction is centered at 0.5 and the
    periodic directions start at origin.

    Intended for 2D periodic systems (monolayers, slabs) where the structure
    is periodic in two directions and has vacuum in one direction.
    The user is prompted to select the vacuum direction.

    Parameters
    ----------
    positions_direct : np.ndarray (N, 3) — fractional coordinates in [0, 1)

    Returns
    -------
    np.ndarray (N, 3) — shifted fractional coordinates in [0, 1)
    """
    
    reference, unwrapped = unwrap(positions_direct)
    center = np.mean(unwrapped, axis=0)
    vacuum = get_direction("vacuum")
    periodic = [i for i in range(3) if i != vacuum]
    new = np.copy(unwrapped)
    new[:, periodic] = unwrapped[:, periodic] - reference[periodic]
    new[:, vacuum]   = unwrapped[:, vacuum] - center[vacuum] + 0.5
    
    return new % 1.0


def shift_bulk(positions_direct):
    """Shift all atoms so the selected atom lands at the cell origin (0, 0, 0)

    Intended for 3D periodic bulk systems where no vacuum is present.
    All atoms are shifted rigidly by the position of the selected reference atom.

    Parameters
    ----------
    positions_direct : np.ndarray (N, 3) — fractional coordinates in [0, 1)

    Returns
    -------
    np.ndarray (N, 3) — shifted fractional coordinates in [0, 1)
    """

    try:
        ref_index = int(input(f"Enter the atom index to shift to origin (1 to {len(positions_direct)}): ")) - 1
        if 0 <= ref_index < len(positions_direct):
            pass
        else:
            print("ERROR! Index out of range. Try again.")
    except ValueError:
        print("ERROR! Must enter a number. Try again.")

    ref = np.copy(positions_direct[ref_index])
    delta = positions_direct - ref
    delta -= np.round(delta)
    unwrapped = ref + delta
    return (unwrapped - ref) % 1.0


def shift_special(total_atoms, positions_direct, species):
    """Shift a selected adsorbate group so its centroid is centered in XY at (0.5, 0.5)
    while the Z coordinates of all atoms are left unchanged.

    Intended for adsorption systems (2D sheet + adsorbate) where the vacuum
    direction is always Z. The user selects the adsorbate atoms by index,
    range, element symbol, or the keyword 'all'. The entire system is then
    shifted rigidly in XY based on the adsorbate centroid only.

    Parameters
    ----------
    total_atoms      : int          — total number of atoms in the system
    positions_direct : np.ndarray (N, 3) — fractional coordinates in [0, 1)
    species          : list[str]    — per-atom element labels

    Returns
    -------
    np.ndarray (N, 3) — shifted fractional coordinates in [0, 1)
    """
    
    adsorbent_atoms = get_adsorbent_atoms(total_atoms, species)
    _, unwrapped = unwrap(positions_direct)
    # re-unwrap around adsorbate reference
    ref_ads = np.copy(positions_direct[adsorbent_atoms[0]])
    delta = positions_direct - ref_ads
    delta -= np.round(delta)
    unwrapped = ref_ads + delta
    center = np.mean(unwrapped[adsorbent_atoms], axis=0)
    new = np.copy(unwrapped)
    new[:, :2] = unwrapped[:, :2] - center[:2] + 0.5
    new[:, 2]  = unwrapped[:, 2]
    
    return new % 1.0


def shift(total_atoms, positions_direct, species):
    """Prompt the user to select a shifting mode and apply the corresponding shift.

    Presents a menu of five shifting strategies corresponding to different
    material dimensionalities and use cases. Dispatches to the appropriate
    shift function based on user input and returns the shifted positions.

    Modes
    -----
    0 — 0D molecule  : all atoms centered at (0.5, 0.5, 0.5)
    1 — 1D wire      : extend direction → origin, transverse → center
    2 — 2D sheet     : vacuum direction → center, periodic → origin
    3 — 3D bulk      : selected atom shifted to origin
    4 — Special      : adsorbate XY centered at 0.5, Z unchanged

    Parameters
    ----------
    total_atoms      : int               — total number of atoms
    positions_direct : np.ndarray (N, 3) — fractional coordinates in [0, 1)
    species          : list[str]         — per-atom element labels

    Returns
    -------
    np.ndarray (N, 3) — shifted fractional coordinates in [0, 1)
    """
    
    print("""
Choices of type of material
  0) 0D (molecule)   -> shift all atoms to center
  1) 1D (wire)       -> origin in extend direction, center in other
  2) 2D (sheet)      -> origin in periodic, center in vacuum
  3) 3D (bulk)       -> shift all atoms to origin
  4) Special!        -> adsorbate XY center, Z free""")

    dispatch = {
        '0': lambda: shift_molecule(positions_direct),
        '1': lambda: shift_wire(positions_direct),
        '2': lambda: shift_sheet(positions_direct),
        '3': lambda: shift_bulk(positions_direct),
        '4': lambda: shift_special(total_atoms, positions_direct, species),
    }

    while True:
        mode = input("Enter choice: ")
        if mode in dispatch:
            return dispatch[mode]()
        elif mode.isdigit():
            print("ERROR!! Must choose type of material")
        else:
            print("ERROR!! Choose again")


def main():
    """Parse arguments, shift atoms with selected method, and write output"""
    
    if '-h' in argv or '--help' in argv or len(argv) != 3:
        usage()
    
    unshift = read_POSCAR(argv[1])
    shift_positions_direct = shift(unshift["total_atoms"], unshift["positions_direct"],  unshift["species"])
    shift_positions_cartesian = direct_to_cartesian(unshift["lattice_matrix"], shift_positions_direct)
    mapping = mapping_elements(unshift["elements"], unshift["atom_counts"], shift_positions_cartesian, shift_positions_direct,
                               unshift["species"], unshift["selective_dynamics"], unshift["flags"])
    labels = define_labels(mapping["elements"], mapping["atom_counts"])
    write_POSCAR(argv[2], unshift["lattice_matrix"], mapping["elements"], mapping["atom_counts"],
                 mapping["positions_cartesian"], mapping["positions_direct"], unshift["selective_dynamics"],
                 mapping["flags"], labels)
    
    print("")


if __name__ == "__main__":
    main()
