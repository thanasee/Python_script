#!/usr/bin/env python

from sys import argv, exit
import os
import readline
import numpy as np


def usage():
    """Print usage information and exit."""
    text = """
Usage:
  vaspTwist.py <POSCAR1> [POSCAR2]

This script supports VASP5 structure file format (i.e. POSCAR) for
generating moire structures from the input file(s), and also prepares
the stacking configurations for each candidate. Only monolayer files
with vacuum space in the z direction are supported.

If TWIST_LIST.dat exists and matches the given POSCAR(s), the saved
candidates are reused instead of re-searching.

This script was developed by Thanasee Thanasarnsurapong.
"""
    print(text)
    exit(0)

MAX_ATOMS  = 500    # Maximum number of atoms in the bilayer supercell
THETA_MIN  = 0.0    # Minimum twist angle (degrees) — heterostrain search range
THETA_MAX  = 180.0  # Maximum twist angle (degrees) — heterostrain search range
MAX_STRAIN = 0.05   # Maximum allowed strain (heterostrain prefilter + final filter)
TWIST_LIST_FILE = "TWIST_LIST.dat"
SNAPSHOT_HOMOBILAYER_FILE = "original.vasp"  # homobilayer: one snapshot, both layers
SNAPSHOT_BOTTOM_FILE      = "bottom.vasp"    # heterobilayer: bottom-layer snapshot
SNAPSHOT_TOP_FILE         = "top.vasp"       # heterobilayer: top-layer snapshot

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

VDW_RADIUS = {
    'H':  1.20, 'He': 1.43, 'Li': 2.12, 'Be': 1.98, 'B':  1.91, 'C':  1.77,
    'N':  1.66, 'O':  1.50, 'F':  1.46, 'Ne': 1.58, 'Na': 2.50, 'Mg': 2.51,
    'Al': 2.25, 'Si': 2.19, 'P':  1.90, 'S':  1.89, 'Cl': 1.82, 'Ar': 1.83,
    'K':  2.73, 'Ca': 2.62, 'Sc': 2.58, 'Ti': 2.46, 'V':  2.42, 'Cr': 2.45,
    'Mn': 2.45, 'Fe': 2.44, 'Co': 2.40, 'Ni': 2.40, 'Cu': 2.38, 'Zn': 2.39,
    'Ga': 2.32, 'Ge': 2.29, 'As': 1.88, 'Se': 1.82, 'Br': 1.86, 'Kr': 1.95,
    'Rb': 3.21, 'Sr': 2.84, 'Y':  2.75, 'Zr': 2.52, 'Nb': 2.56, 'Mo': 2.45,
    'Tc': 2.44, 'Ru': 2.46, 'Rh': 2.44, 'Pd': 2.15, 'Ag': 2.53, 'Cd': 2.49,
    'In': 2.43, 'Sn': 2.42, 'Sb': 2.47, 'Te': 1.99, 'I':  2.04, 'Xe': 2.06,
    'Cs': 3.48, 'Ba': 3.03, 'La': 2.98, 'Ce': 2.88, 'Pr': 2.92, 'Nd': 2.95,
    'Pm': 2.90, 'Sm': 2.87, 'Eu': 2.83, 'Gd': 2.79, 'Tb': 2.87, 'Dy': 2.81,
    'Ho': 2.76, 'Er': 2.75, 'Tm': 2.73, 'Yb': 2.76, 'Lu': 2.68, 'Hf': 2.63,
    'Ta': 2.53, 'W':  2.57, 'Re': 2.49, 'Os': 2.48, 'Ir': 2.41, 'Pt': 2.29,
    'Au': 2.32, 'Hg': 2.45, 'Tl': 2.47, 'Pb': 2.60, 'Bi': 2.54, 'Po': 2.80,
    'At': 2.93, 'Rn': 2.02,
}

def compute_interlayer_gap(bottom_species, bottom_positions_cartesian,
                           top_species, top_positions_cartesian):
    """Compute a physically-motivated interlayer gap from van der Waals radii.

    Sums the van der Waals radii of the topmost atom (highest z) in the
    bottom layer and the bottommost atom (lowest z) in the top layer — the
    pair that will directly face each other once stacked. Inputs should be
    the original unit-cell arrays, not a tiled supercell, since in-plane
    tiling never changes z.

    Parameters
    ----------
    bottom_species             : list[str]
    bottom_positions_cartesian : np.ndarray (N, 3)
    top_species                : list[str]
    top_positions_cartesian    : np.ndarray (M, 3)

    Returns
    -------
    interlayer_gap : float -- interlayer gap in Angstrom
    """

    bottom_top_index = np.argmax(bottom_positions_cartesian[:, 2])
    top_bottom_index = np.argmin(top_positions_cartesian[:, 2])

    bottom_element = bottom_species[bottom_top_index]
    top_element    = top_species[top_bottom_index]

    bottom_radius = VDW_RADIUS.get(bottom_element)
    top_radius    = VDW_RADIUS.get(top_element)

    if bottom_radius is None or top_radius is None:
        missing = bottom_element if bottom_radius is None else top_element
        interlayer_gap = 3.5
        print(f"WARNING! No van der Waals radius known for '{missing}'. "
              f"Falling back to default interlayer gap of {interlayer_gap} Angstrom.")
        return interlayer_gap

    interlayer_gap = bottom_radius + top_radius
    return interlayer_gap


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

    elements = []
    is_number = lines[5].split()[0].isdecimal()
    if is_number:
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
        raw_elements = lines[5].split()
        for name in raw_elements:
            elements.append(name.split('/')[0].split('_')[0])
        atom_counts = [int(x) for x in lines[6].split()]
        selective_dynamics = lines[7].lower().startswith('s')
        position_start = 9 if selective_dynamics else 8

    total_atoms = sum(atom_counts)
    position_stop = position_start + total_atoms

    positions = np.array([[float(x) for x in lines[i].split()[:3]]
                          for i in range(position_start, position_stop)])

    species = [x for i, x in enumerate(elements)
               for _ in range(atom_counts[i])]

    flags = None
    if selective_dynamics:
        flags = np.array([[x for x in lines[i].split()[3:6]]
                          for i in range(position_start, position_stop)])

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
    new_species = list(species).copy()
    new_flags = flags.copy() if (selective_dynamics and flags is not None) else None

    # Group positions and flags by element symbol using the per-atom species list.
    # This is correct even when positions are in layer order (not element order),
    # because we look up each atom's element from species[i] rather than assuming
    # the positions are already contiguous by element.
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

    if sort_elements is None:
        sort_elements = check_elements(elements)

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
            if selective_dynamics and elements_flags is not None:
                sort_flags.extend(elements_flags[element])
            sort_atom_counts.append(len(elements_positions_direct[element]))

        new_positions_cartesian = np.array(sort_positions_cartesian, dtype=float)
        new_positions_direct = np.array(sort_positions_direct, dtype=float)
        new_species = list(sort_species)
        if selective_dynamics and sort_flags is not None:
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
    labels = [f"{symbol}{str(counter).zfill(digits)}" for symbol, number in zip(elements, atom_counts)
              for counter in range(1, number + 1)]

    return labels


def write_POSCAR(filepath, lattice_matrix, elements, atom_counts, positions_cartesian,
                 positions_direct, selective_dynamics, flags, labels, direct=True):
    """Write a VASP5-format POSCAR file, in either Direct or Cartesian coordinates.

    The scale factor is always written as 1.0 because the lattice vectors
    are already stored in absolute Å units. Atom labels are appended as
    inline comments after each position line for readability.

    Parameters
    ----------
    filepath            : str
    lattice_matrix      : np.ndarray (3, 3)  — lattice vectors in Å
    elements            : list[str]          — element symbols in canonical order
    atom_counts         : list[int]          — atoms per element
    positions_cartesian : np.ndarray (N, 3)  — Cartesian coordinates in Å
    positions_direct    : np.ndarray (N, 3)  — fractional coordinates
    selective_dynamics  : bool
    flags               : np.ndarray or None  — per-atom T/F flags
    labels              : list[str]          — per-atom comment labels
    direct              : bool               — True for Direct coordinates, False for Cartesian
    """

    with open(filepath, 'w') as o:
        o.write("Generated by vaspTwist.py\n")
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


def center_sheet(positions_direct):
    """Shift a 2D sheet so the vacuum direction is centered at z = 0.5.

    The vacuum axis is assumed to be z (index 2).

    Parameters
    ----------
    positions_direct : np.ndarray (N, 3) — fractional coordinates in [0, 1)

    Returns
    -------
    np.ndarray (N, 3) — shifted fractional coordinates in [0, 1)
    """

    reference, unwrapped = unwrap(positions_direct)
    center = np.mean(unwrapped, axis=0)
    vacuum = 2
    periodic = [i for i in range(3) if i != vacuum]
    new = np.copy(unwrapped)
    new[:, periodic] = unwrapped[:, periodic] - reference[periodic]
    new[:, vacuum]   = unwrapped[:, vacuum] - center[vacuum] + 0.5

    return new % 1.0


def rotation_matrix(degree):
    """Construct a rotation matrix for rotation about the z-axis.

    Uses the Rodrigues rotation formula:
        R = cos θ · I + sin θ · (u×) + (1 − cos θ) · u⊗u
    where u = [0, 0, 1].

    Parameters
    ----------
    degree : float

    Returns
    -------
    rotate : np.ndarray (3, 3) — rotation matrix
    """

    radian = np.radians(degree)
    sin = np.sin(radian)
    cos = np.cos(radian)
    u = np.array([0., 0., 1.])

    rotate = cos * np.eye(3) + sin * np.cross(np.eye(3), u) + (1 - cos) * np.outer(u, u)

    return rotate


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

    new_flags = None
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


def symmetric_common_vector(v_bottom, v_top):
    """Construct a length-preserving bisector between two supercell vectors.

    Points along the normalized sum of v_bottom and v_top (their angle
    bisector), scaled to the average of their two lengths:
    normalize(v_bottom + v_top) * (|v_bottom| + |v_top|) / 2. For two
    equal-length vectors (homobilayer, zero strain), this is exactly
    "rotate either vector by half the angle between them" — a true
    periodicity vector. A naive average instead shrinks by cos(theta/2),
    giving a vector that isn't a real lattice vector of either layer.

    Parameters
    ----------
    v_bottom, v_top : np.ndarray (3,)

    Returns
    -------
    np.ndarray (3,)
    """

    v_sum = v_bottom + v_top
    norm_sum = np.linalg.norm(v_sum)
    if norm_sum < 1e-10:
        # Antiparallel/degenerate case (180 deg apart) — no well-defined
        # bisector direction; fall back to the raw average.
        return (v_bottom + v_top) / 2.0

    target_length = (np.linalg.norm(v_bottom) + np.linalg.norm(v_top)) / 2.0
    return v_sum / norm_sum * target_length


def build_twisted_bilayer(layer1, layer2_rotated, selective_dynamics,
                          matched_bottom_lattice_matrix, matched_top_lattice_matrix, interlayer_gap):
    """Stack two supercell layers into a twisted bilayer with a given interlayer gap.

    Re-expresses each layer's atoms into a shared common lattice (see
    symmetric_common_vector) so neither layer is privileged, then stacks
    them along z with the given interlayer gap.

    For a homobilayer (zero strain), each layer's matched lattice and the
    common lattice are related by a pure rotation, so this is just a rigid
    rotation of each layer and the cartesian_to_direct wrap below is exact.
    For a heterobilayer (genuine strain), the two layers' matched lattices
    have different SHAPES — there's no lattice either is exactly periodic
    with. To avoid that mismatch corrupting positions on wrap (an atom far
    from the origin landing in the wrong place once mod 1.0 is applied),
    each layer's Cartesian positions are first affinely transformed by the
    exact in-plane map that takes its own matched lattice onto the common
    lattice, making both layers exactly periodic with the common lattice
    before any wrapping happens.

    Parameters
    ----------
    layer1             : dict — bottom layer from build_supercell
    layer2_rotated     : dict — top layer from build_supercell
    selective_dynamics : bool
    matched_bottom_lattice_matrix : np.ndarray (3, 3) — bottom matched supercell vectors
    matched_top_lattice_matrix    : np.ndarray (3, 3) — top matched supercell vectors
    interlayer_gap                 : float — gap between the two layers in Å (see compute_interlayer_gap)

    Returns
    -------
    dict with keys: lattice_matrix, positions_direct, species, flags
    """

    # Common length-preserving bisector lattice — see docstring for why a
    # raw average is unsafe.
    A1_common = symmetric_common_vector(matched_bottom_lattice_matrix[0], matched_top_lattice_matrix[0])
    A2_common = symmetric_common_vector(matched_bottom_lattice_matrix[1], matched_top_lattice_matrix[1])
    a3_common = layer1["lattice_matrix"][2].copy()
    common_lattice = np.array([A1_common, A2_common, a3_common])

    # Affine in-plane transform mapping each layer's own matched lattice
    # exactly onto the common lattice: new_cartesian = old_cartesian @ T,
    # where T = inv(matched_in_plane) @ common_in_plane. Applying this to
    # every atom makes the layer exactly periodic with common_lattice, so
    # the cartesian_to_direct wrap below loses no information regardless of
    # how strained the original match was.
    bottom_in_plane = matched_bottom_lattice_matrix[:2, :2]
    top_in_plane    = matched_top_lattice_matrix[:2, :2]
    common_in_plane = common_lattice[:2, :2]
    T_bottom = np.linalg.inv(bottom_in_plane) @ common_in_plane
    T_top    = np.linalg.inv(top_in_plane)    @ common_in_plane

    # Convert each layer's atoms from their own supercell to Cartesian
    layer1_cartesian = direct_to_cartesian(layer1["lattice_matrix"], layer1["positions_direct"])
    layer2_cartesian = direct_to_cartesian(layer2_rotated["lattice_matrix"], layer2_rotated["positions_direct"])

    # Apply the affine strain transform (in-plane only; z untouched)
    layer1_cartesian[:, :2] = layer1_cartesian[:, :2] @ T_bottom
    layer2_cartesian[:, :2] = layer2_cartesian[:, :2] @ T_top

    # Stack with interlayer gap
    z_max_layer1 = np.max(layer1_cartesian[:, 2])
    z_min_layer2 = np.min(layer2_cartesian[:, 2])
    layer2_cartesian[:, 2] += z_max_layer1 - z_min_layer2 + interlayer_gap

    combined_cartesian = np.vstack((layer1_cartesian, layer2_cartesian))
    combined_species = list(layer1["species"]) + list(layer2_rotated["species"])

    combined_flags = None
    if selective_dynamics and layer1["flags"] is not None and layer2_rotated["flags"] is not None:
        combined_flags = np.vstack((layer1["flags"], layer2_rotated["flags"]))

    # Express all atoms in the common lattice and center along z-axis
    combined_positions_direct = cartesian_to_direct(common_lattice, combined_cartesian)
    combined_positions_direct = center_sheet(combined_positions_direct)

    return {"lattice_matrix":   common_lattice,
            "positions_direct": combined_positions_direct,
            "species":          combined_species,
            "flags":            combined_flags if selective_dynamics else None}


def collect_elements_and_counts(species, known_elements):
    """Collect unique elements present in species and their counts.

    Returns elements in the order they first appear in known_elements,
    restricted to those actually present in species.

    Parameters
    ----------
    species        : list[str]
    known_elements : list[str]

    Returns
    -------
    elements_order : list[str]
    atom_counts    : list[int]
    """

    elements_order = list(dict.fromkeys(e for e in known_elements if e in species))
    atom_counts    = [species.count(e) for e in elements_order]

    return elements_order, atom_counts


def metric_tensor(e1, e2, e3):
    """Compute the 3×3 metric tensor G where G[i,j] = ei · ej.

    Parameters
    ----------
    e1, e2, e3 : np.ndarray (3,)

    Returns
    -------
    G : np.ndarray (3, 3)
    """

    vecs = [e1, e2, e3]
    G = np.zeros((3, 3))
    for i in range(3):
        for j in range(3):
            G[i, j] = vecs[i] @ vecs[j]

    return G


def calculate_strain(matched_bottom_lattice_matrix, matched_top_lattice_matrix):
    """Compute the Lagrangian finite strain between the bottom and top supercell vectors.

    Measures the deformation needed to make the two supercells commensurate
    (zero for a homobilayer at exact CSL angles). Builds 3×3 metric tensors
    G = [ei·ej] for both vector sets, Cholesky-decomposes G = RᵀR, then
    F = R_top·R_bottom⁻¹ − I and ε = ½(F + Fᵀ + FᵀF); returns √(Σλᵢ²/3) over
    ε's eigenvalues. The z-component is set to 1.0 to avoid singularity
    (CellMatch convention for 2D systems).

    Parameters
    ----------
    matched_bottom_lattice_matrix : np.ndarray (3, 3) — bottom supercell vectors in Å
    matched_top_lattice_matrix    : np.ndarray (3, 3) — top supercell vectors in Å

    Returns
    -------
    deformation : float — scalar Lagrangian strain measure
    """

    V1 = np.array([matched_bottom_lattice_matrix[0, 0], matched_bottom_lattice_matrix[0, 1], 0.0])
    V2 = np.array([matched_bottom_lattice_matrix[1, 0], matched_bottom_lattice_matrix[1, 1], 0.0])
    V3 = np.array([0.0,          0.0,          1.0])
    U1 = np.array([matched_top_lattice_matrix[0, 0],    matched_top_lattice_matrix[0, 1],    0.0])
    U2 = np.array([matched_top_lattice_matrix[1, 0],    matched_top_lattice_matrix[1, 1],    0.0])
    U3 = np.array([0.0,          0.0,          1.0])

    G_bottom = metric_tensor(V1, V2, V3)
    G_top    = metric_tensor(U1, U2, U3)

    R_bottom = np.linalg.cholesky(G_bottom).T
    R_top    = np.linalg.cholesky(G_top).T

    F = (R_top @ np.linalg.inv(R_bottom)) - np.eye(3)
    lagrangian = 0.5 * (F + F.T + F.T @ F)

    eigenvalues = np.linalg.eigvalsh(lagrangian)
    deformation = np.sqrt(np.sum(eigenvalues ** 2) / 3.0)

    return float(deformation)


def canonicalize_cell(A1, A2):
    """Build a canonical key for a 2D supercell that is invariant to row order
    and vector sign, so geometrically equivalent cells are treated as duplicates.

    Tries all combinations of sign and row order, rounds to 6 decimal places,
    and returns the lexicographically smallest flattened tuple as the key.

    Parameters
    ----------
    A1, A2 : np.ndarray (3,) — supercell lattice vectors (only xy used)

    Returns
    -------
    tuple — canonical hashable key
    """

    v1 = A1[:2]
    v2 = A2[:2]

    candidates = []
    for s1 in (-1.0, 1.0):
        for s2 in (-1.0, 1.0):
            candidates.append(tuple(np.round(np.concatenate([s1 * v1, s2 * v2]), 6)))
            candidates.append(tuple(np.round(np.concatenate([s2 * v2, s1 * v1]), 6)))

    return min(candidates)


def is_homobilayer(bottom, top, tol=1e-3):
    """Detect whether the bottom and top layers form a true homobilayer.

    Checks in-plane lattice vectors and elemental composition (as a sorted
    multiset, order-independent). tol defaults to 1e-3 Å rather than machine
    precision, since two POSCARs of "the same" relaxed structure from
    separate VASP runs can differ at that level depending on EDIFFG.

    Parameters
    ----------
    bottom : dict — from read_POSCAR
    top    : dict — from read_POSCAR
    tol    : float — absolute tolerance for in-plane lattice vector comparison (Å)

    Returns
    -------
    bool
    """

    lattice_match = np.allclose(bottom["lattice_matrix"][:2, :2],
                                top["lattice_matrix"][:2, :2], atol=tol)

    composition_match = (sorted(zip(bottom["elements"], bottom["atom_counts"])) ==
                         sorted(zip(top["elements"],    top["atom_counts"])))

    return lattice_match and composition_match


def poscar_structure_matches(poscar_a, poscar_b, tol=1e-6):
    """Check whether two parsed POSCARs describe the same in-plane structure.

    Compares in-plane (x, y) lattice vectors and composition exactly, and
    fractional positions by nearest same-species neighbor (periodic
    wrap-around, order-independent). The z/vacuum direction is ignored, so
    differing vacuum padding doesn't count as a difference.

    Used to detect whether a provided POSCAR matches a stored snapshot
    (see write_poscar_snapshot) — more robust than comparing file paths,
    since a renamed/copied file with identical content should match, and
    a same-path file edited in place should not.

    Parameters
    ----------
    poscar_a, poscar_b : dict — from read_POSCAR
    tol                  : float — absolute tolerance in Angstrom (default 1e-6)

    Returns
    -------
    bool
    """

    lattice_2d_a = poscar_a["lattice_matrix"][:2, :2]
    lattice_2d_b = poscar_b["lattice_matrix"][:2, :2]
    if not np.allclose(lattice_2d_a, lattice_2d_b, atol=tol):
        return False

    composition_a = sorted(zip(poscar_a["elements"], poscar_a["atom_counts"]))
    composition_b = sorted(zip(poscar_b["elements"], poscar_b["atom_counts"]))
    if composition_a != composition_b:
        return False

    if poscar_a["total_atoms"] != poscar_b["total_atoms"]:
        return False

    pos_a     = poscar_a["positions_direct"][:, :2]
    pos_b     = poscar_b["positions_direct"][:, :2]
    species_a = poscar_a["species"]
    species_b = poscar_b["species"]

    used = [False] * len(pos_b)
    for i in range(len(pos_a)):
        found = False
        for j in range(len(pos_b)):
            if used[j]:
                continue
            if species_a[i] != species_b[j]:
                continue
            delta = pos_a[i] - pos_b[j]
            delta -= np.round(delta)
            cart_delta = delta @ lattice_2d_a
            if np.linalg.norm(cart_delta) <= tol:
                used[j] = True
                found = True
                break
        if not found:
            return False

    return True


def write_poscar_snapshot(filepath, poscar):
    """Write a structural snapshot to disk at full precision.

    This snapshot is the ground-truth reference used by poscar_structure_matches
    on later invocations — it freezes the structure as it was at search time,
    so a later edit to the original input file (same path, different content)
    is correctly detected as a mismatch rather than silently accepted.

    No quantization step is applied before writing: write_POSCAR already
    formats every coordinate to 16 decimal places, which is far more
    precision than any physically meaningful structural comparison needs,
    so an explicit rounding pass would only throw away information for no
    benefit.

    Parameters
    ----------
    filepath : str
    poscar   : dict — from read_POSCAR
    """

    labels = define_labels(poscar["elements"], poscar["atom_counts"])

    write_POSCAR(filepath, poscar["lattice_matrix"], poscar["elements"], poscar["atom_counts"],
                poscar["positions_cartesian"], poscar["positions_direct"],
                poscar["selective_dynamics"], poscar["flags"], labels)


def csl_angle(m, n, lattice_type):
    """Exact CSL twist angle and cell multiplicity for a homobilayer.

        hexagonal : cos theta = (m^2+4mn+n^2) / (2(m^2+mn+n^2)),  N = m^2+mn+n^2
        square    : cos theta = (m^2-n^2) / (m^2+n^2),            N = m^2+n^2

    Parameters
    ----------
    m, n         : int
    lattice_type : str — 'hexagonal' or 'square' (rectangular/oblique unsupported)

    Returns
    -------
    theta_deg : float or None
    N_cell    : int or None
    """

    if lattice_type == "hexagonal":
        num, den, N = m*m + 4*m*n + n*n, 2*(m*m + m*n + n*n), m*m + m*n + n*n
    elif lattice_type == "square":
        num, den, N = m*m - n*n, m*m + n*n, m*m + n*n
    else:
        return None, None

    if den == 0:
        return None, None
    return float(np.degrees(np.arccos(np.clip(num / den, -1.0, 1.0)))), N


def csl_supercell_matrix(m, n, lattice_type):
    """Integer matrix M such that supercell vectors = M @ primitive vectors.

    Parameters
    ----------
    m, n         : int
    lattice_type : str — 'hexagonal' or 'square'

    Returns
    -------
    np.ndarray (2, 2) or None
    """

    if lattice_type == "hexagonal":
        return np.array([[m, -n], [n, m + n]])
    if lattice_type == "square":
        return np.array([[m, -n], [n, m]])
    return None


def run_search_csl(bottom, top, sort_elements, known_elements, m_max=None):
    """Exact CSL search for a homobilayer — closed-form, no angle grid, no vector search.

    Strain is zero by construction (calculate_strain is a sanity check, not
    a filter). indices1/indices2 store the CSL matrix M's four entries so
    read_twist_list can reconstruct vectors with the same formula shared
    across all search methods.

    Parameters
    ----------
    bottom, top    : dict — from read_POSCAR
    sort_elements  : list or None
    known_elements : list[str]
    m_max          : int or None — upper bound for m; auto-set from MAX_ATOMS if None

    Returns
    -------
    candidates : list[dict] — same schema as run_search_heterostrain
    """

    lattice_type = get_2d_lattice_type(bottom["lattice_matrix"])
    if lattice_type not in ("hexagonal", "square"):
        print(f"WARNING! CSL needs hexagonal or square lattice; detected '{lattice_type}'. "
              f"Falling back to heterostrain-CSL search.")
        return run_search_heterostrain(bottom, top, sort_elements, known_elements)

    n_atoms = bottom["total_atoms"]
    if m_max is None:
        m_max = max(2, int(np.ceil(np.sqrt(MAX_ATOMS / (2 * n_atoms)))))
    print(f"Searching CSL angles ({lattice_type}, m up to {m_max})...")

    a1, a2, a3 = bottom["lattice_matrix"]
    candidates, seen = [], set()

    for m in range(1, m_max + 1):
        for n in range(1, m):              # m > n > 0 — excludes the degenerate
                                            # n=0 case (theta=0 for square,
                                            # theta=60 for hexagonal at every m)
            theta_deg, N_cell = csl_angle(m, n, lattice_type)
            if theta_deg is None:
                continue

            total_atoms_bilayer = 2 * n_atoms * N_cell
            if total_atoms_bilayer > MAX_ATOMS:
                continue

            M = csl_supercell_matrix(m, n, lattice_type)
            A1_bottom = np.append(M[0, 0] * a1[:2] + M[0, 1] * a2[:2], 0.0)
            A2_bottom = np.append(M[1, 0] * a1[:2] + M[1, 1] * a2[:2], 0.0)
            matched_bottom_lattice_matrix = np.array([A1_bottom, A2_bottom, a3])

            config_key = canonicalize_cell(matched_bottom_lattice_matrix[0], matched_bottom_lattice_matrix[1])
            if config_key in seen:
                continue
            seen.add(config_key)

            rot_2d = rotation_matrix(theta_deg)[:2, :2]
            A1_top = np.append(rot_2d @ A1_bottom[:2], 0.0)
            A2_top = np.append(rot_2d @ A2_bottom[:2], 0.0)
            matched_top_lattice_matrix = np.array([A1_top, A2_top, a3])

            strain = calculate_strain(matched_bottom_lattice_matrix, matched_top_lattice_matrix)  # sanity check, ~0

            matrix_indices = (int(M[0, 0]), int(M[0, 1]), int(M[1, 0]), int(M[1, 1]))

            A1_common = symmetric_common_vector(matched_bottom_lattice_matrix[0], matched_top_lattice_matrix[0])
            A2_common = symmetric_common_vector(matched_bottom_lattice_matrix[1], matched_top_lattice_matrix[1])

            candidates.append({"theta": theta_deg,
                               "matched_bottom_lattice_matrix": matched_bottom_lattice_matrix,
                               "matched_top_lattice_matrix":    matched_top_lattice_matrix,
                               "strain": strain,
                               "replicas_bottom": N_cell, "replicas_top": N_cell,
                               "indices1": matrix_indices, "indices2": matrix_indices,
                               "lattice_matrix": np.array([A1_common, A2_common, a3]),
                               "total_atoms": total_atoms_bilayer})

    if len(candidates) == 0:
        print(f"No CSL candidates found with <= {MAX_ATOMS} atoms. Try increasing m_max or MAX_ATOMS.")
        exit(0)

    candidates.sort(key=lambda c: c["total_atoms"])
    return candidates


def find_heterostrain_pairs(primitive_bottom_lattice_matrix, primitive_top_lattice_matrix, n_max, max_strain,
                            theta_min=THETA_MIN, theta_max=THETA_MAX):
    """Find candidate single-vector matches for heterostrain-CSL construction.

    For every pair of integer-indexed lattice vectors (bottom vs. unrotated
    top), computes the rotation angle and uniaxial stretch needed to align
    them in closed form (theta = angle(v_bottom) - angle(v_top_raw), stretch
    = |v_bottom|/|v_top_raw|), vectorized over all index combinations at
    once. Keeps pairs where |stretch - 1| <= max_strain.

    Parameters
    ----------
    primitive_bottom_lattice_matrix : np.ndarray (3, 3)
    primitive_top_lattice_matrix    : np.ndarray (3, 3)
    n_max                  : int   — integer index search range, -n_max..n_max
    max_strain              : float — maximum allowed |stretch - 1|
    theta_min, theta_max    : float — keep only angles within this range (deg)

    Returns
    -------
    pairs : list of tuples (theta, n1, n2, m1, m2, stretch, v_bottom, v_top_raw)
    """

    combined_list = np.arange(-n_max, n_max + 1)
    n1_grid, n2_grid = np.meshgrid(combined_list, combined_list, indexing="ij")
    n1_flat, n2_flat = n1_grid.ravel(), n2_grid.ravel()

    a1, a2 = primitive_bottom_lattice_matrix[0, :2], primitive_bottom_lattice_matrix[1, :2]
    b1, b2 = primitive_top_lattice_matrix[0, :2],    primitive_top_lattice_matrix[1, :2]

    v_bottom_all = n1_flat[:, None] * a1[None, :] + n2_flat[:, None] * a2[None, :]
    v_top_all    = n1_flat[:, None] * b1[None, :] + n2_flat[:, None] * b2[None, :]

    keep_bottom = np.linalg.norm(v_bottom_all, axis=1) > 1e-10
    keep_top    = np.linalg.norm(v_top_all,    axis=1) > 1e-10

    idx_all = np.column_stack([n1_flat, n2_flat])
    v_bottom_all, idx_bottom = v_bottom_all[keep_bottom], idx_all[keep_bottom]
    v_top_all,    idx_top    = v_top_all[keep_top],       idx_all[keep_top]

    norm_bottom  = np.linalg.norm(v_bottom_all, axis=1)
    norm_top     = np.linalg.norm(v_top_all,    axis=1)
    angle_bottom = np.arctan2(v_bottom_all[:, 1], v_bottom_all[:, 0])
    angle_top    = np.arctan2(v_top_all[:, 1],    v_top_all[:, 0])

    stretch = norm_bottom[:, None] / norm_top[None, :]
    theta   = np.degrees(angle_bottom[:, None] - angle_top[None, :]) % 360.0

    mask = ((np.abs(stretch - 1.0) <= max_strain) &
            (theta >= theta_min) & (theta <= theta_max))

    pairs = []
    for r, c in zip(*np.nonzero(mask)):
        n1, n2 = idx_bottom[r]
        m1, m2 = idx_top[c]
        pairs.append((float(theta[r, c]), int(n1), int(n2), int(m1), int(m2),
                     float(stretch[r, c]), v_bottom_all[r], v_top_all[c]))

    return pairs


def polar_decompose_2x2(D):
    """Decompose a 2x2 deformation matrix into rotation and strain components.

    D = R @ S (R orthogonal, S symmetric positive-definite), via
    S = sqrt(DᵀD) then R = D @ S⁻¹. For a heterobilayer match where D maps
    unrotated top vectors onto bottom vectors, R gives the exact twist
    angle reconciling both.

    Parameters
    ----------
    D : np.ndarray, shape (2, 2)

    Returns
    -------
    theta_deg : float or None — rotation angle in degrees, or None if D
                                  is singular or requires a reflection
    strain    : np.ndarray (2, 2) or None — symmetric strain tensor (S - I)
    """

    if abs(np.linalg.det(D)) < 1e-12:
        return None, None

    DtD = D.T @ D
    eigvals, eigvecs = np.linalg.eigh(DtD)
    if np.any(eigvals <= 0):
        return None, None

    sqrt_DtD = eigvecs @ np.diag(np.sqrt(eigvals)) @ eigvecs.T
    R = D @ np.linalg.inv(sqrt_DtD)

    if np.linalg.det(R) < 0:
        return None, None  # reflection component — not a physical twist

    theta_deg = float(np.degrees(np.arctan2(R[1, 0], R[0, 0])) % 360.0)
    strain = sqrt_DtD - np.eye(2)

    return theta_deg, strain


def find_heterostrain_candidates(pairs, primitive_bottom_lattice_matrix, bottom_species,
                                 primitive_top_lattice_matrix, top_species, max_strain):
    """Combine single-vector heterostrain matches into full 2D supercell candidates.

    For a chosen pair of bottom vectors and the corresponding unrotated top
    vectors, polar decomposition finds the single rotation that fits both
    vectors at once (see polar_decompose_2x2). Candidates are filtered on
    non-collinearity, a valid proper rotation, area ratios >= 1, atom count,
    a hard strain cutoff (max_strain), and deduplicated via canonicalize_cell.

    Parameters
    ----------
    pairs                            : list — output of find_heterostrain_pairs
    primitive_bottom_lattice_matrix  : np.ndarray (3, 3)
    bottom_species                   : list[str]
    primitive_top_lattice_matrix     : np.ndarray (3, 3)
    top_species                      : list[str]
    max_strain                       : float

    Returns
    -------
    candidates : list[dict]
    """

    a1, a2 = primitive_bottom_lattice_matrix[0], primitive_bottom_lattice_matrix[1]
    b1, b2 = primitive_top_lattice_matrix[0],    primitive_top_lattice_matrix[1]
    a3 = primitive_bottom_lattice_matrix[2]

    prim_cross_bottom = a1[0] * a2[1] - a1[1] * a2[0]
    prim_cross_top    = b1[0] * b2[1] - b1[1] * b2[0]

    candidates = []
    seen_configurations = set()

    for i in range(len(pairs)):
        _, n1_i, n2_i, m1_i, m2_i, _, v_bottom_i, v_top_raw_i = pairs[i]
        for j in range(i, len(pairs)):
            _, n1_j, n2_j, m1_j, m2_j, _, v_bottom_j, v_top_raw_j = pairs[j]

            V_matrix     = np.array([v_bottom_i,  v_bottom_j])
            U_raw_matrix = np.array([v_top_raw_i, v_top_raw_j])

            sup_cross_bottom = V_matrix[0, 0] * V_matrix[1, 1] - V_matrix[0, 1] * V_matrix[1, 0]
            if abs(sup_cross_bottom) < 1e-10 or abs(prim_cross_bottom) < 1e-10:
                continue
            sup_cross_top = U_raw_matrix[0, 0] * U_raw_matrix[1, 1] - U_raw_matrix[0, 1] * U_raw_matrix[1, 0]
            if abs(sup_cross_top) < 1e-10 or abs(prim_cross_top) < 1e-10:
                continue

            try:
                D = V_matrix @ np.linalg.inv(U_raw_matrix)
            except np.linalg.LinAlgError:
                continue

            theta_deg, _ = polar_decompose_2x2(D)
            if theta_deg is None:
                continue

            omjer1 = round(abs(sup_cross_bottom / prim_cross_bottom))
            omjer2 = round(abs(sup_cross_top    / prim_cross_top))
            if omjer1 < 1 or omjer2 < 1:
                continue

            total_atoms_bilayer = len(bottom_species) * omjer1 + len(top_species) * omjer2
            if total_atoms_bilayer > MAX_ATOMS:
                continue

            rot_2d    = rotation_matrix(theta_deg)[:2, :2]
            A1_top_2d = rot_2d @ U_raw_matrix[0]
            A2_top_2d = rot_2d @ U_raw_matrix[1]

            matched_bottom_lattice_matrix = np.array([np.append(V_matrix[0], 0.0),
                                                       np.append(V_matrix[1], 0.0), a3])
            matched_top_lattice_matrix    = np.array([np.append(A1_top_2d,    0.0),
                                                       np.append(A2_top_2d,    0.0), a3])

            lagrangian_strain = calculate_strain(matched_bottom_lattice_matrix, matched_top_lattice_matrix)
            if lagrangian_strain > max_strain:
                continue

            config_key = canonicalize_cell(matched_bottom_lattice_matrix[0], matched_bottom_lattice_matrix[1])
            if config_key in seen_configurations:
                continue
            seen_configurations.add(config_key)

            A1_common = symmetric_common_vector(matched_bottom_lattice_matrix[0], matched_top_lattice_matrix[0])
            A2_common = symmetric_common_vector(matched_bottom_lattice_matrix[1], matched_top_lattice_matrix[1])

            candidates.append({"theta":                  theta_deg,
                               "matched_bottom_lattice_matrix": matched_bottom_lattice_matrix,
                               "matched_top_lattice_matrix":    matched_top_lattice_matrix,
                               "strain":                 lagrangian_strain,
                               "replicas_bottom":      omjer1,
                               "replicas_top":         omjer2,
                               "indices1":               (n1_i, n2_i, n1_j, n2_j),
                               "indices2":               (m1_i, m2_i, m1_j, m2_j),
                               "lattice_matrix":         np.array([A1_common, A2_common, a3]),
                               "total_atoms":            total_atoms_bilayer})

    return candidates


def run_search_heterostrain(bottom, top, sort_elements, known_elements):
    """Run the heterostrain-CSL search and candidate-building pipeline.

    Replaces a theta-grid scan with closed-form vector matching, so angular
    resolution is not limited by a fixed step size.

    Parameters
    ----------
    bottom, top   : dict — from read_POSCAR
    sort_elements : list or None
    known_elements : list[str]

    Returns
    -------
    candidates : list[dict] — sorted by strain
    """

    primitive_atoms = bottom["total_atoms"] + top["total_atoms"]
    n_max = max(1, int(np.ceil(np.sqrt(MAX_ATOMS / primitive_atoms))))
    print(f"Auto-detected N_MAX = {n_max}"
          f"  (MAX_ATOMS={MAX_ATOMS} / {primitive_atoms} primitive atoms, ceil(sqrt) → {n_max})")

    print("Searching heterostrain-CSL vector matches (closed-form, no angle grid)...")
    pairs = find_heterostrain_pairs(bottom["lattice_matrix"], top["lattice_matrix"],
                                    n_max, MAX_STRAIN, THETA_MIN, THETA_MAX)

    if len(pairs) == 0:
        print("No commensurate heterostrain matches found with the given parameters.")
        exit(0)

    print(f"Found {len(pairs)} raw vector match(es). Filtering candidates (<= {MAX_ATOMS} atoms)...\n")

    candidates = find_heterostrain_candidates(pairs, bottom["lattice_matrix"], bottom["species"],
                                              top["lattice_matrix"], top["species"], MAX_STRAIN)

    if len(candidates) == 0:
        print(f"No candidates found with <= {MAX_ATOMS} atoms. Try relaxing MAX_STRAIN or MAX_ATOMS.")
        exit(0)

    candidates.sort(key=lambda c: c["strain"])
    return candidates


def build_bilayer_for_candidate(candidate, primitive_bottom_lattice_matrix, bottom_positions_cartesian,
                                bottom_species, bottom_selective_dynamics, bottom_flags,
                                primitive_top_lattice_matrix, top_positions_cartesian,
                                top_species, top_selective_dynamics, top_flags,
                                sort_elements, known_elements):
    """Build full atomic positions for a single candidate (generate mode only).

    Runs build_supercell and build_twisted_bilayer to produce the complete
    bilayer structure.  Updates the candidate dict in-place with the position
    data needed by generate_poscars.

    Parameters
    ----------
    candidate                       : dict — from a run_search_* function
    primitive_bottom_lattice_matrix : np.ndarray (3, 3)
    bottom_positions_cartesian      : np.ndarray (N, 3)
    bottom_species                  : list[str]
    bottom_selective_dynamics       : bool
    bottom_flags                    : np.ndarray or None
    primitive_top_lattice_matrix    : np.ndarray (3, 3)
    top_positions_cartesian         : np.ndarray (M, 3)
    top_species                     : list[str]
    top_selective_dynamics          : bool
    top_flags                       : np.ndarray or None
    sort_elements                   : list[str] or None
    known_elements                  : list[str]

    Returns
    -------
    candidate : dict — updated in-place with keys:
        n_bottom, positions_direct, bilayer_species,
        bilayer_flags, elements_order, selective_dynamics
    """

    theta_key = candidate["theta"]
    matched_bottom_lattice_matrix = candidate["matched_bottom_lattice_matrix"]
    matched_top_lattice_matrix    = candidate["matched_top_lattice_matrix"]

    # indices1/indices2 already store the exact integer expansion matrices
    # (bottom, top respectively) — used directly, no need to recover them
    # by inverting the Cartesian A1/A2 vectors. Embedded into 3x3 matrices
    # with [0,0,1] in the third row/column since z is untouched.
    n1, n2, n1p, n2p = candidate["indices1"]
    m1, m2, m1p, m2p = candidate["indices2"]
    expansion_matrix_bottom = np.array([[n1, n2, 0], [n1p, n2p, 0], [0, 0, 1]])
    expansion_matrix_top    = np.array([[m1, m2, 0], [m1p, m2p, 0], [0, 0, 1]])

    # replicas_bottom/replicas_top are already exactly det(expansion_matrix)
    # for each layer (omjer1/omjer2 from heterostrain, or N_cell from CSL) —
    # reused directly rather than recomputed.
    replicas_bottom = candidate["replicas_bottom"]
    replicas_top    = candidate["replicas_top"]

    bottom_unique_elements = list(dict.fromkeys(bottom_species))
    bottom_atom_counts = [bottom_species.count(e) for e in bottom_unique_elements]
    bottom_total_atoms = len(bottom_species)

    top_unique_elements = list(dict.fromkeys(top_species))
    top_atom_counts = [top_species.count(e) for e in top_unique_elements]
    top_total_atoms = len(top_species)

    rotated_top_lattice = primitive_top_lattice_matrix @ rotation_matrix(theta_key).T
    selective_dynamics  = bottom_selective_dynamics or top_selective_dynamics

    layer1 = build_supercell(expansion_matrix_bottom, replicas_bottom, primitive_bottom_lattice_matrix,
                             bottom_atom_counts, bottom_total_atoms, bottom_positions_cartesian,
                             bottom_species, bottom_selective_dynamics, bottom_flags)

    # Rotation applied to both lattice and positions — same rotated frame.
    # Top layer is tiled using its own matched matrix expansion_matrix_top,
    # relative to the rotated top lattice, consistent with the matched cell.
    rot = rotation_matrix(theta_key)
    top_positions_rotated = top_positions_cartesian @ rot.T
    layer2_rotated = build_supercell(expansion_matrix_top, replicas_top, rotated_top_lattice,
                                     top_atom_counts, top_total_atoms, top_positions_rotated,
                                     top_species, top_selective_dynamics, top_flags)

    bilayer_interlayer_gap = compute_interlayer_gap(bottom_species, bottom_positions_cartesian,
                                                     top_species, top_positions_cartesian)

    bilayer = build_twisted_bilayer(layer1, layer2_rotated, selective_dynamics,
                                    matched_bottom_lattice_matrix, matched_top_lattice_matrix,
                                    bilayer_interlayer_gap)

    order_ref = sort_elements if sort_elements is not None else list(dict.fromkeys(known_elements))
    elements_order, _ = collect_elements_and_counts(bilayer["species"], order_ref)

    candidate["n_bottom"]          = len(layer1["species"])
    candidate["positions_direct"]   = bilayer["positions_direct"]
    candidate["bilayer_species"]    = bilayer["species"]
    candidate["bilayer_flags"]      = bilayer["flags"]
    candidate["elements_order"]     = elements_order
    candidate["selective_dynamics"] = selective_dynamics
    return candidate


def get_2d_lattice_type(lattice_matrix):
    """Classify the 2D Bravais lattice type from the in-plane lattice vectors.

    Compares the in-plane lattice lengths (a, b) and the angle γ between them
    to identify the crystal system according to the standard 2D classification:

        hexagonal   : γ = 60° or 120°  and  a = b
        square      : γ = 90°           and  a = b
        rectangular : γ = 90°           and  a ≠ b
        oblique     : all other cases

    Parameters
    ----------
    lattice_matrix : np.ndarray, shape (3, 3) — row vectors of the lattice in Å

    Returns
    -------
    str : one of 'hexagonal', 'square', 'rectangular', 'oblique'
    """

    length_a = np.linalg.norm(lattice_matrix[0])
    length_b = np.linalg.norm(lattice_matrix[1])
    gamma = np.degrees(np.arccos(np.clip((lattice_matrix[0] @ lattice_matrix[1]) /
                                         (length_a * length_b), -1.0, 1.0)))

    if np.abs(gamma - 90.0) < 1e-5:
        return 'square' if np.abs(length_a - length_b) < 1e-8 else 'rectangular'
    elif ((np.abs(gamma - 60.0) < 1e-5 or np.abs(gamma - 120.0) < 1e-5)
          and np.abs(length_a - length_b) < 1e-8):
        return 'hexagonal'
    else:
        return 'oblique'


def get_shift_grid(lattice_type):
    """Return the high-symmetry stacking shift points for the given 2D lattice type.

    The shifts correspond to the irreducible high-symmetry points of the
    stacking space for each 2D Bravais lattice:

        hexagonal   : AA (0,0),  AB (1/3,2/3),  AB' (2/3,1/3)
        square      : AA (0,0),  AB (1/2,0),    AC  (1/2,1/2)
        rectangular : AA (0,0),  AB (1/2,0),    AB' (0,1/2),  AC (1/2,1/2)
        oblique     : AA (0,0),  AB (1/2,0),    AB' (0,1/2),  AC (1/2,1/2)

    Parameters
    ----------
    lattice_type : str — one of 'hexagonal', 'square', 'rectangular', 'oblique'

    Returns
    -------
    list of (float, float, str) — (shift_a, shift_b, label)
    """

    if lattice_type == "hexagonal":
        return [(0.0,       0.0,       "AA"),
                (1.0 / 3.0, 2.0 / 3.0, "AB"),
                (2.0 / 3.0, 1.0 / 3.0, "AB_prime")]

    elif lattice_type == "square":
        return [(0.0, 0.0, "AA"),
                (0.5, 0.0, "AB"),
                (0.5, 0.5, "AC")]

    else:
        # rectangular or oblique
        return [(0.0, 0.0, "AA"),
                (0.5, 0.0, "AB_x"),
                (0.0, 0.5, "AB_y"),
                (0.5, 0.5, "AC")]


def write_twist_list(filepath, candidates, bottom_file, top_file, homobilayer):
    """Write all search candidates to TWIST_LIST.dat, sorted by strain.

    Format mirrors CellMatch's results.dat with an added Theta column.
    The header records the input filenames and the detected homobilayer/
    heterobilayer type, so a later run can reload everything from the file
    alone without re-deriving the type from the POSCARs.

    Parameters
    ----------
    filepath     : str
    candidates   : list[dict]  — sorted by strain (ascending)
    bottom_file  : str
    top_file     : str
    homobilayer  : bool — result of is_homobilayer(bottom, top) for this search
    """

    SEP = "-" * 106

    with open(filepath, 'w') as o:
        o.write("# TWIST_LIST generated by vaspTwist.py\n")
        o.write(f"# bottom = {bottom_file}\n")
        o.write(f"# top    = {top_file}\n")
        o.write(f"# type   = {'homobilayer' if homobilayer else 'heterobilayer'}\n")
        o.write(f"# Total candidates: {len(candidates)}\n")
        o.write(f"# {'---':->49} RESULTS {'---':->49}\n")
        o.write(f"# {SEP}\n")
        o.write(f"# {'|':1}{'index':^7}{'|':1}{'Angle':^15}{'|':1}{'strain':^18}{'|':1}"
                f"{'atoms':^9}{'|':1}{'surf_ratio':^12}{'|':1}"
                f"{'indices1':^23}{'|':1}{'indices2':^23}{'|':1}\n")
        o.write(f"# {SEP}\n")
        for no, c in enumerate(candidates, start=1):
            i11, i12, i21, i22 = c["indices1"]
            j11, j12, j21, j22 = c["indices2"]
            r1, r2 = c["replicas_bottom"], c["replicas_top"]
            o.write(f"| {no:>6}  |  {c['theta']:>15.6f}  |  {c['strain']:>14.8f}  |"
                    f"  {c['total_atoms']:>6}   |"
                    f"  {r1:>4}  {r2:>4}  |"
                    f"  {i11:>4} {i12:>4} {i21:>4} {i22:>4}  |"
                    f"  {j11:>4} {j12:>4} {j21:>4} {j22:>4}  |\n")
        o.write(f"# {SEP}\n")


def read_twist_list_header(filepath):
    """Read only the header of TWIST_LIST.dat to extract input filenames and type.

    The '# type' line is backward-compatible: older TWIST_LIST.dat files
    written before this field existed will simply return homobilayer=None,
    and the caller should fall back to re-deriving it via is_homobilayer().

    Parameters
    ----------
    filepath : str

    Returns
    -------
    bottom_file : str
    top_file    : str
    homobilayer : bool or None — None if the file predates the '# type' field
    """

    bottom_file = None
    top_file    = None
    homobilayer = None
    with open(filepath, 'r') as f:
        for line in f:
            if line.startswith("# bottom ="):
                bottom_file = line.split("=", 1)[1].strip()
            elif line.startswith("# top    ="):
                top_file = line.split("=", 1)[1].strip()
            elif line.startswith("# type   ="):
                homobilayer = (line.split("=", 1)[1].strip().lower() == "homobilayer")
            if bottom_file is not None and top_file is not None and homobilayer is not None:
                break
    return bottom_file, top_file, homobilayer


def display_candidates(candidates):
    """Print a numbered table of candidates."""

    print(f"\nFound {len(candidates)} candidate(s) with <= {MAX_ATOMS} atoms:\n")
    print(f"  {'No.':>4}  {'Angle':>9}  {'Atoms':>5}  {'Ratio1':>6}  {'Ratio2':>6}"
          f"  {'Strain':>14}"
          f"  {'|A1| (Ang)':>10}  {'|A2| (Ang)':>10}  {'Lattice':>12}  {'Stackings':>9}"
          f"  {'indices1':>20}  {'indices2':>20}")
    print("  " + "-" * 147)
    for index, c in enumerate(candidates):
        norm_A1      = np.linalg.norm(c["matched_bottom_lattice_matrix"][0])
        norm_A2      = np.linalg.norm(c["matched_bottom_lattice_matrix"][1])
        lattice_type = get_2d_lattice_type(c["lattice_matrix"])
        n_stackings  = len(get_shift_grid(lattice_type))
        i11, i12, i21, i22 = c["indices1"]
        j11, j12, j21, j22 = c["indices2"]
        print(f"  {index + 1:>4}  {c['theta']:>9.1f}  {c['total_atoms']:>5}"
              f"  {c['replicas_bottom']:>6}  {c['replicas_top']:>6}"
              f"  {c['strain']:>14.8f}"
              f"  {norm_A1:>10.4f}  {norm_A2:>10.4f}  {lattice_type:>12}  {n_stackings:>9}"
              f"  {i11:>4} {i12:>4} {i21:>4} {i22:>4}"
              f"  {j11:>4} {j12:>4} {j21:>4} {j22:>4}")


def prompt_selection(candidates):
    """Prompt the user to select candidates to generate.

    Parameters
    ----------
    candidates : list[dict]

    Returns
    -------
    chosen : list[int] — zero-based indices of selected candidates
    """

    total_poscars = sum(
        len(get_shift_grid(get_2d_lattice_type(c["lattice_matrix"])))
        for c in candidates
    )
    print(f"\n  Selecting 'all' will write {total_poscars} POSCAR(s) total.")
    print( "  Enter 'none' to finish without generating any POSCAR.\n")

    while True:
        raw = input("Enter candidate numbers to generate (e.g. 1 3 5), 'all', or 'none': ").strip()
        if raw.lower() == 'none':
            return []
        if raw.lower() == 'all':
            return list(range(len(candidates)))
        try:
            chosen = [int(x) - 1 for x in raw.split()]
            if all(0 <= i < len(candidates) for i in chosen):
                return chosen
            print(f"ERROR! Numbers must be between 1 and {len(candidates)}.")
        except ValueError:
            print("ERROR! Enter integers separated by spaces, 'all', or 'none'.")


def read_twist_list(filepath, primitive_bottom_lattice_matrix, primitive_top_lattice_matrix):
    """Parse TWIST_LIST.dat and reconstruct candidate dicts for generate mode.

    Data lines start with '|' (not '#').  The matched supercell lattice
    matrices are reconstructed from the stored indices and the POSCAR
    lattice matrices. This single reconstruction formula is shared by CSL,
    heterostrain, and any future search method, since all of them store
    indices1/indices2 as the integer coefficients of a linear combination
    of the bottom/top primitive lattice vectors.

    Parameters
    ----------
    filepath                        : str
    primitive_bottom_lattice_matrix : np.ndarray (3, 3)
    primitive_top_lattice_matrix    : np.ndarray (3, 3)

    Returns
    -------
    candidates : list[dict]  — in file order (sorted by strain)
    """

    a1 = primitive_bottom_lattice_matrix[0]
    a2 = primitive_bottom_lattice_matrix[1]
    a3 = primitive_bottom_lattice_matrix[2]

    candidates = []
    with open(filepath, 'r') as f:
        for line in f:
            line = line.strip()
            if not line.startswith("|"):
                continue
            # Strip leading/trailing '|' and split on '|'
            parts = [p.strip() for p in line.strip("|").split("|")]
            # parts: [index, theta, strain, atoms, surf_ratio, indices1, indices2]
            if len(parts) != 7:
                continue
            try:
                theta       = float(parts[1])
                strain      = float(parts[2])
                total_atoms = int(parts[3])
                r1, r2      = [int(x) for x in parts[4].split()]
                n1, n2, n1p, n2p = [int(x) for x in parts[5].split()]
                m1, m2, m1p, m2p = [int(x) for x in parts[6].split()]
            except ValueError:
                continue

            # Reconstruct supercell vectors from indices + POSCAR lattice
            rot             = rotation_matrix(theta)
            rotated_top_lat = primitive_top_lattice_matrix @ rot.T

            A1_bottom = n1  * a1 + n2  * a2
            A2_bottom = n1p * a1 + n2p * a2
            A1_top    = m1  * rotated_top_lat[0] + m2  * rotated_top_lat[1]
            A2_top    = m1p * rotated_top_lat[0] + m2p * rotated_top_lat[1]
            matched_bottom_lattice_matrix = np.array([A1_bottom, A2_bottom, a3])
            matched_top_lattice_matrix    = np.array([A1_top, A2_top, a3])

            A1_common = symmetric_common_vector(A1_bottom, A1_top)
            A2_common = symmetric_common_vector(A2_bottom, A2_top)

            candidates.append({"theta":                  theta,
                               "matched_bottom_lattice_matrix": matched_bottom_lattice_matrix,
                               "matched_top_lattice_matrix":    matched_top_lattice_matrix,
                               "strain":                 strain,
                               "replicas_bottom":      r1,
                               "replicas_top":         r2,
                               "indices1":               (n1, n2, n1p, n2p),
                               "indices2":               (m1, m2, m1p, m2p),
                               "lattice_matrix":         np.array([A1_common, A2_common, a3]),
                               "total_atoms":            total_atoms})

    return candidates


def run_search_dispatch(bottom, top, sort_elements, known_elements):
    """Detect homobilayer vs heterobilayer and dispatch to the matching search.

    Parameters
    ----------
    bottom, top   : dict — from read_POSCAR
    sort_elements : list or None
    known_elements : list[str]

    Returns
    -------
    candidates : list[dict]
    """

    if is_homobilayer(bottom, top):
        return run_search_csl(bottom, top, sort_elements, known_elements)
    else:
        return run_search_heterostrain(bottom, top, sort_elements, known_elements)


def generate_POSCARs(chosen_indices, candidates, sort_elements, known_elements, working_dir,
                     bottom, top):
    """Build and write POSCARs for the chosen candidates.

    Calls build_bilayer_for_candidate for each selected candidate to construct
    atomic positions on demand — no positions are built until the user selects.

    Parameters
    ----------
    chosen_indices : list[int]  — zero-based indices into candidates
    candidates     : list[dict]
    sort_elements  : list or None
    known_elements : list[str]
    working_dir    : str
    bottom, top    : dict — from read_POSCAR

    Returns
    -------
    output_records : list[dict]
    """

    output_records = []
    written_count  = 0
    p_twist = len(str(len(chosen_indices)))

    for list_no, index in enumerate(chosen_indices, start=1):
        candidate = candidates[index]

        # Build atomic positions now — deferred until user selection
        build_bilayer_for_candidate(
            candidate,
            bottom["lattice_matrix"], bottom["positions_cartesian"],
            bottom["species"], bottom["selective_dynamics"], bottom["flags"],
            top["lattice_matrix"], top["positions_cartesian"],
            top["species"], top["selective_dynamics"], top["flags"],
            sort_elements, known_elements
        )

        theta              = candidate["theta"]
        bilayer_lattice_matrix = candidate["lattice_matrix"]
        total_atoms        = candidate["total_atoms"]
        ratio1             = candidate["replicas_bottom"]
        ratio2             = candidate["replicas_top"]
        strain             = candidate["strain"] * 100.0
        n_bottom           = candidate["n_bottom"]
        selective_dynamics = candidate["selective_dynamics"]

        elements_order, atom_counts = collect_elements_and_counts(
            candidate["bilayer_species"],
            sort_elements if sort_elements is not None else list(dict.fromkeys(known_elements))
        )

        lattice_type = get_2d_lattice_type(bilayer_lattice_matrix)
        shifts       = get_shift_grid(lattice_type)
        p_shift = len(str(len(shifts)))

        base_name = f"{str(list_no).zfill(p_twist)}_{theta:.1f}_{total_atoms}"
        base_dir = os.path.join(working_dir, base_name)
        os.makedirs(base_dir, exist_ok=True)

        print(f"\n  [No.{index + 1}] theta = {theta:.1f} deg | {total_atoms} atoms"
              f" | ratio1 = {ratio1}  ratio2 = {ratio2}"
              f" | strain = {strain:.4f}% | lattice = {lattice_type}")

        for i, (shift_a, shift_b, stack_label) in enumerate(shifts, start=1):

            # Apply stacking shift to top-layer atoms BEFORE mapping_elements.
            # n_bottom correctly identifies the boundary between bottom and top
            # atoms in the layer-ordered positions_direct array.
            # After mapping_elements, atoms are grouped by element, so n_bottom
            # would no longer identify the top layer correctly.
            bilayer_positions_direct = candidate["positions_direct"].copy()
            bilayer_positions_direct[n_bottom:, 0] = (bilayer_positions_direct[n_bottom:, 0] + shift_a) % 1.0
            bilayer_positions_direct[n_bottom:, 1] = (bilayer_positions_direct[n_bottom:, 1] + shift_b) % 1.0

            bilayer_positions_cartesian = direct_to_cartesian(bilayer_lattice_matrix, bilayer_positions_direct)

            mapping = mapping_elements(elements_order, atom_counts,
                                       bilayer_positions_cartesian, bilayer_positions_direct,
                                       candidate["bilayer_species"],
                                       selective_dynamics,
                                       candidate["bilayer_flags"],
                                       sort_elements)

            labels = define_labels(mapping["elements"], mapping["atom_counts"])

            output_name = f"{str(i).zfill(p_shift)}_{stack_label}"
            output_dir  = os.path.join(base_dir, output_name)
            os.makedirs(output_dir, exist_ok=True)
            output_path = os.path.join(output_dir, "POSCAR")

            write_POSCAR(output_path,
                         bilayer_lattice_matrix,
                         mapping["elements"],
                         mapping["atom_counts"],
                         mapping["positions_cartesian"],
                         mapping["positions_direct"],
                         selective_dynamics,
                         mapping["flags"],
                         labels)

            rel_path = os.path.relpath(output_path, working_dir)
            output_records.append({"path":        rel_path,
                                   "theta":       theta,
                                   "total_atoms": total_atoms,
                                   "stack":       stack_label,
                                   "strain":      strain,
                                   "indices1":    candidate["indices1"],
                                   "indices2":    candidate["indices2"]})

            print(f"    Written: {rel_path}  (stacking = {stack_label},"
                  f" shift = ({shift_a:.4f}, {shift_b:.4f}))")
            written_count += 1

    return output_records


def select_and_generate(candidates, sort_elements, known_elements, working_dir, bottom, top):
    """Display candidates, prompt for a selection, and write the chosen POSCARs.

    Shared tail end of both the new-search and existing-TWIST_LIST.dat flows.

    Parameters
    ----------
    candidates     : list[dict]
    sort_elements  : list or None
    known_elements : list[str]
    working_dir    : str
    bottom, top    : dict — from read_POSCAR
    """

    display_candidates(candidates)
    chosen_indices = prompt_selection(candidates)

    if len(chosen_indices) == 0:
        print("\nNo candidates selected. Finished without writing any POSCAR.\n")
        return

    output_records = generate_POSCARs(chosen_indices, candidates, sort_elements, known_elements,
                                      working_dir, bottom, top)

    print(f"\nFinished! Written {len(output_records)} POSCAR(s).\n")


def run_new_search(bottom_file, top_file, bottom, top, working_dir, twist_list_path):
    """Run a fresh commensurate-cell search, write TWIST_LIST.dat, then
    continue straight into candidate selection and POSCAR generation.

    bottom/top are passed in already read — main() reads them once for the
    TWIST_LIST.dat match check and reuses the same dicts here rather than
    reading the files again.

    Parameters
    ----------
    bottom_file, top_file : str
    bottom, top             : dict — from read_POSCAR
    working_dir             : str
    twist_list_path         : str
    """

    homobilayer = is_homobilayer(bottom, top)
    if homobilayer:
        print(f"\nHomobilayer: {bottom_file}")
    else:
        print(f"\nHeterobilayer: {bottom_file}  +  {top_file}")

    bilayer_elements = bottom["elements"] + top["elements"]
    sort_elements    = check_elements(bilayer_elements)
    known_elements   = sort_elements if sort_elements is not None else bilayer_elements

    candidates = run_search_dispatch(bottom, top, sort_elements, known_elements)

    write_twist_list(twist_list_path, candidates, bottom_file, top_file, homobilayer)

    # Snapshot naming depends on homo/hetero status: one shared file for a
    # homobilayer, two separate files for a heterobilayer. Stale files from
    # the other scheme (e.g. left over from a previous hetero search that
    # this run replaced with a homo search) are removed so the directory
    # doesn't accumulate snapshots that no longer correspond to TWIST_LIST.dat.
    homobilayer_snapshot_path = os.path.join(working_dir, SNAPSHOT_HOMOBILAYER_FILE)
    bottom_snapshot_path      = os.path.join(working_dir, SNAPSHOT_BOTTOM_FILE)
    top_snapshot_path         = os.path.join(working_dir, SNAPSHOT_TOP_FILE)

    if homobilayer:
        write_poscar_snapshot(homobilayer_snapshot_path, bottom)
        if os.path.exists(bottom_snapshot_path):
            os.remove(bottom_snapshot_path)
        if os.path.exists(top_snapshot_path):
            os.remove(top_snapshot_path)
        print(f"Saved structure snapshot for future match checks: {SNAPSHOT_HOMOBILAYER_FILE}")
    else:
        write_poscar_snapshot(bottom_snapshot_path, bottom)
        write_poscar_snapshot(top_snapshot_path, top)
        if os.path.exists(homobilayer_snapshot_path):
            os.remove(homobilayer_snapshot_path)
        print(f"Saved structure snapshots for future match checks: "
              f"{SNAPSHOT_BOTTOM_FILE}, {SNAPSHOT_TOP_FILE}")

    print(f"\nFound {len(candidates)} candidate(s). Results written to {TWIST_LIST_FILE}")

    select_and_generate(candidates, sort_elements, known_elements, working_dir, bottom, top)


def run_from_twist_list(twist_list_path, working_dir, saved_bottom, saved_top,
                        saved_homobilayer, bottom, top):
    """Go straight into candidate selection/generation using an already-loaded
    TWIST_LIST.dat header and already-read POSCAR structures.

    All header validation, file-existence checks, and POSCAR reading are
    done once by the caller (main()) — this function trusts its inputs and
    goes straight to reading the candidate list and generating output.

    Parameters
    ----------
    twist_list_path    : str
    working_dir          : str
    saved_bottom          : str
    saved_top             : str
    saved_homobilayer     : bool or None
    bottom, top            : dict — from read_POSCAR
    """

    # Prefer the type recorded in the file; fall back to re-deriving it only
    # for TWIST_LIST.dat files written before the '# type' field existed.
    homobilayer = saved_homobilayer if saved_homobilayer is not None else is_homobilayer(bottom, top)
    if homobilayer:
        print(f"\nHomobilayer (from {TWIST_LIST_FILE}): {saved_bottom}")
    else:
        print(f"\nHeterobilayer (from {TWIST_LIST_FILE}): {saved_bottom}  +  {saved_top}")

    bilayer_elements = bottom["elements"] + top["elements"]
    sort_elements    = check_elements(bilayer_elements)
    known_elements   = sort_elements if sort_elements is not None else bilayer_elements

    candidates = read_twist_list(twist_list_path, bottom["lattice_matrix"], top["lattice_matrix"])
    if len(candidates) == 0:
        print(f"\nERROR! No candidates found in {TWIST_LIST_FILE}.\n")
        exit(1)

    print(f"\nRead {len(candidates)} candidate(s) from {TWIST_LIST_FILE}.")

    select_and_generate(candidates, sort_elements, known_elements, working_dir, bottom, top)


def main():
    """Entry point: POSCAR argument(s) are always required; TWIST_LIST.dat is
    reused only if it matches the given input, otherwise a fresh search runs."""

    if '-h' in argv or '--help' in argv:
        usage()

    if len(argv) > 3 or len(argv) < 2:
        usage()

    bottom_file = argv[1]
    top_file    = argv[2] if len(argv) == 3 else argv[1]

    if not os.path.exists(bottom_file):
        print(f"ERROR! File '{bottom_file}' does not exist.")
        exit(1)
    if not os.path.exists(top_file):
        print(f"ERROR! File '{top_file}' does not exist.")
        exit(1)

    bottom = read_POSCAR(bottom_file)
    top    = read_POSCAR(top_file) if top_file != bottom_file else bottom

    working_dir       = os.getcwd()
    twist_list_path   = os.path.join(working_dir, TWIST_LIST_FILE)
    twist_list_exists = os.path.exists(twist_list_path)

    if twist_list_exists:
        saved_bottom, saved_top, saved_homobilayer = read_twist_list_header(twist_list_path)

        # Which snapshot file(s) to look for is fully determined by the
        # '# type' field already recorded in the header — no separate
        # bookkeeping needed for which naming scheme was used.
        homobilayer_snapshot_path = os.path.join(working_dir, SNAPSHOT_HOMOBILAYER_FILE)
        bottom_snapshot_path      = os.path.join(working_dir, SNAPSHOT_BOTTOM_FILE)
        top_snapshot_path         = os.path.join(working_dir, SNAPSHOT_TOP_FILE)

        header_valid = True
        if saved_bottom is None:
            header_valid = False
        if saved_top is None:
            header_valid = False
        if saved_homobilayer is None:
            header_valid = False
        if header_valid:
            if saved_homobilayer:
                if not os.path.exists(homobilayer_snapshot_path):
                    header_valid = False
            else:
                if not os.path.exists(bottom_snapshot_path):
                    header_valid = False
                if not os.path.exists(top_snapshot_path):
                    header_valid = False

        if not header_valid:
            print(f"\nWARNING! {TWIST_LIST_FILE} has an invalid or incomplete header.")
            print("Running a fresh search from scratch...\n")
        else:
            if saved_homobilayer:
                snapshot_bottom = read_POSCAR(homobilayer_snapshot_path)
                snapshot_top    = snapshot_bottom
            else:
                snapshot_bottom = read_POSCAR(bottom_snapshot_path)
                snapshot_top    = read_POSCAR(top_snapshot_path)

            bottom_match = poscar_structure_matches(bottom, snapshot_bottom)
            top_match    = poscar_structure_matches(top, snapshot_top)

            matches = False
            if bottom_match:
                if top_match:
                    matches = True

            if matches:
                run_from_twist_list(twist_list_path, working_dir, saved_bottom, saved_top,
                                    saved_homobilayer, bottom, top)
                return

            print(f"\nWARNING! Provided POSCAR(s) do not match the structure(s) recorded in {TWIST_LIST_FILE}.")
            print(f"  Recorded : {saved_bottom}  +  {saved_top}")
            print(f"  Provided : {bottom_file}  +  {top_file}")
            print("Re-running the search with the provided POSCAR(s)...\n")

    run_new_search(bottom_file, top_file, bottom, top, working_dir, twist_list_path)


if __name__ == "__main__":
    main()
