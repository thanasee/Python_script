#!/usr/bin/env python

from sys import argv, exit
import os
import readline
import numpy as np


def usage():
    """Print usage information and exit."""
    print("""
Usage: vaspStack.py <input>

This script supports VASP5 structure file format (i.e. POSCAR)
for generated bilayer from input file and also prepared stacking images
only support monolayer file which has vacuum space in z direction.

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


_VDW_RADIUS = [
    1.675, 1.414, 2.799, 2.269, 2.080, 1.910, 1.798, 1.715,
    1.631, 1.554, 2.797, 2.485, 2.412, 2.265, 2.139, 2.063,
    1.981, 1.905, 3.037, 2.792, 2.597, 2.608, 2.557, 2.540,
    2.468, 2.436, 2.395, 2.355, 2.342, 2.277, 2.362, 2.288,
    2.196, 2.186, 2.087, 2.021, 3.080, 2.873, 2.794, 2.651,
    2.600, 2.557, 2.522, 2.489, 2.455, 2.153, 2.395, 2.334,
    2.453, 2.382, 2.312, 2.271, 2.225, 2.167, 3.181, 3.009,
    2.909, 2.890, 2.912, 2.896, 2.880, 2.863, 2.845, 2.785,
    2.814, 2.801, 2.779, 2.764, 2.747, 2.734, 2.728, 2.619,
    2.498, 2.466, 2.436, 2.407, 2.388, 2.348, 2.254, 2.235,
    2.362, 2.342, 2.348, 2.319, 2.304, 2.245, 3.077, 2.966,
    2.886, 2.916, 2.774, 2.705, 2.767, 2.715, 2.709, 2.746,
    2.694, 2.683, 2.672, 2.656, 2.641, 2.644, 3.080, 2.651,
    2.304, 2.288, 2.271, 2.254, 2.236, 2.216, 2.217, 2.174,
    2.186, 2.206, 2.482, 3.000, 2.508, 2.413
]


def get_vdw_radius(element):
    """Look up the van der Waals radius for an element.

    Parameters
    ----------
    element : str — element symbol
 
    Returns
    -------
    float — van der Waals radius in Angstrom, or 3.0 if not found
    """
    try:
        idx = _ELEMENT_SYMBOLS.index(element)
        return _VDW_RADIUS[idx]
    except ValueError:
        return 3.0


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

    bottom_radius = get_vdw_radius(bottom_element)
    top_radius    = get_vdw_radius(top_element)

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
                print("Warning! Empty input — using default unique element order.\n")
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
        o.write("Generated by vaspStack.py\n")
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


def build_second_layer(species, positions_cartesian):
    """Build the second layer by shifting a copy of the monolayer upward in z.

    The vertical shift is set to the monolayer thickness plus a 3.5 Å interlayer
    gap, placing the second layer directly above the first in Cartesian space.

    Parameters
    ----------
    species             : list[str]        — element symbols of the monolayer
    positions_cartesian : np.ndarray, shape (N, 3) — Cartesian coordinates of the
                          monolayer in Å

    Returns
    -------
    thickness                  : float             — layer thickness in Å
                                 (max_z − min_z of the monolayer)
    second_positions_cartesian : np.ndarray, shape (N, 3) — Cartesian coordinates
                                 of the second layer in Å
    """
    interlayer_gap = compute_interlayer_gap(species, positions_cartesian, species, positions_cartesian)
    thickness = np.max(positions_cartesian[:, 2]) - np.min(positions_cartesian[:, 2])

    shift = thickness + interlayer_gap
    
    second_positions_cartesion = positions_cartesian.copy()
    second_positions_cartesion[:, 2] += shift
    
    return thickness, second_positions_cartesion


def flip_sheet(positions_cartesian, flip_mode):
    """Reflect atomic positions of the second layer along a chosen axis.

    For the z-axis, reflection is performed through the layer midplane to
    preserve the interlayer geometry (top layer flipped to bottom orientation).
    For x and y axes, reflection is through the origin — atoms are wrapped
    back into [0, 1) by cartesian_to_direct after this function returns.

        'none' : no reflection
        'z'    : reflect through XY plane at layer midplane (z → 2*mid_z - z)
                 physically flips the layer upside down without displacing it
        'x'    : negate x  (x → -x), wrapped by cartesian_to_direct
        'y'    : negate y  (y → -y), wrapped by cartesian_to_direct
        'xy'   : negate x and y, wrapped by cartesian_to_direct

    Parameters
    ----------
    positions_cartesian : np.ndarray, shape (N, 3) — Cartesian coordinates in Å
    flip_mode           : str — one of 'none', 'z', 'x', 'y', 'xy'

    Returns
    -------
    new_positions_cartesian : np.ndarray, shape (N, 3) — reflected Cartesian coordinates in Å
    """

    new_positions_cartesian = positions_cartesian.copy()

    if flip_mode == 'none':
        pass

    centroid = np.mean(positions_cartesian, axis=0)

    if flip_mode == 'z':
        mid_z = (np.max(positions_cartesian[:, 2]) +
                 np.min(positions_cartesian[:, 2])) / 2.0
        new_positions_cartesian[:, 2] = 2.0 * mid_z - positions_cartesian[:, 2]

    elif flip_mode == 'x':
        new_positions_cartesian[:, 0] = 2.0 * centroid[0] - positions_cartesian[:, 0]

    elif flip_mode == 'y':
        new_positions_cartesian[:, 1] = 2.0 * centroid[1] - positions_cartesian[:, 1]

    elif flip_mode == 'xy':
        new_positions_cartesian[:, 0] = 2.0 * centroid[0] - positions_cartesian[:, 0]
        new_positions_cartesian[:, 1] = 2.0 * centroid[1] - positions_cartesian[:, 1]
    
    return new_positions_cartesian


def rotate_sheet(positions_cartesian, degree):
    """Rotate a 2D sheet in-plane around the z-axis passing through its centroid.

    The rotation is applied in Cartesian space around the geometric center of
    the sheet, keeping all atoms within the original cell by wrapping back to
    fractional coordinates afterwards. The cell itself is not modified.

    Note: meaningful only for angles that preserve the lattice periodicity
    (e.g. 60°, 120°, 180° for hexagonal; 90°, 180° for square). For arbitrary
    twist angles use vaspTwist.py instead.

    Parameters
    ----------
    positions_cartesian : np.ndarray, shape (N, 3) — Cartesian coordinates in Å
    angle_deg           : float — rotation angle in degrees (counterclockwise)

    Returns
    -------
    positions_direct : np.ndarray, shape (N, 3) — rotated fractional coordinates in [0, 1)
    """

    radian = np.radians(degree)
    cos, sin = np.cos(radian), np.sin(radian)

    rotate_matrix = np.array([[ cos, sin, 0.],
                              [-sin, cos, 0.],
                              [  0.,  0., 1.]])

    centroid = np.mean(positions_cartesian, axis=0)
    centroid[2] = 0.0

    rotated  = (positions_cartesian - centroid) @ rotate_matrix.T + centroid

    return rotated


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
    vacuum = 2
    periodic = [i for i in range(3) if i != vacuum]
    new = np.copy(unwrapped)
    new[:, periodic] = unwrapped[:, periodic] - reference[periodic]
    new[:, vacuum]   = unwrapped[:, vacuum] - center[vacuum] + 0.5
    
    return new % 1.0


def get_2d_lattice_type(lattice_matrix):
    """Classify the 2D Bravais lattice type from the in-plane lattice vectors.

    Compares the in-plane lattice lengths (a, b) and the angle γ between them
    to identify the crystal system according to the standard 2D classification:

        square      : γ = 90°  and  a = b
        rectangular : γ = 90°  and  a ≠ b
        hexagonal   : γ = 60° or 120°  and  a = b
        oblique     : all other cases

    Parameters
    ----------
    lattice_matrix : np.ndarray, shape (3, 3) — row vectors of the lattice in Å

    Returns
    -------
    str : one of 'square', 'rectangular', 'hexagonal', 'oblique'
    """

    length_a = np.linalg.norm(lattice_matrix[0])
    length_b = np.linalg.norm(lattice_matrix[1])
    gamma = np.degrees(np.arccos(np.clip((lattice_matrix[0] @ lattice_matrix[1]) /
                                         (length_a * length_b), -1., 1.)))

    if np.abs(gamma - 90.) < 1e-5:
        return 'square' if np.abs(length_a - length_b) < 1e-8 else 'rectangular'
    elif ((np.abs(gamma - 60.) < 1e-5 or np.abs(gamma - 120.) < 1e-5)
          and np.abs(length_a - length_b) < 1e-8):
        return 'hexagonal'
    else:
        return 'oblique'


def get_shift_grid(lattice_type):
    """Return the high-symmetry stacking shift points for the given 2D lattice type.

    The shifts correspond to the irreducible high-symmetry points of the
    stacking space for each 2D Bravais lattice:

        square      : AA (0,0),  AB (1/2,0),    AC  (1/2,1/2)
        rectangular : AA (0,0),  AB (1/2,0),    AB' (0,1/2),  AC   (1/2,1/2)
        hexagonal   : AA (0,0),  AB (1/3,2/3),  AB' (2/3,1/3)
        oblique     : AA (0,0),  AB (1/2,0),    AB' (0,1/2),  AC   (1/2,1/2)

    Parameters
    ----------
    lattice_type : str — one of 'square', 'rectangular', 'hexagonal', 'oblique'

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


def get_rotation_grid(lattice_type):
    """Return the high-symmetry in-plane rotation angles for the given 2D lattice type.

    The angles correspond to symmetry-distinct rotations of the second layer
    within the fixed cell. Only rotations that map the lattice onto itself
    (i.e. preserve periodicity) are included:

        square      : 0°, 90°, 180°
        rectangular : 0°, 180°
        hexagonal   : 0°, 60°, 120°, 180°
        oblique     : 0°, 180°

    A 0° entry corresponds to the unrotated reference stacking.
    360° is excluded as it is equivalent to 0°.

    Parameters
    ----------
    lattice_type : str — one of 'square', 'rectangular', 'hexagonal', 'oblique'

    Returns
    -------
    list of (float, str) — (angle_deg, label) pairs
    """

    if lattice_type == 'hexagonal':
        return [(0.,   'R0'),
                (60.,  'R60'),
                (120., 'R120'),
                (180., 'R180')]

    elif lattice_type == 'square':
        return [(0.,   'R0'),
                (90.,  'R90'),
                (180., 'R180')]

    else:
        # rectangular or oblique
        return [(0.,   'R0'),
                (180., 'R180')]


def get_flip_grid(lattice_type):
    """Return the mirror/flip configurations for the second layer.

    Selects physically distinct flip operations based on lattice symmetry:

        hexagonal   : none, z
        square      : none, z, x, xy
        rectangular : none, z, x, y, xy
        oblique     : none, z

    Parameters
    ----------
    lattice_type : str — one of 'square', 'rectangular', 'hexagonal', 'oblique'

    Returns
    -------
    list of (str, str) — (flip_mode, label) pairs
    """

    if lattice_type == 'hexagonal':
        return [('none', 'F0'),
                ('z',    'Fz')]

    elif lattice_type == 'square':
        return [('none', 'F0'),
                ('z',    'Fz'),
                ('x',    'Fx'),
                ('xy',   'Fxy')]

    elif lattice_type == 'rectangular':
        return [('none', 'F0'),
                ('z',    'Fz'),
                ('x',    'Fx'),
                ('y',    'Fy'),
                ('xy',   'Fxy')]

    else:
        # oblique
        return [('none', 'F0'),
                ('z',    'Fz')]


def shift_sheet(positions_direct, shift_a, shift_b):
    """Translate a sheet by a fractional shift along the a and b lattice directions.

    The shift is applied directly to the fractional coordinates and wrapped
    back into [0, 1) to maintain periodicity. This generates laterally displaced
    stacking configurations for the bilayer grid.

    Parameters
    ----------
    positions_direct : np.ndarray, shape (N, 3) — fractional coordinates in [0, 1)
    shift_a          : float — fractional shift along the a direction
    shift_b          : float — fractional shift along the b direction

    Returns
    -------
    new : np.ndarray, shape (N, 3) — shifted fractional coordinates in [0, 1)
    """
    
    new = positions_direct.copy()
    new[:, 0] += shift_a
    new[:, 1] += shift_b
    new %= 1.0
    
    return new


def build_bilayer(atom_counts, first_positions_direct, second_positions_direct, species, selective_dynamics, flags):
    """Stack two monolayer sheets into a bilayer structure.

    Concatenates the atomic positions of the first and second layers, then
    centers the combined bilayer in the vacuum direction at z = 0.5 (fractional).
    Atom counts and species lists are doubled to reflect both layers.

    Parameters
    ----------
    atom_counts              : list[int]           — atoms per element in one layer
    first_positions_direct   : np.ndarray (N, 3)   — fractional coordinates of layer 1
    second_positions_direct  : np.ndarray (N, 3)   — fractional coordinates of layer 2
    species                  : list[str]           — per-atom element labels for one layer
    selective_dynamics       : bool                — whether Selective Dynamics is used
    flags                    : np.ndarray or None  — per-atom T/F flags for one layer

    Returns
    -------
    dict with keys:
        atom_counts      : list[int]           — doubled atom counts
        positions_direct : np.ndarray (2N, 3)  — centered fractional coordinates
        species          : list[str]           — doubled per-atom species list
        flags            : np.ndarray or None  — doubled T/F flags, or None
    """
    
    new_atom_counts = atom_counts + atom_counts
    new_positions_direct = np.vstack((first_positions_direct, second_positions_direct))
    center_positions_direct = center_sheet(new_positions_direct)
    new_species = list(species) + list(species)
    
    new_flags = None
    if selective_dynamics:
        new_flags = np.vstack((flags, flags))
    
    return {"atom_counts": new_atom_counts,
            "positions_direct": center_positions_direct,
            "species": new_species,
            "flags": new_flags if selective_dynamics else None}


def write_stack_list(filepath, flips, rotations, shifts, working_dir):
    """Write a summary file listing all generated stacking configurations.

    Records the index, directory path, and the flip, rotation, and shift
    labels for each generated bilayer configuration. This file serves as
    a reference for identifying and submitting stacking calculations.

    Parameters
    ----------
    filepath    : str                            — path to the output summary file
    flips       : list[tuple[str, str]]          — (flip_mode, label) entries
    rotations   : list[tuple[float, str]]        — (angle_deg, label) entries
    shifts      : list[tuple[float, float, str]] — (shift_a, shift_b, label) entries
    """

    with open(filepath, 'w') as o:
        o.write(f"{'No.':<6}  {'Flip':<6}  {'Rotation':<10}  "
                f"{'Stack':<10}  {'shift_a':>10}  {'shift_b':>10}  "
                f"{'Path'}\n")
        o.write("-" * 110 + "\n")
        i = 1
        for _, flip_label in flips:
            for _, rot_label in rotations:
                for shift_a, shift_b, stack_label in shifts:
                    output_dir = os.path.join(working_dir,
                                              f"{i}_{flip_label}_{rot_label}_{stack_label}")
                    o.write(f"{i:<6}  {flip_label:<6}  {rot_label:<10}  "
                            f"{stack_label:<10}  {shift_a:>10.6f}  {shift_b:>10.6f}  "
                            f"{output_dir}\n")
                    i += 1


def main():
    """Parse argument, build second layer, stack both layers, and write output files"""
    
    if '-h' in argv or '--help' in argv or len(argv) != 2:
        usage()

    working_dir = os.getcwd()
    monolayer = read_POSCAR(argv[1])
    _, second_positions_cartesion = build_second_layer(monolayer["positions_cartesian"])
    
    bilayer_elements = monolayer["elements"] + monolayer["elements"]
    sort_elements = check_elements(bilayer_elements)
    lattice_type = get_2d_lattice_type(monolayer["lattice_matrix"])
    print(f"Detected 2D lattice type: {lattice_type}")
    shifts    = get_shift_grid(lattice_type)
    rotations = get_rotation_grid(lattice_type)
    flips     = get_flip_grid(lattice_type)
    total = len(flips) * len(rotations) * len(shifts)
    
    place = len(str(total))
    
    i = 1
    for flip_mode, flip_label in flips:
        flipped_positions_cartesian = flip_sheet(second_positions_cartesion, flip_mode)

        for degree, rot_label in rotations:
            if np.abs(degree) > 1e-8:
                rotated_positions_cartesian = rotate_sheet(flipped_positions_cartesian, degree)
            else:
                rotated_positions_cartesian = flipped_positions_cartesian.copy()
            rotated_positions_direct = cartesian_to_direct(monolayer["lattice_matrix"], rotated_positions_cartesian)

            for shift_a, shift_b, stack_label in shifts:
                shift_positions_direct = shift_sheet(rotated_positions_direct, shift_a, shift_b)
                bilayer = build_bilayer(monolayer["atom_counts"], monolayer["positions_direct"], shift_positions_direct,
                                        monolayer["species"], monolayer["selective_dynamics"], monolayer["flags"])
                bilayer_positions_cartesian = direct_to_cartesian(monolayer["lattice_matrix"], bilayer["positions_direct"])
                mapping = mapping_elements(bilayer_elements, bilayer["atom_counts"], bilayer_positions_cartesian, bilayer["positions_direct"],
                                           bilayer["species"], monolayer["selective_dynamics"], bilayer["flags"], sort_elements)
                labels = define_labels(mapping["elements"], mapping["atom_counts"])
                output_name = f"{str(i).zfill(place)}_{flip_label}_{rot_label}_{stack_label}"
                output_dir = os.path.join(working_dir, output_name)
                os.makedirs(output_dir, exist_ok=True)
                output_path = os.path.join(output_dir, "POSCAR")
                write_POSCAR(output_path, monolayer["lattice_matrix"], mapping["elements"], mapping["atom_counts"],
                             mapping["positions_cartesian"], mapping["positions_direct"], monolayer["selective_dynamics"],
                             mapping["flags"], labels)
                i += 1

    
    print(f"Written {total} POSCARs Finished!\n")
    
    stack_list = "STACK_LIST.dat"
    stack_list_path = os.path.join(working_dir, stack_list)
    write_stack_list(stack_list_path, flips, rotations, shifts, working_dir)
    print(f"Stack list written to: {stack_list}\n")


if __name__ == "__main__":
    main()
