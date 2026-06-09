#!/usr/bin/env python

from sys import argv, exit
import os
import numpy as np


def usage():
    """Print usage information and exit."""
    print("""
Usage: ElasticTensor2D.py <mode> [structure file]

This script supports VASP5 Structure file format (i.e. POSCAR)
for applying strain to a structure file.

Mode:
- pre  <structure_file> : prepare structure files for elastic tensor calculations
- post                  : get energy and calculate elastic tensor

This script was developed by Thanasee Thanasarnsurapong.
""")
    exit(0)


ANGSTROM = 1e-10
ALL_STRAIN = ['C11', 'C22', 'C11_C22_2C12', 'C66']
STRAIN_RANGE = np.linspace(-0.02, 0.02, 9)
ZERO_STRAIN_IDX = len(STRAIN_RANGE) // 2


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
            "positions_cartesian":new_positions_cartesian,
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
    labels = [f"{symbol}{str(counter).zfill(digits)}" for symbol, number in zip(elements, atom_counts)
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


def get_strain_types(crystal_system):
    """Return the list of strain deformation types required for the given crystal system.

    Square, rectangular, and hexagonal lattices require four strain types to
    determine C11, C22, C12, and C66. Oblique lattices additionally require
    two mixed strain types to resolve the off-diagonal constants C16 and C26.

    Parameters
    ----------
    crystal_system : str — output of get_2d_lattice_type()

    Returns
    -------
    list[str] — strain type labels corresponding to keys in build_strain_matrix()
    """

    strain_types = ALL_STRAIN.copy()
    if crystal_system == 'oblique':
        strain_types.extend(['C11_C66_2C16', 'C22_C66_2C26'])

    return strain_types


def build_strain_matrix(strain_type, strain):
    """Construct the 3×3 strain tensor for a given deformation type and magnitude.

    Each strain type applies a specific deformation pattern to the lattice:
        C11           : uniaxial strain along x
        C22           : uniaxial strain along y
        C11_C22_2C12  : biaxial strain (x and y simultaneously)
        C66           : in-plane shear strain
        C11_C66_2C16  : combined uniaxial-x and shear (oblique only)
        C22_C66_2C26  : combined uniaxial-y and shear (oblique only)

    The shear components are halved (strain/2) to maintain the engineering
    convention where γ_xy = 2·ε_xy.

    Parameters
    ----------
    strain_type : str   — one of the keys listed above
    strain      : float — strain magnitude δ (dimensionless)

    Returns
    -------
    np.ndarray, shape (3, 3) — the strain tensor ε
    """

    d = strain
    h = strain / 2

    strain_matrices = {
        'C11':          np.array([[d, 0, 0], [0, 0, 0], [0, 0, 0]], dtype=float),
        'C22':          np.array([[0, 0, 0], [0, d, 0], [0, 0, 0]], dtype=float),
        'C11_C22_2C12': np.array([[d, 0, 0], [0, d, 0], [0, 0, 0]], dtype=float),
        'C66':          np.array([[0, h, 0], [h, 0, 0], [0, 0, 0]], dtype=float),
        'C11_C66_2C16': np.array([[d, h, 0], [h, 0, 0], [0, 0, 0]], dtype=float),
        'C22_C66_2C26': np.array([[0, h, 0], [h, d, 0], [0, 0, 0]], dtype=float),
    }

    return strain_matrices[strain_type]


def applying_strain(lattice_matrix, strain_matrix):
    """Apply a strain tensor to a lattice matrix using the deformation gradient.

    The strained lattice is computed as:
        a' = a · (I + ε)
    where ε is the strain tensor and the lattice vectors are row vectors.

    Parameters
    ----------
    lattice_matrix : np.ndarray (3, 3) — original lattice vectors in Å
    strain_matrix  : np.ndarray (3, 3) — strain tensor ε

    Returns
    -------
    np.ndarray (3, 3) — strained lattice vectors in Å
    """

    return lattice_matrix @ (np.eye(3) + strain_matrix)


def read_OUTCAR(filepath):
    """Extract the final total energy and convergence status from a VASP OUTCAR file.

    Convergence is determined by the presence of 'reached required accuracy'
    anywhere in the file. The final energy is the last occurrence of the line
    containing 'energy  without entropy=', which corresponds to the energy
    after the last ionic step.

    Parameters
    ----------
    filepath : str — path to the OUTCAR file

    Returns
    -------
    energy    : float or None — total energy in eV, or None if not found
    converged : bool          — True if the calculation reached required accuracy
    """

    if not os.path.exists(filepath):
        print(f"ERROR! File not found: {filepath}")
        return None, False

    with open(filepath, 'r') as f:
        outcar_lines = f.readlines()

    converged = any('reached required accuracy' in line for line in outcar_lines)

    if not converged:
        print(f"WARNING! {filepath} has not reached required accuracy.")

    energy = None
    for line in reversed(outcar_lines):
        if 'energy  without entropy=' in line:
            energy = float(line.split()[-1])
            break

    if energy is None:
        print(f"WARNING! Could not extract energy from {filepath}.")

    return energy, converged


def fitting_strain_energy(strain_type, strain_range, strain_energy, area):
    """Fit the strain-energy curve to a quadratic and return the elastic constant.

    The strain energy density U(δ)/A is fit to:
        U(δ)/A = a · δ**2
    The corresponding elastic constant is then:
        C = 2a   (in eV/Å**2)
    and converted to N/m using the SI conversion factor e / Å**2.

    The energy is referenced to the unstrained value (index ZERO_STRAIN_IDX)
    and normalized by the in-plane area A.

    The strain-energy data for each strain type is also written to a .dat file
    named StrainVsEnergy_<strain_type>.dat.

    Parameters
    ----------
    strain_type  : str             — label used for the output filename
    strain_range : np.ndarray (M,) — strain values δ
    strain_energy: np.ndarray (M,) — total energies in eV
    area         : float           — in-plane cell area in Å**2

    Returns
    -------
    float — elastic constant in N/m
    """

    from scipy.constants import e
    from scipy.optimize import curve_fit

    eVpA2_to_Npm = e / ANGSTROM ** 2

    def quadratic(x, a):
        return a * x ** 2

    energy_per_area = (strain_energy - strain_energy[ZERO_STRAIN_IDX]) / area

    output_file = f"StrainVsEnergy_{strain_type}.dat"
    with open(output_file, 'w') as o:
        o.write(f"# Strain vs Energy per area — {strain_type}\n")
        o.write("# Strain    U(eV/Å²)\n")
        for s, u in zip(strain_range, energy_per_area):
            o.write(f"  {s:>+6.2f}   {u:>14.6f}\n")

    coef, _ = curve_fit(quadratic, strain_range, energy_per_area)
    constant = 2 * coef[0] * eVpA2_to_Npm

    return constant


def collect_fitting_coef(strain_types, strain_range, area):
    """Collect energies from all strain OUTCAR files and fit each strain-energy curve.

    For each strain type, reads OUTCAR files from directories named
    '<strain_type>/strain<±δ>' and checks convergence. If any calculation
    in a strain set is unconverged or missing, that strain type is skipped
    and its constant is set to None.

    Parameters
    ----------
    strain_types : list[str]       — strain type labels
    strain_range : np.ndarray (M,) — strain values δ
    area         : float           — in-plane cell area in Å**2

    Returns
    -------
    dict[str, float or None] — fitted elastic constant (N/m) per strain type,
                               or None if that strain type could not be fitted
    """

    constants = {}

    for strain_type in strain_types:

        strain_energy = []
        all_converged = True

        for strain in strain_range:
            strain_path = os.path.join(strain_type, f"strain{strain:+.2f}")
            outcar = os.path.join(strain_path, 'OUTCAR')
            energy, converged = read_OUTCAR(outcar)
            strain_energy.append(energy)
            if not converged:
                all_converged = False

        if not all_converged or None in strain_energy:
            print(f"WARNING! Skipping '{strain_type}' — incomplete or unconverged calculations.")
            constants[strain_type] = None
            continue

        strain_energy = np.array(strain_energy, dtype=float)
        constants[strain_type] = fitting_strain_energy(strain_type, strain_range, strain_energy, area)

    return constants


def obtain_elastic_tensor(constants, crystal_system):
    """Derive the 2D elastic tensor from the fitted strain-energy constants.

    The independent elastic constants are obtained from the fitted combinations:
        C12 = (C11 + C22 + 2C12  −  C11  −  C22) / 2
        C16 = (C11 + C66 + 2C16  −  C11  −  C66) / 2   [oblique only]
        C26 = (C22 + C66 + 2C26  −  C22  −  C66) / 2   [oblique only]

    The tensor is assembled as a symmetric 3×3 matrix in Voigt notation:
        | C11  C12  C16 |
        | C12  C22  C26 |
        | C16  C26  C66 |

    Parameters
    ----------
    constants      : dict[str, float or None] — output of collect_fitting_coef()
    crystal_system : str                      — output of get_2d_lattice_type()

    Returns
    -------
    np.ndarray (3, 3) or None — elastic tensor in N/m, or None if any constant is missing
    """

    missing = [k for k, v in constants.items() if v is None]
    if missing:
        for key in missing:
            print(f"ERROR! {key} is not calculated.")
        return None

    C11 = constants['C11']
    C22 = constants['C22']
    C66 = constants['C66']
    C12 = (constants['C11_C22_2C12'] - C11 - C22) / 2
    C16 = 0.
    C26 = 0.

    if crystal_system == 'oblique':
        C16 = (constants['C11_C66_2C16'] - C11 - C66) / 2
        C26 = (constants['C22_C66_2C26'] - C22 - C66) / 2

    elastic_tensor = np.array([[C11, C12, C16],
                               [C12, C22, C26],
                               [C16, C26, C66]])

    return elastic_tensor


def check_stability(elastic_tensor, lattice_matrix, area_vector, area):
    """Check mechanical stability of the 2D material via eigenvalue positivity.

    A 2D material is mechanically stable if and only if all eigenvalues of its
    elastic tensor are positive. The stability threshold is scaled by the vacuum
    layer thickness (extracted from the out-of-plane lattice vector component
    projected onto the normal direction) divided by 10 to convert from 3D (GPa·Å)
    to 2D (N/m) units.

    Parameters
    ----------
    elastic_tensor : np.ndarray (3, 3) — 2D elastic tensor in N/m
    lattice_matrix : np.ndarray (3, 3) — lattice vectors in Å
    area_vector    : np.ndarray (3,)   — cross product of a and b vectors
    area           : float             — in-plane cell area in Å**2

    Returns
    -------
    bool — True if mechanically stable, False otherwise
    """

    vector_n  = area_vector / area
    factor_2d = np.abs(lattice_matrix[2] @ vector_n) / 10

    return np.all(np.linalg.eigvalsh(elastic_tensor) > 1e-5 * factor_2d)


def compute_mechanical_properties(elastic_tensor):
    """Compute the angular dependence of Young's modulus, Poisson's ratio, and shear modulus.

    Uses the 2D compliance tensor S = C⁻¹ to evaluate directional mechanical
    properties as a function of in-plane angle θ (0° to 360° in 0.1° steps)
    via the standard transformation relations for an anisotropic 2D solid:

        E(θ)  = 1 / A(θ)
        ν(θ)  = −B(θ) / A(θ)
        G(θ)  = 1 / C(θ)

    where A, B, C are trigonometric combinations of the compliance components
    Sij following the formulation for the most general (oblique) case, which
    reduces correctly for higher-symmetry systems.

    Parameters
    ----------
    elastic_tensor : np.ndarray (3, 3) — 2D elastic tensor in N/m

    Returns
    -------
    dict with keys:
        degrees       : np.ndarray (M,) — angles in degrees
        young_modulus : np.ndarray (M,) — E(θ) in N/m
        poisson_ratio : np.ndarray (M,) — ν(θ) dimensionless
        shear_modulus : np.ndarray (M,) — G(θ) in N/m
    """

    degrees = np.arange(0, 360.0, 0.1)
    radians = np.deg2rad(degrees)
    sin = np.sin(radians)
    cos = np.cos(radians)

    compliance_tensor = np.linalg.inv(elastic_tensor)
    S11 = compliance_tensor[0, 0]
    S22 = compliance_tensor[1, 1]
    S12 = compliance_tensor[0, 1]
    S66 = compliance_tensor[2, 2]
    S16 = compliance_tensor[0, 2]
    S26 = compliance_tensor[1, 2]

    A = (S11 * cos**4 + S22 * sin**4
         + (2 * S12 + S66) * cos**2 * sin**2
         + 2 * S16 * cos**3 * sin
         + 2 * S26 * cos * sin**3)
    B = ((S11 + S22 - S66) * cos**2 * sin**2
         + S12 * (cos**4 + sin**4)
         + (S26 - S16) * (cos**3 * sin - cos * sin**3))
    C = (4 * (S11 + S22 - 2 * S12) * cos**2 * sin**2
         + S66 * (cos**2 - sin**2)**2
         + 4 * (S16 - S26) * (cos**3 * sin - cos * sin**3))

    young_modulus = 1 / A
    poisson_ratio = -B / A
    shear_modulus = 1 / C

    return {"degrees":       degrees,
            "young_modulus": young_modulus,
            "poisson_ratio": poisson_ratio,
            "shear_modulus": shear_modulus}


def write_mechanical_properties(properties):
    """Write angular mechanical properties to three output .dat files.

    Writes Young.dat, Poisson.dat, and Shear.dat. For Poisson's ratio, an
    additional absolute-value column is included if any negative values are
    present (indicating an auxetic material).

    Parameters
    ----------
    properties : dict — output of compute_mechanical_properties()
    """

    degrees       = properties["degrees"]
    young_modulus = properties["young_modulus"]
    poisson_ratio = properties["poisson_ratio"]
    shear_modulus = properties["shear_modulus"]

    with open('Young.dat', 'w') as o:
        o.write("# Young's Modulus\n")
        o.write("#  Degree(°)  Y(N/m)\n")
        for dg, y in zip(degrees, young_modulus):
            o.write(f" {dg:>6.2f}     {y:>12.8f}\n")

    with open('Poisson.dat', 'w') as o:
        o.write("# Poisson's Ratio\n")
        if (poisson_ratio < 0.).any():
            o.write("#  Degree(°) v             |v|\n")
            for dg, nu in zip(degrees, poisson_ratio):
                o.write(f" {dg:>6.2f}   {nu:>12.8f}   {abs(nu):>12.8f}\n")
        else:
            o.write("#  Degree(°) v\n")
            for dg, nu in zip(degrees, poisson_ratio):
                o.write(f" {dg:>6.2f}   {nu:>12.8f}\n")

    with open('Shear.dat', 'w') as o:
        o.write("# Shear Modulus\n")
        o.write("#  Degree(°)  G(N/m)\n")
        for dg, g in zip(degrees, shear_modulus):
            o.write(f" {dg:>6.2f}     {g:>12.8f}\n")


def write_elastic_tensor(elastic_tensor):
    """Write the 2D elastic tensor to Elastic.dat and print it to stdout.

    The tensor is written in Voigt notation as a 3×3 matrix with a descriptive
    header indicating the component layout and units (N/m).

    Parameters
    ----------
    elastic_tensor : np.ndarray (3, 3) — 2D elastic tensor in N/m
    """

    C = elastic_tensor
    header = ("# Elastic tensor (N/m)\n"
              "#     C11         C12         C16\n"
              "#     C12         C22         C26\n"
              "#     C16         C26         C66\n\n")
    rows = (f"   {C[0,0]:>11.4f} {C[0,1]:>11.4f} {C[0,2]:>11.4f}\n"
            f"   {C[1,0]:>11.4f} {C[1,1]:>11.4f} {C[1,2]:>11.4f}\n"
            f"   {C[2,0]:>11.4f} {C[2,1]:>11.4f} {C[2,2]:>11.4f}\n")

    with open('Elastic.dat', 'w') as o:
        o.write(header + rows)

    print("\n" + header + rows)


def mode_pre(filepath):
    """Run the pre-processing mode: generate strained POSCAR files for all strain types.

    Reads the input structure, detects the 2D crystal system, and writes:
    - unstrain/POSCAR   : the reference (zero-strain) structure
    - <strain_type>/strain<±δ>/POSCAR : one POSCAR per (strain_type, δ) combination

    The strained lattice is generated by applying the deformation gradient
    F = I + ε to the original lattice matrix, while atomic fractional
    coordinates are kept fixed (ions are not relaxed at this stage).

    Parameters
    ----------
    filepath : str — path to the input POSCAR file (passed as argv[2])
    """

    poscar = read_POSCAR(filepath)
    crystal_system = get_2d_lattice_type(poscar["lattice_matrix"])
    strain_types = get_strain_types(crystal_system)

    print(f"\nDetected crystal system: {crystal_system}")
    print(f"Strain types to compute: {strain_types}\n")

    mapping = mapping_elements(poscar["elements"], poscar["atom_counts"], poscar["positions_cartesian"],
                               poscar["positions_direct"], poscar["species"], poscar["selective_dynamics"],
                               poscar["flags"])
    labels = define_labels(mapping["elements"], mapping["atom_counts"])

    # Write unstrained reference structure
    unstrain_path = 'unstrain'
    os.makedirs(unstrain_path, exist_ok=True)
    write_POSCAR(os.path.join(unstrain_path, 'POSCAR'), poscar["lattice_matrix"], mapping["elements"], mapping["atom_counts"],
                 poscar["positions_cartesian"], poscar["positions_direct"], mapping["selective_dynamics"],
                 mapping["flags"], labels)

    # Write strained structures
    for strain_type in strain_types:
        for strain in STRAIN_RANGE:
            strain_path = os.path.join(strain_type, f"strain{strain:+.2f}")
            os.makedirs(strain_path, exist_ok=True)
            strain_matrix = build_strain_matrix(strain_type, strain)
            new_lattice_matrix = applying_strain(poscar["lattice_matrix"], strain_matrix)
            write_POSCAR(os.path.join(strain_path, 'POSCAR'), new_lattice_matrix, mapping["elements"], mapping["atom_counts"],
                         mapping["positions_cartesian"], mapping["positions_direct"], mapping["selective_dynamics"],
                         mapping["flags"], labels)

    print(f"Done. Strained POSCARs written for {len(strain_types)} strain types "
          f"× {len(STRAIN_RANGE)} strain values.\n")


def mode_post():
    """Run the post-processing mode: parse OUTCAR files and compute the elastic tensor.

    Reads the reference structure from unstrain/POSCAR to determine the crystal
    system and in-plane area. Then:
    1. Parses all strained OUTCAR files via collect_fitting_coef()
    2. Assembles the elastic tensor via obtain_elastic_tensor()
    3. Writes the tensor to Elastic.dat and prints it
    4. Checks mechanical stability via eigenvalue positivity
    5. Computes and writes angular Young's modulus, Poisson's ratio, and shear modulus
    """

    unstrain_poscar = os.path.join('unstrain', 'POSCAR')
    poscar = read_POSCAR(unstrain_poscar)

    crystal_system = get_2d_lattice_type(poscar["lattice_matrix"])
    strain_types = get_strain_types(crystal_system)

    print(f"\nDetected crystal system: {crystal_system}")

    area_vector = np.cross(poscar["lattice_matrix"][0], poscar["lattice_matrix"][1])
    area = np.linalg.norm(area_vector)

    # Collect fitted elastic constants
    constants = collect_fitting_coef(strain_types, STRAIN_RANGE, area)

    # Assemble full elastic tensor
    elastic_tensor = obtain_elastic_tensor(constants, crystal_system)
    if elastic_tensor is None:
        print("ERROR! Elastic tensor could not be assembled. Exiting.")
        exit(1)

    write_elastic_tensor(elastic_tensor)

    # Mechanical stability check
    stable = check_stability(elastic_tensor, poscar["lattice_matrix"], area_vector, area)
    if stable:
        print("This material is mechanically stable.\n")
    else:
        print("This material is mechanically unstable!!\n")
        exit(0)

    # Angular mechanical properties
    properties = compute_mechanical_properties(elastic_tensor)
    write_mechanical_properties(properties)
    print("Mechanical properties written to Young.dat, Poisson.dat, Shear.dat.\n")


def main():
    """Parse command-line arguments and dispatch to the appropriate mode.

    Usage:
        ElasticTensor2D.py pre  <structure_file>
        ElasticTensor2D.py post
        ElasticTensor2D.py -h | --help
    """

    if '-h' in argv or '--help' in argv or len(argv) < 2:
        usage()

    mode = argv[1]

    if mode == 'pre':
        if len(argv) < 3:
            print("ERROR! 'pre' mode requires a structure file as argument.")
            usage()
        mode_pre(argv[2])
    elif mode == 'post':
        mode_post()
    else:
        print(f"ERROR! Unknown mode: '{mode}'. Use 'pre' or 'post'.")
        usage()


if __name__ == "__main__":
    main()
