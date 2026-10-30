#!/usr/bin/env python

from sys import argv, exit
import os, re
import readline
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

Method:
1) energy : energy-strain method
2) stress : stress-strain method (requires converged ENCUT, ~1.3*ENMAX)

This script was developed by Thanasee Thanasarnsurapong.
""")
    exit(0)


ANGSTROM = 1e-10
ALL_STRAIN = ['C11', 'C22', 'C11_C22_2C12', 'C66']
STRAIN_RANGE = np.linspace(-0.02, 0.02, 9)
ZERO_STRAIN_IDX = len(STRAIN_RANGE) // 2
METHOD_CHOICES = {'1': ['energy'], 'energy': ['energy'],
                  '2': ['stress'], 'stress': ['stress'],
                  '3': ['energy', 'stress'], 'both': ['energy', 'stress']}

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
                print("WARNING! Empty input — using default unique element order.")
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
        o.write("Generated by ElasticTensor2D.py\n")
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


def get_strain_types(crystal_system, method):
    """Return the list of strain deformation types required for the given crystal system and method.

    Energy-strain: square, rectangular, and hexagonal lattices require four
    strain types to determine C11, C22, C12, and C66. Oblique lattices
    additionally require two mixed strain types to resolve C16 and C26.

    Stress-strain: the three single-component strain types C11, C22, and C66
    give one column of the elastic tensor each, for every lattice type.

    Parameters
    ----------
    crystal_system : str — output of get_2d_lattice_type()
    method         : str — 'energy' or 'stress'

    Returns
    -------
    list[str] — strain type labels corresponding to keys in build_strain_matrix()
    """

    if method == 'stress':
        return ['C11', 'C22', 'C66']

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

    epsilon = strain
    delta = strain * 0.5

    strain_matrices = {
        'C11':          np.array([[epsilon, 0., 0.], [0., 0., 0.], [0., 0., 0.]], dtype=float),
        'C22':          np.array([[0., 0., 0.], [0., epsilon, 0.], [0., 0., 0.]], dtype=float),
        'C11_C22_2C12': np.array([[epsilon, 0., 0.], [0., epsilon, 0.], [0., 0., 0.]], dtype=float),
        'C66':          np.array([[0., delta, 0.], [delta, 0., 0.], [0., 0., 0.]], dtype=float),
        'C11_C66_2C16': np.array([[epsilon, delta, 0.], [delta, 0., 0.], [0., 0., 0.]], dtype=float),
        'C22_C66_2C26': np.array([[0., delta, 0.], [delta, epsilon, 0.], [0., 0., 0.]], dtype=float),
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
    """Extract the final energy, stress tensor, and convergence status from a VASP OUTCAR file.

    Convergence is determined by the presence of 'reached required accuracy'
    anywhere in the file. The final energy is the last value E(sigma->0) on the
    line containing 'energy  without entropy='. The final stress is the last
    'in kB' line, parsed with a regular expression so that fused fixed-width
    numbers (e.g. '-1234.56789-1234.56789') are split correctly.

    Parameters
    ----------
    filepath : str — path to the OUTCAR file

    Returns
    -------
    dict with keys:
        energy    : float or None         — total energy in eV
        stress    : np.ndarray (6,) or None — stress in kB, VASP order and sign (XX YY ZZ XY YZ ZX)
        converged : bool                  — True if the calculation reached required accuracy
    """

    outcar = {"energy": None, "stress": None, "converged": False}

    if not os.path.exists(filepath):
        print(f"ERROR!\nFile: {filepath} does not exist.")
        return outcar

    with open(filepath, 'r') as f:
        outcar_lines = f.readlines()

    outcar["converged"] = any('reached required accuracy' in line for line in outcar_lines)
    if not outcar["converged"]:
        print(f"WARNING! {filepath} has not reached required accuracy.")

    for line in reversed(outcar_lines):
        if outcar["energy"] is None and 'energy  without entropy=' in line:
            outcar["energy"] = float(line.split()[-1])
        if outcar["stress"] is None and line.strip().startswith('in kB'):
            values = re.findall(r'[-+]?\d+\.\d+', line)
            if len(values) == 6:
                outcar["stress"] = np.array(values, dtype=float)
        if outcar["energy"] is not None and outcar["stress"] is not None:
            break

    return outcar


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
            o.write(f"  {s:>+6.3f}   {u:>14.6f}\n")

    coef, _ = curve_fit(quadratic, strain_range, energy_per_area)
    constant = 2 * coef[0] * eVpA2_to_Npm

    return constant


def fitting_strain_stress(strain_type, strain_range, strain_stress, thickness):
    """Fit the strain-stress curves linearly and return one column of the elastic tensor.

    The VASP stress (kB, pressure sign convention) is converted to the 2D stress
    in N/m by flipping the sign and multiplying by the out-of-plane cell length:
        σ_2D = −σ_kB × 0.01 × L_z
    Each of σxx, σyy, and σxy is fit to σ(δ) = slope · δ + intercept, where the
    intercept absorbs any residual stress of the reference structure.

    The strain-stress data for each strain type is also written to a .dat file
    named StrainVsStress_<strain_type>.dat.

    Parameters
    ----------
    strain_type   : str               — label used for the output filename
    strain_range  : np.ndarray (M,)   — strain values δ
    strain_stress : np.ndarray (M, 6) — stress in kB, VASP order and sign
    thickness     : float             — out-of-plane cell length L_z in Å

    Returns
    -------
    np.ndarray (3,) — slopes of σxx, σyy, σxy in N/m
    """

    stress_2d = -strain_stress[:, [0, 1, 3]] * 0.01 * thickness  # kB·Å to N/m

    output_file = f"StrainVsStress_{strain_type}.dat"
    with open(output_file, 'w') as o:
        o.write(f"# Strain vs 2D stress — {strain_type}\n")
        o.write("# Strain         XX(N/m)         YY(N/m)         XY(N/m)\n")
        for s, (sxx, syy, sxy) in zip(strain_range, stress_2d):
            o.write(f"  {s:>+6.3f}   {sxx:>14.6f}  {syy:>14.6f}  {sxy:>14.6f}\n")

    slopes = np.polyfit(strain_range, stress_2d, 1)[0]

    return slopes


def collect_outcars(crystal_system, methods, strain_range):
    """Read every OUTCAR needed by the selected methods once.

    Parameters
    ----------
    crystal_system : str             — output of get_2d_lattice_type()
    methods        : list[str]       — selected methods ('energy' and/or 'stress')
    strain_range   : np.ndarray (M,) — strain values δ

    Returns
    -------
    dict[str, list[dict]] — read_OUTCAR() results per strain type, ordered as strain_range
    """

    strain_types = list(dict.fromkeys(strain_type for method in methods
                                      for strain_type in get_strain_types(crystal_system, method)))

    outcars = {}
    for strain_type in strain_types:
        outcars[strain_type] = [read_OUTCAR(os.path.join(strain_type, f"strain{strain:+.3f}", 'OUTCAR'))
                                for strain in strain_range]

    return outcars


def collect_fitting_coef(method, crystal_system, outcars, strain_range, area, thickness):
    """Fit each strain set of the given method from the collected OUTCAR data.

    Energy-strain requires a converged energy at every strain point.
    Stress-strain requires a converged stress tensor at every strain point.
    If any point fails, that strain type is skipped and its result is set
    to None.

    Parameters
    ----------
    method         : str                    — 'energy' or 'stress'
    crystal_system : str                    — output of get_2d_lattice_type()
    outcars        : dict[str, list[dict]]  — output of collect_outcars()
    strain_range   : np.ndarray (M,)        — strain values δ
    area           : float                  — in-plane cell area in Å**2
    thickness      : float                  — out-of-plane cell length L_z in Å

    Returns
    -------
    dict[str, float or np.ndarray or None]
        energy : fitted elastic constant (N/m) per strain type
        stress : slopes of σxx, σyy, σxy (N/m) per strain type
    """

    constants = {}

    for strain_type in get_strain_types(crystal_system, method):

        valid = True
        for strain, outcar in zip(strain_range, outcars[strain_type]):
            filepath = os.path.join(strain_type, f"strain{strain:+.3f}", 'OUTCAR')
            if not outcar["converged"]:  # missing or unconverged, already reported by read_OUTCAR()
                valid = False
                continue
            if method == 'energy' and outcar["energy"] is None:
                print(f"WARNING! Could not extract energy from {filepath}.")
                valid = False
            if method == 'stress' and outcar["stress"] is None:
                print(f"WARNING! Could not extract stress from {filepath}.")
                valid = False

        if not valid:
            print(f"WARNING! Skipping '{strain_type}' for {method}-strain — incomplete or unconverged calculations.")
            constants[strain_type] = None
            continue

        if method == 'energy':
            strain_energy = np.array([outcar["energy"] for outcar in outcars[strain_type]], dtype=float)
            constants[strain_type] = fitting_strain_energy(strain_type, strain_range, strain_energy, area)
        else:
            strain_stress = np.array([outcar["stress"] for outcar in outcars[strain_type]], dtype=float)
            constants[strain_type] = fitting_strain_stress(strain_type, strain_range, strain_stress, thickness)

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

    missing = [key for key, value in constants.items() if value is None]
    if missing:
        for key in missing:
            print(f"ERROR! {key} is not calculated.")
        return None

    C11 = constants['C11']
    C22 = constants['C22']
    C66 = constants['C66']
    C12 = (constants['C11_C22_2C12'] - C11 - C22) * 0.5
    C16 = 0.
    C26 = 0.

    if crystal_system == 'oblique':
        C16 = (constants['C11_C66_2C16'] - C11 - C66) * 0.5
        C26 = (constants['C22_C66_2C26'] - C22 - C66) * 0.5

    elastic_tensor = np.array([[C11, C12, C16],
                               [C12, C22, C26],
                               [C16, C26, C66]])

    return elastic_tensor


def obtain_elastic_tensor_stress(constants, crystal_system):
    """Assemble the 2D elastic tensor from the fitted strain-stress slopes.

    Each strain set gives one column of the elastic tensor:
        C11 set (εxx = δ) : C11, C12, C16
        C22 set (εyy = δ) : C12, C22, C26
        C66 set (γxy = δ) : C16, C26, C66
    The raw tensor is symmetrized as (C + Cᵀ)/2 and printed together with the
    asymmetry of the off-diagonal terms. For square, rectangular, and hexagonal
    lattices, C16 and C26 are then forced to zero by symmetry.

    Parameters
    ----------
    constants      : dict[str, np.ndarray or None] — output of collect_fitting_coef()
    crystal_system : str                           — output of get_2d_lattice_type()

    Returns
    -------
    np.ndarray (3, 3) or None — elastic tensor in N/m, or None if any column is missing
    """

    missing = [key for key, value in constants.items() if value is None]
    if missing:
        for key in missing:
            print(f"ERROR! {key} is not calculated.")
        return None

    raw_tensor = np.column_stack([constants['C11'], constants['C22'], constants['C66']])
    dC12 = np.abs(raw_tensor[0, 1] - raw_tensor[1, 0])
    dC16 = np.abs(raw_tensor[0, 2] - raw_tensor[2, 0])
    dC26 = np.abs(raw_tensor[1, 2] - raw_tensor[2, 1])

    elastic_tensor = (raw_tensor + raw_tensor.T) * 0.5

    print("\n# Elastic tensor from stress-strain (N/m, symmetrized, before forcing)\n"
          f"# Cell type: {crystal_system}\n"
          f"   {elastic_tensor[0,0]:>11.4f} {elastic_tensor[0,1]:>11.4f} {elastic_tensor[0,2]:>11.4f}\n"
          f"   {elastic_tensor[1,0]:>11.4f} {elastic_tensor[1,1]:>11.4f} {elastic_tensor[1,2]:>11.4f}\n"
          f"   {elastic_tensor[2,0]:>11.4f} {elastic_tensor[2,1]:>11.4f} {elastic_tensor[2,2]:>11.4f}\n"
          f"# Asymmetry before symmetrization: |dC12| = {dC12:.4f}, |dC16| = {dC16:.4f}, |dC26| = {dC26:.4f}")

    if np.abs(elastic_tensor[2, 2]) < 1.:
        print("WARNING! C66 ≈ 0 — check that the C66 set is sheared.")

    if crystal_system != 'oblique':
        elastic_tensor[0, 2] = elastic_tensor[2, 0] = 0.
        elastic_tensor[1, 2] = elastic_tensor[2, 1] = 0.
        print(f"C16, C26 forced to 0 for {crystal_system} cell")

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
    factor_2d = np.abs(lattice_matrix[2] @ vector_n) * 0.1  # Angstrom to nm

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

    degrees = np.arange(0.0, 360.0, 0.1)
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

    A = (S11 * cos**4. + S22 * sin**4.
         + (2. * S12 + S66) * cos**2. * sin**2.
         + 2. * S16 * cos**3. * sin
         + 2. * S26 * cos * sin**3.)
    B = ((S11 + S22 - S66) * cos**2. * sin**2.
         + S12 * (cos**4. + sin**4.)
         + (S26 - S16) * (cos**3. * sin - cos * sin**3.))
    C = (4. * (S11 + S22 - 2. * S12) * cos**2. * sin**2.
         + S66 * (cos**2. - sin**2.)**2.
         + 4. * (S26 - S16) * (cos**3. * sin - cos * sin**3.))

    young_modulus = 1. / A
    poisson_ratio = -1. * B / A
    shear_modulus = 1. / C

    return {"degrees":       degrees,
            "young_modulus": young_modulus,
            "poisson_ratio": poisson_ratio,
            "shear_modulus": shear_modulus}


def write_mechanical_properties(properties, suffix=""):
    """Write angular mechanical properties to three output .dat files.

    Writes Young<suffix>.dat, Poisson<suffix>.dat, and Shear<suffix>.dat. For
    Poisson's ratio, an additional absolute-value column is included if any
    negative values are present (indicating an auxetic material).

    Parameters
    ----------
        properties : dict — output of compute_mechanical_properties()
        suffix     : str  — filename suffix ('' or '_<method>')
    """

    degrees       = properties["degrees"]
    young_modulus = properties["young_modulus"]
    poisson_ratio = properties["poisson_ratio"]
    shear_modulus = properties["shear_modulus"]

    with open(f'Young{suffix}.dat', 'w') as o:
        o.write("# Young's Modulus\n")
        o.write("#  Degree(°)  Y(N/m)\n")
        for dg, y in zip(degrees, young_modulus):
            o.write(f" {dg:>6.2f}     {y:>12.8f}\n")

    with open(f'Poisson{suffix}.dat', 'w') as o:
        o.write("# Poisson's Ratio\n")
        if (poisson_ratio < 0.).any():
            o.write("#  Degree(°) v             |v|\n")
            for dg, nu in zip(degrees, poisson_ratio):
                o.write(f" {dg:>6.2f}   {nu:>12.8f}   {abs(nu):>12.8f}\n")
        else:
            o.write("#  Degree(°) v\n")
            for dg, nu in zip(degrees, poisson_ratio):
                o.write(f" {dg:>6.2f}   {nu:>12.8f}\n")

    with open(f'Shear{suffix}.dat', 'w') as o:
        o.write("# Shear Modulus\n")
        o.write("#  Degree(°)  G(N/m)\n")
        for dg, g in zip(degrees, shear_modulus):
            o.write(f" {dg:>6.2f}     {g:>12.8f}\n")


def write_elastic_tensor(elastic_tensor, crystal_system, method, suffix=""):
    """Write the 2D elastic tensor to Elastic<suffix>.dat and print it to stdout.

    The tensor is written in Voigt notation as a 3×3 matrix with a descriptive
    header indicating the method, the cell type, the component layout, and
    units (N/m).

    Parameters
    ----------
    elastic_tensor : np.ndarray (3, 3) — 2D elastic tensor in N/m
    crystal_system : str               — output of get_2d_lattice_type()
    method         : str               — 'energy' or 'stress'
    suffix         : str               — filename suffix ('' or '_<method>')
    """

    C = elastic_tensor
    header = (f"# Elastic tensor (N/m) — {method}-strain\n"
              f"# Cell type: {crystal_system}\n"
              "#     C11         C12         C16\n"
              "#     C12         C22         C26\n"
              "#     C16         C26         C66\n\n")
    rows = (f"   {C[0,0]:>11.4f} {C[0,1]:>11.4f} {C[0,2]:>11.4f}\n"
            f"   {C[1,0]:>11.4f} {C[1,1]:>11.4f} {C[1,2]:>11.4f}\n"
            f"   {C[2,0]:>11.4f} {C[2,1]:>11.4f} {C[2,2]:>11.4f}\n")

    with open(f'Elastic{suffix}.dat', 'w') as o:
        o.write(header + rows)

    print("\n" + header + rows)


def compare_methods(tensors, crystal_system):
    """Print and write the elastic constants from both methods side by side.

    The relative difference is (stress − energy) / energy in percent. C16 and
    C26 rows are included only for oblique lattices.

    Parameters
    ----------
    tensors        : dict[str, np.ndarray (3, 3)] — elastic tensors keyed by 'energy' and 'stress'
    crystal_system : str                          — output of get_2d_lattice_type()
    """

    components = [('C11', 0, 0), ('C22', 1, 1), ('C12', 0, 1), ('C66', 2, 2)]
    if crystal_system == 'oblique':
        components += [('C16', 0, 2), ('C26', 1, 2)]

    lines = ["# Comparison of elastic constants (N/m)",
             f"# Cell type: {crystal_system}",
             "#  Cij      energy-strain   stress-strain    diff(%)"]
    for name, i, j in components:
        energy = tensors['energy'][i, j]
        stress = tensors['stress'][i, j]
        diff = f"{(stress - energy) / energy * 100:>10.2f}" if np.abs(energy) > 1e-8 else f"{'—':>10s}"
        lines.append(f"   {name}   {energy:>14.4f}  {stress:>14.4f}  {diff}")
    text = "\n".join(lines) + "\n"

    with open('Elastic_compare.dat', 'w') as o:
        o.write(text)

    print("\n" + text)


def select_method():
    """Prompt the user to select the calculation method until a valid choice is entered.

    Accepts the menu number or the method name (case-insensitive).

    Returns
    -------
    list[str] — ['energy'], ['stress'], or ['energy', 'stress']
    """

    print("\nSelect the calculation method:\n"
          "  1) energy-strain (energy)\n"
          "  2) stress-strain (stress)\n"
          "  3) both          (both)")
    while True:
        choice = input("Enter choice: ").strip().lower()
        if choice in METHOD_CHOICES:
            return list(METHOD_CHOICES[choice])
        print("ERROR! Invalid choice. Enter 1, 2, 3 or energy, stress, both.")


def write_METHOD(methods):
    """Write the selected method to the METHOD file as a canonical name.

    Parameters
    ----------
    methods : list[str] — output of select_method()
    """

    with open('METHOD', 'w') as o:
        o.write(("both" if len(methods) == 2 else methods[0]) + "\n")


def read_METHOD():
    """Read the calculation method from the METHOD file.

    Accepts the same keys as select_method() (menu number or method name).

    Returns
    -------
    list[str] or None — selected methods, or None if the file is missing or invalid
    """

    if not os.path.exists('METHOD'):
        return None

    with open('METHOD', 'r') as f:
        content = f.read().strip()

    if content.lower() in METHOD_CHOICES:
        return list(METHOD_CHOICES[content.lower()])

    print(f"WARNING! Invalid METHOD file: '{content}'")
    return None


def mode_pre(filepath):
    """Run the pre-processing mode: generate strained POSCAR files for all strain types.

    Reads the input structure, detects the 2D crystal system, prompts for the
    calculation method, and writes:
    - METHOD            : the selected method (energy, stress, or both)
    - unstrain/POSCAR   : the reference (zero-strain) structure
    - <strain_type>/strain<±δ>/POSCAR : one POSCAR per (strain_type, δ) combination

    The strain types are the union of those required by the selected methods.
    The strained lattice is generated by applying the deformation gradient
    F = I + ε to the original lattice matrix, while atomic fractional
    coordinates are kept fixed (ions are not relaxed at this stage).

    Parameters
    ----------
    filepath : str — path to the input POSCAR file (passed as argv[2])
    """

    poscar = read_POSCAR(filepath)
    crystal_system = get_2d_lattice_type(poscar["lattice_matrix"])
    print(f"\nDetected cell type: {crystal_system}")

    methods = select_method()
    write_METHOD(methods)

    strain_types = list(dict.fromkeys(strain_type for method in methods
                                      for strain_type in get_strain_types(crystal_system, method)))
    print(f"\nStrain types to compute: {strain_types}\n")

    mapping = mapping_elements(poscar["elements"], poscar["atom_counts"], poscar["positions_cartesian"],
                               poscar["positions_direct"], poscar["species"], poscar["selective_dynamics"],
                               poscar["flags"])
    labels = define_labels(mapping["elements"], mapping["atom_counts"])

    # Write unstrained reference structure
    unstrain_path = 'unstrain'
    os.makedirs(unstrain_path, exist_ok=True)
    write_POSCAR(os.path.join(unstrain_path, 'POSCAR'), poscar["lattice_matrix"], mapping["elements"], mapping["atom_counts"],
                 mapping["positions_cartesian"], mapping["positions_direct"], poscar["selective_dynamics"],
                 mapping["flags"], labels)

    # Write strained structures
    for strain_type in strain_types:
        for strain in STRAIN_RANGE:
            strain_path = os.path.join(strain_type, f"strain{strain:+.3f}")
            os.makedirs(strain_path, exist_ok=True)
            strain_matrix = build_strain_matrix(strain_type, strain)
            new_lattice_matrix = applying_strain(poscar["lattice_matrix"], strain_matrix)
            write_POSCAR(os.path.join(strain_path, 'POSCAR'), new_lattice_matrix, mapping["elements"], mapping["atom_counts"],
                         mapping["positions_cartesian"], mapping["positions_direct"], poscar["selective_dynamics"],
                         mapping["flags"], labels)

    print(f"Done. Strained POSCARs written for {len(strain_types)} strain types "
          f"× {len(STRAIN_RANGE)} strain values.\n")


def mode_post():
    """Run the post-processing mode: parse OUTCAR files and compute the elastic tensor.

    Reads the reference structure from unstrain/POSCAR to determine the crystal
    system, in-plane area, and out-of-plane thickness, and the method from the
    METHOD file (or the prompt if the file is missing or invalid). Then:
    1. Reads all required OUTCAR files once via collect_outcars()
    2. For each method:
       a. Fits the strain sets via collect_fitting_coef()
       b. Assembles the elastic tensor via obtain_elastic_tensor() or obtain_elastic_tensor_stress()
       c. Writes the tensor to Elastic<suffix>.dat and prints it
       d. Checks mechanical stability via eigenvalue positivity
       e. Computes and writes angular Young's modulus, Poisson's ratio, and shear modulus
    3. Compares both methods if both were selected

    A method that fails or is unstable does not stop the other method.
    """

    unstrain_poscar = os.path.join('unstrain', 'POSCAR')
    poscar = read_POSCAR(unstrain_poscar)

    crystal_system = get_2d_lattice_type(poscar["lattice_matrix"])
    print(f"\nDetected cell type: {crystal_system}")

    methods = read_METHOD()
    if methods is None:
        methods = select_method()
    else:
        print(f"Method (from METHOD): {'both' if len(methods) == 2 else methods[0]}")

    area_vector = np.cross(poscar["lattice_matrix"][0], poscar["lattice_matrix"][1])
    area = np.linalg.norm(area_vector)
    thickness = np.abs(poscar["lattice_matrix"][2] @ (area_vector / area))

    outcars = collect_outcars(crystal_system, methods, STRAIN_RANGE)

    tensors = {}
    for method in methods:

        suffix = f"_{method}" if len(methods) == 2 else ""
        print(f"\n========== {method}-strain method ==========")

        constants = collect_fitting_coef(method, crystal_system, outcars, STRAIN_RANGE, area, thickness)

        if method == 'energy':
            elastic_tensor = obtain_elastic_tensor(constants, crystal_system)
        else:
            elastic_tensor = obtain_elastic_tensor_stress(constants, crystal_system)

        if elastic_tensor is None:
            print(f"ERROR! Elastic tensor could not be assembled for {method}-strain.")
            continue

        write_elastic_tensor(elastic_tensor, crystal_system, method, suffix)
        tensors[method] = elastic_tensor

        # Mechanical stability check
        stable = check_stability(elastic_tensor, poscar["lattice_matrix"], area_vector, area)
        if stable:
            print("This material is mechanically stable.\n")
        else:
            print("This material is mechanically unstable!!\n")
            continue

        # Angular mechanical properties
        properties = compute_mechanical_properties(elastic_tensor)
        write_mechanical_properties(properties, suffix)
        print(f"Mechanical properties written to Young{suffix}.dat, Poisson{suffix}.dat, Shear{suffix}.dat.\n")

    if len(tensors) == 2:
        compare_methods(tensors, crystal_system)


def main():
    """Parse command-line arguments and dispatch to the appropriate mode."""

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
