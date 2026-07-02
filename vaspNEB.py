#!/usr/bin/env python

from sys import argv, exit
import os
import readline
import numpy as np


def usage():
    """Print usage information and exit."""
    print("""
Usage: vaspNEB.py <initial> <final> <images> [OPTIONS]

This script generates NEB image folders with POSCAR files by interpolating between two endpoint structures.
The initial and final structures must be provided as POSCAR files, and the number of intermediate images (excluding endpoints) must be specified.
By default, the script uses the IDPP method for interpolation, which produces better initial paths and reduces NEB iterations.
The -linear option can be used to switch to simple linear interpolation instead.

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
    lattice_matrix   : np.ndarray, shape (3, 3) — row vectors of the lattice in Å
    positions_direct : np.ndarray, shape (N, 3) — fractional coordinates

    Returns
    -------
    positions_cartesian : np.ndarray, shape (N, 3) — Cartesian coordinates in Å
    """

    positions = positions_direct % 1.0
    positions_cartesian = positions @ lattice_matrix

    return positions_cartesian


def cartesian_to_direct(lattice_matrix, positions_cartesian):
    """Convert Cartesian coordinates to fractional (Direct) coordinates.

    Uses the relation:  r_direct = r_cart @ lattice_matrix^-1

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


def define_labels(elements, atom_counts):
    """Generate per-atom labels used as inline comments in the POSCAR position block.

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
        o.write("Generated by vaspNEB.py\n")
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


def validate_POSCARs(initial, final):
    """Validate that two POSCAR dictionaries describe the same system.

    Checks element symbols, per-element atom counts, and warns when lattice
    matrices differ (dynamic cell). Exits with an error message on any
    structural mismatch.

    Parameters
    ----------
    initial : dict — output of read_POSCAR for the initial image
    final : dict — output of read_POSCAR for the final image

    Returns
    -------
    dyn_cell : bool — True if lattice matrices differ between images
    """

    # Check species list
    if initial["elements"] != final["elements"]:
        print("ERROR! Element symbols do not match between the two POSCARs.")
        print(f"  Initial : {initial['elements']}")
        print(f"  Final   : {final['elements']}")
        exit(1)

    # Check per-element atom counts
    if initial["atom_counts"] != final["atom_counts"]:
        print("ERROR! Atom counts do not match between the two POSCARs.")
        for el, n1, n2 in zip(initial["elements"], initial["atom_counts"], final["atom_counts"]):
            if n1 != n2:
                print(f"  Element {el}: initial = {n1}, final = {n2}")
        exit(1)

    # Check total atom count (redundant but explicit)
    if initial["total_atoms"] != final["total_atoms"]:
        print("ERROR! Total atom counts do not match between the two POSCARs.")
        print(f"  Initial : {initial['total_atoms']}")
        print(f"  Final   : {final['total_atoms']}")
        exit(1)

    # Check lattice matrices — warn but allow (dynamic cell)
    dynamic_cell = False
    if not np.allclose(initial["lattice_matrix"], final["lattice_matrix"], atol=1e-6):
        print("WARNING: Lattice matrices differ between the two POSCARs.")
        print("  Dynamic cell interpolation will be applied.")
        print("  Initial lattice (Å):")
        for row in initial["lattice_matrix"]:
            print(f"    {row[0]:12.8f}  {row[1]:12.8f}  {row[2]:12.8f}")
        print("  Final lattice (Å):")
        for row in final["lattice_matrix"]:
            print(f"    {row[0]:12.8f}  {row[1]:12.8f}  {row[2]:12.8f}")
        print("  I hope you know what you are doing.")
        dynamic_cell = True

    return dynamic_cell


def mic_displacement(position_cartesian_i, position_cartesian_f, lattice_matrix):
    """Compute minimum-image-convention (MIC) displacement vectors.

    Converts the displacement to fractional coordinates, wraps each component
    to the range (-0.5, 0.5], then converts back to Cartesian. This ensures
    that the interpolation always takes the shortest path across any periodic
    boundary.

    Parameters
    ----------
    position_cartesian_i : np.ndarray (N, 3) — Cartesian positions of initial image in Å
    position_cartesian_f : np.ndarray (N, 3) — Cartesian positions of final image in Å
    lattice_matrix : np.ndarray (3, 3) — lattice row vectors in Å

    Returns
    -------
    displacement_cartesian : np.ndarray (N, 3) — MIC Cartesian displacement vectors in Å
    """

    displacement_cartesian_raw = position_cartesian_f - position_cartesian_i
    displacement_direct = displacement_cartesian_raw @ np.linalg.inv(lattice_matrix)
    displacement_direct -= np.round(displacement_direct)   # MIC wrap per atom independently
    displacement_cartesian = displacement_direct @ lattice_matrix

    return displacement_cartesian


def interpolate_linear(initial, final, n_images, dynamic_cell):
    """Generate NEB images by linear interpolation in Cartesian space with MIC.

    For fixed-cell structures, all images share the initial lattice matrix.
    For dynamic cell structures, the lattice matrix is also linearly
    interpolated between the initial and final values.

    Parameters
    ----------
    initial      : dict — read_POSCAR output for the initial image
    final      : dict — read_POSCAR output for the final image
    n_images : int  — number of intermediate images (excluding endpoints)
    dynamic_cell : bool — whether to interpolate the lattice matrix

    Returns
    -------
    images : list[dict] — list of (n_images + 2) dicts, each with keys:
        lattice_matrix   : np.ndarray (3, 3)
        positions_direct : np.ndarray (N, 3)
    """

    total_frames = n_images + 2
    lattice_matrix_initial = initial["lattice_matrix"]
    lattice_matrix_final = final["lattice_matrix"]
    positions_initial = initial["positions_cartesian"]

    displacement_cartesian = mic_displacement(positions_initial, final["positions_cartesian"], lattice_matrix_initial)

    images = []
    for k in range(total_frames):
        t = k / (total_frames - 1)
        lattice_k = lattice_matrix_initial + t * (lattice_matrix_final - lattice_matrix_initial) if dynamic_cell else lattice_matrix_initial.copy()
        positions_cartesian_k = positions_initial + t * displacement_cartesian
        positions_direct_k = cartesian_to_direct(lattice_k, positions_cartesian_k)
        images.append({"lattice_matrix": lattice_k,
                       "positions_direct": positions_direct_k})

    return images


def interpolate_idpp(initial, final, n_images, dynamic_cell):
    """Generate NEB images using the IDPP method (Image Dependent Pair Potential).

    IDPP minimises a cost function based on pairwise inter-atomic distances
    so that the initial path better resembles the true reaction pathway.
    This reduces the number of NEB iterations required in practice.

    Reference: Smidstrup et al., J. Chem. Phys. 140, 214106 (2014).
               https://doi.org/10.1063/1.4878664

    For dynamic cell cases the method falls back to linear cell interpolation
    while applying IDPP only to the atomic coordinates.

    Parameters
    ----------
    initial      : dict — read_POSCAR output for the initial image
    final      : dict — read_POSCAR output for the final image
    n_images : int  — number of intermediate images (excluding endpoints)
    dynamic_cell : bool — whether to interpolate the lattice matrix

    Returns
    -------
    images : list[dict] — list of (n_images + 2) dicts, each with keys:
        lattice_matrix   : np.ndarray (3, 3)
        positions_direct : np.ndarray (N, 3)
    """

    try:
        from scipy.optimize import minimize
    except ImportError:
        print("WARNING: SciPy not found. Falling back to linear interpolation.")
        return interpolate_linear(initial, final, n_images, dynamic_cell)

    total_frames = n_images + 2
    lattice_matrix_initial = initial["lattice_matrix"]
    lattice_matrix_final = final["lattice_matrix"]
    positions_initial = initial["positions_cartesian"]
    positions_final = final["positions_cartesian"]
    n_atoms = initial["total_atoms"]

    # Precompute linearly interpolated lattices for each frame
    lattices = []
    for k in range(total_frames):
        t = k / (total_frames - 1)
        lattice_matrix_k = lattice_matrix_initial + t * (lattice_matrix_final - lattice_matrix_initial) if dynamic_cell else lattice_matrix_initial.copy()
        lattices.append(lattice_matrix_k)

    # Seed with linear interpolation (MIC-aware)
    displacement_cartesian = mic_displacement(positions_initial, positions_final, lattice_matrix_initial)
    linear_cartesian = np.array([positions_initial + (k / (total_frames - 1)) * displacement_cartesian
                             for k in range(total_frames)])   # (F, N, 3)

    # Target pairwise distances: linear interpolation of endpoint distances
    def pairwise_distances(cart):
        """Return upper-triangle inter-atomic distances for one frame."""
        diff = cart[:, np.newaxis, :] - cart[np.newaxis, :, :]  # (N, N, 3)
        return np.sqrt(np.sum(diff ** 2, axis=-1))

    d_ini = pairwise_distances(positions_initial)
    d_fin = pairwise_distances(positions_final)

    # IDPP cost function: sum over images and atom pairs of (d - d_target)^2 / d_target^4
    def idpp_cost(x_flat):
        x = x_flat.reshape(n_images, n_atoms, 3)
        cost = 0.0
        frames_mid = [linear_cartesian[0]] + [x[k] for k in range(n_images)] + [linear_cartesian[-1]]
        for k, cart_k in enumerate(frames_mid):
            t = k / (total_frames - 1)
            d_target = d_ini + t * (d_fin - d_ini)
            d_target = np.where(d_target < 1e-10, 1e-10, d_target)
            diff = cart_k[:, np.newaxis, :] - cart_k[np.newaxis, :, :]
            d_k = np.sqrt(np.sum(diff ** 2, axis=-1))
            mask = np.triu(np.ones((n_atoms, n_atoms), dtype=bool), k=1)
            residual = (d_k[mask] - d_target[mask]) ** 2 / d_target[mask] ** 4
            cost += residual.sum()
        return cost

    x0 = linear_cartesian[1:-1].flatten()
    result = minimize(idpp_cost, x0, method='L-BFGS-B',
                      options={'maxiter': 1000, 'ftol': 1e-10, 'gtol': 1e-6})

    optimized_cartesian = result.x.reshape(n_images, n_atoms, 3)

    images = []
    for k in range(total_frames):
        lattice_matrix_k = lattices[k]
        if k == 0:
            cartesian_k = positions_initial
        elif k == total_frames - 1:
            cartesian_k = positions_final
        else:
            cartesian_k = optimized_cartesian[k - 1]
        positions_direct_k = cartesian_to_direct(lattice_matrix_k, cartesian_k)
        images.append({"lattice_matrix": lattice_matrix_k,
                       "positions_direct": positions_direct_k})

    return images


def define_dir(total_frames):
    """Generate directory names for the images based on the number of intermediate images.

    The directories are named '00', '01', ..., with zero-padding to ensure proper sorting.
    The total number of images (including endpoints) must not exceed 100.

    Parameters
    ----------
    total_frames : int — total number of images (including endpoints)

    Returns
    -------
    dir_names : list[str] — list of directory names for all images including endpoints
    """

    digits = len(str(total_frames)) + 1
    dir_names = [f"{str(k).zfill(digits)}" for k in range(total_frames)]

    return dir_names


def main():
    """Parse arguments, validate POSCARs, interpolate, and write image folders."""

    if '-h' in argv or '--help' in argv or len(argv) < 4:
        usage()

    poscar_initial = argv[1]
    poscar_final = argv[2]

    try:
        n_images = int(argv[3])
    except ValueError:
        print("ERROR! N_images must be an integer.")
        exit(1)

    if n_images < 1:
        print("ERROR! N_images must be at least 1.")
        exit(1)

    total_frames = n_images + 2
    use_idpp = '-linear' not in argv

    # Read POSCARs
    print(f"\nReading {poscar_initial} ...")
    initial = read_POSCAR(poscar_initial)
    print(f"Reading {poscar_final} ...")
    final = read_POSCAR(poscar_final)

    # Validate
    print("\nValidating structures ...")
    dynamic_cell = validate_POSCARs(initial, final)

    # Interpolate
    method = "IDPP" if use_idpp else "Linear"
    print(f"\nInterpolating {n_images} intermediate image(s) using {method} method ...")
    if use_idpp:
        images = interpolate_idpp(initial, final, n_images, dynamic_cell)
    else:
        images = interpolate_linear(initial, final, n_images, dynamic_cell)

    labels = define_labels(initial["elements"], initial["atom_counts"])

    # Write image directories
    print("\nWriting image directories ...")
    dir_names = define_dir(total_frames)
    for dir_name, image in zip(dir_names, images):
        if not os.path.isdir(dir_name):
            os.mkdir(dir_name)
        write_POSCAR(os.path.join(dir_name, "POSCAR"), image["lattice_matrix"], initial["elements"], initial["atom_counts"],
                     image["positions_cartesian"], image["positions_direct"], initial["selective_dynamics"],
                     initial["flags"], labels)

    print("\nOK, all set up here.")
    print(f"For later analysis, put OUTCARs in folders {dir_names[0]} and {dir_names[-1]}.")


if __name__ == "__main__":
    main()
