#!/usr/bin/env python

from sys import argv, exit
import os
import re
import readline
import numpy as np
import h5py as h5


def usage():
    """Print usage information and exit."""
    print("""
Usage: convergePhono3py.py

This script obtain lattice thermal conductivity depends on q-mesh from HDF5 files

This script was developed by Thanasee Thanasarnsurapong.
""")
    exit(0)


def find_kappa_files():
    """Find and return a sorted list of phono3py kappa HDF5 files in the current directory.

    Scans the current directory for files matching the pattern 'kappa-m*.hdf5',
    which are the default output files from phono3py thermal conductivity calculations.
    Files are sorted in ascending order by the mesh number extracted from the filename
    (e.g., kappa-m20.hdf5 < kappa-m30.hdf5 < kappa-m40.hdf5).

    Returns
    -------
    list of str
        Sorted list of matching filenames.

    Exits
    -----
    Exits with an error message if no matching files are found.
    """

    files = [f for f in os.listdir() if f.startswith("kappa-m") and f.endswith(".hdf5")]
    if not files:
        print("ERROR! kappa hdf5 files are not found.")
        exit(0)
    return sorted(files, key=mesh_number)


def mesh_number(filename):
    """Extract the integer mesh number from a kappa-mXXX.hdf5 filename."""
    match = re.search(r"kappa-m(\d+)\.hdf5", filename)
    return int(match.group(1)) if match else float('inf')


def load(f, key):
    """Load a dataset from an open HDF5 file and suppress numerical noise."""
    if key not in f:
        return None
    arr = np.array(f[key])
    return np.where(arr < 1e-12, 0.0, arr)


def read_HDF5(filepath):
    """Read thermal conductivity data from a phono3py HDF5 output file.

    Loads the q-mesh, temperature grid, and all available kappa arrays from the file.
    Values below 1e-12 are set to zero to suppress numerical noise.
    Keys not present in the file are returned as None.

    Supported phono3py calculation modes and their corresponding keys:
        --br / --lbte          : 'kappa', 'kappa_RTA'
        --wigner --br or --lbte: 'kappa_C'
        --wigner --br          : 'kappa_P_RTA', 'kappa_TOT_RTA'
        --wigner --lbte        : 'kappa_P_exact', 'kappa_TOT_exact'

    Parameters
    ----------
    filepath : str
        Path to the phono3py HDF5 file (e.g., 'kappa-m20.hdf5').

    Returns
    -------
    dict
        Dictionary with keys: 'mesh', 'temperature', 'kappa', 'kappa_RTA',
        'kappa_C', 'kappa_P_RTA', 'kappa_TOT_RTA', 'kappa_P_exact', 'kappa_TOT_exact'.
        Each value is a numpy array or None if not present in the file.
    """

    with h5.File(filepath, 'r') as f:
        data = {'mesh'           : np.array(f["mesh"])        if 'mesh'        in f else None,
                'temperature'    : np.array(f["temperature"]) if 'temperature' in f else None,
                # --br / --lbte
                'kappa'          : load(f, 'kappa'),
                'kappa_RTA'      : load(f, 'kappa_RTA'),
                # --wigner --br or --wigner --lbte
                'kappa_C'        : load(f, 'kappa_C'),
                # --wigner --br
                'kappa_P_RTA'    : load(f, 'kappa_P_RTA'),
                'kappa_TOT_RTA'  : load(f, 'kappa_TOT_RTA'),
                # --wigner --lbte
                'kappa_P_exact'  : load(f, 'kappa_P_exact'),
                'kappa_TOT_exact': load(f, 'kappa_TOT_exact')}

    return data


def validate(data, filepath):
    """Check that the HDF5 file contains at least one usable set of kappa variables.

    The script requires either:
        - 'kappa' (from --lbte), or
        - the complete Wigner-RTA set: 'kappa_C', 'kappa_P_RTA', 'kappa_TOT_RTA'
          (from --wigner --br)

    Parameters
    ----------
    data : dict
        Data dictionary returned by read_HDF5().
    filepath : str
        Filename used in the error message if validation fails.

    Exits
    -----
    Exits with an error message if neither condition is satisfied.
    """
    k   = data['kappa']
    kC  = data['kappa_C']
    kPR = data['kappa_P_RTA']
    kTR = data['kappa_TOT_RTA']
    if k is None and (kC is None or kPR is None or kTR is None):
        print(f"ERROR! Essential variables are not found in {filepath}.")
        exit(0)


def choose_temperature(temperature, filepath, last_temperature, current_temp):
    """Select the target temperature for a given file.

    If the temperature grid of the current file differs from the previous file
    (or this is the first file), the user is prompted to choose a temperature.
    If the grid is unchanged, the previously chosen temperature is reused silently.
    If the file contains only one temperature, it is selected automatically.

    Parameters
    ----------
    temperature : numpy.ndarray
        Temperature grid from the current HDF5 file.
    filepath : str
        Filename shown in the prompt when asking the user to choose.
    last_temperature : numpy.ndarray or None
        Temperature grid from the previous file, or None for the first file.
    current_temp : float or None
        The temperature chosen for the previous file, reused if grid is unchanged.

    Returns
    -------
    float
        The selected target temperature in Kelvin.
    """
    if last_temperature is None or not np.array_equal(temperature, last_temperature):
        if len(temperature) == 1:
            return temperature[0]
        else:
            print(f"\n[{filepath}] List of temperatures:")
            print("   ".join(map(str, temperature)))
            while True:
                try:
                    target_temp = float(input("Select target temperature (K): "))
                    break
                except ValueError:
                    print("Invalid input! Please enter a number. Using the first temperature in the list.")
            return target_temp


def get_temp_index(temperature, target_temp, filepath):
    """Return the array index corresponding to the target temperature.

    Parameters
    ----------
    temperature : numpy.ndarray
        Temperature grid from the current HDF5 file.
    target_temp : float
        The target temperature in Kelvin.
    filepath : str
        Filename used in the error message if the temperature is not found.

    Returns
    -------
    int
        Index of target_temp in the temperature array.

    Exits
    -----
    Exits with an error message if target_temp is not found in the array.
    """
    index = np.where(temperature == target_temp)[0]
    if len(index) == 0:
        print(f"ERROR! Temperature {target_temp} not found in {filepath}.")
        exit(0)
    return index[0]


def extract_row(mesh, kappa_arr, temp_index, renorm_factor=1.0):
    """Extract one data row for a given temperature from a kappa array.
 
    Builds a tuple containing the three q-mesh dimensions followed by the
    six independent components of the thermal conductivity tensor at the
    specified temperature index, scaled by renorm_factor.
 
    For 3D materials renorm_factor = 1.0 (no change).
    For 2D materials renorm_factor = (c · n̂) / t_eff, converting the
    phono3py periodic-cell kappa to the true 2D sheet conductivity.
 
    Parameters
    ----------
    mesh : numpy.ndarray
        Array of three integers [Q_x, Q_y, Q_z] representing the q-mesh dimensions.
    kappa_arr : numpy.ndarray
        Kappa array of shape (n_temperatures, 6), where the 6 components are
        xx, yy, zz, yz, xz, xy in W/m-K.
    temp_index : int
        Index of the target temperature in the kappa array.
    renorm_factor : float, optional
        Multiplicative renormalization factor applied to all 6 kappa components.
        Default is 1.0 (3D, no renormalization).
 
    Returns
    -------
    tuple of float
        (Q_x, Q_y, Q_z, xx, yy, zz, yz, xz, xy)
    """
    kappa_components = kappa_arr[temp_index, :6] * renorm_factor
    return (mesh[0], mesh[1], mesh[2],
            kappa_components[0], kappa_components[1], kappa_components[2],
            kappa_components[3], kappa_components[4], kappa_components[5])


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


def compute_2d_thickness(poscar):
    """Compute the effective 2D layer thickness using Alvarez vdW radii.
 
    Atomic positions are projected onto the unit normal of the ab-plane to
    correctly handle non-orthogonal cells where the c vector is tilted.
    The renormalization factor accounts for this tilt.
 
    Thickness formula:
        t_eff = z_range + r_vdW_top + r_vdW_bottom
 
    Renormalization factor:
        factor_2d = (c · n̂) / t_eff
        where n̂ = (a × b) / |a × b|  (unit normal to the ab-plane)
 
    Parameters
    ----------
    poscar : dict
        Dictionary returned by read_POSCAR().
 
    Returns
    -------
    factor_2d   : float
        Renormalization factor: kappa_2D = kappa_phono3py * factor_2d.
    t_eff       : float
        Effective layer thickness in Å (z_range + r_vdW_top + r_vdW_bottom).
    c_proj      : float
        Projection of the c vector onto the ab-plane normal in Å.
    z_range     : float
        Distance between outermost atomic planes along the normal in Å.
    projected   : ndarray, shape (nAtom,)
        Projected position of each atom along the ab-plane normal.
    """
    lattice_matrix      = poscar["lattice_matrix"]
    species             = poscar["species"]
    positions_cartesian = poscar["positions_cartesian"]
 
    # Unit normal to the ab-plane
    area_vector = np.cross(lattice_matrix[0], lattice_matrix[1])
    vector_n    = area_vector / np.linalg.norm(area_vector)
 
    # Projection of c onto the normal (scalar thickness of the periodic cell)
    c_proj = np.abs(lattice_matrix[2] @ vector_n)
 
    # Project all atomic positions onto the normal
    projected = positions_cartesian @ vector_n
 
    # Identify outermost atoms
    idx_top = np.argmax(projected)
    idx_bot = np.argmin(projected)
    z_range = projected[idx_top] - projected[idx_bot]
 
    # vdW radii; fall back to 2.00 Å if element not in table
    r_top = VDW_RADIUS.get(species[idx_top], 2.00)
    r_bot = VDW_RADIUS.get(species[idx_bot], 2.00)
 
    t_eff     = z_range + r_top + r_bot
    factor_2d = c_proj / t_eff
 
    return factor_2d, t_eff, c_proj, z_range, projected
 
 
def ask_dimensionality():
    """Interactively ask the user for the material dimensionality.
 
    For 2D materials, reads POSCAR, computes the effective layer thickness,
    and returns the renormalization factor kappa_2D = kappa_phono3py * factor_2d
    where factor_2d = (c · n̂) / t_eff.
 
    Returns
    -------
    renorm_factor : float
        Renormalization factor to multiply all kappa values by.
        1.0 for 3D; (c · n̂) / t_eff for 2D.
    renorm_info   : str or None
        Human-readable description of the renormalization written to summary file.
        None for 3D.
    """
    print("\nMaterial dimensionality:")
    print("  [1] 3D — no renormalization")
    print("  [2] 2D — renormalize kappa by effective layer thickness")
 
    while True:
        choice = input("Select (1 or 2): ").strip()
        if choice in ('1', '2'):
            break
        print("  Please enter 1 or 2.")
 
    if choice == '1':
        return 1.0, None
 
    # 2D: locate POSCAR
    poscar_path = 'POSCAR'
    if not os.path.exists('POSCAR'):
        print(f"ERROR! POSCAR file does not exist. Using 3D (no renormalization).")
        return 1.0, None
 
    poscar = read_POSCAR(poscar_path)
    factor_2d, t_eff, c_proj, z_range, _ = compute_2d_thickness(poscar)
 
    print("\n  2D renormalization summary:")
    print(f"    c projection onto ab-normal : {c_proj:.5f} Å")
    print(f"    z range (atom-atom)         : {z_range:.5f} Å")
    print(f"    Effective thickness         : {t_eff:.5f} Å  (z_range + r_vdW_top + r_vdW_bottom)")
    print(f"    Renormalization factor      : {factor_2d:.6f}")
    print(f"    kappa_2D = kappa_phono3py × {factor_2d:.6f}")
 
    renorm_info = (f"2D renormalization: c_proj = {c_proj:.5f} A, "
                   f"t_eff = {t_eff:.5f} A (z_range = {z_range:.5f} A), "
                   f"factor = {factor_2d:.6f}")
 
    return factor_2d, renorm_info


def write_dat(filename, rows, display=False, renorm_info=None):
    """Write kappa-vs-mesh data to a .dat file and optionally print to stdout.
 
    Each row contains the q-mesh dimensions (Q_x, Q_y, Q_z) and the six
    thermal conductivity tensor components (xx, yy, zz, yz, xz, xy) in W/m-K.
    For 2D materials, values are already renormalized before being passed in.
 
    Parameters
    ----------
    filename : str
        Output filename (e.g., 'KappaVsMesh.dat').
    rows : numpy.ndarray
        2D array of shape (n_meshes, 9) where each row is
        (Q_x, Q_y, Q_z, xx, yy, zz, yz, xz, xy).
    display : bool, optional
        If True, also print the filename, header, and data to stdout.
        Default is False.
    renorm_info : str or None, optional
        If provided, written as a comment line in the .dat file header
        describing the 2D renormalization parameters. Default is None.
    """
    header1 = "# Thermal conductivity(W/m-K) vs Q-Mesh"
    header2 = "#  Q_x   Q_y   Q_z        xx          yy          zz          yz          xz          xy"
    row_fmt = (" {0:>5.0f} {1:>5.0f} {2:>5.0f}"
               "  {3:>10.3f}  {4:>10.3f}  {5:>10.3f}  {6:>10.3f}  {7:>10.3f}  {8:>10.3f}")
 
    with open(filename, 'w') as o:
        o.write(header1 + "\n")
        if renorm_info is not None:
            o.write(renorm_info + "\n")
        o.write(header2 + "\n")
        for item in rows:
            o.write(row_fmt.format(*item) + "\n")
        o.write("\n")
 
    if display:
        print(filename)
        print(header1)
        print(header2)
        for item in rows:
            print(row_fmt.format(*item))
        print("")


def main():
    """Parses arguments, sort kappa HDF5 files, and write outputs."""
    if '-h' in argv or len(argv) != 1:
        usage()

    files = find_kappa_files()
    
    # Ask dimensionality once before processing any file
    renorm_factor, renorm_info = ask_dimensionality()

    # Accumulators: one list of row-tuples per kappa type
    collected = {'kappa'          : [],
                 'kappa_RTA'      : [],
                 'kappa_C'        : [],
                 'kappa_P_RTA'    : [],
                 'kappa_TOT_RTA'  : [],
                 'kappa_P_exact'  : [],
                 'kappa_TOT_exact': []}

    last_temperature = None
    target_temp      = None

    for filepath in files:
        data        = read_HDF5(filepath)
        validate(data, filepath)
        temperature = data['temperature']
        target_temp = choose_temperature(temperature, filepath, last_temperature, target_temp)
        last_temperature = temperature

        temp_index = get_temp_index(temperature, target_temp, filepath)
        mesh       = data['mesh']

        for key in collected:
            if data[key] is not None:
                collected[key].append(extract_row(mesh, data[key], temp_index, renorm_factor))

    # Output configuration: filename and whether to display to stdout
    output_map = {'kappa'          : ('KappaVsMesh.dat',           True),
                  'kappa_RTA'      : ('Kappa_RTAVsMesh.dat',       False),
                  'kappa_C'        : ('Kappa_CVsMesh.dat',         False),
                  'kappa_P_RTA'    : ('Kappa_P_RTAVsMesh.dat',     True),
                  'kappa_TOT_RTA'  : ('Kappa_TOT_RTAVsMesh.dat',   False),
                  'kappa_P_exact'  : ('Kappa_P_exactVsMesh.dat',   False),
                  'kappa_TOT_exact': ('Kappa_TOT_exactVsMesh.dat', False)}

    for key, (filename, display) in output_map.items():
        rows = collected[key]
        if rows:
            write_dat(filename, np.array(rows), display=display, renorm_info=renorm_info)


if __name__ == "__main__":
    main()
