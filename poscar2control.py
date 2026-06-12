#!/usr/bin/env python

from sys import argv, exit
import os
import numpy as np


def usage():
    """Print usage information and exit."""
    print("""
Usage: poscar2control.py <input>

This script convert POSCAR format to CONTROL format for ShengBTE.

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


def get_vacuum_index():
    """
    Prompt the user to specify whether the system is a 2D material.
 
    If 2D, asks for the vacuum direction (a, b, or c) and returns the
    corresponding axis index. If 3D, returns None.
 
    Returns
    -------
    int or None
        0 for a, 1 for b, 2 for c, or None for 3D bulk material.
    """
    while True:
        dim = input("2D materials or not (Y/N): ").strip().lower()
        if dim[0] == 'y':
            while True:
                vacuum = input("Enter vacuum direction (a/b/c): ").strip().lower()
                if vacuum in ("a", "b", "c"):
                    break
                else:
                    print("Input must be a, b, or c!")
            return {"a": 0, "b": 1, "c": 2}[vacuum]
        elif dim[0] == 'n':
            return None
        else:
            print("Input must be Y or N!")
 
 
def get_ngrid(reciprocal_matrix, index):
    """
    Prompt the user for a q-point spacing and compute the q-point mesh.
 
    The mesh size along each reciprocal direction is computed as:
        n_i = round(|b_i| / (spacing * 2π))
    where |b_i| is the length of the i-th reciprocal lattice vector in
    units of 2π/Å and spacing is in units of 2π/Å.
    The vacuum direction (if any) is forced to 1.
 
    Parameters
    ----------
    reciprocal_matrix : ndarray (3, 3), reciprocal lattice vectors as rows
                        in units of 2π/Å
    index             : int or None, axis index of the vacuum direction,
                        or None for 3D
 
    Returns
    -------
    ngrid : list of int, q-point mesh size [n1, n2, n3]
    """
    while True:
        spacing = input("Enter q-point spacing in 2π/Å: ").strip()
        try:
            qspacing = float(spacing) * 2 * np.pi
            if qspacing > 0:
                ngrid = []
                for i in range(3):
                    if i == index:
                        ngrid.append(1)
                    else:
                        n = max(1, int(np.round(np.linalg.norm(reciprocal_matrix[i]) / qspacing)))
                        ngrid.append(n)
                return ngrid
            else:
                print("Spacing must be positive.")
        except ValueError:
            print("Invalid input. Please enter a number.")
 
 
def get_supercell_matrix():
    """
    Prompt the user for a supercell matrix used in the force constant calculation.
 
    Expects 3 positive integers corresponding to the supercell expansion
    along each lattice direction, matching the supercell used in VASP
    for the force constant calculation.
 
    Returns
    -------
    supercell_matrix : str, space-separated integers e.g. "4 4 4"
    """
    while True:
        supercell_matrix = input("Enter Supercell Matrix (3 elements): ")
        try:
            check_supercell = np.array(list(map(int, supercell_matrix.split())))
            if len(check_supercell) == 3 and all(val > 0 for val in check_supercell):
                return supercell_matrix
            else:
                print("Input must be 3 components and positive integers!")
        except ValueError:
            print("Invalid input. Please enter integer numbers separated by spaces.")
 
 
def get_phonon_flags():
    """
    Prompt the user for the phonon process order and generate the
    corresponding ShengBTE &flags namelist block.
 
    For 3-phonon: sets convergence=.true. and nthreads=-1 (use all threads).
    For 4-phonon: asks for CPU or GPU parallelization. CPU sets
    convergence=.true., GPU sets convergence=.false. (iterative solver
    not supported on GPU).
 
    Returns
    -------
    str : the closing portion of the &flags namelist block including &end
    """
    while True:
        n_phonon = input("Enter n phonon process: ")
        try:
            n_phonon = int(n_phonon)
            if n_phonon == 3:
                return """        convergence=.true.
        nthreads=-1
&end"""
            elif n_phonon == 4:
                while True:
                    device = input("Enter parallel devices (CPU/GPU): ").strip().upper()
                    if device[0] == 'C':
                        print("Your parallel devices use CPU.")
                        convergence = '.true.'
                        break
                    elif device[0] == 'G':
                        print("Your parallel devices use GPU.")
                        convergence = '.false.'
                        break
                    else:
                        print("Input must be CPU or GPU!")
                return f"""        convergence={convergence}
        four_phonon=.true.
        four_phonon_iteration={convergence}
&end"""
            else:
                print("Input must be 3 or 4!")
        except ValueError:
            print("Invalid input. Please enter integer numbers.")

 
def write_CONTROL(filepath, lattice_matrix, elements, atom_counts, positions_direct,
                  ngrid, supercell_matrix, phonon_flags):
    """
    Write the ShengBTE CONTROL file in Fortran namelist format.
 
    Constructs the &allocations, &crystal, &parameters, and &flags
    namelists and writes them to the specified file. The lattice factor
    lfactor is set to 0.1 to convert Angstrom to nanometers as required
    by ShengBTE.
 
    Parameters
    ----------
    filepath         : str, output file path
    lattice_matrix   : ndarray (3, 3), lattice vectors as rows in Angstrom
    elements         : list of str, element symbols in order
    atom_counts      : list of int, number of atoms per element
    positions_direct : ndarray (N, 3), fractional coordinates in [0, 1)
    ngrid            : list of int, q-point mesh [n1, n2, n3]
    supercell_matrix : str, space-separated supercell expansion integers
    phonon_flags     : str, closing &flags namelist content including &end
    """
    total_atoms = sum(atom_counts)
    elements_str = " ".join([f'"{element}"' for element in elements])
    types = " ".join([str(i + 1) for i, count in enumerate(atom_counts)
                      for _ in range(count)])
 
    control = "&allocations"
    control += f"""\n        nelements= {len(elements)},
        natoms= {total_atoms},
        ngrid(:)= {ngrid[0]} {ngrid[1]} {ngrid[2]}
&end
&crystal
        lfactor={.1:.16f},
"""
 
    for i, vec in enumerate(lattice_matrix, 1):
        control += f"        lattvec(:,{i})= {vec[0]:.16f}  {vec[1]:.16f}  {vec[2]:.16f},\n"
 
    control += f"""        elements= {elements_str}
        types= {types},
"""
 
    for i, position in enumerate(positions_direct, 1):
        comma = "," if i < total_atoms else ""
        control += f"        positions(:,{i})= {position[0]:.16f}  {position[1]:.16f}  {position[2]:.16f}{comma}\n"
 
    control += f"""        scell(:)= {supercell_matrix}
&end
&parameters
        T=300.
        scalebroad=1.0
&end
&flags
        nonanalytic=.false.
        isotopes=.true.
        autoisotopes=.true.
        nanowire=.false.
        onlyharmonic=.false.
"""
    control += phonon_flags
 
    with open(filepath, 'w') as o:
        o.write(control)
 
 
def main():
    """
    Parses arguments, reads the POSCAR, collects user inputs interactively, and writes the CONTROL file.
    """
    if '-h' in argv or '--help' in argv or len(argv) != 2:
        usage()
 
    poscar = read_POSCAR(argv[1])
 
    reciprocal_matrix = 2. * np.pi * np.linalg.inv(poscar["lattice_matrix"]).T
    vac_idx = get_vacuum_index()
    ngrid = get_ngrid(reciprocal_matrix, vac_idx)
    supercell_matrix = get_supercell_matrix()
    phonon_flags = get_phonon_flags()
 
    output_file = "CONTROL.initial"
    write_CONTROL(output_file, poscar["lattice_matrix"], poscar["elements"], poscar["atom_counts"],
                  poscar["positions_direct"], ngrid, supercell_matrix, phonon_flags)
 
    print(f"Written: {output_file}")
 
 
if __name__ == "__main__":
    main()
 
