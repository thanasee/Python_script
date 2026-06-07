#!/usr/bin/env python

from sys import argv, exit
import os
import numpy as np
from scipy.constants import e, angstrom

_eV_PER_A3 = e / angstrom ** 3
_KBAR_TO_EV_PER_A3 = 1.0 / _eV_PER_A3 * 1e8


def usage():
    """Print usage information and exit."""
    print("""
Usage: mlab2extxyz.py <ML_AB input> <output.extxyz>

This script exports ML_AB data file to extended XYZ (.extxyz) format.
It automatically converts stress from kbar to eV/A^3
with a negative sign applied.

Stress conversion:
    stress(eV/A^3) = - stress(kbar) / (e * angstrom ** 3) * 1e8

This script was developed by Thanasee Thanasarnsurapong.
""")
    exit(0)


def read_lines(input_file):
    """
    Read all lines from the ML_AB file.

    Parameters
    ----------
    input_file : str
        Path to the ML_AB input file.

    Returns
    -------
    list of str
        Lines of the file with trailing newlines stripped.
    """
    with open(input_file, 'r') as f:
        return [line.rstrip('\n') for line in f]


def find_line(lines, start, keyword):
    """
    Search for a keyword in lines starting from a given index.

    Parameters
    ----------
    lines : list of str
        All lines of the file.
    start : int
        Line index to begin the search from.
    keyword : str
        Substring to search for.

    Returns
    -------
    int
        Index of the first matching line, or -1 if not found.
    """
    for i in range(start, len(lines)):
        if keyword in lines[i]:
            return i
    return -1


def read_global_header(lines):
    """
    Read the global header section of the ML_AB file.

    Parameters
    ----------
    lines : list of str
        All lines of the file.

    Returns
    -------
    all_elements : list of str
        Element symbols declared in the global header.
    config_indices : list of int
        Line indices of all 'Configuration num.' markers.
    """
    atom_types_index = find_line(lines, 0, 'The atom types in the data file')
    if atom_types_index == -1:
        print("ERROR! Cannot find atom types header in ML_AB file.")
        exit(1)

    n_types_index = find_line(lines, 0, 'The maximum number of atom type')
    if n_types_index == -1:
        print("ERROR! Cannot find the maximum number of atom type header.")
        exit(1)

    try:
        max_atom_type = int(lines[n_types_index + 2].split()[0])
    except Exception:
        print("ERROR! Cannot read the number of atom types.")
        exit(1)

    all_elements = []
    line_index = atom_types_index + 2
    while len(all_elements) < max_atom_type:
        if line_index >= len(lines):
            print("ERROR! Unexpected end of file while reading atom types.")
            exit(1)
        row = lines[line_index].split()
        if row:
            all_elements.extend(row)
        line_index += 1
    all_elements = all_elements[:max_atom_type]

    config_indices = [i for i, line in enumerate(lines) if 'Configuration num.' in line]
    if not config_indices:
        print("ERROR! No configuration blocks are found in ML_AB file.")
        exit(1)

    return all_elements, config_indices


def read_species(lines, start, config_count):
    """
    Read atom types, counts, and expand to per-atom species list.

    Parameters
    ----------
    lines : list of str
        All lines of the file.
    start : int
        Line index of the 'Configuration num.' marker.
    config_count : int
        1-based configuration index (used in error messages).

    Returns
    -------
    species_per_atom : list of str
        Element symbol for each atom in order.
    n_atoms : int
        Total number of atoms in this configuration.
    """
    number_type_index = find_line(lines, start, 'The number of atom types')
    if number_type_index == -1:
        print(f"ERROR! Cannot find number of atom types for configuration {config_count}.")
        exit(1)
    n_types = int(lines[number_type_index + 2].split()[0])

    number_atom_index = find_line(lines, start, 'The number of atoms')
    if number_atom_index == -1:
        print(f"ERROR! Cannot find number of atoms for configuration {config_count}.")
        exit(1)
    n_atoms = int(lines[number_atom_index + 2].split()[0])

    atom_number_index = find_line(lines, start, 'Atom types and atom numbers')
    if atom_number_index == -1:
        print(f"ERROR! Cannot find atom types and atom numbers for configuration {config_count}.")
        exit(1)

    species = []
    atom_counts = []
    for i in range(atom_number_index + 2, atom_number_index + 2 + n_types):
        data = lines[i].split()
        if len(data) < 2:
            print(f"ERROR! Cannot parse atom type/count for configuration {config_count}.")
            exit(1)
        species.append(data[0])
        atom_counts.append(int(data[1]))

    species_per_atom = [sym for sym, count in zip(species, atom_counts) for _ in range(count)]
    if len(species_per_atom) != n_atoms:
        print(f"ERROR! The number of expanded species does not match the number of atoms "
              f"in configuration {config_count}.")
        exit(1)

    return species_per_atom, n_atoms


def read_lattice(lines, start, config_count):
    """
    Read primitive lattice vectors for a configuration.

    Parameters
    ----------
    lines : list of str
        All lines of the file.
    start : int
        Line index of the 'Configuration num.' marker.
    config_count : int
        1-based configuration index (used in error messages).

    Returns
    -------
    np.ndarray, shape (3, 3)
        Lattice vectors in Angstroms, one vector per row.
    """
    lattice_index = find_line(lines, start, 'Primitive lattice vectors (ang.)')
    if lattice_index == -1:
        print(f"ERROR! Cannot find lattice vectors for configuration {config_count}.")
        exit(1)
    return np.array(
        [[float(x) for x in lines[i].split()[:3]] for i in range(lattice_index + 2, lattice_index + 5)],
        dtype=float,
    )


def read_positions(lines, start, n_atoms, config_count):
    """
    Read Cartesian atomic positions for a configuration.

    Parameters
    ----------
    lines : list of str
        All lines of the file.
    start : int
        Line index of the 'Configuration num.' marker.
    n_atoms : int
        Number of atoms in this configuration.
    config_count : int
        1-based configuration index (used in error messages).

    Returns
    -------
    np.ndarray, shape (n_atoms, 3)
        Atomic positions in Angstroms.
    """
    position_index = find_line(lines, start, 'Atomic positions (ang.)')
    if position_index == -1:
        print(f"ERROR! Cannot find atomic positions for configuration {config_count}.")
        exit(1)
    return np.array(
        [[float(x) for x in lines[i].split()[:3]]
         for i in range(position_index + 2, position_index + 2 + n_atoms)],
        dtype=float,
    )


def read_energy(lines, start, config_count):
    """
    Read total DFT energy for a configuration.

    Parameters
    ----------
    lines : list of str
        All lines of the file.
    start : int
        Line index of the 'Configuration num.' marker.
    config_count : int
        1-based configuration index (used in error messages).

    Returns
    -------
    float
        Total energy in eV.
    """
    energy_index = find_line(lines, start, 'Total energy (eV)')
    if energy_index == -1:
        print(f"ERROR! Cannot find total energy for configuration {config_count}.")
        exit(1)
    return float(lines[energy_index + 2].split()[0])


def read_forces(lines, start, n_atoms, config_count):
    """
    Read atomic forces for a configuration.

    Parameters
    ----------
    lines : list of str
        All lines of the file.
    start : int
        Line index of the 'Configuration num.' marker.
    n_atoms : int
        Number of atoms in this configuration.
    config_count : int
        1-based configuration index (used in error messages).

    Returns
    -------
    np.ndarray, shape (n_atoms, 3)
        Atomic forces in eV/Angstrom.
    """
    force_index = find_line(lines, start, 'Forces (eV ang.^-1)')
    if force_index == -1:
        print(f"ERROR! Cannot find forces for configuration {config_count}.")
        exit(1)
    return np.array(
        [[float(x) for x in lines[i].split()[:3]]
         for i in range(force_index + 2, force_index + 2 + n_atoms)],
        dtype=float,
    )


def read_stress(lines, start, config_count):
    """
    Read stress tensor and convert from kbar to eV/Angstrom^3.

    The ML_AB file stores stress as two rows:
        Row at offset +4: xx  yy  zz
        Row at offset +8: xy  yz  zx

    The negative sign is applied to match the extxyz convention
    (compressive stress is negative).

    Conversion: stress(eV/A^3) = -stress(kbar) * 1e8 / (e / angstrom^3)

    Parameters
    ----------
    lines : list of str
        All lines of the file.
    start : int
        Line index of the 'Configuration num.' marker.
    config_count : int
        1-based configuration index (used in error messages).

    Returns
    -------
    np.ndarray, shape (3, 3)
        Full symmetric stress matrix in eV/Angstrom^3.
        Layout: [[xx, xy, zx], [xy, yy, yz], [zx, yz, zz]]
    """
    stress_index = find_line(lines, start, 'Stress (kbar)')
    if stress_index == -1:
        print(f"ERROR! Cannot find stress for configuration {config_count}.")
        exit(1)

    stress_xx_yy_zz = [float(x) for x in lines[stress_index + 4].split()[:3]]
    stress_xy_yz_zx = [float(x) for x in lines[stress_index + 8].split()[:3]]

    # Reorder ML_AB layout (xx yy zz / xy yz zx) into Voigt order: xx yy zz yz zx xy
    stress_voigt_kbar = np.array([
        stress_xx_yy_zz[0],  # xx
        stress_xx_yy_zz[1],  # yy
        stress_xx_yy_zz[2],  # zz
        stress_xy_yz_zx[1],  # yz
        stress_xy_yz_zx[2],  # zx
        stress_xy_yz_zx[0],  # xy
    ], dtype=float)

    stress_voigt = -stress_voigt_kbar * _KBAR_TO_EV_PER_A3

    # Expand Voigt [xx, yy, zz, yz, zx, xy] to full symmetric 3x3 matrix
    return np.array([
        [stress_voigt[0], stress_voigt[5], stress_voigt[4]],
        [stress_voigt[5], stress_voigt[1], stress_voigt[3]],
        [stress_voigt[4], stress_voigt[3], stress_voigt[2]],
    ], dtype=float)


def read_system_name(lines, start, config_count):
    """
    Read the system name label for a configuration.

    Parameters
    ----------
    lines : list of str
        All lines of the file.
    start : int
        Line index of the 'Configuration num.' marker.
    config_count : int
        1-based configuration index (used in error messages).

    Returns
    -------
    str
        System name string.
    """
    system_name_index = find_line(lines, start, 'System name')
    if system_name_index == -1:
        print(f"ERROR! Cannot find system name for configuration {config_count}.")
        exit(1)
    return lines[system_name_index + 2].strip()


def write_config(f, config_count, system_name, n_atoms,
                 lattice, positions, forces, energy, stress_matrix,
                 species_per_atom):
    """
    Write a single configuration frame to an open extxyz file.

    Parameters
    ----------
    f : file object
        Open output file in write mode.
    config_count : int
        1-based configuration index written to config_index field.
    system_name : str
        System name label.
    n_atoms : int
        Number of atoms.
    lattice : np.ndarray, shape (3, 3)
        Lattice vectors in Angstroms.
    positions : np.ndarray, shape (n_atoms, 3)
        Atomic positions in Angstroms.
    forces : np.ndarray, shape (n_atoms, 3)
        Atomic forces in eV/Angstrom.
    energy : float
        Total energy in eV.
    stress_matrix : np.ndarray, shape (3, 3)
        Full symmetric stress tensor in eV/Angstrom^3.
    species_per_atom : list of str
        Element symbol for each atom.
    """
    lattice_text = ' '.join(f'{v:.16f}' for v in lattice.reshape(-1))
    stress_text = ' '.join(f'{v:.16f}' for v in stress_matrix.reshape(-1))

    f.write(f'{n_atoms}\n')
    f.write(
        f'Lattice="{lattice_text}" '
        f'Properties=species:S:1:pos:R:3:forces:R:3 '
        f'energy={energy:.16f} '
        f'stress="{stress_text}" '
        f'pbc="T T T" '
        f'config_index={config_count} '
        f'system_name="{system_name}"\n'
    )
    for symbol, pos, frc in zip(species_per_atom, positions, forces):
        f.write(
            f'{symbol:>2s} '
            f'{pos[0]:20.16f} {pos[1]:20.16f} {pos[2]:20.16f} '
            f'{frc[0]:20.16f} {frc[1]:20.16f} {frc[2]:20.16f}\n'
        )


def convert(input_file, output_file):
    """
    Convert an ML_AB file to extended XYZ (.extxyz) format.

    Parameters
    ----------
    input_file : str
        Path to the ML_AB input file.
    output_file : str
        Path to the output .extxyz file.
    """
    lines = read_lines(input_file)
    _all_elements, config_indices = read_global_header(lines)

    with open(output_file, 'w') as o:
        for config_count, start in enumerate(config_indices, start=1):
            system_name = read_system_name(lines, start, config_count)
            species_per_atom, n_atoms = read_species(lines, start, config_count)
            lattice = read_lattice(lines, start, config_count)
            positions = read_positions(lines, start, n_atoms, config_count)
            energy = read_energy(lines, start, config_count)
            forces = read_forces(lines, start, n_atoms, config_count)
            stress_matrix = read_stress(lines, start, config_count)

            write_config(o, config_count, system_name, n_atoms,
                         lattice, positions, forces, energy, stress_matrix,
                         species_per_atom)

    print(f'\nWrite {len(config_indices)} configurations to {output_file}\n')


def main():
    """Parse arguments, convert to extended XYZ format, and write output"""
    if '-h' in argv or '--help' in argv or len(argv) != 3:
        usage()

    input_file = argv[1]
    output_file = argv[2]

    if not os.path.exists(input_file):
        print(f"ERROR!\nFile: {input_file} does not exist.")
        exit(1)

    convert(input_file, output_file)


if __name__ == '__main__':
    main()
