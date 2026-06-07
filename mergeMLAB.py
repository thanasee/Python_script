#!/usr/bin/env python

from sys import argv, exit
import os
from collections import OrderedDict

def usage():
    """Print usage information and exit."""
    print("""
Usage: mergeMLAB.py <input1> <input2> [input3 ...] <output>

This script merges ML_AB-format data files into one file.
It combines the header information, unifies basis sets for each atom type,
and rewrites configuration numbering in the merged output.

This script was developed by Thanasee Thanasarnsurapong.
""")
    exit(0)


_LATTICE_FMT_STR = (
    "{:.16f}  {:.16f}  {:.16f}\n{:.16f}  {:.16f}  {:.16f}\n{:.16f}  {:.16f}  {:.16f}"
)
_BASIS_FMT_STR = """**************************************************
     Basis set for {}
--------------------------------------------------
{}"""
_CONFIGURATION_FMT_STR = """**************************************************
     Configuration num.    {}
==================================================
     System name
--------------------------------------------------
     {}
==================================================
     The number of atom types
--------------------------------------------------
       {}
==================================================
     The number of atoms
--------------------------------------------------
         {}
**************************************************
     Atom types and atom numbers
--------------------------------------------------
{}
==================================================
     CTIFOR
--------------------------------------------------
   {}
==================================================
     Primitive lattice vectors (ang.)
--------------------------------------------------
{}
==================================================
     Atomic positions (ang.)
--------------------------------------------------
{}
==================================================
     Total energy (eV)
--------------------------------------------------
  {}
==================================================
     Forces (eV ang.^-1)
--------------------------------------------------
{}
==================================================
     Stress (kbar)
--------------------------------------------------
     XX YY ZZ
--------------------------------------------------
 {}
--------------------------------------------------
     XY YZ ZX
--------------------------------------------------
 {}"""
_ML_AB_FMT_STR = """ {}
**************************************************
     The number of configurations
--------------------------------------------------
        {}
**************************************************
     The maximum number of atom type
--------------------------------------------------
       {}
**************************************************
     The atom types in the data file
--------------------------------------------------
     {}
**************************************************
     The maximum number of atoms per system
--------------------------------------------------
            {}
**************************************************
     The maximum number of atoms per atom type
--------------------------------------------------
            {}
**************************************************
     Reference atomic energy (eV)
--------------------------------------------------
   {}
**************************************************
     Atomic mass
--------------------------------------------------
   {}
**************************************************
     The numbers of basis sets per atom type
--------------------------------------------------
       {}
{}
{}
"""


def find_config_indices(lines):
    """
    Find line indices of all configuration block headers.

    Parameters
    ----------
    lines : list of str
        All lines from an ML_AB file.

    Returns
    -------
    list of int
        Line indices where 'Configuration num.' appears.
    """
    return [i for i, line in enumerate(lines) if 'Configuration num.' in line]


def read_title_value(lines, title):
    """
    Read a single scalar value from a titled section.

    Parameters
    ----------
    lines : list of str
        All lines from an ML_AB file.
    title : str
        Exact section title string to search for.

    Returns
    -------
    str
        The value on the second line after the title (skipping the separator).

    Raises
    ------
    ValueError
        If the section title is not found.
    """
    for i, line in enumerate(lines):
        if line.strip() == title:
            return lines[i + 2].strip()
    raise ValueError(f"Cannot find section '{title}'")


def read_title_block(lines, title):
    """
    Read all data lines from a titled section until the next separator.

    Parameters
    ----------
    lines : list of str
        All lines from an ML_AB file.
    title : str
        Exact section title string to search for.

    Returns
    -------
    list of str
        Non-empty data lines within the section, stripped of trailing newlines.

    Raises
    ------
    ValueError
        If the section title is not found.
    """
    for i, line in enumerate(lines):
        if line.strip() == title:
            start = i + 2
            end = start
            while end < len(lines):
                stripped = lines[end].strip()
                if stripped and set(stripped) <= set('*='):
                    break
                end += 1
            return [line.rstrip('\n') for line in lines[start:end] if line.strip()]
    raise ValueError(f"Cannot find section '{title}'")


def parse_basis_sections(header_lines):
    """
    Parse all 'Basis set for <element>' sections from the file header.

    Each basis set entry is a pair of integers (n1, n2) representing
    the radial and angular quantum number indices of the basis function.

    Parameters
    ----------
    header_lines : list of str
        Lines from the file header (before the first configuration block).

    Returns
    -------
    OrderedDict
        Mapping of element symbol (str) to list of (n1, n2) int tuples,
        preserving the order in which elements appear.
    """
    basis_map = OrderedDict()
    i = 0
    while i < len(header_lines):
        stripped = header_lines[i].strip()
        if stripped.startswith('Basis set for '):
            element = stripped.replace('Basis set for ', '').strip()
            start = i + 2
            end = start
            while end < len(header_lines):
                test = header_lines[end].strip()
                if test.startswith('Basis set for ') or 'Configuration num.' in test:
                    break
                if test and set(test) <= set('*='):
                    break
                end += 1
            rows = []
            for row in header_lines[start:end]:
                if row.strip():
                    parts = row.split()
                    if len(parts) >= 2:
                        rows.append((int(parts[0]), int(parts[1])))
            basis_map[element] = rows
            i = end
            continue
        i += 1
    return basis_map


def read_MLAB(filename):
    """
    Parse a complete ML_AB file into a structured dictionary.

    Reads the global header (atom types, reference energies, atomic masses,
    basis sets) and all configuration blocks.

    Parameters
    ----------
    filename : str
        Path to the ML_AB file to read.

    Returns
    -------
    dict with keys:
        'version'        : str           -- Version string from the first line.
        'ref_energy_map' : OrderedDict   -- Element -> reference energy (eV).
        'mass_map'       : OrderedDict   -- Element -> atomic mass (amu).
        'basis_map'      : OrderedDict   -- Element -> list of (n1, n2) tuples.
        'configs'        : list of dict  -- Parsed configuration blocks.
    """
    if not os.path.exists(filename):
        print(f"ERROR!\nFile: {filename} does not exist.")
        exit(1)

    with open(filename, 'r') as f:
        lines = f.readlines()

    config_indices = find_config_indices(lines)
    if not config_indices:
        raise ValueError(f"No configuration blocks found in {filename}")

    header_lines = lines[:config_indices[0]]
    config_blocks = []
    for idx, start in enumerate(config_indices):
        end = config_indices[idx + 1] if idx + 1 < len(config_indices) else len(lines)
        config_blocks.append(lines[start:end])

    atom_types = []
    for line in read_title_block(header_lines, 'The atom types in the data file'):
        atom_types.extend(line.split())

    ref_energy_values = [float(x) for line in read_title_block(header_lines, 'Reference atomic energy (eV)') for x in line.split()]
    mass_values = [float(x) for line in read_title_block(header_lines, 'Atomic mass') for x in line.split()]

    if not (len(atom_types) == len(ref_energy_values) == len(mass_values)):
        raise ValueError(f"Header atom-type metadata length mismatch in {filename}")

    ref_energy_map = OrderedDict((element, energy) for element, energy in zip(atom_types, ref_energy_values))
    mass_map = OrderedDict((element, mass) for element, mass in zip(atom_types, mass_values))
    basis_map = parse_basis_sections(header_lines)

    parsed_configs = []
    for block in config_blocks:
        parsed_configs.append(parse_config_block(block))

    return {
        'version': header_lines[0].rstrip('\n'),
        'ref_energy_map': ref_energy_map,
        'mass_map': mass_map,
        'basis_map': basis_map,
        'configs': parsed_configs,
    }


def parse_config_block(block_lines):
    """
    Parse a single configuration block into a structured dictionary.

    Parameters
    ----------
    block_lines : list of str
        Lines belonging to one configuration block, starting from the
        'Configuration num.' header line.

    Returns
    -------
    dict with keys:
        'system_name'      : str           -- System label.
        'ctifor'           : str           -- CTIFOR value.
        'lattice'          : list of str   -- Three lattice vector lines (ang.).
        'positions'        : list of str   -- Atomic position lines (ang.).
        'energy'           : str           -- Total energy line (eV).
        'forces'           : list of str   -- Force lines (eV/ang.).
        'stress_xx_yy_zz'  : str           -- Diagonal stress components (kbar).
        'stress_xy_yz_zx'  : str           -- Off-diagonal stress components (kbar).
        'atom_counts'      : OrderedDict   -- Element -> atom count for this config.
    """
    lines = [line.rstrip('\n') for line in block_lines]

    config = {
        'system_name': '',
        'ctifor': '',
        'lattice': [],
        'positions': [],
        'energy': '',
        'forces': [],
        'stress_xx_yy_zz': '',
        'stress_xy_yz_zx': '',
        'atom_counts': OrderedDict(),
    }

    i = 0
    while i < len(lines):
        stripped = lines[i].strip()

        if stripped == 'System name':
            config['system_name'] = lines[i + 2]
            i += 3
            continue

        if stripped == 'Atom types and atom numbers':
            j = i + 2
            while j < len(lines):
                test = lines[j].strip()
                if not test:
                    j += 1
                    continue
                if set(test) <= set('*='):
                    break
                parts = lines[j].split()
                if len(parts) >= 2:
                    config['atom_counts'][parts[0]] = int(parts[1])
                j += 1
            i = j
            continue

        if stripped == 'CTIFOR':
            config['ctifor'] = lines[i + 2]
            i += 3
            continue

        if stripped == 'Primitive lattice vectors (ang.)':
            config['lattice'] = [lines[i + 2], lines[i + 3], lines[i + 4]]
            i += 5
            continue

        if stripped == 'Atomic positions (ang.)':
            n_atoms = sum(config['atom_counts'].values())
            config['positions'] = lines[i + 2:i + 2 + n_atoms]
            i += 2 + n_atoms
            continue

        if stripped == 'Total energy (eV)':
            config['energy'] = lines[i + 2]
            i += 3
            continue

        if stripped == 'Forces (eV ang.^-1)':
            n_atoms = sum(config['atom_counts'].values())
            config['forces'] = lines[i + 2:i + 2 + n_atoms]
            i += 2 + n_atoms
            continue

        if stripped == 'XX YY ZZ':
            config['stress_xx_yy_zz'] = lines[i + 2]
            i += 3
            continue

        if stripped == 'XY YZ ZX':
            config['stress_xy_yz_zx'] = lines[i + 2]
            i += 3
            continue

        i += 1

    return config


def unique_preserve_order(items):
    """
    Return deduplicated list preserving first-occurrence order.

    Parameters
    ----------
    items : list
        Input list, potentially containing duplicates.

    Returns
    -------
    list
        List with duplicates removed, order of first occurrence retained.
    """
    unique = []
    for item in items:
        if item not in unique:
            unique.append(item)
    return unique


def format_values(values, scientific=False):
    """
    Format a flat list of floats into fixed-width lines of up to 3 values each.

    Each value is formatted to width 23 with 15 decimal places, in either
    fixed-point or scientific notation. Lines contain at most 3 values,
    matching the ML_AB column layout.

    Parameters
    ----------
    values : list of float
        Values to format.
    scientific : bool, optional
        If True, use scientific notation (e.g. '-1.234567890123456E+01').
        If False (default), use fixed-point notation.

    Returns
    -------
    list of str
        Formatted lines, each containing 1–3 values concatenated.
    """
    chunks = []
    for i, value in enumerate(values):
        if scientific:
            entry = f"{value:>23.15E}"
        else:
            entry = f"{value:>23.15f}"
        chunks.append(entry)
    return [''.join(chunks[i:i + 3]) for i in range(0, len(chunks), 3)]


def write_MLAB(output_file, merged_version, global_types, ref_energy_map, mass_map, basis_map, configs):
    """
    Write a merged ML_AB file from unified header data and configuration list.

    Computes global header statistics (max atom types, max atoms per system,
    max atoms per type) from the full configuration list, writes all header
    sections, basis sets, and renumbered configuration blocks in ML_AB format.

    Parameters
    ----------
    output_file : str
        Path to the output ML_AB file to write.
    merged_version : str
        Version string written to the first line.
    global_types : list of str
        Ordered list of all element symbols across merged files.
    ref_energy_map : OrderedDict
        Element -> reference energy (eV).
    mass_map : OrderedDict
        Element -> atomic mass (amu).
    basis_map : OrderedDict
        Element -> list of (n1, n2) basis function index tuples.
    configs : list of dict
        All parsed configuration dictionaries (from parse_config_block).
    """

    max_atom_types_per_system = max(len(config['atom_counts']) for config in configs)
    max_atoms_per_system = max(sum(config['atom_counts'].values()) for config in configs)
    max_per_type = max(
        config['atom_counts'].get(element, 0)
        for element in global_types
        for config in configs
    )

    # --- Header fields ---
    atom_types_str = "\n     ".join(
        " ".join(f"{e:<2s}" for e in global_types[i:i+3])
        for i in range(0, len(global_types), 3)
    )
    ref_energy_str = "\n   ".join(
        "  ".join(f"{ref_energy_map[e]:.16f}" for e in global_types[i:i+3])
        for i in range(0, len(global_types), 3)
    )
    mass_str = "\n   ".join(
        "  ".join(f"{mass_map[e]:.16f}" for e in global_types[i:i+3])
        for i in range(0, len(global_types), 3)
    )
    basis_count_str = "\n       ".join(
        "   ".join(f"{len(basis_map[e])}" for e in global_types[i:i+3])
        for i in range(0, len(global_types), 3)
    )

    # --- Basis set blocks ---
    basis_blocks = "\n".join(
        _BASIS_FMT_STR.format(
            element,
            "\n".join(f"{n1:>11d}{n2:>7d}" for n1, n2 in basis_map[element])
        )
        for element in global_types
    )

    # --- Configuration blocks ---
    config_blocks = "\n".join(
        _CONFIGURATION_FMT_STR.format(
            i,
            config['system_name'].strip(),
            len([e for e in global_types if config['atom_counts'].get(e, 0) > 0]),
            sum(config['atom_counts'].values()),
            "\n".join(
                f"     {e:<2s}{config['atom_counts'][e]:>7d}"
                for e in global_types if config['atom_counts'].get(e, 0) > 0
            ),
            config['ctifor'].strip(),
            "\n".join(line.rstrip() for line in config['lattice']),
            "\n".join(line.rstrip() for line in config['positions']),
            config['energy'].strip(),
            "\n".join(line.rstrip() for line in config['forces']),
            config['stress_xx_yy_zz'].strip(),
            config['stress_xy_yz_zx'].strip(),
        )
        for i, config in enumerate(configs, start=1)
    )

    with open(output_file, 'w') as f:
        f.write(_ML_AB_FMT_STR.format(
            merged_version.strip(),
            len(configs),
            max_atom_types_per_system,
            atom_types_str,
            max_atoms_per_system,
            max_per_type,
            ref_energy_str,
            mass_str,
            basis_count_str,
            basis_blocks,
            config_blocks,
        ))


def main():
    """Parse arguments, merge files, and write output"""
    if '-h' in argv or '--help' in argv or len(argv) < 4:
        usage()

    input_files = argv[1:-1]
    output_file = argv[-1]
    
    parsed_files = [read_MLAB(filename) for filename in input_files]
    
    version = parsed_files[0]['version']
    for parsed in parsed_files[1:]:
        if parsed['version'] != version:
            print("Warning! Input versions are different. The first file version will be used.")
            break
    
    all_types = []
    for parsed in parsed_files:
        all_types.extend(list(parsed['ref_energy_map'].keys()))
    global_types = unique_preserve_order(all_types)
    
    ref_energy_map = OrderedDict()
    mass_map = OrderedDict()
    basis_map = OrderedDict((element, []) for element in global_types)
    configs = []
    
    for element in global_types:
        for parsed in parsed_files:
            if element in parsed['ref_energy_map']:
                if element not in ref_energy_map:
                    ref_energy_map[element] = parsed['ref_energy_map'][element]
                elif ref_energy_map[element] != parsed['ref_energy_map'][element]:
                    print(f"Warning! Reference energy mismatch for '{element}'. First-file value will be used.")
            if element in parsed['mass_map']:
                if element not in mass_map:
                    mass_map[element] = parsed['mass_map'][element]
                elif mass_map[element] != parsed['mass_map'][element]:
                    print(f"Warning! Atomic mass mismatch for '{element}'. First-file value will be used.")
    
    for element in global_types:
        basis_rows = []
        for parsed in parsed_files:
            if element in parsed['basis_map']:
                basis_rows.extend(parsed['basis_map'][element])
        basis_map[element] = unique_preserve_order(basis_rows)
    
    for parsed in parsed_files:
        configs.extend(parsed['configs'])
    
    write_MLAB(output_file, version, global_types, ref_energy_map, mass_map, basis_map, configs)
    print(f"\nMerged {len(input_files)} files into {output_file}")
    print(f"Total configurations: {len(configs)}")
    print("Atom types in merged file: " + " ".join(global_types)+"\n")


if __name__ == "__main__":
    main()
