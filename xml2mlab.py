#!/usr/bin/env python

from sys import argv, exit
import os
import xml.etree.ElementTree as ET

# ---------------------------------------------------------------------------
# Standard atomic masses (u) for elements commonly used in VASP calculations
# ---------------------------------------------------------------------------
_ATOMIC_MASS = {
    'H' : 1.00800, 'He': 4.00260, 'Li': 6.94000, 'Be': 9.01220,
    'B' : 10.8100, 'C' : 12.0110, 'N' : 14.0070, 'O' : 15.9990,
    'F' : 18.9984, 'Ne': 20.1797, 'Na': 22.9898, 'Mg': 24.3050,
    'Al': 26.9815, 'Si': 28.0855, 'P' : 30.9738, 'S' : 32.0600,
    'Cl': 35.4500, 'Ar': 39.9480, 'K' : 39.0983, 'Ca': 40.0780,
    'Sc': 44.9559, 'Ti': 47.8670, 'V' : 50.9415, 'Cr': 51.9961,
    'Mn': 54.9380, 'Fe': 55.8450, 'Co': 58.9332, 'Ni': 58.6934,
    'Cu': 63.5460, 'Zn': 65.3800, 'Ga': 69.7230, 'Ge': 72.6300,
    'As': 74.9216, 'Se': 78.9600, 'Br': 79.9040, 'Kr': 83.7980,
    'Rb': 85.4678, 'Sr': 87.6200, 'Y' : 88.9059, 'Zr': 91.2240,
    'Nb': 92.9064, 'Mo': 95.9600, 'Tc': 98.0000, 'Ru': 101.070,
    'Rh': 102.906, 'Pd': 106.420, 'Ag': 107.868, 'Cd': 112.411,
    'In': 114.818, 'Sn': 118.710, 'Sb': 121.760, 'Te': 127.600,
    'I' : 126.904, 'Xe': 131.293, 'Cs': 132.905, 'Ba': 137.327,
    'La': 138.905, 'Ce': 140.116, 'Pr': 140.908, 'Nd': 144.240,
    'Pm': 145.000, 'Sm': 150.360, 'Eu': 151.964, 'Gd': 157.250,
    'Tb': 158.925, 'Dy': 162.500, 'Ho': 164.930, 'Er': 167.259,
    'Tm': 168.934, 'Yb': 173.040, 'Lu': 174.967, 'Hf': 178.490,
    'Ta': 180.948, 'W' : 183.840, 'Re': 186.207, 'Os': 190.230,
    'Ir': 192.217, 'Pt': 195.078, 'Au': 196.967, 'Hg': 200.590,
    'Tl': 204.383, 'Pb': 207.200, 'Bi': 208.980, 'Po': 209.000,
    'At': 210.000, 'Rn': 222.000, 'Fr': 223.000, 'Ra': 226.000,
    'Ac': 227.000, 'Th': 232.038, 'Pa': 231.036, 'U' : 238.029,
}

# ---------------------------------------------------------------------------
# ML_AB format strings
# ---------------------------------------------------------------------------
_HEADER_FMT_STR = """ 1.0 Version
**************************************************
     The number of configurations
--------------------------------------------------
{nconf:>8}
**************************************************
     The maximum number of atom type
--------------------------------------------------
{ntype:>7}
**************************************************
     The atom types in the data file
--------------------------------------------------
{atom_types}
**************************************************
     The maximum number of atoms per system
--------------------------------------------------
{max_atoms:>14}
**************************************************
     The maximum number of atoms per atom type
--------------------------------------------------
{max_per_type}
**************************************************
     Reference atomic energy (eV)
--------------------------------------------------
{ref_energy}
**************************************************
     Atomic mass
--------------------------------------------------
{atomic_mass}
**************************************************
     The numbers of basis sets per atom type
--------------------------------------------------
{nbasis}
{basis_blocks}"""

_CONFIGURATION_FMT_STR = """**************************************************
     Configuration num.    {idx:>6}
==================================================
     System name
--------------------------------------------------
     {system}
==================================================
     The number of atom types
--------------------------------------------------
{ntype:>7}
==================================================
     The number of atoms
--------------------------------------------------
{natoms:>9}
**************************************************
     Atom types and atom numbers
--------------------------------------------------
{atom_type_numbers}
==================================================
     Primitive lattice vectors (ang.)
--------------------------------------------------
{lattice}
==================================================
     Atomic positions (ang.)
--------------------------------------------------
{positions}
==================================================
     Total energy (eV)
--------------------------------------------------
{energy}
==================================================
     Forces (eV ang.^-1)
--------------------------------------------------
{forces}
==================================================
     Stress (kbar)
--------------------------------------------------
     XX YY ZZ
--------------------------------------------------
{stress_diag}
--------------------------------------------------
     XY YZ ZX
--------------------------------------------------
{stress_offdiag}"""


def usage():
    """Print usage and exit."""
    print("""
Usage: xml2mlab.py <vasprun1.xml> [vasprun2.xml ...]

Converts AIMD vasprun.xml trajectories to VASP ML_AB training data format.
Multiple files are concatenated into a single ML_AB output file.

  - Step selection (equilibration skip, stride) set interactively
  - CTIFOR is omitted (external training data convention)
  - Reference atomic energies set to 0.0 (re-fitted during training)
  - Basis sets written as dummy entries (compatible with ML_MODE = select)
  - Element mismatch across files: warning issued, union taken

Output: ML_AB (written in current directory)
""")
    exit(0)


def get_std_atomic_mass(symbol):
    """
    Return the standard atomic mass for an element symbol.

    Parameters
    ----------
    symbol : str
        Element symbol, e.g. 'Si', 'Pb'.

    Returns
    -------
    float
        Atomic mass in u. Returns 1.0 if symbol not found (with warning).
    """
    mass = _ATOMIC_MASS.get(symbol.strip().capitalize())
    if mass is None:
        print("Warning: atomic mass not found for '{}', using 1.0".format(symbol))
        return 1.0
    return mass


def count_steps(xml_path):
    """
    Count the number of ionic steps (calculation blocks) in a vasprun.xml.

    Parameters
    ----------
    xml_path : str
        Path to vasprun.xml.

    Returns
    -------
    int
        Number of <calculation> elements found.
    """
    count = 0
    for event, elem in ET.iterparse(xml_path, events=('end',)):
        if elem.tag == 'calculation':
            count += 1
            elem.clear()
    return count


def read_species_info(xml_path):
    """
    Read atom species, counts, and SYSTEM tag from a vasprun.xml.

    Parameters
    ----------
    xml_path : str
        Path to vasprun.xml.

    Returns
    -------
    dict with keys:
        elements : list of str
            Element symbols in VASP order, e.g. ['Si', 'O'].
        counts : list of int
            Number of atoms per element, same order as elements.
        system : str
            SYSTEM tag value (truncated to 40 characters).
        nions : int
            Total number of atoms.
        ntyp : int
            Number of atom types.
    """
    tree = ET.parse(xml_path)
    root = tree.getroot()

    # SYSTEM tag
    system = ''
    for item in root.iter('i'):
        if item.get('name') == 'SYSTEM':
            system = (item.text or '').strip()[:40]
            break

    # Element symbols and counts from atominfo/array[@name="atomtypes"]
    elements = []
    counts   = []
    atomtypes = root.find('.//atominfo/array[@name="atomtypes"]/set')
    if atomtypes is not None:
        for rc in atomtypes.findall('rc'):
            fields = rc.findall('c')
            if len(fields) >= 2:
                counts.append(int(fields[0].text.strip()))
                elements.append(fields[1].text.strip())

    nions = sum(counts)
    ntyp  = len(elements)

    return {
        'elements': elements,
        'counts'  : counts,
        'system'  : system,
        'nions'   : nions,
        'ntyp'    : ntyp,
    }


def build_global_meta(info_list, xml_paths):
    """
    Aggregate per-file species info into global ML_AB header metadata.

    Warns if element sets differ across files. Takes the union in first-seen
    order. Computes max atoms per system and max atoms per atom type.

    Parameters
    ----------
    info_list : list of dict
        Output of read_species_info() for each file.
    xml_paths : list of str
        File paths, used in warning messages.

    Returns
    -------
    dict with keys:
        elements       : list of str  — union element list, first-seen order
        max_atoms      : int          — max NIONS across all files
        max_per_type   : dict         — {element: max count} across all files
        ref_energy     : dict         — {element: 0.0}
        atomic_mass    : dict         — {element: standard mass}
    """
    # Build union element list (first-seen order) and detect mismatches
    union   = []
    seen    = set()
    el_sets = [set(info['elements']) for info in info_list]
    all_eq  = all(s == el_sets[0] for s in el_sets)

    if not all_eq:
        print("Warning: element mismatch across files")
        for path, info in zip(xml_paths, info_list):
            print("  {}   ->  {}".format(path, '  '.join(info['elements'])))

    for info in info_list:
        for el in info['elements']:
            if el not in seen:
                union.append(el)
                seen.add(el)

    if not all_eq:
        print("  Global atom types (union, first-seen order): {}".format(
            '  '.join(union)))
        print("Continuing...")

    # Max atoms per system
    max_atoms = max(info['nions'] for info in info_list)

    # Max atoms per atom type across all files
    max_per_type = {el: 0 for el in union}
    for info in info_list:
        for el, cnt in zip(info['elements'], info['counts']):
            if cnt > max_per_type[el]:
                max_per_type[el] = cnt

    return {
        'elements'    : union,
        'max_atoms'   : max_atoms,
        'max_per_type': max_per_type,
        'ref_energy'  : {el: 0.0 for el in union},
        'atomic_mass' : {el: get_std_atomic_mass(el) for el in union},
    }


def prompt_skip_stride(xml_paths, step_counts):
    """
    Interactively prompt the user for equilibration skip and stride values.

    Prints a file summary table, asks for skip and stride, prints a
    selection summary, then asks for confirmation.

    Parameters
    ----------
    xml_paths : list of str
        Paths to vasprun.xml files.
    step_counts : list of int
        Number of ionic steps in each file.

    Returns
    -------
    tuple of (int, int)
        (skip, stride) values to apply uniformly to all files.
    """
    n = len(xml_paths)
    print("\nFound {} vasprun.xml file{}:".format(n, 's' if n > 1 else ''))
    for i, (path, steps) in enumerate(zip(xml_paths, step_counts), 1):
        print("  [{}] {}   ->  {} ionic steps".format(i, path, steps))

    print("\n--- Step selection ---")

    while True:
        raw = input("Skip first N steps (equilibration)? [default 0]: ").strip()
        if raw == '':
            skip = 0
            break
        if raw.isdigit():
            skip = int(raw)
            break
        print("  Enter a non-negative integer.")

    while True:
        raw = input("Use every Nth step (stride)?        [default 1]: ").strip()
        if raw == '':
            stride = 1
            break
        if raw.isdigit() and int(raw) >= 1:
            stride = int(raw)
            break
        print("  Enter a positive integer.")

    print("\n--- Summary ---")
    total = 0
    for i, (path, steps) in enumerate(zip(xml_paths, step_counts), 1):
        available = max(0, steps - skip)
        selected  = len(range(skip, steps, stride))
        end_step  = steps - 1
        print("  [{}] {}   ->  {} configs selected   (steps {}–{}, stride {})".format(
            i, path, selected, skip, end_step, stride))
        total += selected
    print("  Total: {} configurations -> ML_AB".format(total))

    while True:
        ans = input("\nProceed? [y/n]: ").strip().lower()
        if ans == 'y':
            return skip, stride
        if ans == 'n':
            print("Aborted.")
            exit(0)


def parse_lattice(calc):
    """
    Extract the 3x3 lattice matrix from a <calculation> element.

    Parameters
    ----------
    calc : xml.etree.ElementTree.Element
        A <calculation> XML element from vasprun.xml.

    Returns
    -------
    list of list of float
        3x3 lattice matrix in Angstrom, row = lattice vector.
    """
    basis = calc.find('.//structure/crystal/varray[@name="basis"]')
    lat   = []
    for v in basis.findall('v'):
        lat.append([float(x) for x in v.text.split()])
    return lat


def parse_positions(calc, lat):
    """
    Extract atomic positions and convert from direct to Cartesian coordinates.

    Parameters
    ----------
    calc : xml.etree.ElementTree.Element
        A <calculation> XML element from vasprun.xml.
    lat : list of list of float
        3x3 lattice matrix (row = lattice vector) in Angstrom.

    Returns
    -------
    list of list of float
        Cartesian positions in Angstrom, one entry per atom.
    """
    pos_el = calc.find('.//structure/varray[@name="positions"]')
    cart   = []
    for v in pos_el.findall('v'):
        d = [float(x) for x in v.text.split()]
        # Cartesian = d[0]*a1 + d[1]*a2 + d[2]*a3
        xyz = [
            d[0]*lat[0][0] + d[1]*lat[1][0] + d[2]*lat[2][0],
            d[0]*lat[0][1] + d[1]*lat[1][1] + d[2]*lat[2][1],
            d[0]*lat[0][2] + d[1]*lat[1][2] + d[2]*lat[2][2],
        ]
        cart.append(xyz)
    return cart


def parse_energy(calc):
    """
    Extract the total energy (without entropy) from a <calculation> element.

    Parameters
    ----------
    calc : xml.etree.ElementTree.Element
        A <calculation> XML element from vasprun.xml.

    Returns
    -------
    float
        Total energy in eV (e_wo_entrp).
    """
    for item in calc.iter('i'):
        if item.get('name') == 'e_wo_entrp':
            return float(item.text.strip())
    raise ValueError("e_wo_entrp not found in calculation block")


def parse_forces(calc):
    """
    Extract forces on all atoms from a <calculation> element.

    Parameters
    ----------
    calc : xml.etree.ElementTree.Element
        A <calculation> XML element from vasprun.xml.

    Returns
    -------
    list of list of float
        Forces in eV/Angstrom, one [fx, fy, fz] per atom.
    """
    forces_el = calc.find('.//varray[@name="forces"]')
    forces    = []
    for v in forces_el.findall('v'):
        forces.append([float(x) for x in v.text.split()])
    return forces


def parse_stress(calc):
    """
    Extract the stress tensor from a <calculation> element.

    The vasprun.xml stress is a 3x3 matrix in kbar. Components are mapped to
    ML_AB order: XX, YY, ZZ (diagonal) and XY, YZ, ZX (off-diagonal).
    No sign flip is applied (consistent with native VASP MLFF output).

    Parameters
    ----------
    calc : xml.etree.ElementTree.Element
        A <calculation> XML element from vasprun.xml.

    Returns
    -------
    tuple of (list of float, list of float)
        (stress_diag, stress_offdiag) where:
        stress_diag    = [XX, YY, ZZ]
        stress_offdiag = [XY, YZ, ZX]
    """
    stress_el = calc.find('.//varray[@name="stress"]')
    rows      = []
    for v in stress_el.findall('v'):
        rows.append([float(x) for x in v.text.split()])
    # rows[0] = [XX, XY, XZ], rows[1] = [YX, YY, YZ], rows[2] = [ZX, ZY, ZZ]
    stress_diag    = [rows[0][0], rows[1][1], rows[2][2]]          # XX YY ZZ
    stress_offdiag = [rows[0][1], rows[1][2], rows[2][0]]          # XY YZ ZX
    return stress_diag, stress_offdiag


def iter_calculations(xml_path, skip, stride):
    """
    Iterate over ionic steps in a vasprun.xml, yielding parsed config dicts.

    Uses iterparse for memory-efficient streaming. Steps before `skip` and
    steps not on the stride grid are discarded.

    Parameters
    ----------
    xml_path : str
        Path to vasprun.xml.
    skip : int
        Number of ionic steps to skip at the start (0-indexed).
    stride : int
        Yield every stride-th step after skip (stride=1 yields all).

    Yields
    ------
    dict with keys:
        lattice        : list of list of float   — 3x3 matrix (Ang)
        positions      : list of list of float   — Cartesian (Ang)
        energy         : float                   — eV
        forces         : list of list of float   — eV/Ang
        stress_diag    : list of float           — [XX, YY, ZZ] kbar
        stress_offdiag : list of float           — [XY, YZ, ZX] kbar
    """
    step = 0
    for event, elem in ET.iterparse(xml_path, events=('end',)):
        if elem.tag != 'calculation':
            continue
        if step >= skip and (step - skip) % stride == 0:
            lat  = parse_lattice(elem)
            pos  = parse_positions(elem, lat)
            eng  = parse_energy(elem)
            frc  = parse_forces(elem)
            sd, so = parse_stress(elem)
            yield {
                'lattice'       : lat,
                'positions'     : pos,
                'energy'        : eng,
                'forces'        : frc,
                'stress_diag'   : sd,
                'stress_offdiag': so,
            }
        step += 1
        elem.clear()


def _fmt_blocked(values, fmt, per_line=3):
    """
    Format a flat list of values into lines of at most per_line entries.

    Parameters
    ----------
    values : list
        Values to format.
    fmt : str
        Python format spec for each value, e.g. '{:.5f}'.
    per_line : int
        Maximum entries per line (default 3, per ML_AB spec).

    Returns
    -------
    str
        Multi-line string, no trailing newline.
    """
    lines = []
    for i in range(0, len(values), per_line):
        chunk = values[i:i + per_line]
        lines.append('  ' + '  '.join(fmt.format(v) for v in chunk))
    return '\n'.join(lines)


def _fmt_atom_type_numbers(elements, counts):
    """
    Format the 'Atom types and atom numbers' block for one configuration.

    Parameters
    ----------
    elements : list of str
        Element symbols present in this configuration.
    counts : list of int
        Number of atoms per element.

    Returns
    -------
    str
        Multi-line string, one '  El  N' entry per line.
    """
    lines = []
    for el, cnt in zip(elements, counts):
        lines.append('     {:<2}  {:>5}'.format(el, cnt))
    return '\n'.join(lines)


def _fmt_vec3(v):
    """
    Format a 3-component vector as a single ML_AB data line.

    Parameters
    ----------
    v : list of float
        Three-component vector.

    Returns
    -------
    str
        Space-separated string with 16-decimal fixed-point values.
    """
    return '  {:>22.16f}  {:>22.16f}  {:>22.16f}'.format(v[0], v[1], v[2])


def write_config(f, idx, cfg, info):
    """
    Write one configuration block to an open file handle.

    Parameters
    ----------
    f : file object
        Open output file.
    idx : int
        1-based configuration index.
    cfg : dict
        Parsed config dict from iter_calculations().
    info : dict
        Species info dict from read_species_info() for this file.
    """
    lattice_str  = '\n'.join(_fmt_vec3(v) for v in cfg['lattice'])
    pos_str      = '\n'.join(_fmt_vec3(p) for p in cfg['positions'])
    forces_str   = '\n'.join(_fmt_vec3(fv) for fv in cfg['forces'])
    stress_d_str = _fmt_vec3(cfg['stress_diag'])
    stress_o_str = _fmt_vec3(cfg['stress_offdiag'])
    atom_str     = _fmt_atom_type_numbers(info['elements'], info['counts'])
    energy_str   = '  {:>22.14f}'.format(cfg['energy'])

    block = _CONFIGURATION_FMT_STR.format(
        idx             = idx,
        system          = info['system'],
        ntype           = info['ntyp'],
        natoms          = info['nions'],
        atom_type_numbers = atom_str,
        lattice         = lattice_str,
        positions       = pos_str,
        energy          = energy_str,
        forces          = forces_str,
        stress_diag     = stress_d_str,
        stress_offdiag  = stress_o_str,
    )
    f.write(block)
    f.write('\n')


def write_header(f, meta, n_configs):
    """
    Write the global ML_AB header to an open file handle.

    Reference atomic energies are set to 0.0. Basis sets are written as
    dummy entries (one line '1  1' per element), compatible with VASP's
    ML_MODE = select which re-selects reference configurations.

    Parameters
    ----------
    f : file object
        Open output file.
    meta : dict
        Global metadata from build_global_meta().
    n_configs : int
        Total number of configurations to be written.
    """
    elements = meta['elements']
    ntype    = len(elements)

    # Atom types block: <=3 per line, 2-char padded
    type_lines = []
    for i in range(0, ntype, 3):
        chunk = elements[i:i+3]
        type_lines.append('     ' + '  '.join('{:<2}'.format(e) for e in chunk))
    atom_types_str = '\n'.join(type_lines)

    # Max atoms per atom type: <=3 per line
    max_per_type_vals = [meta['max_per_type'][el] for el in elements]
    max_per_type_str  = _fmt_blocked(max_per_type_vals, '{:>14}')

    # Reference atomic energy: 0.0 for all, <=3 per line
    ref_vals = [0.0 for _ in elements]
    ref_str  = _fmt_blocked(ref_vals, '{:>22.13f}')

    # Atomic mass: <=3 per line
    mass_vals = [meta['atomic_mass'][el] for el in elements]
    mass_str  = _fmt_blocked(mass_vals, '{:>22.13f}')

    # Numbers of basis sets: 1 per element (dummy), <=3 per line
    nbasis_vals = [1 for _ in elements]
    nbasis_str  = _fmt_blocked(nbasis_vals, '{:>4}')

    # Dummy basis set blocks: one '          1      1' line per element
    basis_block_lines = []
    for el in elements:
        basis_block_lines.append(
            "**************************************************\n"
            "     Basis set for {}\n"
            "--------------------------------------------------\n"
            "          1      1".format(el)
        )
    basis_blocks_str = '\n'.join(basis_block_lines)

    header = _HEADER_FMT_STR.format(
        nconf        = n_configs,
        ntype        = ntype,
        atom_types   = atom_types_str,
        max_atoms    = meta['max_atoms'],
        max_per_type = max_per_type_str,
        ref_energy   = ref_str,
        atomic_mass  = mass_str,
        nbasis       = nbasis_str,
        basis_blocks = basis_blocks_str,
    )
    f.write(header)
    f.write('\n')


def write_MLAB(xml_paths, info_list, meta, skip, stride, outfile='ML_AB'):
    """
    Collect all configurations and write a complete ML_AB file.

    Streams configurations from each vasprun.xml via iter_calculations(),
    collects them in memory, then writes the header followed by all configs.

    Parameters
    ----------
    xml_paths : list of str
        Paths to vasprun.xml files.
    info_list : list of dict
        Species info from read_species_info() for each file.
    meta : dict
        Global metadata from build_global_meta().
    skip : int
        Number of ionic steps to skip at the start of each file.
    stride : int
        Step stride applied to each file.
    outfile : str
        Output filename (default 'ML_AB').
    """
    # Collect all (cfg, info) pairs
    all_configs = []
    for xml_path, info in zip(xml_paths, info_list):
        for cfg in iter_calculations(xml_path, skip, stride):
            all_configs.append((cfg, info))

    n_configs = len(all_configs)

    print("Writing {} ...".format(outfile), end=' ', flush=True)
    with open(outfile, 'w') as f:
        write_header(f, meta, n_configs)
        for idx, (cfg, info) in enumerate(all_configs, 1):
            write_config(f, idx, cfg, info)
    print("done.")


def main():
    """Entry point for xml2mlab.py."""
    if '-h' in argv or '--help' in argv or len(argv) < 2:
        usage()

    xml_paths = argv[1:]

    # Validate files exist
    for path in xml_paths:
        if not os.path.isfile(path):
            print("Error: file not found: {}".format(path))
            exit(1)

    # Count steps (fast pre-scan)
    step_counts = []
    for path in xml_paths:
        step_counts.append(count_steps(path))

    # Read species info per file
    info_list = []
    for path in xml_paths:
        info_list.append(read_species_info(path))

    # Build global metadata (union elements, warn on mismatch)
    meta = build_global_meta(info_list, xml_paths)

    # Interactive step selection
    skip, stride = prompt_skip_stride(xml_paths, step_counts)

    # Write ML_AB
    write_MLAB(xml_paths, info_list, meta, skip, stride)


if __name__ == '__main__':
    main()
