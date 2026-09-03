#!/usr/bin/env python

from sys import argv, exit
import os
import re
import readline
import numpy as np
from itertools import groupby
from collections import Counter


def usage():
    """Print usage information and exit."""
    print("""
Usage: vaspMagmom.py <POSCAR>

This script writes an initial MAGMOM tag to INCAR based on element composition
read from a POSCAR/CONTCAR file, using standard per-element default moments.

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

_ELEMENT_UNPAIRED = [
    1.0,  0.6,  1.0,  0.6,  1.0,  2.0,  3.0,  2.0,
    1.0,  0.6,  1.0,  0.6,  1.0,  2.0,  3.0,  2.0,
    1.0,  0.6,  1.0,  0.6,  1.0,  2.0,  3.0,  6.0,
    5.0,  4.0,  3.0,  2.0,  1.0,  0.6,  1.0,  2.0,
    3.0,  2.0,  1.0,  0.6,  1.0,  0.6,  1.0,  2.0,
    5.0,  6.0,  5.0,  4.0,  3.0,  0.6,  1.0,  0.6,
    1.0,  2.0,  3.0,  2.0,  1.0,  0.6,  1.0,  0.6,
    1.0,  2.0,  3.0,  4.0,  5.0,  6.0,  7.0,  8.0,
    5.0,  4.0,  3.0,  2.0,  1.0,  0.6,  1.0,  2.0,
    3.0,  4.0,  5.0,  4.0,  3.0,  2.0,  1.0,  0.6,
    1.0,  2.0,  3.0,  2.0,  1.0,  0.6,  1.0,  0.6,
    1.0,  2.0,  3.0,  4.0,  5.0,  6.0,  7.0,  8.0,
    5.0,  4.0,  3.0,  2.0,  1.0,  0.6,  1.0,  2.0,
    3.0,  4.0,  5.0,  4.0,  3.0,  2.0,  1.0,  0.6,
    1.0,  2.0,  3.0,  2.0,  1.0,  0.6
]
 
_ELEMENT_MAGMOM = [
    0.6,  0.6,  0.6,  0.6,  0.6,  0.6,  0.6,  0.6,
    0.6,  0.6,  0.6,  0.6,  0.6,  0.6,  0.6,  0.6,
    0.6,  0.6,  0.6,  0.6,  1.0,  2.0,  3.0,  5.0,
    5.0,  5.0,  3.0,  2.0,  1.0,  0.6,  0.6,  0.6,
    0.6,  0.6,  0.6,  0.6,  0.6,  0.6,  1.0,  2.0,
    3.0,  4.0,  5.0,  4.0,  3.0,  2.0,  1.0,  0.6,
    0.6,  0.6,  0.6,  0.6,  0.6,  0.6,  0.6,  0.6,
    0.6,  1.0,  2.0,  3.0,  4.0,  5.0,  7.0,  7.0,
    6.0,  5.0,  4.0,  3.0,  2.0,  1.0,  0.6,  2.0,
    3.0,  4.0,  5.0,  4.0,  3.0,  2.0,  1.0,  0.6,
    0.6,  0.6,  0.6,  0.6,  0.6,  0.6,  0.6,  0.6,
    0.6,  0.6,  0.6,  2.0,  0.6,  0.6,  0.6,  0.6,
    0.6,  0.6,  0.6,  0.6,  0.6,  0.6,  0.6,  0.6,
    0.6,  0.6,  0.6,  0.6,  0.6,  0.6,  0.6,  0.6,
    0.6,  0.6,  0.6,  0.6,  0.6,  0.6
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
 
 
def get_default_magmom(element):
    """Look up the default initial magnetic moment (mu_B) for an element.
 
    Parameters
    ----------
    element : str — element symbol
 
    Returns
    -------
    float — default moment from _ELEMENT_MAGMOM, or 0.6 if not found
    """
    try:
        idx = _ELEMENT_SYMBOLS.index(element)
        return _ELEMENT_MAGMOM[idx]
    except ValueError:
        return 0.6


def get_unpaired_electrons(element):
    """Look up the free-atom ground-state unpaired electron count for an element.
 
    Parameters
    ----------
    element : str — element symbol
 
    Returns
    -------
    float — value from _ELEMENT_UNPAIRED, or 0.6 if not found
    """
    try:
        idx = _ELEMENT_SYMBOLS.index(element)
        return _ELEMENT_UNPAIRED[idx]
    except ValueError:
        return 0.6
 
 
def build_default_magmom_values(unique_elements, element_counts):
    """Look up the initial magnetic moment for every unique element.

    Uses the _ELEMENT_MAGMOM bulk-default table by default. Leaving MAGMOM
    unset lets VASP fall back to its own initial guess, which can trigger
    convergence errors in some magnetic calculations — writing an explicit
    value avoids that.

    Single-atom override: if an element's bulk default is 0.6 (this
    table's non-magnetic placeholder) and exactly one atom of that
    element is present in the whole structure — an isolated dopant,
    adatom, or defect rather than a bulk species — the placeholder is
    replaced with that element's number of unpaired electrons (Hund's
    rule, neutral free-atom ground state, from _ELEMENT_UNPAIRED). A
    lone atom can carry its free-atom moment even when the element is
    conventionally non-magnetic in bulk, so leaving it at the placeholder
    would bias the calculation toward the wrong spin state. Elements
    with a genuinely nonzero bulk default are never touched by this
    override, regardless of atom count.

    Parameters
    ----------
    unique_elements : list[str]      — unique element symbols, first-occurrence order
    element_counts  : dict[str, int] — element -> total atom count in the structure

    Returns
    -------
    dict[str, float] — element -> magnetic moment (mu_B)
    """
    magmom_values = {}
    for element in unique_elements:
        default = get_default_magmom(element)
        if default == 0.6 and element_counts[element] == 1:
            default = float(get_unpaired_electrons(element))
        magmom_values[element] = default
    return magmom_values
 
 
def build_magmom_string(species, magmom_values):
    """Expand per-element moments to per-atom order and run-length-encode.
 
    Consecutive atoms sharing the same moment are compressed into VASP's
    'count*value' shorthand, matching the atom order of the POSCAR file.
 
    Parameters
    ----------
    species        : list[str]         — per-atom element labels, in POSCAR file order
    magmom_values  : dict[str, float]  — element -> magnetic moment
 
    Returns
    -------
    str — MAGMOM tag value, e.g. '3*5.0 2*0.6 4*1.0'
    """
    per_atom = list(zip(species, (round(magmom_values[s], 1) for s in species)))
    groups = []
    for (_, value), group in groupby(per_atom):
        count = len(list(group))
        groups.append(f"{count}*{value:.1f}")
    return " ".join(groups)
 
 
_TAG_PATTERN = re.compile(r'^\s*([A-Za-z_][A-Za-z0-9_]*)\s*=')
 
 
def read_INCAR(filepath):
    """Read an existing INCAR file into a list of lines.
 
    Parameters
    ----------
    filepath : str
 
    Returns
    -------
    list[str] — lines of the file
    """
    with open(filepath, 'r') as f:
        return f.readlines()
 
 
def default_INCAR_lines(magmom_string):
    """Build a fresh INCAR from the standard spin-polarized template.
 
    Used only when no INCAR is found in the current directory.
 
    Parameters
    ----------
    magmom_string : str — MAGMOM tag value
 
    Returns
    -------
    list[str] — INCAR lines, ready to write
    """
    template = [
        ("ISMEAR",  "0"),
        ("SIGMA",   "0.05"),
        ("LWAVE",   ".FALSE."),
        ("LCHARG",  ".FALSE."),
        ("ISPIN",   "2"),
        ("MAGMOM",  magmom_string),
        ("IBRION",  "2"),
        ("NSW",     "200"),
        ("NELMIN",  "5"),
        ("EDIFF",   "1E-5"),
        ("EDIFFG",  "-1E-2"),
        ("ENCUT",   "520"),
    ]
    return [f"{tag} = {value}\n" for tag, value in template]
 
 
def find_tag_line(lines, tag):
    """Return the index of the (uncommented) line that sets `tag`, or None.
 
    Parameters
    ----------
    lines : list[str]
    tag   : str — INCAR tag name, e.g. 'MAGMOM'
 
    Returns
    -------
    int or None
    """
    for i, line in enumerate(lines):
        stripped = line.strip()
        if stripped.startswith('#') or stripped.startswith('!'):
            continue
        match = _TAG_PATTERN.match(line)
        if match and match.group(1).upper() == tag.upper():
            return i
    return None
 
 
def set_tag(lines, tag, value):
    """Insert or replace a 'TAG = value' line in an INCAR line list, in place.
 
    Parameters
    ----------
    lines : list[str] — modified in place
    tag   : str
    value : str
    """
    new_line = f"{tag} = {value}\n"
    idx = find_tag_line(lines, tag)
    if idx is not None:
        lines[idx] = new_line
    else:
        if lines and not lines[-1].endswith('\n'):
            lines[-1] += '\n'
        lines.append(new_line)
 
 
def ensure_ispin(lines):
    """Ensure ISPIN = 2 is set, overriding any other value (e.g. ISPIN = 1).
 
    Parameters
    ----------
    lines : list[str] — current INCAR lines
 
    Returns
    -------
    bool — True if 'ISPIN = 2' should be (re)written
    """
    idx = find_tag_line(lines, 'ISPIN')
    if idx is not None:
        value = lines[idx].split('=', 1)[1].strip()
        current = value.split()[0] if value.split() else ''
        if current == '2':
            return False
        return True
 
    return True
 
 
def write_INCAR(filepath, lines):
    """Write INCAR lines to a file.
 
    Parameters
    ----------
    filepath : str
    lines    : list[str]
    """
    with open(filepath, 'w') as f:
        f.writelines(lines)
 
 
def main():
    """Parse arguments, build the MAGMOM tag from POSCAR composition, and overwrite INCAR in place."""
 
    if '-h' in argv or '--help' in argv or len(argv) != 2:
        usage()
 
    poscar_file = argv[1]
    incar_file = "INCAR"
 
    poscar = read_POSCAR(poscar_file)
 
    # First-occurrence order avoids reordering atoms; MAGMOM must follow the
    # exact atom order already present in the POSCAR file.
    unique_elements = list(dict.fromkeys(poscar["species"]))
    element_counts = Counter(poscar["species"])
 
    magmom_values = build_default_magmom_values(unique_elements, element_counts)
    magmom_string = build_magmom_string(poscar["species"], magmom_values)
 
    if os.path.exists(incar_file):
        lines = read_INCAR(incar_file)
        set_tag(lines, "MAGMOM", magmom_string)
        if ensure_ispin(lines):
            set_tag(lines, "ISPIN", "2")
    else:
        lines = default_INCAR_lines(magmom_string)
 
    write_INCAR(incar_file, lines)
 
 
if __name__ == "__main__":
    main()
