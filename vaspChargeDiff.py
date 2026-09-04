#!/usr/bin/env python

from sys import argv, exit
import os
import numpy as np


def usage():
    """Print usage information and exit."""
    print("""
Usage: vaspChargeDiff.py <CHGCAR_AB> <CHGCAR_A> <CHGCAR_B>

Computes Delta_rho = rho_AB - rho_A - rho_B and writes CHGDIFF.vasp.
CHGCAR_AB is the combined system; CHGCAR_A/B are the isolated fragments in
the same cell. All three must share the same grid dimensions.

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


def _looks_like_grid_header(line, expected_ngrid=None):
    """True if ``line`` is an 'NGXF NGYF NGZF' header (three positive ints,
    optionally matching ``expected_ngrid``)."""
    tokens = line.split()
    if len(tokens) != 3 or not all(t.isdigit() for t in tokens):
        return False
    dims = tuple(int(t) for t in tokens)
    if any(d <= 0 for d in dims):
        return False
    if expected_ngrid is not None and dims != expected_ngrid:
        return False
    return True


def _read_grid_block(lines, line_idx):
    """
    Read one 'NGXF NGYF NGZF' header plus its NGXF*NGYF*NGZF values, and
    reshape in Fortran order (x fastest) to match VASP's own write order.
    Returns (ngrid, grid, line_idx_after_block).
    """
    ngrid = tuple(int(x) for x in lines[line_idx].split()[:3])
    n_values = ngrid[0] * ngrid[1] * ngrid[2]
    line_idx += 1

    values = []
    while len(values) < n_values:
        values.extend(float(x) for x in lines[line_idx].split())
        line_idx += 1

    if len(values) != n_values:
        print(f"ERROR! Grid block declared {n_values} values "
              f"(NGXF*NGYF*NGZF = {ngrid}) but {len(values)} were read.")
        exit(1)

    grid = np.array(values[:n_values]).reshape(ngrid, order='F')
    return ngrid, grid, line_idx


def _skip_augmentation(lines, line_idx, total_atoms):
    """
    Skip a PAW 'augmentation occupancies' block if present right after a
    density grid (one header + values per atom). Returns line_idx
    unchanged if no such block is there.
    """
    peek = line_idx
    while peek < len(lines) and lines[peek].strip() == '':
        peek += 1
    if peek >= len(lines) or not lines[peek].strip().lower().startswith('augmentation occupancies'):
        return line_idx

    line_idx = peek
    for _ in range(total_atoms):
        header = lines[line_idx].split()
        n_values = int(header[-1])
        line_idx += 1
        count = 0
        while count < n_values:
            count += len(lines[line_idx].split())
            line_idx += 1
    return line_idx


def read_CHGCAR(filepath):
    """
    Read a CHGCAR-format volumetric file (CHGCAR, CHG, and PARCHG all share
    this layout): structural header (VASP4/5/6, same as read_POSCAR) plus
    one or two density grids. Grid values are returned exactly as stored,
    i.e. rho(r) * V_cell -- not divided by cell volume -- so they can be
    written back out as valid CHGCAR files with no renormalization.

    Non-collinear (LSORBIT/LNONCOLLINEAR) files, which carry three
    magnetization components (mx, my, mz) instead of one, raise an error.

    Returns a dict: lattice_matrix, elements, atom_counts, total_atoms,
    positions_direct, ngrid, ispin, density_tot, density_mag (None unless
    ISPIN = 2).
    """
    if not os.path.exists(filepath):
        print(f"ERROR!\nFile: {filepath} does not exist.")
        exit(1)

    with open(filepath, 'r') as f:
        lines = f.readlines()

    # ---- structural header (same layout as POSCAR) ----
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

    # Detect VASP4 vs VASP5/6 format by checking whether line 6 starts with
    # a number (VASP4 has no element-symbol line, so the user is prompted).
    is_number = lines[5].split()[0].isdecimal()
    if is_number:
        elements = []
        for i in range(len(lines[5].split())):
            while True:
                name = input(f"Enter the name of species No. {i + 1:>3}: ").strip()
                if name in _ELEMENT_SYMBOLS:
                    break
                else:
                    print("The name of species must be a valid element symbol.")
            elements.append(name)
        atom_counts = [int(x) for x in lines[5].split()]
        position_start = 7
    else:
        raw_elements = lines[5].split()
        elements = [name.split('/')[0].split('_')[0] for name in raw_elements]
        atom_counts = [int(x) for x in lines[6].split()]
        position_start = 8

    total_atoms = sum(atom_counts)
    position_stop = position_start + total_atoms
    positions = np.array([[float(x) for x in lines[i].split()[:3]]
                          for i in range(position_start, position_stop)])
    # CHGCAR atomic positions are always Direct (fractional) coordinates.
    positions_direct = positions % 1.0

    # ---- volumetric grid(s) ----
    line_idx = position_stop
    while lines[line_idx].strip() == '':
        line_idx += 1

    ngrid, density_tot, line_idx = _read_grid_block(lines, line_idx)
    line_idx = _skip_augmentation(lines, line_idx, total_atoms)

    ispin = 1
    density_mag = None
    peek = line_idx
    while peek < len(lines) and lines[peek].strip() == '':
        peek += 1
    if peek < len(lines) and _looks_like_grid_header(lines[peek], ngrid):
        ispin = 2
        _, density_mag, line_idx = _read_grid_block(lines, peek)
        line_idx = _skip_augmentation(lines, line_idx, total_atoms)

        # A third density block would mean a non-collinear (LSORBIT) CHGCAR.
        peek2 = line_idx
        while peek2 < len(lines) and lines[peek2].strip() == '':
            peek2 += 1
        if peek2 < len(lines) and _looks_like_grid_header(lines[peek2], ngrid):
            print("ERROR! This CHGCAR appears to carry more than 2 density "
                  "blocks (mx/my/mz), suggesting a non-collinear "
                  "(LSORBIT/LNONCOLLINEAR) calculation, which is not "
                  "currently supported.")
            exit(1)

    return {"lattice_matrix":    lattice_matrix,
            "elements":          elements,
            "atom_counts":       atom_counts,
            "total_atoms":       total_atoms,
            "positions_direct":  positions_direct,
            "ngrid":             ngrid,
            "ispin":             ispin,
            "density_tot":       density_tot,
            "density_mag":       density_mag}


def write_CHGCAR(filepath, lattice_matrix, elements, atom_counts,
                 positions_direct, ngrid, density_tot, density_mag=None,
                 comment="Generated by vaspChargeDiff.py"):
    """
    Write a CHGCAR-format file (structural header plus one or two density
    grids) in VASP's native rho*V_cell convention and x-fastest write
    order. No PAW augmentation-occupancies block is written, so this is
    fine for visualization (e.g. VESTA) but not for restarting VASP.
    """
    with open(filepath, 'w') as f:
        f.write(f"{comment}\n")
        f.write("   1.00000000000000\n")
        for row in lattice_matrix:
            f.write(f"    {row[0]:>13.8f}{row[1]:>13.8f}{row[2]:>13.8f}\n")
        f.write("   " + "  ".join(elements) + "\n")
        f.write("   " + "  ".join(str(c) for c in atom_counts) + "\n")
        f.write("Direct\n")
        for pos in positions_direct:
            f.write(f"  {pos[0]:>10.6f}{pos[1]:>10.6f}{pos[2]:>10.6f}\n")
        f.write("\n")

        def _write_grid(grid):
            f.write(f"{ngrid[0]:>5}{ngrid[1]:>5}{ngrid[2]:>5}\n")
            flat = grid.flatten(order='F')
            for i in range(0, len(flat), 5):
                f.write("".join(f"{v:>19.11E}" for v in flat[i:i + 5]) + "\n")

        _write_grid(density_tot)
        if density_mag is not None:
            f.write("\n")
            _write_grid(density_mag)


def task_charge_difference(chgcar_AB, chgcar_A, chgcar_B):
    """
    Charge-Density Difference -- Δρ = ρ_AB - ρ_A - ρ_B, e.g. for analyzing
    electron transfer in an adsorbate (A) + substrate (B) system. All three
    CHGCARs must share the same grid dimensions.
    """
    if not (chgcar_AB["ngrid"] == chgcar_A["ngrid"] == chgcar_B["ngrid"]):
        print(f"ERROR! Grid mismatch: AB{chgcar_AB['ngrid']}, "
              f"A{chgcar_A['ngrid']}, B{chgcar_B['ngrid']} must be "
              f"identical for a charge-density difference.")
        exit(1)

    for label, other in (("A", chgcar_A), ("B", chgcar_B)):
        if not np.allclose(chgcar_AB["lattice_matrix"], other["lattice_matrix"], atol=1e-4):
            print(f"Warning: lattice vectors of CHGCAR_{label} differ from "
                  f"CHGCAR_AB; the difference may not be physically "
                  f"meaningful. Continuing anyway.")

    return chgcar_AB["density_tot"] - chgcar_A["density_tot"] - chgcar_B["density_tot"]


def main():
    """Parse arguments, read the three CHGCARs, compute the difference, and write output."""
    if '-h' in argv or '--help' in argv or len(argv) != 4:
        usage()

    path_AB, path_A, path_B = argv[1], argv[2], argv[3]

    chgcar_AB = read_CHGCAR(path_AB)
    chgcar_A = read_CHGCAR(path_A)
    chgcar_B = read_CHGCAR(path_B)
    print(f"Read {path_AB}: grid {chgcar_AB['ngrid']}, {chgcar_AB['total_atoms']} atoms")
    print(f"Read {path_A}: grid {chgcar_A['ngrid']}, {chgcar_A['total_atoms']} atoms")
    print(f"Read {path_B}: grid {chgcar_B['ngrid']}, {chgcar_B['total_atoms']} atoms")

    diff = task_charge_difference(chgcar_AB, chgcar_A, chgcar_B)

    out_path = "CHGDIFF.vasp"
    write_CHGCAR(out_path, chgcar_AB["lattice_matrix"], chgcar_AB["elements"],
                chgcar_AB["atom_counts"], chgcar_AB["positions_direct"],
                chgcar_AB["ngrid"], diff)
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
