#!/usr/bin/env python

from sys import argv, exit
import os
import readline
import numpy as np


def usage():
    """Print usage information and exit."""
    print("""
Usage: vaspCharge.py <CHGCAR>

Reads a CHGCAR file (CHG/PARCHG share the same layout) and prompts for one
of the following:

   1) Slice of Charge Density        -> SLICE_X.grd, SLICE_Y.grd, CHGSLICE.grd
   2) Charge Density                 -> CHGTOT.vasp
   3) Spin Density                   -> CHGMAG.vasp (ISPIN = 2 only)
   4) Spin-Up & Spin-Down Density    -> CHGUP.vasp, CHGDW.vasp (ISPIN = 2 only)
   5) Linear-Average Charge Density  -> LAVG_X.grd, LAVG_Y.grd, CHGLAVG.grd
   6) Planar-Average Charge Density  -> PLANAR_AVERAGE.dat
   7) Macroscopic-Average Charge Density -> MACROSCOPIC_AVERAGE.dat
   8) Charge Density Along a Specified Path -> CHG_LINE.dat
   9) Build Supercell of Charge Density -> CHGCAR_SUPERCELL.vasp
  10) STM Simulation                 -> STM_X/Y.grd + STM_HEIGHT.grd or STM_CURRENT.grd

Non-collinear CHGCAR files are not supported, but will be supported in a future version.

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
                 positions_direct, ngrid, density_tot, density_mag=None):
    """
    Write a CHGCAR-format file (structural header plus one or two density
    grids) in VASP's native rho*V_cell convention and x-fastest write
    order. No PAW augmentation-occupancies block is written, so this is
    fine for visualization (e.g. VESTA) but not for restarting VASP.
    """
    with open(filepath, 'w') as o:
        o.write("Generated by vaspCharge.py\n")
        o.write(f"   {1.0:.8f}\n")
        for row in lattice_matrix:
            o.write(f"    {row[0]:>13.8f}{row[1]:>13.8f}{row[2]:>13.8f}\n")
        o.write("   " + "  ".join(elements) + "\n")
        o.write("   " + "  ".join(str(c) for c in atom_counts) + "\n")
        o.write("Direct\n")
        for pos in positions_direct:
            o.write(f"  {pos[0]:>10.6f}{pos[1]:>10.6f}{pos[2]:>10.6f}\n")
        o.write("\n")

        def _write_grid(grid):
            o.write(f"{ngrid[0]:>5}{ngrid[1]:>5}{ngrid[2]:>5}\n")
            flat = grid.flatten(order='F')
            for i in range(0, len(flat), 5):
                o.write("".join(f"{v:>19.11E}" for v in flat[i:i + 5]) + "\n")

        _write_grid(density_tot)
        if density_mag is not None:
            o.write("\n")
            _write_grid(density_mag)


def task_charge_density(chgcar):
    """2) Charge Density -- total electron density, exactly as stored."""
    return chgcar["density_tot"]


def task_spin_density(chgcar):
    """3) Spin Density -- magnetization (rho_up - rho_down)."""
    if chgcar["ispin"] != 2:
        print("ERROR! Task 3 (Spin Density) requires a spin-polarized "
              "CHGCAR (ISPIN = 2); this file has no magnetization block.")
        exit(1)
    return chgcar["density_mag"]


def task_spin_updown(chgcar):
    """
    4) Spin-Up & Spin-Down Density -- derived from the total and
    magnetization blocks: rho_up = (tot + mag) / 2, rho_dw = (tot - mag) / 2.
    """
    if chgcar["ispin"] != 2:
        print("ERROR! Task 4 (Spin-Up & Spin-Down Density) requires a "
              "spin-polarized CHGCAR (ISPIN = 2); this file has no "
              "magnetization block.")
        exit(1)
    density_up = (chgcar["density_tot"] + chgcar["density_mag"]) / 2.0
    density_dw = (chgcar["density_tot"] - chgcar["density_mag"]) / 2.0
    return density_up, density_dw


def cell_volume(lattice_matrix):
    """Return the unit-cell volume (Å**3) as |det(lattice_matrix)|."""
    return abs(np.linalg.det(lattice_matrix))


def prompt_axis(prompt_text="Enter the lattice axis (A/B/C): "):
    """Prompt for A/B/C (re-prompting on invalid input); returns 0/1/2."""
    valid = ('A', 'B', 'C')
    axis = input(prompt_text).strip().upper()
    while axis not in valid:
        axis = input(f"Invalid entry, must be A, B, or C. {prompt_text}").strip().upper()
    return {'A': 0, 'B': 1, 'C': 2}[axis]


def _build_plane_XY(lattice_matrix, ngrid, axis_idx):
    """
    Build Cartesian X, Y coordinate grids for the plane perpendicular to
    ``axis_idx``, spanned by the other two lattice vectors. Uses only
    their x/y components, which is exact for a slab-in-vacuum cell but
    approximate for a strongly non-orthogonal one.
    """
    other_idx = [i for i in range(3) if i != axis_idx]
    n1, n2 = ngrid[other_idx[0]], ngrid[other_idx[1]]
    vec1, vec2 = lattice_matrix[other_idx[0]], lattice_matrix[other_idx[1]]

    i1, i2 = np.meshgrid(np.arange(n1), np.arange(n2), indexing='ij')
    frac1, frac2 = i1 / n1, i2 / n2
    X = frac1 * vec1[0] + frac2 * vec2[0]
    Y = frac1 * vec1[1] + frac2 * vec2[1]
    return X, Y


def _axis_positions(lattice_matrix, ngrid, axis_idx):
    """1D array of Cartesian distances (Å) along ``axis_idx`` at each grid point."""
    n = ngrid[axis_idx]
    length = np.linalg.norm(lattice_matrix[axis_idx])
    return np.arange(n) / n * length


def _slice_along_axis(grid, axis_idx, frac):
    """2D cross-section of ``grid`` perpendicular to ``axis_idx`` at fractional
    coordinate ``frac``, linearly interpolated between the two nearest planes."""
    n = grid.shape[axis_idx]
    pos = (frac % 1.0) * n
    i0 = int(np.floor(pos)) % n
    i1 = (i0 + 1) % n
    w = pos - np.floor(pos)
    s0 = np.take(grid, i0, axis=axis_idx)
    s1 = np.take(grid, i1, axis=axis_idx)
    return (1 - w) * s0 + w * s1


def _write_grd(path, array):
    """Write a plain whitespace-delimited 2D grid, readable by plotGrid.py."""
    np.savetxt(path, array, fmt='%.6f')


def task_slice(chgcar):
    """
    1) Slice of Charge Density -- 2D cross-section of the physical
    density (e/A^3) on a plane perpendicular to a chosen axis, at a
    chosen fractional coordinate (interpolated between grid planes).
    Writes SLICE_X.grd, SLICE_Y.grd, CHGSLICE.grd (readable by plotGrid.py).
    """
    axis_idx = prompt_axis("Enter the lattice axis normal to the slice plane (A/B/C): ")
    frac = float(input("Enter the fractional coordinate along that axis (0-1): "))

    volume = cell_volume(chgcar["lattice_matrix"])
    density = chgcar["density_tot"] / volume
    density_slice = _slice_along_axis(density, axis_idx, frac)

    X, Y = _build_plane_XY(chgcar["lattice_matrix"], chgcar["ngrid"], axis_idx)

    _write_grd("SLICE_X.grd", X)
    _write_grd("SLICE_Y.grd", Y)
    _write_grd("CHGSLICE.grd", density_slice)
    print("Wrote SLICE_X.grd, SLICE_Y.grd, CHGSLICE.grd")


def task_linear_average(chgcar):
    """
    5) Linear-Average Charge Density -- average the physical density
    (e/A^3) along ONE chosen axis, leaving a 2D map over the other two.
    Writes LAVG_X.grd, LAVG_Y.grd, CHGLAVG.grd for plotGrid.py.
    """
    axis_idx = prompt_axis("Enter the lattice axis to average over (A/B/C): ")
    volume = cell_volume(chgcar["lattice_matrix"])
    density = chgcar["density_tot"] / volume
    density_avg = density.mean(axis=axis_idx)

    X, Y = _build_plane_XY(chgcar["lattice_matrix"], chgcar["ngrid"], axis_idx)

    _write_grd("LAVG_X.grd", X)
    _write_grd("LAVG_Y.grd", Y)
    _write_grd("CHGLAVG.grd", density_avg)
    print("Wrote LAVG_X.grd, LAVG_Y.grd, CHGLAVG.grd -- ready for plotGrid.py")


def _planar_average_profile(chgcar, axis_idx):
    """
    Average the physical density (e/A^3) over the plane perpendicular to
    ``axis_idx``, giving a 1D profile. Returns (positions, profile).
    """
    other_idx = tuple(i for i in range(3) if i != axis_idx)
    volume = cell_volume(chgcar["lattice_matrix"])
    density = chgcar["density_tot"] / volume
    profile = density.mean(axis=other_idx)
    positions = _axis_positions(chgcar["lattice_matrix"], chgcar["ngrid"], axis_idx)
    return positions, profile


def task_planar_average(chgcar):
    """
    6) Planar-Average Charge Density -- 1D profile of the physical
    density (e/A^3) along a chosen axis. Writes PLANAR_AVERAGE.dat.
    """
    axis_idx = prompt_axis("Enter the lattice axis to profile along (A/B/C): ")
    positions, profile = _planar_average_profile(chgcar, axis_idx)

    out_path = "PLANAR_AVERAGE.dat"
    with open(out_path, 'w') as f:
        f.write("#Position(Angstrom) Planar-Averaged-Density(e/A^3)\n")
        for pos, val in zip(positions, profile):
            f.write(f"{pos:12.6f}{val:16.8f}\n")
    print(f"Wrote {out_path}")


def _circular_moving_average(profile, window_points):
    """Periodic boxcar moving average via FFT convolution (profile is periodic)."""
    n = len(profile)
    window_points = min(max(window_points, 1), n)
    kernel = np.zeros(n)
    half = window_points // 2
    kernel[:half + 1] = 1.0
    if half > 0:
        kernel[-half:] = 1.0
    kernel /= kernel.sum()
    return np.real(np.fft.ifft(np.fft.fft(profile) * np.fft.fft(kernel)))


def task_macroscopic_average(chgcar):
    """
    7) Macroscopic-Average Charge Density -- smooths the planar-average
    profile (task 6) with a periodic moving-window average, commonly used
    for work-function / band-alignment analysis. Window length should
    normally be one interplanar spacing. Writes MACROSCOPIC_AVERAGE.dat.
    """
    axis_idx = prompt_axis("Enter the lattice axis to profile along (A/B/C): ")
    positions, profile = _planar_average_profile(chgcar, axis_idx)

    window_length = float(input("Enter the macroscopic-averaging window "
                                 "length in Angstrom (e.g. one interplanar "
                                 "spacing): "))
    dz = positions[1] - positions[0]
    window_points = max(1, round(window_length / dz))

    macro = _circular_moving_average(profile, window_points)

    out_path = "MACROSCOPIC_AVERAGE.dat"
    with open(out_path, 'w') as f:
        f.write("#Position(Angstrom) Planar-Average(e/A^3) Macroscopic-Average(e/A^3)\n")
        for pos, pl, mc in zip(positions, profile, macro):
            f.write(f"{pos:12.6f}{pl:16.8f}{mc:16.8f}\n")
    print(f"Wrote {out_path}")


def _trilinear_interp_periodic(grid, frac_coord):
    """Trilinear interpolation at an arbitrary fractional point, wrapping at cell boundaries."""
    nx, ny, nz = grid.shape
    fx, fy, fz = (c % 1.0 for c in frac_coord)
    x, y, z = fx * nx, fy * ny, fz * nz

    x0, y0, z0 = int(np.floor(x)) % nx, int(np.floor(y)) % ny, int(np.floor(z)) % nz
    x1, y1, z1 = (x0 + 1) % nx, (y0 + 1) % ny, (z0 + 1) % nz
    wx, wy, wz = x - np.floor(x), y - np.floor(y), z - np.floor(z)

    c000, c100 = grid[x0, y0, z0], grid[x1, y0, z0]
    c010, c110 = grid[x0, y1, z0], grid[x1, y1, z0]
    c001, c101 = grid[x0, y0, z1], grid[x1, y0, z1]
    c011, c111 = grid[x0, y1, z1], grid[x1, y1, z1]

    c00 = c000 * (1 - wx) + c100 * wx
    c10 = c010 * (1 - wx) + c110 * wx
    c01 = c001 * (1 - wx) + c101 * wx
    c11 = c011 * (1 - wx) + c111 * wx

    c0 = c00 * (1 - wy) + c10 * wy
    c1 = c01 * (1 - wy) + c11 * wy

    return c0 * (1 - wz) + c1 * wz


def _prompt_fractional_point(prompt_text):
    """Interactively prompt for three whitespace-separated fractional coordinates."""
    while True:
        parts = input(prompt_text).split()
        if len(parts) == 3:
            try:
                return np.array([float(x) for x in parts])
            except ValueError:
                pass
        print("Invalid entry, enter three numbers separated by spaces "
              "(fractional a b c coordinates).")


def task_line_profile(chgcar):
    """
    8) Charge Density Along a Specified Path -- interpolated physical
    density (e/A^3) along a line between two fractional points. Writes
    CHG_LINE.dat.
    """
    point1 = _prompt_fractional_point("Enter the starting point (fractional a b c): ")
    point2 = _prompt_fractional_point("Enter the ending point (fractional a b c): ")
    n_points = int(input("Enter the number of sampling points along the path: "))
    if n_points < 2:
        print("ERROR! The number of sampling points must be at least 2.")
        exit(1)

    volume = cell_volume(chgcar["lattice_matrix"])
    density = chgcar["density_tot"] / volume
    lattice_matrix = chgcar["lattice_matrix"]

    cart1 = point1 @ lattice_matrix
    cart2 = point2 @ lattice_matrix
    path_length = np.linalg.norm(cart2 - cart1)

    distances = np.linspace(0.0, path_length, n_points)
    values = np.empty(n_points)
    for i, t in enumerate(np.linspace(0.0, 1.0, n_points)):
        frac = (1 - t) * point1 + t * point2
        values[i] = _trilinear_interp_periodic(density, frac)

    out_path = "CHG_LINE.dat"
    with open(out_path, 'w') as f:
        f.write("#Distance(Angstrom) Density(e/A^3)\n")
        for d, v in zip(distances, values):
            f.write(f"{d:12.6f}{v:16.8f}\n")
    print(f"Wrote {out_path}")


def task_build_supercell(chgcar):
    """
    9) Build Supercell of Charge Density -- tiles the density grid and
    structure by integer multiples (n1, n2, n3); diagonal transformations
    only, no arbitrary 3x3 matrix. Tiled values are also scaled by
    n1*n2*n3, since CHGCAR stores rho(r)*V_cell and plain tiling would
    leave values normalized to the original (smaller) cell's volume.
    Writes CHGCAR_SUPERCELL.vasp.
    """
    print("\nBuild Supercell of Charge Density: diagonal transformation only "
          "(n1 x n2 x n3 integer multiples).")
    n1 = int(input("Enter the multiple along axis A: "))
    n2 = int(input("Enter the multiple along axis B: "))
    n3 = int(input("Enter the multiple along axis C: "))
    if n1 < 1 or n2 < 1 or n3 < 1:
        print("ERROR! Supercell multiples must be positive integers.")
        exit(1)

    replication_factor = n1 * n2 * n3
    new_lattice = chgcar["lattice_matrix"] * np.array([[n1], [n2], [n3]])

    new_density_tot = replication_factor * np.tile(chgcar["density_tot"], (n1, n2, n3))
    new_density_mag = None
    if chgcar["ispin"] == 2:
        new_density_mag = replication_factor * np.tile(chgcar["density_mag"], (n1, n2, n3))

    # Replicate atomic positions element-by-element so the output stays
    # grouped into contiguous element blocks, matching new_atom_counts.
    multiples = np.array([n1, n2, n3])
    position_blocks = []
    start = 0
    for count in chgcar["atom_counts"]:
        block = chgcar["positions_direct"][start:start + count]
        start += count
        replicas = [(block + np.array([i, j, k])) / multiples
                    for i in range(n1) for j in range(n2) for k in range(n3)]
        position_blocks.append(np.vstack(replicas))
    new_positions = np.vstack(position_blocks)
    new_atom_counts = [c * replication_factor for c in chgcar["atom_counts"]]

    new_ngrid = (chgcar["ngrid"][0] * n1, chgcar["ngrid"][1] * n2, chgcar["ngrid"][2] * n3)

    out_path = "CHGCAR_SUPERCELL.vasp"
    write_CHGCAR(out_path, new_lattice, chgcar["elements"], new_atom_counts,
                new_positions, new_ngrid, new_density_tot, new_density_mag)
    print(f"Wrote {out_path} ({n1}x{n2}x{n3} supercell, grid {new_ngrid})")


def task_stm_simulation(chgcar):
    """
    10) STM Simulation -- Tersoff-Hamann-style image from a (partial)
    charge density, e.g. a PARCHG near E_F. Assumes the surface normal is
    axis C with vacuum on the +C side. Constant-height mode reports
    I(x,y) at a fixed height; constant-current mode reports the height
    z(x,y) where density first crosses a chosen threshold, scanning in
    from vacuum. Writes STM_X.grd, STM_Y.grd, and STM_HEIGHT.grd or
    STM_CURRENT.grd.
    """
    volume = cell_volume(chgcar["lattice_matrix"])
    density = chgcar["density_tot"] / volume

    print("""
STM Simulation modes:
1) Constant-height
2) Constant-current""")
    while True:
        mode = input("Enter Choice: ").strip()
        if mode in ('1', '2'):
            break
        print("ERROR! Choose again.")

    X, Y = _build_plane_XY(chgcar["lattice_matrix"], chgcar["ngrid"], 2)
    _write_grd("STM_X.grd", X)
    _write_grd("STM_Y.grd", Y)

    if mode == '1':
        frac_c = float(input("Enter the fractional height along axis C (0-1): "))
        image = _slice_along_axis(density, 2, frac_c)
        out_path = "STM_HEIGHT.grd"
        _write_grd(out_path, image)
    else:
        threshold = float(input("Enter the target density threshold (e/A^3): "))
        nz = density.shape[2]
        c_length = np.linalg.norm(chgcar["lattice_matrix"][2])

        height_map = np.full(density.shape[:2], np.nan)
        found = np.zeros(density.shape[:2], dtype=bool)
        for k in range(nz - 1, -1, -1):
            layer = density[:, :, k]
            newly_crossed = (~found) & (layer >= threshold)
            height_map[newly_crossed] = (k / nz) * c_length
            found |= newly_crossed
        if not found.all():
            print("Warning: the density threshold was not reached (scanning "
                  "in from the vacuum side, +C) for some (x, y) points; "
                  "those entries are written as NaN.")

        out_path = "STM_CURRENT.grd"
        _write_grd(out_path, height_map)

    print(f"Wrote STM_X.grd, STM_Y.grd, {out_path}")


def main():
    """Parse arguments, read the CHGCAR, dispatch to the chosen task, and write output."""
    if '-h' in argv or '--help' in argv or len(argv) != 2:
        usage()

    chgcar_path = argv[1]

    chgcar = read_CHGCAR(chgcar_path)
    print(f"Read {chgcar_path}: grid {chgcar['ngrid']}, "
          f"{chgcar['total_atoms']} atoms, ISPIN = {chgcar['ispin']}")

    print("""
Choices of charge-density analysis:
 1) Slice of Charge Density
 2) Charge Density
 3) Spin Density
 4) Spin-Up & Spin-Down Density
 5) Linear-Average Charge Density
 6) Planar-Average Charge Density
 7) Macroscopic-Average Charge Density
 8) Charge Density Along a Specified Path
 9) Build Supercell of Charge Density
10) STM Simulation""")
    valid_choices = [str(i) for i in range(1, 11)]
    while True:
        task = input("Enter Choice: ").strip()
        if task in valid_choices:
            break
        print("ERROR! Choose again.")

    common_args = (chgcar["lattice_matrix"], chgcar["elements"], chgcar["atom_counts"],
                   chgcar["positions_direct"], chgcar["ngrid"])

    if task == '1':
        task_slice(chgcar)

    elif task == '2':
        density_tot = task_charge_density(chgcar)
        out_path = "CHGTOT.vasp"
        write_CHGCAR(out_path, *common_args, density_tot)
        print(f"Wrote {out_path}")

    elif task == '3':
        density_mag = task_spin_density(chgcar)
        out_path = "CHGMAG.vasp"
        write_CHGCAR(out_path, *common_args, density_mag)
        print(f"Wrote {out_path}")

    elif task == '4':
        density_up, density_dw = task_spin_updown(chgcar)
        up_path = "CHGUP.vasp"
        dw_path = "CHGDW.vasp"
        write_CHGCAR(up_path, *common_args, density_up)
        write_CHGCAR(dw_path, *common_args, density_dw)
        print(f"Wrote {up_path}, {dw_path}")

    elif task == '5':
        task_linear_average(chgcar)

    elif task == '6':
        task_planar_average(chgcar)

    elif task == '7':
        task_macroscopic_average(chgcar)

    elif task == '8':
        task_line_profile(chgcar)

    elif task == '9':
        task_build_supercell(chgcar)

    elif task == '10':
        task_stm_simulation(chgcar)


if __name__ == "__main__":
    main()
