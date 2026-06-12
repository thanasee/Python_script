#!/usr/bin/env python

from sys import argv, exit
import os
import re
import numpy as np


def usage():
    """Print usage information and exit."""
    print("""
Usage: analyzeNEB.py [nj]

Analyze VASP NEB outputs and generate summary files.

Arguments:
    nj   (optional)  Number of spline interpolation sub-steps per interval (default: 20)

Outputs:
    neb.dat        Cumulative distance, energy, tangent force per image
    nebss.dat      SSNEB-normalised neb.dat (SSNEB mode only)
    spline.dat     Dense cubic spline along the MEP
    spliness.dat   Dense cubic spline, SSNEB-normalised (SSNEB mode only)
    exts.dat       Extrema (transition states / minima) along the MEP
    extsss.dat     Extrema, SSNEB-normalised (SSNEB mode only)
    nebef.dat      Image-by-image force and energy table
    movie          Concatenated POSCAR-format movie (from CONTCARs)
    movie.xyz      Concatenated XYZ movie
    <img>/fe.dat   Per-image force/energy convergence history

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
        o.write("Generated by analyzeNEB.py\n")
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


def _last_match(filepath, pattern):
    """Return the last line in filepath containing pattern, stripped.

    Parameters
    ----------
    filepath : str
    pattern  : str — substring to match

    Returns
    -------
    str — last matching line, or ''
    """
    result = ''
    with open(filepath) as f:
        for line in f:
            if pattern in line:
                result = line
    return result.rstrip()


def _all_matches(filepath, pattern):
    """Return all lines in filepath containing pattern.

    Parameters
    ----------
    filepath : str
    pattern  : str

    Returns
    -------
    list[str]
    """
    results = []
    with open(filepath) as f:
        for line in f:
            if pattern in line:
                results.append(line.rstrip())
    return results


def _parse_energy(outcar):
    """Extract the final ionic-step energy without entropy (eV) from OUTCAR.

    Parameters
    ----------
    outcar : str — path to OUTCAR

    Returns
    -------
    float — energy in eV
    """
    line = _last_match(outcar, 'energy  without entropy')
    if not line:
        print(f"ERROR! Could not parse energy from {outcar}")
        exit(1)
    return float(line.split()[-1])


def _parse_nions(outcar):
    """Extract NIONS (total number of ions) from OUTCAR.

    Parameters
    ----------
    outcar : str

    Returns
    -------
    int
    """
    line = _last_match(outcar, 'NIONS')
    return int(line.split()[-1])


def _parse_max_force(outcar):
    """Extract the maximum Cartesian force (eV/Angstrom) from the last ionic step.

    Parameters
    ----------
    outcar : str

    Returns
    -------
    float
    """
    line = _last_match(outcar, 'max at')
    if not line:
        return 0.0
    # 'RMS ... max at atom ...' — force value is the token after 'at atom N'
    parts = line.split()
    try:
        idx = parts.index('max') + 3
        return float(parts[idx])
    except (ValueError, IndexError):
        return 0.0


def _parse_total_forces(outcar, nions):
    """Extract Cartesian forces on all atoms from the last ionic step.

    Reads the block following 'TOTAL-FORCE (eV/Angst)' and takes the last
    nions rows, which correspond to the final ionic step.

    Parameters
    ----------
    outcar : str
    nions  : int — number of ions

    Returns
    -------
    np.ndarray, shape (nions, 3) — forces in eV/Angstrom
    """
    forces = []
    collect = False
    count = 0
    with open(outcar) as f:
        for line in f:
            if 'TOTAL-FORCE (eV/Angst)' in line:
                forces = []
                collect = True
                count = 0
                continue
            if collect:
                parts = line.split()
                if len(parts) >= 6:
                    try:
                        forces.append([float(parts[3]), float(parts[4]), float(parts[5])])
                        count += 1
                        if count == nions:
                            collect = False
                    except ValueError:
                        collect = False
    if len(forces) >= nions:
        return np.array(forces[-nions:], dtype=float)
    return np.zeros((nions, 3))


def _parse_stress(outcar):
    """Extract total stress (kBar) from the last ionic step in OUTCAR.

    Parameters
    ----------
    outcar : str

    Returns
    -------
    float
    """
    line = _last_match(outcar, 'Stress total and')
    if not line:
        return 0.0
    try:
        return float(line.split()[-1])
    except (ValueError, IndexError):
        return 0.0


def _parse_volume(outcar):
    """Extract unit-cell volume (Angstrom^3) from OUTCAR.

    Parameters
    ----------
    outcar : str

    Returns
    -------
    float
    """
    line = _last_match(outcar, 'volume of cell')
    if not line:
        return 0.0
    try:
        return float(line.split()[-1])
    except (ValueError, IndexError):
        return 0.0


def _parse_magmom(outcar):
    """Extract total magnetic moment (mu_B) from the last ionic step in OUTCAR.

    Parameters
    ----------
    outcar : str

    Returns
    -------
    float
    """
    line = _last_match(outcar, 'number of electron')
    if not line:
        return 0.0
    try:
        return float(line.split()[-1])
    except (ValueError, IndexError):
        return 0.0


def _detect_ssneb(outcar_01):
    """Detect SSNEB mode by checking LNEBCELL flag in 01/OUTCAR.

    Parameters
    ----------
    outcar_01 : str — path to 01/OUTCAR

    Returns
    -------
    bool — True when LNEBCELL = T
    """
    line = _last_match(outcar_01, 'LNEBCELL')
    return bool(line) and line.split()[-1].upper() == 'T'


def _poscar_path(image_dir):
    """Return CONTCAR path if it exists, else POSCAR path.

    Parameters
    ----------
    image_dir : str

    Returns
    -------
    str
    """
    contcar = os.path.join(image_dir, 'CONTCAR')
    poscar  = os.path.join(image_dir, 'POSCAR')
    return contcar if os.path.isfile(contcar) else poscar


def _inter_image_distance(poscar_i, poscar_j):
    """Compute the Euclidean inter-image distance (Angstrom) with minimum-image convention.

    Distance = sqrt( sum_atoms |delta_r|^2 ) where delta_r are minimum-image
    Cartesian displacements. Mirrors the dist.pl VTST script.

    Parameters
    ----------
    poscar_i : dict — image i   (from read_POSCAR)
    poscar_j : dict — image j   (from read_POSCAR)

    Returns
    -------
    float — distance in Angstrom
    """
    lat = poscar_i['lattice_matrix']
    # fractional difference with minimum-image wrapping
    frac_delta = poscar_i['positions_direct'] - poscar_j['positions_direct']
    frac_delta -= np.round(frac_delta)
    cart_delta = frac_delta @ lat
    return float(np.sqrt(np.sum(cart_delta ** 2)))


def _endpoint_tangent_force(poscar_self, poscar_neighbor, outcar):
    """Compute the tangent-projected force for an NEB endpoint image.

    Projects the total DFT force onto the unit tangent vector pointing from
    poscar_self toward poscar_neighbor (into the chain), as in nebforces.pl.

    Parameters
    ----------
    poscar_self     : dict — endpoint image (read_POSCAR)
    poscar_neighbor : dict — adjacent image  (read_POSCAR)
    outcar          : str  — OUTCAR of the endpoint image

    Returns
    -------
    float — tangent force component in eV/Angstrom
    """
    nions = poscar_self['total_atoms']
    forces = _parse_total_forces(outcar, nions)

    lat = poscar_self['lattice_matrix']
    frac_delta = poscar_neighbor['positions_direct'] - poscar_self['positions_direct']
    frac_delta -= np.round(frac_delta)
    tangent = (frac_delta @ lat).ravel()
    norm = np.linalg.norm(tangent)
    if norm < 1e-10:
        return 0.0
    tangent /= norm
    return float(np.dot(forces.ravel(), tangent))


def compute_barrier(directories, ssneb=False):
    """Compute NEB barrier data from image OUTCARs and write neb.dat / nebss.dat.

    For each image the cumulative inter-image distance, relative energy, and
    tangent force are extracted. Endpoint forces are corrected by projecting
    the DFT force onto the local tangent (nebforces.pl logic). In SSNEB mode
    distances and forces are read from 'NEBCELL: distance/projections' tags.

    Parameters
    ----------
    directories : list[str] — image directory names in sorted order (e.g. ['00','01',...])
    ssneb       : bool      — True for solid-state NEB (LNEBCELL=T)

    Returns
    -------
    list[dict], one per image, with keys:
        index    : int   — image index
        dist_cum : float — cumulative path length in Angstrom
        energy   : float — energy relative to image 00 in eV
        force    : float — tangent force in eV/Angstrom
        dir      : int   — directory index
    """
    nions    = _parse_nions(os.path.join(directories[0], 'OUTCAR'))
    energy0  = None
    dist_cum = 0.0
    records  = []

    for i, d in enumerate(directories):
        outcar = os.path.join(d, 'OUTCAR')
        if not os.path.isfile(outcar):
            print(f"ERROR! No OUTCAR in {d}")
            exit(1)

        energy = _parse_energy(outcar)
        if energy0 is None:
            energy0 = energy
        rel_energy = energy - energy0

        # --- inter-image distance ---
        if i == 0:
            dist = 0.0
        elif ssneb:
            # SSNEB: read pre-computed distance from OUTCAR tag
            line = _last_match(outcar, 'NEBCELL: distance')
            parts = line.split()
            # non-final images: field at index -3; final image: reuse previous dist
            dist = float(parts[-3]) if i < len(directories) - 1 else records[-1]['_dist']
        else:
            p_i   = read_POSCAR(_poscar_path(directories[i]))
            p_im1 = read_POSCAR(_poscar_path(directories[i - 1]))
            dist  = _inter_image_distance(p_i, p_im1)

        dist_cum += dist

        # --- tangent force ---
        if ssneb:
            line  = _last_match(outcar, 'NEBCELL: projections')
            force = float(line.split()[-1]) if line else 0.0
        else:
            line  = _last_match(outcar, 'NEB: projections')
            force = float(line.split()[-1]) if line else 0.0

            # endpoint correction (nebforces.pl): project full DFT force onto tangent
            if i == 0:
                p_self  = read_POSCAR(_poscar_path(directories[0]))
                p_neigh = read_POSCAR(_poscar_path(directories[1]))
                force   = _endpoint_tangent_force(p_self, p_neigh, outcar)
            elif i == len(directories) - 1:
                p_self  = read_POSCAR(_poscar_path(directories[-1]))
                p_neigh = read_POSCAR(_poscar_path(directories[-2]))
                force   = _endpoint_tangent_force(p_self, p_neigh, outcar)

        records.append({
            'index':    i,
            'dist_cum': dist_cum,
            'energy':   rel_energy,
            'force':    force,
            '_dist':    dist,
            'dir':      int(d),
        })

    # write neb.dat
    with open('neb.dat', 'w') as f:
        for r in records:
            f.write(f"{r['index']:3d} {r['dist_cum']:12.6f} {r['energy']:12.6f} "
                    f"{r['force']:12.6f} {r['dir']:3d}\n")

    # write nebss.dat (SSNEB-normalised per atom)
    if ssneb:
        sq = np.sqrt(nions)
        with open('nebss.dat', 'w') as f:
            for r in records:
                f.write(f"{r['index']:3d} {r['dist_cum'] / sq:12.6f} "
                        f"{r['energy'] / nions:12.6f} {r['force'] / sq:12.6f} "
                        f"{r['dir']:3d}\n")

    return records


def compute_spline(input_file, output_spline, output_exts, nj=20):
    """Fit a piecewise Hermite cubic spline to a NEB path and locate extrema.

    Reads (index, distance, energy, force) columns from input_file.
    The cubic on interval i, parameterised by f in [0, 1], is:
        E(f) = d*f^3 + c*f^2 + b*f + a
    with Hermite boundary conditions:
        E(0) = E_i,   E(1) = E_{i+1}
        dE/df|_0 = -F_i * dR,  dE/df|_1 = -F_{i+1} * dR
    Extrema are found analytically from dE/df = 0 (quadratic or cubic roots).

    Parameters
    ----------
    input_file    : str — neb.dat or nebss.dat
    output_spline : str — spline.dat or spliness.dat
    output_exts   : str — exts.dat or extsss.dat
    nj            : int — sub-steps per interval (default 20)
    """
    data = np.loadtxt(input_file)
    R = data[:, 1]
    E = data[:, 2]
    F = data[:, 3]
    n = len(R)

    # Hermite cubic coefficients per interval
    a = np.zeros(n - 1)
    b = np.zeros(n - 1)
    c = np.zeros(n - 1)
    d = np.zeros(n - 1)

    for i in range(n - 1):
        dR  = R[i + 1] - R[i]
        F1  = F[i]     * dR
        F2  = F[i + 1] * dR
        Ud  = E[i + 1] - E[i]
        Fs  = F1 + F2
        a[i] =  E[i]
        b[i] = -F1
        c[i] =  3 * Ud + F1 + Fs
        d[i] = -2 * Ud - Fs

    # Dense interpolated path
    rows = []
    for i in range(n):
        rows.append(f"{i}\t{R[i]:.8f}\t{E[i]:.8f}\t{F[i]:.8f}")
        if i < n - 1:
            dR = R[i + 1] - R[i]
            for j in range(1, nj):
                f    = j / nj
                Rspl = R[i] + f * dR
                Espl = d[i]*f**3 + c[i]*f**2 + b[i]*f + a[i]
                Fspl = -(3*d[i]*f**2 + 2*c[i]*f + b[i]) / dR
                rows.append(f"{i + f:.4f}\t{Rspl:.8f}\t{Espl:.8f}\t{Fspl:.8f}")

    with open(output_spline, 'w') as f:
        f.write('\n'.join(rows) + '\n')

    # Extrema: solve 3*d*f^2 + 2*c*f + b = 0 on each interval
    extrema = {}
    for i in range(n - 1):
        disc = c[i]**2 - 3 * b[i] * d[i]
        if disc < 0:
            continue
        candidates = []
        if d[i] == 0 and c[i] != 0:
            # quadratic: b + 2*c*f = 0 → f = -b / (2*c)
            candidates.append(-b[i] / (2 * c[i]))
        elif d[i] != 0:
            sq = np.sqrt(disc)
            candidates.append((-c[i] + sq) / (3 * d[i]) * -1)
            candidates.append((-c[i] - sq) / (3 * d[i]) * -1)
        for f in candidates:
            if 0.0 <= f <= 1.0:
                extrema[i + f] = d[i]*f**3 + c[i]*f**2 + b[i]*f + a[i]

    label = 'Extremum' if output_exts == 'exts.dat' else 'Extrema'
    with open(output_exts, 'w') as f:
        for k, (pos, val) in enumerate(sorted(extrema.items()), 1):
            f.write(f"{label} {k} found at image {pos:9.6f} with energy: {val:9.6f}\n")


def report_ef(directories, output='nebef.dat'):
    """Write and print a force/energy table for all NEB images.

    Columns: image index, max force (eV/Angstrom), absolute energy (eV),
    energy relative to image 00 (eV).

    Parameters
    ----------
    directories : list[str]
    output      : str — output filename (default 'nebef.dat')
    """
    energy0 = None
    lines   = []
    for i, d in enumerate(directories):
        outcar = os.path.join(d, 'OUTCAR')
        energy = _parse_energy(outcar)
        force  = _parse_max_force(outcar)
        if energy0 is None:
            energy0 = energy
        rel  = energy - energy0
        lines.append(f"{i:4d} {force:16.6f} {energy:16.6f} {rel:16.6f}")

    with open(output, 'w') as f:
        f.write('\n'.join(lines) + '\n')
    print('\n'.join(lines))


def report_efs(directories, output='nebefs.dat'):
    """Write and print an extended force/energy/stress/volume/magmom table.

    Intended for SSNEB runs. Columns: image, max force, stress (kBar),
    volume (Angstrom^3), magnetic moment (mu_B), relative energy (eV).

    Parameters
    ----------
    directories : list[str]
    output      : str — output filename (default 'nebefs.dat')
    """
    header = (f" {'Image':>4}  {'Force':>12}  {'Stress':>12}  "
              f"{'Volume':>6}  {'Magnet':>11}  {'Rel Energy':>12}")
    print(header)
    energy0 = None
    lines   = []
    for i, d in enumerate(directories):
        outcar = os.path.join(d, 'OUTCAR')
        energy = _parse_energy(outcar)
        force  = _parse_max_force(outcar)
        stress = _parse_stress(outcar)
        volume = _parse_volume(outcar)
        mag    = _parse_magmom(outcar)
        if energy0 is None:
            energy0 = energy
        rel  = energy - energy0
        line = (f"{i:4d} {force:12.8f} {stress:12.8f} {volume:6.2f} "
                f"{mag:11.8f} {rel:12.8f}")
        lines.append(line)
        print(line)

    with open(output, 'w') as f:
        f.write('\n'.join(lines) + '\n')


def _poscar_to_xyz(poscar, comment=''):
    """Convert a read_POSCAR dict to an XYZ format string.

    Parameters
    ----------
    poscar  : dict — from read_POSCAR()
    comment : str  — second (comment) line of the XYZ block

    Returns
    -------
    str — complete XYZ block for this image
    """
    n       = poscar['total_atoms']
    species = poscar['species']
    cart    = poscar['positions_cartesian']
    lines   = [str(n), comment]
    for sym, pos in zip(species, cart):
        lines.append(f"{sym}  {pos[0]:12.8f}  {pos[1]:12.8f}  {pos[2]:12.8f}")
    return '\n'.join(lines)


def make_movie(directories):
    """Generate POSCAR-format and XYZ movies from image CONTCARs (or POSCARs).

    For each image, reads the structure and annotates with the force and
    energy from OUTCAR. Writes:
        movie     — concatenated POSCAR-format structures
        movie.xyz — concatenated XYZ structures

    Parameters
    ----------
    directories : list[str] — all image directories including endpoints
    """
    poscar_blocks = []
    xyz_blocks    = []
    energy0       = None

    for d in directories:
        src    = _poscar_path(d)
        poscar = read_POSCAR(src)
        outcar = os.path.join(d, 'OUTCAR')

        ene_rel   = 0.0
        force_val = 0.0
        if os.path.isfile(outcar):
            ene_val   = _parse_energy(outcar)
            force_val = _parse_max_force(outcar)
            if energy0 is None:
                energy0 = ene_val
            ene_rel = ene_val - energy0

        annotation = f"F: {force_val:.6f}  E: {ene_rel:.6f}"

        # XYZ block
        xyz_blocks.append(_poscar_to_xyz(poscar, comment=annotation))

        # POSCAR block — write to temp file then read back as string
        tmp = f"_neb_tmp_{d}.vasp"
        labels = define_labels(poscar['elements'], poscar['atom_counts'])
        write_POSCAR(tmp, poscar['lattice_matrix'], poscar['elements'], poscar['atom_counts'],
                     poscar['positions_cartesian'], poscar['positions_direct'], poscar['selective_dynamics'],
                     poscar['flags'], labels)
        with open(tmp) as tf:
            block = tf.read()
        os.remove(tmp)
        poscar_blocks.append(block)

    with open('movie.xyz', 'w') as f:
        f.write('\n'.join(xyz_blocks) + '\n')

    with open('movie', 'w') as f:
        f.write(''.join(poscar_blocks))

    print("movie and movie.xyz written.")


def report_convergence(directories):
    """Write per-image fe.dat files tracking force and energy vs ionic step.

    Skips endpoint images (first and last in directories). For each interior
    image, reads all 'energy  without entropy' and 'FORCES: m' lines from
    OUTCAR and writes <image>/fe.dat with columns:
        step, max_force (eV/Angstrom), absolute_energy (eV), relative_energy (eV).

    Parameters
    ----------
    directories : list[str] — all image directories including endpoints
    """
    interior = directories[1:-1]
    for d in interior:
        print(f"In {d} ...")
        outcar   = os.path.join(d, 'OUTCAR')
        energies = _all_matches(outcar, 'energy  without entropy')
        forces   = _all_matches(outcar, 'FORCES: m')

        n = min(len(energies), len(forces))
        if n == 0:
            print(f"  Warning: no data in {outcar}")
            continue

        energy0 = None
        rows    = []
        for j in range(n):
            e_parts = energies[j].split()
            f_parts = forces[j].split()
            try:
                e_val = float(e_parts[-1])
                f_val = float(f_parts[4])
            except (IndexError, ValueError):
                continue
            if energy0 is None:
                energy0 = e_val
            rows.append(f"{j:5d} {f_val:20.8f} {e_val:20.6f} {e_val - energy0:20.6g}")

        with open(os.path.join(d, 'fe.dat'), 'w') as f:
            f.write('\n'.join(rows) + '\n')
        print(f"  {n} steps written to {d}/fe.dat")


def find_image_dirs():
    """Find and sort NEB image directories matching the two-digit pattern (00, 01, ...).

    Returns
    -------
    list[str] — sorted directory names
    """
    dirs = [d for d in os.listdir('.')
            if os.path.isdir(d) and re.fullmatch(r'[0-9]{2}', d)]
    return sorted(dirs)


def main():
    if '-h' in argv or len(argv) > 2:
        usage()

    nj = 20
    if len(argv) == 2:
        if not argv[1].isdigit() or int(argv[1]) < 1:
            print("ERROR! nj must be a positive integer.")
            exit(1)
        nj = int(argv[1])

    # directories
    directories = find_image_dirs()
    if not directories:
        print("ERROR! No NEB image directories (00, 01, ...) found.")
        exit(1)
    print(f"\nImage directories: {' '.join(directories)}\n")

    # SSNEB auto-detect
    outcar_01 = os.path.join(directories[1] if len(directories) > 1 else directories[0],
                             'OUTCAR')
    ssneb = os.path.isfile(outcar_01) and _detect_ssneb(outcar_01)
    if ssneb:
        print("SSNEB mode detected (LNEBCELL = T)\n")

    # barrier + spline
    print("Do compute_barrier ; compute_spline")
    compute_barrier(directories, ssneb=ssneb)
    compute_spline('neb.dat', 'spline.dat', 'exts.dat', nj=nj)
    if ssneb and os.path.isfile('nebss.dat'):
        compute_spline('nebss.dat', 'spliness.dat', 'extsss.dat', nj=nj)

    # nebef
    print("Do report_ef")
    report_ef(directories)

    # nebefs (SSNEB only)
    if ssneb:
        print("Do report_efs")
        report_efs(directories)

    # movie
    print("Do make_movie")
    make_movie(directories)

    # convergence
    print("Do report_convergence")
    report_convergence(directories)

    # final summary
    print("\nForces and Energy:")
    if os.path.isfile('nebef.dat'):
        with open('nebef.dat') as f:
            print(f.read())
    if os.path.isfile('exts.dat'):
        with open('exts.dat') as f:
            print(f.read())


if __name__ == '__main__':
    main()
