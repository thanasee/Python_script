#!/usr/bin/env python

from sys import argv, exit
import os
import re
import readline
import numpy as np
import h5py
from ase import Atoms
from ase.io import read


def usage():
    """Print usage information and exit."""
    print("""
Usage: vaspVibration.py <POSCAR input> <input file> [scaling factor]
       vaspVibration.py <vaspout.h5> [scaling factor]

This script extracts vibrational modes and writes them in XSF format.
Supports VASP (OUTCAR: Γ modes of the dynamical matrix or phonon dispersion with PHON_NWRITE = -3;
vaspout.h5: phonon dispersion) and Phonopy (band.yaml/mesh.yaml/qpoints.yaml) outputs with write eigenvectors.
For a phonon dispersion, <POSCAR input> is the supercell POSCAR of the VASP run.

This script was developed by Thanasee Thanasarnsurapong.
""")
    exit(0)


def read_structure(poscar_file):
    """Read atomic structure from a POSCAR file using ASE.

    Parameters
    ----------
    poscar_file : str
        Path to the POSCAR file.

    Returns
    -------
    structure : ase.Atoms
        ASE Atoms object containing the crystal structure.
    """

    if not os.path.exists(poscar_file):
        print(f"ERROR!\nFile: {poscar_file} does not exist.")
        exit(1)

    return read(poscar_file)


def choose_qpoint(qpoints, labels, candidates, kind, source):
    """Remove duplicated candidate q-points and prompt for one if more than one remains.

    Parameters
    ----------
    qpoints : np.ndarray, shape (n_qpoints, 3)
        Fractional q-point coordinates.
    labels : list of str
        Label of each q-point ('' if none).
    candidates : list of int
        Indices of the q-points that can be chosen.
    kind : str
        Description of the candidates printed in the menu.
    source : str
        Input file name printed in the menu.

    Returns
    -------
    iq : int
        Index of the chosen q-point.
    """

    unique = []
    for i in candidates:
        if not any(np.allclose(qpoints[i], qpoints[j], atol=1e-6) for j in unique):
            unique.append(i)

    if len(unique) == 1:
        return unique[0]

    print(f"\nFound {len(unique)} {kind} in {source}:")
    for n, i in enumerate(unique, start=1):
        q = qpoints[i]
        print(f"  {n:>3d}) {labels[i]:<8s} [{q[0]:10.6f}{q[1]:10.6f}{q[2]:10.6f} ]")
    while True:
        try:
            choice = int(input(f"Enter the q-point index (1 to {len(unique)}): ")) - 1
            if 0 <= choice < len(unique):
                return unique[choice]
            print("ERROR! Index out of range. Try again.")
        except ValueError:
            print("ERROR! Must enter a number. Try again.")


def real_modes(eigenvectors, masses):
    """Remove the global phase of each mode, keep the real part, and divide by sqrt(m).

    The sign of each mode is kept, so real eigenvectors are unchanged.

    Parameters
    ----------
    eigenvectors : np.ndarray, shape (n_bands, n_atoms, 3), complex
        Mass-weighted eigenvectors at one q-point.
    masses : np.ndarray, shape (n_atoms,)
        Atomic masses.

    Returns
    -------
    modes : np.ndarray, shape (n_bands, n_atoms, 3)
        Mass-unweighted real displacement vectors.
    """

    eigenvectors = eigenvectors.copy()
    for b in range(len(eigenvectors)):
        z = eigenvectors[b].flat[np.abs(eigenvectors[b]).argmax()]
        # Phase in [-π/2, π/2) keeps the sign of real eigenvectors; a (nearly) purely imaginary
        # component is rotated to positive real, so a rounded -0/+0 real part cannot flip the sign
        if abs(z.real) < 1e-4 * abs(z):
            theta = np.sign(z.imag) * np.pi / 2
        else:
            theta = (np.angle(z) + np.pi / 2) % np.pi - np.pi / 2
        eigenvectors[b] *= np.exp(-1j * theta)

    if np.abs(eigenvectors.imag).max() > 1e-4:
        print("Warning! Eigenvectors at this q-point are complex. Only the real part is written.")

    return eigenvectors.real / np.sqrt(masses[None, :, None])


def dispersion_modes(cell, symbols, positions_direct, qpoints, eigenvectors, qpoints_file, source):
    """Choose a q-point of a VASP phonon dispersion and convert its eigenvectors to displacements.

    Candidate q-points are the start and end of the path, both ends of each
    segment, and every point where the path changes direction. Labels are
    read from a line-mode QPOINTS file if present. VASP eigenvectors refer to
    the cell in which each ion is stored, so ions outside [0, 1) are wrapped
    and their components multiplied by exp(-2πi q·n).

    Parameters
    ----------
    cell : np.ndarray, shape (3, 3)
        Primitive lattice vectors (Å).
    symbols : list of str
        Element symbol of each primitive-cell ion.
    positions_direct : np.ndarray, shape (n_atoms, 3)
        Fractional positions of the primitive-cell ions as stored by VASP.
    qpoints : np.ndarray, shape (n_qpoints, 3)
        Fractional q-point coordinates.
    eigenvectors : np.ndarray, shape (n_qpoints, n_bands, n_atoms, 3), complex
        Mass-weighted eigenvectors.
    qpoints_file : str
        Path to the QPOINTS file used for labels.
    source : str
        Input file name printed in the menu.

    Returns
    -------
    structure : ase.Atoms
        Primitive cell with wrapped positions.
    modes : np.ndarray, shape (n_bands, n_atoms, 3)
        Mass-unweighted real displacement vectors at the chosen q-point.
    """

    nq = len(qpoints)
    steps = np.diff(qpoints, axis=0)
    norms = np.linalg.norm(steps, axis=1)
    candidates = {0, nq - 1}
    for i in range(1, nq - 1):
        if (norms[i - 1] < 1e-6 or norms[i] < 1e-6 or
                not np.allclose(steps[i - 1] / norms[i - 1], steps[i] / norms[i], atol=1e-3)):
            candidates.add(i)
    candidates = sorted(candidates)

    labels = [''] * nq
    if os.path.exists(qpoints_file):
        with open(qpoints_file, 'r') as f:
            lines = f.readlines()
        if (len(lines) > 4 and lines[2].strip().lower().startswith('l')
                and not lines[3].strip().lower().startswith(('c', 'k'))):
            for line in lines[4:]:
                tokens = line.split()
                if len(tokens) < 4:
                    continue
                try:
                    coordinate = np.array([float(x) for x in tokens[:3]])
                except ValueError:
                    continue
                label = ' '.join(tokens[3:]).lstrip('!').strip()
                for i in range(nq):
                    if np.allclose(qpoints[i], coordinate, atol=1e-4):
                        labels[i] = label

    kind = "q-points" if len(candidates) == nq else "high-symmetry q-points"
    iq = choose_qpoint(qpoints, labels, candidates, kind, source)

    shift = np.floor(positions_direct)
    structure = Atoms(symbols, scaled_positions=positions_direct - shift, cell=cell, pbc=True)
    phase = np.exp(-2j * np.pi * shift @ qpoints[iq])
    modes = real_modes(eigenvectors[iq] * phase[None, :, None], structure.get_masses())

    return structure, modes


def read_modes_outcar(outcar_file, structure):
    """Read vibrational modes from a VASP OUTCAR file.

    Reads the 'Eigenvectors and eigenvalues of the dynamical matrix' block
    (Γ modes of the POSCAR cell) and/or the phonon dispersion section
    (LPHON_DISPERSION with PHON_NWRITE = -3). If both are present, the user
    is prompted to choose one. The 'Eigenvectors after division by SQRT(mass)'
    block (NWRITE = 3) is not read.

    Parameters
    ----------
    outcar_file : str
        Path to the OUTCAR file.
    structure : ase.Atoms
        Structure of the VASP run (supercell for a phonon dispersion).

    Returns
    -------
    structure : ase.Atoms
        Structure of the modes (primitive cell for a phonon dispersion).
    modes : np.ndarray, shape (n_modes, n_atoms, 3)
        Mass-unweighted Cartesian displacement vectors for each mode.
    ascending : bool
        True if the modes are in ascending frequency order (phonon dispersion),
        False for descending order (Γ block).
    """

    if not os.path.exists(outcar_file):
        print(f"ERROR!\nFile: {outcar_file} does not exist.")
        exit(1)

    with open(outcar_file, 'r') as f:
        outcar_lines = f.readlines()

    total_ions = None
    for line in outcar_lines:
        if 'NIONS =' in line:
            total_ions = int(line.split()[-1])
            break

    if total_ions is None:
        print("ERROR!\n'NIONS' not found in OUTCAR.")
        exit(1)

    if total_ions != len(structure):
        print("ERROR!\nShape mismatch between eigenvectors and atomic positions.")
        exit(1)

    index_start = None
    index_phonons = None
    index_primitive = None
    for i, line in enumerate(outcar_lines):
        if 'Eigenvectors and eigenvalues of the dynamical matrix' in line:
            index_start = i + 2
        elif line.strip() == 'Phonons':
            index_phonons = i
        elif line.strip() == 'Primitive cell':
            index_primitive = i

    # Phonon dispersion: one block per q-point; eigenvectors (PHON_NWRITE = -3) are written as
    # 3N rows (ion-major x, y, z) of n_bands complex numbers 're +imi'
    qpoints = []
    eigenvectors = []
    if index_phonons is not None:
        complex_pattern = re.compile(r'([-+]?\d+\.\d+(?:[eE][-+]?\d+)?)\s*([-+]\d+\.\d+(?:[eE][-+]?\d+)?)i')
        i = index_phonons + 1
        while i < len(outcar_lines):
            stripped = outcar_lines[i].strip()
            if stripped.startswith('q-point No.'):
                qpoints.append([float(x) for x in outcar_lines[i + 1].split()[1:4]])
                i += 2
            elif stripped == 'eigenvectors':
                values = []
                i += 1
                while i < len(outcar_lines):
                    stripped = outcar_lines[i].strip()
                    if not stripped or stripped.startswith('q-point No.') or set(stripped) <= set('-'):
                        break
                    values.extend(complex(float(a), float(b)) for a, b in complex_pattern.findall(outcar_lines[i]))
                    i += 1
                eigenvectors.append(values)
            elif stripped and set(stripped) <= set('-'):
                break
            else:
                i += 1

    use_dispersion = False
    if index_phonons is not None and len(eigenvectors) == 0:
        print("Warning! Phonon dispersion in OUTCAR has no eigenvectors (needs PHON_NWRITE = -3 or vaspout.h5).")
    elif index_phonons is not None:
        if index_start is None:
            use_dispersion = True
        else:
            print("""
Found both Γ modes and phonon dispersion in OUTCAR
  1) Γ modes of the dynamical matrix (POSCAR cell)
  2) Phonon dispersion (primitive cell)""")
            while True:
                choice = input("Enter choice: ").strip()
                if choice in ('1', '2'):
                    use_dispersion = choice == '2'
                    break
                print("ERROR! Must choose 1 or 2. Try again.")

    if use_dispersion:
        if index_primitive is None:
            print("ERROR!\nPrimitive cell not found in OUTCAR.")
            exit(1)

        lattice = None
        index_positions = None
        ion_index = []
        for i in range(index_primitive, len(outcar_lines)):
            line = outcar_lines[i]
            if 'direct lattice vectors' in line and lattice is None:
                lattice = np.array([[float(x) for x in outcar_lines[i + k].split()[:3]] for k in (1, 2, 3)])
            elif 'position of ions in fractional coordinates' in line:
                index_positions = i + 1
            elif 'ion indices of the primitive-cell ions' in line:
                for row in outcar_lines[i + 2:]:
                    tokens = row.split()
                    if len(tokens) != 2 or not all(t.isdigit() for t in tokens):
                        break
                    ion_index.append(int(tokens[1]))
                break

        if lattice is None or index_positions is None or len(ion_index) == 0:
            print("ERROR!\nPrimitive cell not found in OUTCAR.")
            exit(1)

        natoms = len(ion_index)
        positions = np.array([[float(x) for x in outcar_lines[index_positions + k].split()[:3]]
                              for k in range(natoms)])
        symbols = [structure.get_chemical_symbols()[k - 1] for k in ion_index]

        nq = len(qpoints)
        sizes = {len(v) for v in eigenvectors}
        if len(eigenvectors) != nq or len(sizes) != 1 or sizes.pop() % (3 * natoms) != 0:
            print("ERROR!\nUnexpected eigenvector layout in the OUTCAR phonon dispersion.")
            exit(1)
        nbands = len(eigenvectors[0]) // (3 * natoms)
        eigenvectors = (np.array(eigenvectors).reshape(nq, 3 * natoms, nbands)
                        .transpose(0, 2, 1).reshape(nq, nbands, natoms, 3))

        qpoints_file = os.path.join(os.path.dirname(os.path.abspath(outcar_file)), 'QPOINTS')
        structure, modes = dispersion_modes(lattice, symbols, positions, np.array(qpoints),
                                            eigenvectors, qpoints_file, outcar_file)
        return structure, modes, True

    if index_start is None:
        print("ERROR!\nEigenvector block not found in OUTCAR.")
        exit(1)

    # Read mode by mode: 'f  =' or 'f/i=' line, 'X Y Z dx dy dz' line, one line per ion.
    # Stops at the first other non-empty line (e.g. 'Eigenvectors after division by SQRT(mass)').
    modes = []
    i = index_start
    while i < len(outcar_lines):
        line = outcar_lines[i]
        if '2PiTHz' in line and ('f  =' in line or 'f/i=' in line):
            modes.append([row.split()[3:6] for row in outcar_lines[i + 2:i + 2 + total_ions]])
            i += 2 + total_ions
        elif line.strip() == '':
            i += 1
        else:
            break

    if len(modes) == 0:
        print("ERROR!\nEigenvector block not found in OUTCAR.")
        exit(1)

    # Mass-unweight: OUTCAR stores mass-weighted eigenvectors; divide by sqrt(m)
    modes = np.array(modes, dtype=float) / np.sqrt(structure.get_masses()[None, :, None])

    return structure, modes, False


def read_modes_h5(h5_file):
    """Read the phonon dispersion (LPHON_DISPERSION) from a VASP vaspout.h5 file.

    Parameters
    ----------
    h5_file : str
        Path to the vaspout.h5 file.

    Returns
    -------
    structure : ase.Atoms
        Primitive cell with wrapped positions.
    modes : np.ndarray, shape (n_bands, n_atoms, 3)
        Mass-unweighted real displacement vectors at the chosen q-point.
    """

    if not os.path.exists(h5_file):
        print(f"ERROR!\nFile: {h5_file} does not exist.")
        exit(1)

    with h5py.File(h5_file, 'r') as f:
        if 'results/phonons/eigenvectors' not in f:
            print("ERROR!\nNo phonon dispersion (LPHON_DISPERSION) found in vaspout.h5.")
            exit(1)
        group = f['results/phonons']
        raw = group['eigenvectors'][()]
        qpoints = group['qpoint_coords'][()]
        cell = group['primitive/lattice_vectors'][()] * group['primitive/scale'][()]
        positions = group['primitive/position_ions'][()]
        types = [t.decode() if isinstance(t, bytes) else str(t) for t in group['primitive/ion_types'][()]]
        counts = group['primitive/number_ion_types'][()]

    # eigenvectors: (q-point, band, ion, xyz, real/imaginary)
    eigenvectors = raw[..., 0] + 1j * raw[..., 1]
    symbols = [t.strip() for t, c in zip(types, counts) for _ in range(int(c))]

    qpoints_file = os.path.join(os.path.dirname(os.path.abspath(h5_file)), 'QPOINTS')
    return dispersion_modes(cell, symbols, positions, qpoints, eigenvectors, qpoints_file, h5_file)


def read_modes_phonopy(yaml_file, structure):
    """Read vibrational frequencies and eigenvectors from a Phonopy YAML file.

    Parses band.yaml, mesh.yaml, or qpoints.yaml by string splitting.
    Candidate q-points are the segment end points for band.yaml (with
    'segment_nqpoint') or all q-points otherwise, with duplicated
    q-positions removed. If more than one candidate remains, the user is
    prompted to choose one. The Bloch phase exp(2πi q·r) is applied and
    the global phase of each mode is removed before the real part is
    kept; a warning is printed if the imaginary part is not negligible.

    Parameters
    ----------
    yaml_file : str
        Path to the Phonopy YAML file.
    structure : ase.Atoms
        ASE Atoms object, used to obtain atomic masses for mass-unweighting.

    Returns
    -------
    frequency : np.ndarray, shape (n_bands,)
        Phonon frequencies at the chosen q-point (THz).
    modes : np.ndarray, shape (n_bands, n_atoms, 3)
        Mass-unweighted real part of eigenvectors at the chosen q-point.
    """

    if not os.path.exists(yaml_file):
        print(f"ERROR!\nFile: {yaml_file} does not exist.")
        exit(1)

    with open(yaml_file, 'r') as f:
        content = f.read()

    if 'phonon:' not in content:
        print("ERROR!\nNo q-point data found in YAML file.")
        exit(1)

    header, body = content.split('phonon:', 1)
    q_points = body.split('q-position:')[1:]

    if len(q_points) == 0:
        print("ERROR!\nNo q-point data found in YAML file.")
        exit(1)

    # q-position and optional label of each q-point
    positions = []
    labels = []
    for qp in q_points:
        positions.append(np.array([float(x) for x in qp.split(']')[0].strip(' [').split(',')]))
        label = ''
        for line in qp.split('band:')[0].splitlines():
            if line.strip().startswith('label:'):
                label = line.split(':', 1)[1].strip().strip('\'"')
        labels.append(label)

    # Segment end points of band.yaml are the high-symmetry q-points
    segments = []
    if 'segment_nqpoint:' in header:
        for line in header.split('segment_nqpoint:')[1].splitlines()[1:]:
            if not line.strip().startswith('-'):
                break
            segments.append(int(line.split('-')[1]))
    is_band = len(segments) > 0 and sum(segments) == len(q_points)
    if is_band:
        ends = np.cumsum(segments)
        candidates = sorted({i for n, e in zip(segments, ends) for i in (e - n, e - 1)})
        # Phonopy 4.x writes labels only in the header list, one [start, end] pair per segment
        segment_labels = []
        if 'labels:' in header:
            for line in header.split('labels:')[1].splitlines()[1:]:
                if not line.strip().startswith('-'):
                    break
                segment_labels.append([s.strip().strip('\'"') for s in line.strip()[1:].strip(' []').split(',')])
        if len(segment_labels) == len(segments):
            for (n, e), pair in zip(zip(segments, ends), segment_labels):
                labels[e - n] = labels[e - n] or pair[0]
                labels[e - 1] = labels[e - 1] or pair[-1]
    else:
        candidates = list(range(len(q_points)))

    kind = "high-symmetry q-points" if is_band else "q-points"
    iq = choose_qpoint(positions, labels, candidates, kind, yaml_file)

    bands = q_points[iq].split('frequency:')[1:]
    nbands = len(bands)
    natoms = len(bands[0].split('atom')[1:]) if nbands > 0 else 0

    if natoms == 0:
        print("ERROR!\nNo eigenvectors found in YAML file. Rerun Phonopy with --eigenvectors.")
        exit(1)

    if natoms != len(structure):
        print("ERROR!\nShape mismatch between eigenvectors and atomic positions.")
        exit(1)

    # Extract frequencies and eigenvector components (real, imaginary) at the chosen q-point
    frequency = np.array(
        [float(bands[b].split('eigenvector')[0]) for b in range(nbands)],
        dtype=float
    )

    eigenvectors = np.zeros((nbands, natoms, 3), dtype=complex)
    for b in range(nbands):
        atom_blocks = bands[b].split('atom')[1:]
        for a in range(natoms):
            for d in range(3):
                component = atom_blocks[a].split('[')[d + 1].split(']')[0].split(',')
                eigenvectors[b, a, d] = complex(float(component[0]), float(component[1]))

    # Bloch phase exp(2πi q·r) of each atom (not included in Phonopy eigenvectors)
    eigenvectors *= np.exp(2j * np.pi * structure.get_scaled_positions() @ positions[iq])[None, :, None]

    return frequency, real_modes(eigenvectors, structure.get_masses())


def write_xsf_modes(structure, modes, scale, ascending):
    """Write each vibrational mode to a separate XSF file.

    Each output file mode_N.xsf contains the crystal lattice vectors
    and atomic positions augmented with displacement vectors, suitable
    for visualization in VESTA or XCrySDen. Mode numbering follows
    ascending order for Phonopy input and VASP phonon dispersion, and
    descending order (highest frequency first) for the OUTCAR Γ block.

    Parameters
    ----------
    structure : ase.Atoms
        ASE Atoms object containing the crystal structure.
    modes : np.ndarray, shape (n_modes, n_atoms, 3)
        Displacement vectors for each mode (mass-unweighted).
    scale : float
        Scaling factor applied to displacement vectors for visualization.
    ascending : bool
        If True, number modes in ascending order (Phonopy, VASP phonon dispersion).
        If False, number modes in descending order (OUTCAR Γ block).
    """

    total_modes = modes.shape[0]
    total_atoms = len(structure)
    symbols     = structure.get_chemical_symbols()
    place       = len(str(total_modes))
    if modes.shape[1] != total_atoms:
        print("ERROR!\nShape mismatch between eigenvectors and atomic positions.")
        exit(1)

    for j in range(total_modes):
        vector = modes[j] * scale
        positions_vector = np.hstack((structure.positions, vector))

        mode_index  = j + 1 if ascending else total_modes - j
        output_name = f"mode_{str(mode_index).zfill(place)}.xsf"

        with open(output_name, 'w') as o:
            o.write("CRYSTAL\n")
            o.write("PRIMVEC\n")
            o.write("\n".join(
                ' '.join(f'{a:20.16f}' for a in lattice_row)
                for lattice_row in structure.cell
            ))
            o.write("\nPRIMCOORD\n")
            o.write(f"{total_atoms:3d} 1\n")
            o.write("\n".join(
                f'{symbols[k]:3s}' + ' '.join(f'{a:20.16f}' for a in positions_vector[k])
                for k in range(total_atoms)
            ))


def main():
    """Parse arguments, read inputs, and write vibrational mode XSF files."""

    if '-h' in argv or '--help' in argv or len(argv) < 2:
        usage()

    if argv[1].endswith(".h5"):
        if len(argv) > 3:
            usage()
        scale = float(argv[2]) if len(argv) == 3 else 1.0
        structure, modes = read_modes_h5(argv[1])
        ascending = True
    else:
        if not (3 <= len(argv) <= 4):
            usage()
        poscar_file = argv[1]
        input_file  = argv[2]
        scale       = float(argv[3]) if len(argv) == 4 else 1.0

        structure   = read_structure(poscar_file)
        if input_file.endswith(".h5"):
            structure, modes = read_modes_h5(input_file)
            ascending = True
        elif input_file.endswith(".yaml"):
            _, modes = read_modes_phonopy(input_file, structure)
            ascending = True
        else:
            structure, modes, ascending = read_modes_outcar(input_file, structure)

    write_xsf_modes(structure, modes, scale, ascending)


if __name__ == '__main__':
    main()
