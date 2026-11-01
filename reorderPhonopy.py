#!/usr/bin/env python

from sys import argv, exit
import os
import numpy as np
import yaml
try:
    from yaml import CLoader as Loader
except ImportError:
    from yaml import Loader


def usage():
    """Print usage instructions and exit."""
    print("""
Usage: reorderPhonopy.py <input band.yaml> <output band.yaml>

Reconnect phonon branches between path segments in a Phonopy band.yaml file.

This script was developed by Thanasee Thanasarnsurapong.
""")
    exit(0)


def yaml_eigvecs_to_complex(modes):
    """
    Convert eigenvectors from yaml [real, imag] format to complex ndarray.

    Phonopy stores eigenvectors in yaml as a nested list of shape
    (n_atoms, 3, 2) where the last axis is [real, imag]. This function
    converts them to the complex ndarray of shape (3*n_atoms,) used
    internally by Phonopy's band connection algorithm.

    Parameters
    ----------
    modes : list of shape (n_modes, n_atoms, 3, 2)
        Eigenvectors for all modes at one q-point from band.yaml.

    Returns
    -------
    eigvecs : np.ndarray of shape (3*n_atoms, n_modes), complex
        Complex eigenvector matrix matching Phonopy's internal format.
    """
    n_modes = len(modes)
    n_atoms = len(modes[0])
    n_dim   = 3 * n_atoms

    eigvecs = np.zeros((n_dim, n_modes), dtype=complex)

    for k, mode in enumerate(modes):
        idx = 0
        for atom in mode:
            for comp in atom:
                eigvecs[idx, k] = complex(comp[0], comp[1])
                idx += 1

    return eigvecs


def estimate_band_connection(prev_eigvecs, eigvecs, prev_band_order):
    """
    Connect neighboring q-points by eigenvector similarity.

    Replicates Phonopy's internal ``estimate_band_connection`` function
    (phonopy/phonon/band_structure.py) exactly, using the same greedy
    algorithm and overlap metric.

    Parameters
    ----------
    prev_eigvecs : np.ndarray of shape (3*n_atoms, n_modes), complex
        Eigenvector matrix at the previous q-point (already reordered).
    eigvecs : np.ndarray of shape (3*n_atoms, n_modes), complex
        Eigenvector matrix at the current q-point (raw order).
    prev_band_order : list of int
        Current global band ordering accumulated from previous steps.

    Returns
    -------
    band_order : list of int
        Updated global band ordering for the current segment.
    """
    metric = np.abs(np.dot(prev_eigvecs.conjugate().T, eigvecs))

    connection_order = []

    for overlaps in metric:
        maxval    = 0
        maxindex  = 0

        for i in reversed(range(len(metric))):
            val = overlaps[i]

            if i in connection_order:
                continue

            if val > maxval:
                maxval   = val
                maxindex = i

        connection_order.append(maxindex)

    band_order = [connection_order[x] for x in prev_band_order]

    return band_order


def reorder_band_structure(data):
    """
    Reconnect phonon branches across path segments.

    Assumes Phonopy already performed band connection within each segment.
    Applies the same greedy eigenvector overlap algorithm across segment
    boundaries to ensure globally consistent branch labeling.

    Parameters
    ----------
    data : dict
        Parsed band.yaml dictionary.

    Returns
    -------
    ordered_bands : list of length n_qpoints
        Reordered band dicts for all q-points.
    """
    all_bands = [pt['band'] for pt in data['phonon']]
    all_modes = [[b['eigenvector'] for b in pt['band']] for pt in data['phonon']]

    segment_nqpoint = data['segment_nqpoint']
    cumulative = [0] + list(np.cumsum(segment_nqpoint))

    n_bands    = len(all_bands[0])
    band_order = list(range(n_bands))

    ordered_bands = [list(qp) for qp in all_bands]

    for s in range(1, len(segment_nqpoint)):

        prev_end   = cumulative[s] - 1
        curr_start = cumulative[s]
        curr_end   = cumulative[s + 1]

        # Convert boundary eigenvectors to complex ndarray (Phonopy format)
        prev_eigvecs = yaml_eigvecs_to_complex(all_modes[prev_end])
        curr_eigvecs = yaml_eigvecs_to_complex(all_modes[curr_start])

        band_order = estimate_band_connection(
            prev_eigvecs,
            curr_eigvecs,
            band_order
        )

        # Apply permutation to entire segment
        for q in range(curr_start, curr_end):
            ordered_bands[q] = [ordered_bands[q][i] for i in band_order]

    return ordered_bands


def write_band_yaml(output, data, ordered_bands):
    """
    Write reordered band structure to a yaml file matching Phonopy format exactly.

    Parameters
    ----------
    data : dict
        Original band.yaml data dictionary.
    ordered_bands : list
        Reordered band data for each q-point.
    output : str
        Output file path.
    """
    def vec3_recip(vec, comment):
        inner = ','.join(f'{v:13.8f}' for v in vec)
        return f'- [{inner} ] # {comment}'

    def vec3_lattice(vec, comment):
        inner = ','.join(f'{v:22.15f}' for v in vec)
        return f'- [{inner} ] # {comment}'

    def vec3_coords(vec):
        inner = ','.join(f'{v:19.15f}' for v in vec)
        return f'[{inner} ]'

    def vec3_qpos(vec):
        inner = ','.join(f'{v:13.7f}' for v in vec)
        return f'[{inner} ]'

    def fmt_evec_comp(comp):
        return f'[ {comp[0]:18.14f},{comp[1]:18.14f} ]'

    lines = []

    # Keys are written in input order; keys without a dedicated format
    # (e.g. calculator, length_unit) are passed through via yaml.dump
    for key, value in data.items():
        if key == 'phonon':
            continue
        elif key in ('nqpoint', 'npath', 'natom'):
            lines.append(f'{key}: {value}')
        elif key == 'segment_nqpoint':
            lines.append('segment_nqpoint:')
            for nq in value:
                lines.append(f'- {nq}')
        elif key == 'labels':
            lines.append('labels:')
            for start, end in value:
                lines.append(f"- [ '{start}', '{end}' ]")
        elif key == 'reciprocal_lattice':
            lines.append('reciprocal_lattice:')
            for vec, label in zip(value, ['a*', 'b*', 'c*']):
                lines.append(vec3_recip(vec, label))
        elif key == 'lattice':
            lines.append('lattice:')
            for vec, label in zip(value, ['a', 'b', 'c']):
                lines.append(vec3_lattice(vec, label))
        elif key == 'points':
            lines.append('points:')
            for i, pt in enumerate(value):
                lines.append(f'- symbol: {pt["symbol"]} # {i + 1}')
                lines.append(f'  coordinates: {vec3_coords(pt["coordinates"])}')
                lines.append(f'  mass: {pt["mass"]:.6f}')
        else:
            lines.append(yaml.dump({key: value}, default_flow_style=False,
                                   sort_keys=False).rstrip('\n'))

    lines.append('')
    lines.append('phonon:')

    for pt, band_data in zip(data['phonon'], ordered_bands):
        lines.append(f'- q-position: {vec3_qpos(pt["q-position"])}')
        lines.append(f'  distance: {pt["distance"]:12.7f}')
        if 'label' in pt:
            lines.append(f'  label: {pt["label"]}')
        lines.append('  band:')
        for b_idx, band in enumerate(band_data):
            lines.append(f'  - # {b_idx + 1}')
            lines.append(f'    frequency: {band["frequency"]:16.10f}')
            if 'group_velocity' in band:
                inner = ','.join(f'{v:14.7f}' for v in band['group_velocity'])
                lines.append(f'    group_velocity: [{inner} ]')
            lines.append('    eigenvector:')
            for a_idx, atom_evec in enumerate(band['eigenvector']):
                lines.append(f'    - # atom {a_idx + 1}')
                for comp in atom_evec:
                    lines.append(f'      - {fmt_evec_comp(comp)}')

    lines.append('')

    with open(output, 'w') as f:
        f.write('\n'.join(lines) + '\n')


def main():
    """Read input, reorder bands across segments, and write output."""
    if '-h' in argv or '--help' in argv or len(argv) != 3:
        usage()

    filepath = argv[1]
    output   = argv[2]

    if not os.path.exists(filepath):
        print(f"ERROR!\nFile: {filepath} does not exist.")
        exit(1)

    with open(filepath, 'r') as f:
        data = yaml.load(f, Loader=Loader)

    if 'segment_nqpoint' not in data:
        print("ERROR!")
        print("band.yaml does not contain segment information.")
        exit(1)

    if 'eigenvector' not in data['phonon'][0]['band'][0]:
        print("ERROR!")
        print("band.yaml does not contain eigenvectors.")
        print("Run phonopy with --eigenvectors.")
        exit(1)

    ordered_bands = reorder_band_structure(data)
    write_band_yaml(output, data, ordered_bands)


if __name__ == "__main__":
    main()
