#!/usr/bin/env python

import os
import subprocess
from sys import argv, exit
from glob import glob
from ase.io import read
from hiphive import ClusterSpace, ForceConstantPotential, enforce_rotational_sum_rules
from hiphive.force_constants import ForceConstants
from hiphive.cutoffs import estimate_maximum_cutoff
from hiphive.utilities import extract_parameters


def usage():
    """Print usage information and exit."""
    print("""
Usage: enforceIFC.py <input FORCE_CONSTANTS> [output FORCE_CONSTANTS]

This script enforces the rotational sum rules on the second-order interatomic force constants (IFC2) of a supercell structure.
It reads the primitive and supercell structures, extracts the IFC2 from a specified file, applies the rotational sum rules, and writes the modified IFC2 back to a file.

This script was developed by Thanasee Thanasarnsurapong.
""")
    exit(0)


def read_structure(filename):
    """
    Read an atomic structure from a file.

    Parameters
    ----------
    filename : str
        Path to the structure file (e.g., POSCAR, SPOSCAR).

    Returns
    -------
    ase.Atoms
        Atomic structure read from the file.
    """
    return read(filename)


def build_cluster_space(primitive, supercell, cutoff_margin=0.01):
    """
    Build a cluster space for the given primitive and supercell structures.

    Parameters
    ----------
    primitive : ase.Atoms
        Primitive unit cell structure.
    supercell : ase.Atoms
        Supercell structure used to estimate the maximum cutoff radius.
    cutoff_margin : float, optional
        Margin subtracted from the maximum cutoff to avoid incomplete
        neighbor shells at the boundary, in Angstrom. Default is 0.01.

    Returns
    -------
    hiphive.ClusterSpace
        Cluster space constructed with a single 2-body cutoff.
    """
    max_cutoff = estimate_maximum_cutoff(supercell) - cutoff_margin
    cutoffs = [max_cutoff]
    cluster_space = ClusterSpace(primitive, cutoffs)
    return cluster_space

def read_phonopy_fc2(supercell, fc_file="FORCE_CONSTANTS", fc_format="text"):
    """
    Read phonopy FC2 force constants from a file.

    Parameters
    ----------
    supercell : ase.Atoms
        Supercell structure corresponding to the force constants.
    fc_file : str, optional
        Path to the force constants file. Default is 'FORCE_CONSTANTS'.
    fc_format : str, optional
        File format, either 'text' or 'hdf5'. Default is 'text'.

    Returns
    -------
    hiphive.force_constants.ForceConstants
        Force constants object read from the file.
    """
    return ForceConstants.read_phonopy(supercell, fc_file, format=fc_format)

def project_fc2_to_parameters(force_constants, cluster_space):
    """
    Project FC2 force constants onto the cluster space parameters.

    Parameters
    ----------
    force_constants : hiphive.force_constants.ForceConstants
        Force constants to project.
    cluster_space : hiphive.ClusterSpace
        Cluster space defining the basis for projection.

    Returns
    -------
    numpy.ndarray
        Parameter vector representing the force constants in the cluster space.
    """
    return extract_parameters(force_constants, cluster_space)


def apply_rotational_sum_rules(cluster_space, parameters, sum_rules=("Huang", "Born-Huang")):
    """
    Enforce rotational sum rules on the force constant parameters.

    Parameters
    ----------
    cluster_space : hiphive.ClusterSpace
        Cluster space corresponding to the parameters.
    parameters : numpy.ndarray
        Parameter vector to enforce sum rules on.
    sum_rules : tuple of str, optional
        Sum rules to enforce. Default is ('Huang', 'Born-Huang').

    Returns
    -------
    numpy.ndarray
        Parameter vector with rotational sum rules enforced.
    """
    return enforce_rotational_sum_rules(cluster_space, parameters, list(sum_rules))


def parameters_to_force_constants(cluster_space, parameters, supercell):
    """
    Reconstruct force constants from cluster space parameters.

    Parameters
    ----------
    cluster_space : hiphive.ClusterSpace
        Cluster space defining the force constant basis.
    parameters : numpy.ndarray
        Parameter vector to reconstruct force constants from.
    supercell : ase.Atoms
        Supercell structure to evaluate the force constants on.

    Returns
    -------
    hiphive.force_constants.ForceConstants
        Reconstructed force constants object.
    """
    fcp = ForceConstantPotential(cluster_space, parameters)
    return fcp.get_force_constants(supercell)


def write_phonopy_fc2(force_constants, fc_file, fc_format="text"):
    """
    Write FC2 force constants to a phonopy-compatible file.

    Parameters
    ----------
    force_constants : hiphive.force_constants.ForceConstants
        Force constants object to write.
    fc_file : str
        Path to the output file.
    fc_format : str, optional
        File format, either 'text' or 'hdf5'. Default is 'text'.
    """
    force_constants.write_to_phonopy(fc_file, format=fc_format)


def main():
    """Parse arguments, read structures and force constants, apply sum rules, and write modified force constants."""
    if '-h' in argv or '--help' in argv or len(argv) > 3:
        usage()
    
    primitive_file = "POSCAR"
    if not os.path.isfile(primitive_file):
        print(f"Error: Primitive structure file '{primitive_file}' not found.")
        exit(1)

    supercell_file = "SPOSCAR"
    if not os.path.isfile(supercell_file):
        print(f"Error: Supercell structure file '{supercell_file}' not found.")
        exit(1)

    input_fc_file = argv[1] if len(argv) > 1 else "FORCE_CONSTANTS"
    if not os.path.isfile(input_fc_file):
        if not os.path.isfile("phonopy_params.yaml"):
            vasprun_files = sorted(f for f in glob("vasprun.xml-*") if f != "vasprun.xml-sposcar")
            subcommand = f"phonopy --fz vasprun.xml-sposcar {' '.join(vasprun_files)} --sp"
            subprocess.run(subcommand, shell=True, check=True)
    
        command = "phonopy-load phonopy_params.yaml --writefc --full-fc"
        subprocess.run(command, shell=True, check=True)
        input_fc_file = "FORCE_CONSTANTS"
    input_format = "hdf5" if input_fc_file.endswith(".hdf5") else "text"

    output_fc_file = argv[2] if len(argv) > 2 else (
    os.path.splitext(input_fc_file)[0] + "_rot" +
    (".hdf5" if input_format == "hdf5" else "")
    )
    output_format = "hdf5" if output_fc_file.endswith(".hdf5") else "text"

    cutoff_margin = 1e-5
    sum_rules = ("Huang", "Born-Huang")
    # ---------------------

    primitive = read_structure(primitive_file)
    supercell = read_structure(supercell_file)

    cluster_space = build_cluster_space(primitive, supercell, cutoff_margin)
    force_constants = read_phonopy_fc2(supercell, input_fc_file, input_format)
    parameters = project_fc2_to_parameters(force_constants, cluster_space)
    parameters_rot = apply_rotational_sum_rules(cluster_space, parameters, sum_rules)
    force_constants_rot = parameters_to_force_constants(cluster_space, parameters_rot, supercell)
    write_phonopy_fc2(force_constants_rot, output_fc_file, output_format)


if __name__ == "__main__":
    main()
