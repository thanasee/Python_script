#!/usr/bin/env python

from sys import argv, exit
import os
import readline
import numpy as np
from ase.io import read


def usage():
    """Print usage information and exit."""
    print("""
Usage: vaspPiezoelectric.py <POSCAR input> <OUTCAR input>

This script gets piezoelectric stress tensor from OUTCAR file
and calculates piezoelectric strain tensor by getting elastic coefficients.
Piezoelectric stress tensor = electronic + ionic (relaxed-ion) if the ionic part is found,
otherwise electronic only (clamped-ion).
Elastic tensor is read from OUTCAR (last TOTAL ELASTIC MODULI block) -> Elastic.dat -> manual input.

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


def read_piezo_stress(outcar_file):
    """Read the piezoelectric stress tensor from a VASP OUTCAR file.

    Reads the electronic part 'PIEZOELECTRIC TENSOR (including local field
    effects)' and, when present, adds the ionic part 'PIEZOELECTRIC TENSOR
    IONIC CONTR' to give the relaxed-ion tensor. Columns are reordered from
    VASP convention (xx, yy, zz, xy, yz, xz) to Voigt notation (1, 2, 3, 4, 5, 6).

    Parameters
    ----------
    outcar_file : str
        Path to the VASP OUTCAR file.

    Returns
    -------
    outcar_lines : list of str
        All lines read from the OUTCAR file, passed downstream for elastic
        tensor extraction without re-reading the file.
    piezostress_coef : numpy.ndarray, shape (3, 6)
        Piezoelectric stress tensor in Voigt notation, in units of C/m**2.
    piezo_label : str
        'relaxed-ion' (electronic + ionic) or 'clamped-ion' (electronic only).
    """

    if not os.path.exists(outcar_file):
        print(f"ERROR!\nFile: {outcar_file} does not exist.")
        exit(1)

    with open(outcar_file, 'r') as f:
        outcar_lines = f.readlines()

    electronic_index = None
    ionic_index      = None
    for i, line in enumerate(outcar_lines):
        if 'PIEZOELECTRIC TENSOR (including local field effects)' in line and electronic_index is None:
            electronic_index = i
        elif 'PIEZOELECTRIC TENSOR IONIC CONTR' in line:
            ionic_index = i

    if electronic_index is None:
        print("ERROR!\nThe 'PIEZOELECTRIC TENSOR (including local field effects)' section was not found in the OUTCAR file.")
        exit(1)

    piezostress_vasp = np.zeros((3, 6))
    for name, index in (('PIEZOELECTRIC TENSOR (including local field effects)', electronic_index),
                        ('PIEZOELECTRIC TENSOR IONIC CONTR', ionic_index)):
        if index is None:
            continue
        rows = [line.split() for line in outcar_lines[index + 3:index + 6]]
        if [row[0] if row else '' for row in rows] != ['x', 'y', 'z'] or any(len(row) != 7 for row in rows):
            print(f"ERROR!\nThe '{name}' section in the OUTCAR file has an unexpected format.")
            exit(1)
        print(f"The '{name}' section was found in the OUTCAR file.")
        piezostress_vasp += np.array([[float(x) for x in row[1:]] for row in rows])

    if ionic_index is not None:
        piezo_label = 'relaxed-ion'
        print("Piezoelectric stress tensor = electronic + ionic (relaxed-ion).")
    else:
        piezo_label = 'clamped-ion'
        print("Warning! The 'PIEZOELECTRIC TENSOR IONIC CONTR' section was not found.\n"
              "Piezoelectric stress tensor = electronic only (clamped-ion).")

    # Reorder from VASP convention (xx,yy,zz,xy,yz,xz) to Voigt notation (11,22,33,44,55,66)
    piezostress_coef = piezostress_vasp[:, [0, 1, 2, 4, 5, 3]]

    return outcar_lines, piezostress_coef, piezo_label


def read_elastic_tensor(outcar_lines):
    """Read the elastic stiffness tensor from VASP OUTCAR lines.

    Uses the last 'TOTAL ELASTIC MODULI (kBar)' block, which includes the
    ionic relaxation contribution when VASP prints it (IBRION=6, ISIF>=3).
    Values are converted from kBar to GPa, and columns/rows are reordered
    from VASP convention (xx, yy, zz, xy, yz, xz) to Voigt notation
    (1, 2, 3, 4, 5, 6).

    Parameters
    ----------
    outcar_lines : list of str
        All lines from the VASP OUTCAR file, as returned by read_piezo_stress().

    Returns
    -------
    elastic_coef : numpy.ndarray, shape (6, 6), or None
        Elastic stiffness tensor in Voigt notation, in units of GPa.
        Returns None if the section is not found.
    elastic_label : str or None
        'relaxed-ion' if an ionic elastic contribution is present in OUTCAR,
        otherwise 'clamped-ion'. Returns None if the section is not found.
    """

    moduli_indices = [i for i, line in enumerate(outcar_lines) if 'TOTAL ELASTIC MODULI (kBar)' in line]

    if not moduli_indices:
        return None, None

    moduli_index = moduli_indices[-1]
    ionic = len(moduli_indices) > 1 or any('ELASTIC MODULI' in line and 'IONIC' in line for line in outcar_lines)
    elastic_label = 'relaxed-ion' if ionic else 'clamped-ion'

    rows = [line.split() for line in outcar_lines[moduli_index + 3:moduli_index + 9]]
    if [row[0] if row else '' for row in rows] != ['XX', 'YY', 'ZZ', 'XY', 'YZ', 'ZX'] or any(len(row) != 7 for row in rows):
        print("ERROR!\nThe 'TOTAL ELASTIC MODULI (kBar)' section in the OUTCAR file has an unexpected format.")
        exit(1)

    print(f"The 'TOTAL ELASTIC MODULI (kBar)' section was found in the OUTCAR file "
          f"({len(moduli_indices)} block(s), using the last one: {elastic_label}).")
    elastic_vasp = np.array([[float(x) for x in row[1:]] for row in rows]) / 10  # Convert kBar to GPa

    # Reorder from VASP convention (xx,yy,zz,xy,yz,xz) to Voigt notation (1,2,3,4,5,6)
    elastic_coef = elastic_vasp[:, [0, 1, 2, 4, 5, 3]][[0, 1, 2, 4, 5, 3], :]

    return elastic_coef, elastic_label


def read_elastic_from_file():
    """Read the elastic stiffness tensor from a local 'Elastic.dat' file.

    The file must contain either a 3x3 matrix (2D materials, in N/m) or a
    6x6 matrix (in GPa), with comment lines starting with '#' ignored.

    Returns
    -------
    elastic_coef : numpy.ndarray, shape (3, 3) or (6, 6), or None
        Elastic stiffness tensor read from file.
        Returns None if 'Elastic.dat' does not exist.
        Exits if the matrix shape is neither (3, 3) nor (6, 6).
    """

    if not os.path.exists('Elastic.dat'):
        return None

    with open('Elastic.dat', 'r') as f:
        elastic_lines = [line.strip() for line in f.readlines()
                         if not line.lstrip().startswith("#") and line.strip()]

    try:
        elastic_coef = np.array([list(map(float, line.split())) for line in elastic_lines])
    except ValueError:
        elastic_coef = np.array([])

    if elastic_coef.shape == (3, 3):
        print("Elastic tensor was read from Elastic.dat (3x3, N/m). Your material should be 2D materials.")
        return elastic_coef
    elif elastic_coef.shape == (6, 6):
        print("Elastic tensor was read from Elastic.dat (6x6, GPa).")
        return elastic_coef
    else:
        print("ERROR!\nElastic.dat must contain a 3x3 (N/m) or 6x6 (GPa) matrix.")
        exit(1)


def read_elastic_manual():
    """Prompt the user to enter the elastic stiffness tensor manually.

    Accepts either 9 components (3x3, for 2D materials in N/m) or 36
    components (6x6 in GPa, for 3D materials or 2D materials before thickness
    conversion) entered as space-separated values on a single line. Loops
    until valid input is provided.

    Returns
    -------
    elastic_coef : numpy.ndarray, shape (3, 3) or (6, 6)
        Elastic stiffness tensor entered by the user.
    """

    print("""Enter elastic tensor manually:
For 2D materials (N/m)
C11 C12 C16
C12 C22 C26
C16 C26 C66
For 3D materials, or 2D materials as VASP cell tensor (GPa)
C11 C12 C13 C14 C15 C16
C12 C22 C23 C24 C25 C26
C13 C23 C33 C34 C35 C36
C14 C24 C34 C44 C45 C46
C15 C25 C35 C45 C55 C56
C16 C26 C36 C46 C56 C66""")

    while True:
        try:
            elastic_flat = np.array(input().split(), dtype=float)
        except ValueError:
            print("Warning! Input must be numbers.")
            continue
        if len(elastic_flat) == 9:
            return elastic_flat.reshape(3, 3)
        elif len(elastic_flat) == 36:
            return elastic_flat.reshape(6, 6)
        else:
            print("Warning! Input must be 9 or 36 components.")


def get_elastic_tensor(outcar_lines):
    """Obtain the elastic stiffness tensor using a three-level fallback chain.

    OUTCAR -> local 'Elastic.dat' -> manual input.

    Parameters
    ----------
    outcar_lines : list of str
        All lines from the VASP OUTCAR file, as returned by read_piezo_stress().

    Returns
    -------
    elastic_coef : numpy.ndarray, shape (3, 3) or (6, 6)
        Elastic stiffness tensor in GPa (6x6) or N/m (3x3).
    elastic_source : str
        'OUTCAR', 'Elastic.dat', or 'manual'.
    elastic_label : str
        'relaxed-ion' or 'clamped-ion' when read from OUTCAR, otherwise 'unknown'.
    """

    elastic_coef, elastic_label = read_elastic_tensor(outcar_lines)

    if elastic_coef is not None:
        return elastic_coef, 'OUTCAR', elastic_label

    print("The 'TOTAL ELASTIC MODULI (kBar)' section was not found in the OUTCAR file.\n"
          "Read Elastic coefficient from another instead.")

    elastic_coef = read_elastic_from_file()
    if elastic_coef is not None:
        return elastic_coef, 'Elastic.dat', 'unknown'

    return read_elastic_manual(), 'manual', 'unknown'


def compute_piezo_2d(structure, piezostress_coef, elastic_coef):
    """Compute the 2D piezoelectric stress tensor and elastic tensor for a 2D material.

    Calculates the vacuum-layer thickness (factor_2d) from the out-of-plane
    lattice vector and uses it to convert bulk VASP tensors to 2D sheet
    quantities. Only the in-plane components (indices 1, 2, 6 in Voigt
    notation, corresponding to xx, yy, xy) are retained.

    Unit conversions
    ----------------
    piezostress : C/m**2 × Å  →  10**-10 C/m
    elastic     : GPa × Å / 10  →  N/m  (any 6x6 input)

    Parameters
    ----------
    structure : ase.Atoms
        Crystal structure read from POSCAR, used to compute the vacuum
        layer thickness.
    piezostress_coef : numpy.ndarray, shape (3, 6)
        Full 3D piezoelectric stress tensor in Voigt notation (C/m**2), as
        returned by read_piezo_stress().
    elastic_coef : numpy.ndarray, shape (3, 3) or (6, 6)
        Elastic stiffness tensor: (6, 6) in GPa (OUTCAR, Elastic.dat, or
        manual input) or (3, 3) in N/m (Elastic.dat or manual input).

    Returns
    -------
    piezostress_2d : numpy.ndarray, shape (3, 3)
        In-plane piezoelectric stress tensor in units of 10**-10 C/m.
    elastic_2d : numpy.ndarray, shape (3, 3)
        In-plane elastic stiffness tensor in units of N/m.
    factor_2d : float
        Out-of-plane cell thickness in Å, used as the 2D conversion
        factor and passed to check_stability_2d().
    """

    vector_a = structure.cell[0]
    vector_b = structure.cell[1]
    vector_c = structure.cell[2]

    area_vector = np.cross(vector_a, vector_b)
    vector_n  = area_vector / np.linalg.norm(area_vector)
    factor_2d = np.abs(vector_c @ vector_n) # Angstrom

    piezostress_2d = piezostress_coef[:, [0, 1, 5]] * factor_2d  # Convert C/m^2*Angstrom to 10^-10 C/m

    if elastic_coef.shape == (6, 6):
        elastic_2d = elastic_coef[np.ix_([0, 1, 5], [0, 1, 5])] * factor_2d / 10  # Convert GPa*Angstrom to N/m
    elif elastic_coef.shape == (3, 3):
        elastic_2d = elastic_coef  # Already in N/m
    else:
        print("ERROR!\nElastic tensor for 2D materials must be 3x3 or 6x6. Got shape:", elastic_coef.shape)
        exit(1)

    return piezostress_2d, elastic_2d, factor_2d


def check_stability_2d(elastic_2d, factor_2d):
    """Check the mechanical stability of a 2D material from its elastic tensor.

    A 2D material is mechanically stable if all eigenvalues of the in-plane
    elastic tensor are positive (Born stability criteria). The threshold is
    scaled by factor_2d to remain consistent with the Å-based unit convention
    used in the 2D branch.

    Parameters
    ----------
    elastic_2d : numpy.ndarray, shape (3, 3)
        In-plane elastic stiffness tensor in units of N/m.
    factor_2d : float
        Out-of-plane cell thickness in Å, used to scale the stability
        threshold.
    """

    if np.all(np.linalg.eigvalsh(elastic_2d) > 1e-5 * factor_2d):
        print("This material is mechanically stable.")
    else:
        print("This material is mechanically unstable!!")
        exit(1)


def write_elastic_2d(elastic_2d, write_file=True):
    """Write the 2D elastic stiffness tensor to file and print to screen.

    Saves the in-plane elastic tensor components to 'Elastic.dat' (skipped
    when write_file is False, i.e. Elastic.dat was the input) and prints
    the same values to standard output.

    Parameters
    ----------
    elastic_2d : numpy.ndarray, shape (3, 3)
        In-plane elastic stiffness tensor in units of N/m, with components
        ordered as [[C11, C12, C16], [C12, C22, C26], [C16, C26, C66]].
    write_file : bool
        Write 'Elastic.dat' if True.
    """

    C11 = elastic_2d[0, 0]; C22 = elastic_2d[1, 1]; C66 = elastic_2d[2, 2]
    C12 = elastic_2d[0, 1]; C16 = elastic_2d[0, 2]; C26 = elastic_2d[1, 2]

    header = (
        "# Elastic tensor(N/m)\n"
        "#       C11         C12         C16\n"
        "#       C12         C22         C26\n"
        "#       C16         C26         C66\n\n"
    )
    rows = (
        f"   {C11:>11.4f} {C12:>11.4f} {C16:>11.4f}\n"
        f"   {C12:>11.4f} {C22:>11.4f} {C26:>11.4f}\n"
        f"   {C16:>11.4f} {C26:>11.4f} {C66:>11.4f}\n"
    )
    if write_file:
        with open('Elastic.dat', 'w') as o:
            o.write(header + rows)
    else:
        print("Elastic.dat was used as input and is not overwritten.")

    print("\n# Elastic tensor(N/m)\n"
          "#     C11         C12         C16\n"
          "#     C12         C22         C26\n"
          "#     C16         C26         C66\n\n"
          f"   {C11:>11.4f} {C12:>11.4f} {C16:>11.4f}\n"
          f"   {C12:>11.4f} {C22:>11.4f} {C26:>11.4f}\n"
          f"   {C16:>11.4f} {C26:>11.4f} {C66:>11.4f}\n")


def write_piezostress_2d(piezostress_2d, piezo_label):
    """Write the 2D piezoelectric stress tensor to file and print to screen.

    Saves the in-plane piezoelectric stress tensor components to
    'Piezoelectric_Stress.dat' and prints the same values to standard output.

    Parameters
    ----------
    piezostress_2d : numpy.ndarray, shape (3, 3)
        In-plane piezoelectric stress tensor in units of 10**-10 C/m, with
        components ordered as [[e11, e12, e16], [e21, e22, e26],
        [e31, e32, e36]].
    piezo_label : str
        'relaxed-ion' or 'clamped-ion', written in the header.
    """

    e11 = piezostress_2d[0, 0]; e12 = piezostress_2d[0, 1]; e16 = piezostress_2d[0, 2]
    e21 = piezostress_2d[1, 0]; e22 = piezostress_2d[1, 1]; e26 = piezostress_2d[1, 2]
    e31 = piezostress_2d[2, 0]; e32 = piezostress_2d[2, 1]; e36 = piezostress_2d[2, 2]

    header = (
        f"# Piezoelectric Stress(10^-10 C/m), {piezo_label}\n"
        "#       e11         e12         e16\n"
        "#       e21         e22         e26\n"
        "#       e31         e32         e36\n\n"
    )
    rows = (
        f"   {e11:>11.4f} {e12:>11.4f} {e16:>11.4f}\n"
        f"   {e21:>11.4f} {e22:>11.4f} {e26:>11.4f}\n"
        f"   {e31:>11.4f} {e32:>11.4f} {e36:>11.4f}\n"
    )
    with open('Piezoelectric_Stress.dat', 'w') as o:
        o.write(header + rows)

    print("\n" + header + rows)


def write_piezostrain_2d(piezostrain_2d, strain_label):
    """Write the 2D piezoelectric strain tensor to file and print to screen.

    Saves the in-plane piezoelectric strain tensor components to
    'Piezoelectric_Strain.dat' and prints the same values to standard output.

    Parameters
    ----------
    piezostrain_2d : numpy.ndarray, shape (3, 3)
        In-plane piezoelectric strain tensor in units of pm/V, with components
        ordered as [[d11, d12, d16], [d21, d22, d26], [d31, d32, d36]].
    strain_label : str
        Clamped/relaxed-ion origin of e and C, written in the header.
    """

    d11 = piezostrain_2d[0, 0]; d12 = piezostrain_2d[0, 1]; d16 = piezostrain_2d[0, 2]
    d21 = piezostrain_2d[1, 0]; d22 = piezostrain_2d[1, 1]; d26 = piezostrain_2d[1, 2]
    d31 = piezostrain_2d[2, 0]; d32 = piezostrain_2d[2, 1]; d36 = piezostrain_2d[2, 2]

    header = (
        f"# Piezoelectric Strain(pm/V), {strain_label}\n"
        "#       d11         d12         d16\n"
        "#       d21         d22         d26\n"
        "#       d31         d32         d36\n\n"
    )
    rows = (
        f"   {d11:>11.4f} {d12:>11.4f} {d16:>11.4f}\n"
        f"   {d21:>11.4f} {d22:>11.4f} {d26:>11.4f}\n"
        f"   {d31:>11.4f} {d32:>11.4f} {d36:>11.4f}\n"
    )
    with open('Piezoelectric_Strain.dat', 'w') as o:
        o.write(header + rows)

    print("\n" + header + rows)


def run_2d(structure, piezostress_coef, elastic_coef, elastic_source, piezo_label, strain_label):
    """Run the full 2D piezoelectric calculation pipeline.

    Orchestrates the 2D workflow in order: compute 2D tensors, write elastic
    tensor, write piezoelectric stress tensor, check mechanical stability,
    compute the piezoelectric strain tensor (d = e · S), and write the result.

    The piezoelectric strain tensor is obtained from:
        d = e · S,  where S = C**-1 (compliance tensor)

    Unit conversion for piezostrain:
        e [10**-10 C/m] · S [m/N] × 100  →  d [pm/V]

    Parameters
    ----------
    structure : ase.Atoms
        Crystal structure read from POSCAR.
    piezostress_coef : numpy.ndarray, shape (3, 6)
        Full piezoelectric stress tensor in Voigt notation (C/m**2).
    elastic_coef : numpy.ndarray, shape (3, 3) or (6, 6)
        Elastic stiffness tensor from OUTCAR, file, or manual input.
    elastic_source : str
        'OUTCAR', 'Elastic.dat', or 'manual'.
    piezo_label, strain_label : str
        Header labels for the stress and strain output files.
    """

    piezostress_2d, elastic_2d, factor_2d = compute_piezo_2d(structure, piezostress_coef, elastic_coef)

    write_elastic_2d(elastic_2d, elastic_source != 'Elastic.dat')
    write_piezostress_2d(piezostress_2d, piezo_label)
    check_stability_2d(elastic_2d, factor_2d)

    compliance_2d  = np.linalg.inv(elastic_2d)
    piezostrain_2d = (piezostress_2d @ compliance_2d) * 1e2  # 10^-10 m/V to pm/V

    write_piezostrain_2d(piezostrain_2d, strain_label)


def compute_piezo_3d(piezostress_coef, elastic_coef):
    """Validate and prepare the 3D piezoelectric and elastic tensors.

    Checks that the elastic tensor is 6x6 (required for 3D materials) and
    returns deep copies of both tensors to prevent in-place modification of
    the original arrays.

    Parameters
    ----------
    piezostress_coef : numpy.ndarray, shape (3, 6)
        Full piezoelectric stress tensor in Voigt notation (C/m**2), as
        returned by read_piezo_stress().
    elastic_coef : numpy.ndarray, shape (6, 6)
        Elastic stiffness tensor in GPa. Must be 6x6; exits with an error
        message if the shape does not match.

    Returns
    -------
    piezostress_3d : numpy.ndarray, shape (3, 6)
        Copy of the piezoelectric stress tensor (C/m**2).
    elastic_3d : numpy.ndarray, shape (6, 6)
        Copy of the elastic stiffness tensor (GPa).
    """

    if elastic_coef.shape != (6, 6):
        print("ERROR!\nElastic tensor for 3D materials must be 6x6. Got shape:", elastic_coef.shape)
        exit(1)

    piezostress_3d = np.copy(piezostress_coef)
    elastic_3d     = np.copy(elastic_coef)

    return piezostress_3d, elastic_3d


def check_stability_3d(elastic_3d):
    """Check the mechanical stability of a 3D material from its elastic tensor.

    A 3D material is mechanically stable if all eigenvalues of the elastic
    stiffness tensor are positive (Born stability criteria).

    Parameters
    ----------
    elastic_3d : numpy.ndarray, shape (6, 6)
        Elastic stiffness tensor in units of GPa.
    """

    if np.all(np.linalg.eigvalsh(elastic_3d) > 1e-5):
        print("This material is mechanically stable.")
    else:
        print("This material is mechanically unstable!!")
        exit(1)


def write_elastic_3d(elastic_3d, write_file=True):
    """Write the 3D elastic stiffness tensor to file and print to screen.

    Saves the full 6x6 elastic tensor to 'Elastic.dat' (skipped when
    write_file is False, i.e. Elastic.dat was the input) and prints the
    same values to standard output.

    Parameters
    ----------
    elastic_3d : numpy.ndarray, shape (6, 6)
        Elastic stiffness tensor in units of GPa in Voigt notation.
    write_file : bool
        Write 'Elastic.dat' if True.
    """

    C = elastic_3d

    header = (
        "# Elastic tensor(GPa)\n"
        "#       C11         C12         C13         C14         C15         C16\n"
        "#       C12         C22         C23         C24         C25         C26\n"
        "#       C13         C23         C33         C34         C35         C36\n"
        "#       C14         C24         C34         C44         C45         C46\n"
        "#       C15         C25         C35         C45         C55         C56\n"
        "#       C16         C26         C36         C46         C56         C66\n\n"
    )
    rows = "".join("   " + " ".join(f"{C[i, j]:>11.4f}" for j in range(6)) + "\n" for i in range(6))

    if write_file:
        with open('Elastic.dat', 'w') as o:
            o.write(header + rows)
    else:
        print("Elastic.dat was used as input and is not overwritten.")

    print("\n" + header + rows)


def write_piezostress_3d(piezostress_3d, piezo_label):
    """Write the 3D piezoelectric stress tensor to file and print to screen.

    Saves the full 3x6 piezoelectric stress tensor to 'Piezoelectric_Stress.dat'
    and prints the same values to standard output.

    Parameters
    ----------
    piezostress_3d : numpy.ndarray, shape (3, 6)
        Piezoelectric stress tensor in Voigt notation in units of C/m**2, with
        rows indexed by the electric field direction (x, y, z) and columns
        by the strain component (1, 2, 3, 4, 5, 6).
    piezo_label : str
        'relaxed-ion' or 'clamped-ion', written in the header.
    """

    E = piezostress_3d

    header = (
        f"# Piezoelectric Stress(C/m^2), {piezo_label}\n"
        "#       e11         e12         e13         e14         e15         e16\n"
        "#       e21         e22         e23         e24         e25         e26\n"
        "#       e31         e32         e33         e34         e35         e36\n\n"
    )
    rows = "".join("   " + " ".join(f"{E[i, j]:>11.4f}" for j in range(6)) + "\n" for i in range(3))

    with open('Piezoelectric_Stress.dat', 'w') as o:
        o.write(header + rows)

    print("\n" + header + rows)


def write_piezostrain_3d(piezostrain_3d, strain_label):
    """Write the 3D piezoelectric strain tensor to file and print to screen.

    Saves the full 3x6 piezoelectric strain tensor to 'Piezoelectric_Strain.dat'
    and prints the same values to standard output.

    Parameters
    ----------
    piezostrain_3d : numpy.ndarray, shape (3, 6)
        Piezoelectric strain tensor in Voigt notation in units of pm/V, with
        rows indexed by the electric field direction (x, y, z) and columns
        by the strain component (1, 2, 3, 4, 5, 6).
    strain_label : str
        Clamped/relaxed-ion origin of e and C, written in the header.
    """

    D = piezostrain_3d

    header = (
        f"# Piezoelectric Strain(pm/V), {strain_label}\n"
        "#       d11         d12         d13         d14         d15         d16\n"
        "#       d21         d22         d23         d24         d25         d26\n"
        "#       d31         d32         d33         d34         d35         d36\n\n"
    )
    rows = "".join("   " + " ".join(f"{D[i, j]:>11.4f}" for j in range(6)) + "\n" for i in range(3))

    with open('Piezoelectric_Strain.dat', 'w') as o:
        o.write(header + rows)

    print("\n" + header + rows)


def run_3d(piezostress_coef, elastic_coef, elastic_source, piezo_label, strain_label):
    """Run the full 3D piezoelectric calculation pipeline.

    Orchestrates the 3D workflow in order: validate and copy tensors, write
    elastic tensor, write piezoelectric stress tensor, check mechanical
    stability, compute the piezoelectric strain tensor (d = e · S), and write
    the result.

    The piezoelectric strain tensor is obtained from:
        d = e · S,  where S = C**-1 (compliance tensor)

    Unit conversion for piezostrain:
        e [C/m**2] · S [1/GPa] × 1000  →  d [pm/V]

    Parameters
    ----------
    piezostress_coef : numpy.ndarray, shape (3, 6)
        Full piezoelectric stress tensor in Voigt notation (C/m**2).
    elastic_coef : numpy.ndarray, shape (6, 6)
        Elastic stiffness tensor in GPa.
    elastic_source : str
        'OUTCAR', 'Elastic.dat', or 'manual'.
    piezo_label, strain_label : str
        Header labels for the stress and strain output files.
    """

    piezostress_3d, elastic_3d = compute_piezo_3d(piezostress_coef, elastic_coef)

    write_elastic_3d(elastic_3d, elastic_source != 'Elastic.dat')
    write_piezostress_3d(piezostress_3d, piezo_label)
    check_stability_3d(elastic_3d)

    compliance_3d  = np.linalg.inv(elastic_3d)
    piezostrain_3d = (piezostress_3d @ compliance_3d) * 1e3  # C/m^2 / GPa to pm/V

    write_piezostrain_3d(piezostrain_3d, strain_label)


def main():
    """Parse arguments, read inputs, and dispatch to 2D or 3D analysis."""

    if '-h' in argv or '--help' in argv or len(argv) != 3:
        usage()

    structure = read_structure(argv[1])
    outcar_lines, piezostress_coef, piezo_label     = read_piezo_stress(argv[2])
    elastic_coef, elastic_source, elastic_label     = get_elastic_tensor(outcar_lines)

    if elastic_label == 'unknown':
        print(f"Warning! Check that the elastic tensor from {elastic_source} is {piezo_label}, the same as e.")
    elif elastic_label != piezo_label:
        print(f"Warning! e is {piezo_label} but the elastic tensor is {elastic_label}; d mixes both.")
    strain_label = f"e: {piezo_label}, C: {elastic_label} ({elastic_source})"

    print("""Choices of type of material
1) 2D materials
2) bulk (3D) materials""")

    while True:
        input_type = input("Enter choice: ")
        if input_type == '1':
            run_2d(structure, piezostress_coef, elastic_coef, elastic_source, piezo_label, strain_label)
            break
        elif input_type == '2':
            run_3d(piezostress_coef, elastic_coef, elastic_source, piezo_label, strain_label)
            break
        else:
            print("Warning! Wrong input")


if __name__ == '__main__':
    main()
