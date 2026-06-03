#!/usr/bin/env python

from sys import argv, exit
import os
import numpy as np
from ase.io import read
from ase.spacegroup.symmetrize import check_symmetry
from scipy.constants import Avogadro, h, k


def usage():
    """Print usage information and exit."""
    print("""
Usage: vaspMechanics.py <POSCAR input> <OUTCAR input>

This script calculates Young's modulus and Poisson's Ratio as functions of crystal orientation.
Output files can plot by Origin Program or plotMechanics.py script. (For 2D)
Check mechanical stability too.
This script read POSCAR and OUTCAR files by default.

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
        exit(0)

    return read(poscar_file)


def read_elastic_tensor(outcar_file):
    """Read and parse the elastic tensor from a VASP OUTCAR file.

    Extracts the 'TOTAL ELASTIC MODULI (kBar)' section, converts units
    from kBar to GPa, and reorders indices from VASP convention
    (xx, yy, zz, xy, yz, xz) to Voigt notation (1, 2, 3, 4, 5, 6).

    Parameters
    ----------
    outcar_file : str
        Path to the OUTCAR file.

    Returns
    -------
    elastic_coef : np.ndarray, shape (6, 6)
        Elastic tensor in Voigt notation (GPa).
    """

    if not os.path.exists(outcar_file):
        print(f"ERROR!\nFile: {outcar_file} does not exist.")
        exit(0)

    with open(outcar_file, 'r') as f:
        outcar_lines = f.readlines()

    moduli_index = None
    for i, line in enumerate(outcar_lines):
        if 'TOTAL ELASTIC MODULI (kBar)' in line:
            moduli_index = i
            break

    if moduli_index is None:
        print("The 'TOTAL ELASTIC MODULI (kBar)' section was not found in the OUTCAR file.")
        exit(0)

    moduli_lines = outcar_lines[moduli_index + 3:moduli_index + 9]
    elastic_vasp = np.array([[float(x) for x in line.split()[1:]]
                             for line in moduli_lines]) / 10  # Convert unit kBar to GPa

    # Reorder from VASP convention (xx,yy,zz,xy,yz,xz) to Voigt notation (1,2,3,4,5,6)
    elastic_coef = elastic_vasp[:, [0, 1, 2, 4, 5, 3]][[0, 1, 2, 4, 5, 3], :]

    return elastic_coef


def get_2d_lattice_type(structure):
    """Determine the 2D lattice type from cell parameters.

    Classifies the in-plane lattice as square, rectangular, hexagonal,
    or oblique based on lattice lengths a, b and the angle gamma.

    Parameters
    ----------
    structure : ase.Atoms
        ASE Atoms object containing the crystal structure.

    Returns
    -------
    lattice_type : str
        One of 'square', 'rectangular', 'hexagonal', or 'oblique'.
    """

    length_a = structure.cell.cellpar()[0]
    length_b = structure.cell.cellpar()[1]
    gamma    = structure.cell.cellpar()[5]

    if np.abs(gamma - 90.) < 1e-5:
        return 'square' if np.abs(length_a - length_b) < 1e-8 else 'rectangular'
    elif (np.abs(gamma - 60.) < 1e-5 or np.abs(gamma - 120.) < 1e-5) and np.abs(length_a - length_b) < 1e-8:
        return 'hexagonal'
    else:
        return 'oblique'


def compute_elastic_2d(structure, elastic_coef):
    """Extract the 2D elastic tensor from the 3D VASP elastic tensor.

    Projects the 3D elastic tensor onto the 2D in-plane subspace by
    multiplying by the out-of-plane thickness (factor_2d), converting
    units from GPa to N/m. Off-diagonal terms C16 and C26 are set to
    zero for non-oblique lattices as required by 2D lattice symmetry.

    Parameters
    ----------
    structure : ase.Atoms
        ASE Atoms object containing the crystal structure.
    elastic_coef : np.ndarray, shape (6, 6)
        3D elastic tensor in Voigt notation (GPa).

    Returns
    -------
    elastic_2d : np.ndarray, shape (3, 3)
        2D elastic tensor in Voigt notation (N/m).
    C11, C22, C12, C66 : float
        In-plane elastic constants (N/m).
    C16, C26 : float
        Off-diagonal elastic constants (N/m); zero for non-oblique lattices.
    factor_2d : float
        Out-of-plane thickness in nm, used for GPa*Angstrom to N/m conversion.
    """

    vector_a = structure.cell[0]
    vector_b = structure.cell[1]
    vector_c = structure.cell[2]

    area_vector = np.cross(vector_a, vector_b)
    vector_n    = area_vector / np.linalg.norm(area_vector)
    factor_2d   = np.abs(vector_c @ vector_n) / 10  # Angstrom to nm (for GPa*nm = N/m conversion)

    elastic_2d = elastic_coef[np.ix_([0, 1, 5], [0, 1, 5])] * factor_2d  # Convert unit GPa*Angstrom to N/m

    lattice_type = get_2d_lattice_type(structure)
    print(f"This material is {lattice_type}.")

    oblique = (lattice_type == 'oblique')

    C11 = elastic_2d[0, 0]
    C22 = elastic_2d[1, 1]
    C12 = elastic_2d[0, 1]
    C66 = elastic_2d[2, 2]
    C16 = elastic_2d[0, 2] if oblique else 0.
    C26 = elastic_2d[1, 2] if oblique else 0.

    return elastic_2d, C11, C22, C12, C66, C16, C26, factor_2d


def write_elastic_2d(C11, C22, C12, C66, C16, C26):
    """Print and write the 2D elastic tensor to Elastic.dat.

    Parameters
    ----------
    C11, C22, C12, C66, C16, C26 : float
        2D elastic constants (N/m).
    """

    header = (
        "# Elastic tensor(N/m)\n"
        "#     C11         C12         C16\n"
        "#     C12         C22         C26\n"
        "#     C16         C26         C66\n\n"
    )
    rows = (
        f"   {C11:>11.4f} {C12:>11.4f} {C16:>11.4f}\n"
        f"   {C12:>11.4f} {C22:>11.4f} {C26:>11.4f}\n"
        f"   {C16:>11.4f} {C26:>11.4f} {C66:>11.4f}\n"
    )
    with open('Elastic.dat', 'w') as o:
        o.write(header + rows)

    print("\n# Elastic tensor(N/m)\n"
          "#     C11         C12         C16\n"
          "#     C12         C22         C26\n"
          "#     C16         C26         C66\n\n"
          f"   {C11:>11.4f} {C12:>11.4f} {C16:>11.4f}\n"
          f"   {C12:>11.4f} {C22:>11.4f} {C26:>11.4f}\n"
          f"   {C16:>11.4f} {C26:>11.4f} {C66:>11.4f}\n")


def check_stability_2d(elastic_2d, factor_2d):
    """Check mechanical stability of a 2D material via eigenvalue criterion.

    A 2D material is mechanically stable if all eigenvalues of the
    2D elastic tensor are positive. Exits if the material is unstable.

    Parameters
    ----------
    elastic_2d : np.ndarray, shape (3, 3)
        2D elastic tensor in Voigt notation (N/m).
    factor_2d : float
        Out-of-plane thickness (nm), used to set the eigenvalue threshold.
    """

    if np.all(np.linalg.eigvalsh(elastic_2d) > 1e-5 * factor_2d):
        print("This material is mechanically stable.")
    else:
        print("This material is mechanically unstable!!")
        exit(0)


def compute_directional_properties_2d(elastic_2d):
    """Compute angle-dependent mechanical properties of a 2D material.

    Calculates Young's modulus, Poisson's ratio, and shear modulus as
    functions of in-plane angle (0 to 360 degrees) using the compliance
    tensor rotation formulas. Also computes the maximum and minimum
    values of each property and their corresponding angles.

    Parameters
    ----------
    elastic_2d : np.ndarray, shape (3, 3)
        2D elastic tensor in Voigt notation (N/m).

    Returns
    -------
    degrees : np.ndarray
        Angles from 0 to 360 degrees in 0.1 degree steps.
    young_modulus : np.ndarray
        Young's modulus at each angle (N/m).
    poisson_ratio : np.ndarray
        Poisson's ratio at each angle (dimensionless).
    shear_modulus : np.ndarray
        Shear modulus at each angle (N/m).
    max_min : dict
        Maximum and minimum values of Young's modulus (N/m), Poisson's
        ratio (dimensionless), and shear modulus (N/m), along with the
        corresponding angles (degrees).
    """

    degrees = np.arange(0, 360.0, 0.1)
    radians = np.radians(degrees)
    sin = np.sin(radians)
    cos = np.cos(radians)

    compliance_2d = np.linalg.inv(elastic_2d)

    S11 = compliance_2d[0, 0]
    S22 = compliance_2d[1, 1]
    S12 = compliance_2d[0, 1]
    S66 = compliance_2d[2, 2]
    S16 = compliance_2d[0, 2]
    S26 = compliance_2d[1, 2]

    # Calculate Young's modulus, Poisson's ratio, and shear modulus from Sij
    A = (S11 * cos**4 + S22 * sin**4
         + (2 * S12 + S66) * cos**2 * sin**2
         + 2 * S16 * cos**3 * sin
         + 2 * S26 * cos * sin**3)
    young_modulus = 1 / A

    B = ((S11 + S22 - S66) * cos**2 * sin**2
         + S12 * (cos**4 + sin**4)
         + (S26 - S16) * (cos**3 * sin - cos * sin**3))
    poisson_ratio = -B / A

    C = (4 * (S11 + S22 - 2 * S12) * cos**2 * sin**2
         + S66 * (cos**2 - sin**2)**2
         + 4 * (S16 - S26) * (cos**3 * sin - cos * sin**3))
    shear_modulus = 1 / C

    max_min = {
        "max_young":         np.max(young_modulus),
        "max_young_angle":   degrees[np.argmax(young_modulus)],
        "min_young":         np.min(young_modulus),
        "min_young_angle":   degrees[np.argmin(young_modulus)],
        "max_poisson":       np.max(poisson_ratio),
        "max_poisson_angle": degrees[np.argmax(poisson_ratio)],
        "min_poisson":       np.min(poisson_ratio),
        "min_poisson_angle": degrees[np.argmin(poisson_ratio)],
        "max_shear":         np.max(shear_modulus),
        "max_shear_angle":   degrees[np.argmax(shear_modulus)],
        "min_shear":         np.min(shear_modulus),
        "min_shear_angle":   degrees[np.argmin(shear_modulus)],
    }

    return degrees, young_modulus, poisson_ratio, shear_modulus, max_min


def write_directional_properties_2d(degrees, young_modulus, poisson_ratio, shear_modulus):
    """Write angle-dependent mechanical properties to output files.

    Writes Young's modulus to Young.dat, Poisson's ratio to Poisson.dat
    (with an extra |v| column if any value is negative), and shear
    modulus to Shear.dat.

    Parameters
    ----------
    degrees : np.ndarray
        Angles from 0 to 360 degrees.
    young_modulus : np.ndarray
        Young's modulus at each angle (N/m).
    poisson_ratio : np.ndarray
        Poisson's ratio at each angle (dimensionless).
    shear_modulus : np.ndarray
        Shear modulus at each angle (N/m).
    """

    with open('Young.dat', 'w') as o:
        o.write("# Young's Modulus\n")
        o.write("#  Degree(\u00B0)  Y(N/m)\n")
        for dg, y in zip(degrees, young_modulus):
            o.write(f" {dg:>6.1f}      {y:>12.8f}\n")

    with open('Poisson.dat', 'w') as o:
        o.write("# Poisson's Ratio\n")
        if (poisson_ratio < 0.).any():
            o.write("#  Degree(\u00B0) v             |v|\n")
            for dg, nu in zip(degrees, poisson_ratio):
                o.write(f" {dg:>6.1f}    {nu:>12.8f}   {abs(nu):>12.8f}\n")
        else:
            o.write("#  Degree(\u00B0) v\n")
            for dg, nu in zip(degrees, poisson_ratio):
                o.write(f" {dg:>6.1f}   {nu:>12.8f}\n")

    with open('Shear.dat', 'w') as o:
        o.write("# Shear modulus\n")
        o.write("#  Degree(\u00B0)  G(N/m)\n")
        for dg, g in zip(degrees, shear_modulus):
            o.write(f" {dg:>6.1f}      {g:>12.8f}\n")


def write_max_min(max_min):
    """Print and write extreme values of directional mechanical properties to Extreme.dat.

    Reports the maximum and minimum of Young's modulus, Poisson's ratio,
    and shear modulus together with the angles at which they occur.

    Parameters
    ----------
    max_min : dict
        Dictionary of extreme values as returned by
        compute_directional_properties_2d(). Keys: max_young,
        max_young_angle, min_young, min_young_angle, max_poisson,
        max_poisson_angle, min_poisson, min_poisson_angle, max_shear,
        max_shear_angle, min_shear, min_shear_angle.
    """

    max_young         = max_min["max_young"]
    max_young_angle   = max_min["max_young_angle"]
    min_young         = max_min["min_young"]
    min_young_angle   = max_min["min_young_angle"]
    max_poisson       = max_min["max_poisson"]
    max_poisson_angle = max_min["max_poisson_angle"]
    min_poisson       = max_min["min_poisson"]
    min_poisson_angle = max_min["min_poisson_angle"]
    max_shear         = max_min["max_shear"]
    max_shear_angle   = max_min["max_shear_angle"]
    min_shear         = max_min["min_shear"]
    min_shear_angle   = max_min["min_shear_angle"]

    print(f"\nMax Young's modulus: {max_young:>.4f} N/m at {max_young_angle:>.1f} degrees")
    print(f"Min Young's modulus: {min_young:>.4f} N/m at {min_young_angle:>.1f} degrees")
    print(f"\nMax Poisson's ratio: {max_poisson:>.4f} at {max_poisson_angle:>.1f} degrees")
    print(f"Min Poisson's ratio: {min_poisson:>.4f} at {min_poisson_angle:>.1f} degrees")
    print(f"\nMax Shear modulus: {max_shear:>.4f} N/m at {max_shear_angle:>.1f} degrees")
    print(f"Min Shear modulus: {min_shear:>.4f} N/m at {min_shear_angle:>.1f} degrees")

    with open("Extreme.dat", "w") as f:
        f.write("# Extreme values of directional mechanical properties\n")
        f.write(f"Max Young's modulus: {max_young:>.4f} N/m at {max_young_angle:>.1f} degrees\n")
        f.write(f"Min Young's modulus: {min_young:>.4f} N/m at {min_young_angle:>.1f} degrees\n")
        f.write(f"\nMax Poisson's ratio: {max_poisson:>.4f} at {max_poisson_angle:>.1f} degrees\n")
        f.write(f"Min Poisson's ratio: {min_poisson:>.4f} at {min_poisson_angle:>.1f} degrees\n")
        f.write(f"\nMax Shear modulus: {max_shear:>.4f} N/m at {max_shear_angle:>.1f} degrees\n")
        f.write(f"Min Shear modulus: {min_shear:>.4f} N/m at {min_shear_angle:>.1f} degrees\n")


def run_2d(structure, elastic_coef):
    """Run the full 2D mechanical analysis pipeline.

    Computes the 2D elastic tensor, writes it to file, checks mechanical
    stability, then computes and writes angle-dependent Young's modulus,
    Poisson's ratio, and shear modulus, and reports their extreme values.

    Parameters
    ----------
    structure : ase.Atoms
        ASE Atoms object containing the crystal structure.
    elastic_coef : np.ndarray, shape (6, 6)
        3D elastic tensor in Voigt notation (GPa).
    """

    elastic_2d, C11, C22, C12, C66, C16, C26, factor_2d = compute_elastic_2d(structure, elastic_coef)
    write_elastic_2d(C11, C22, C12, C66, C16, C26)
    check_stability_2d(elastic_2d, factor_2d)
    degrees, young_modulus, poisson_ratio, shear_modulus, max_min = compute_directional_properties_2d(elastic_2d)
    write_directional_properties_2d(degrees, young_modulus, poisson_ratio, shear_modulus)
    write_max_min(max_min)


def get_crystal_system(structure):
    """Determine the crystal system from the spacegroup number.

    Uses ASE's check_symmetry to obtain the spacegroup number and maps
    it to one of the seven crystal systems (with Tetragonal and Trigonal
    each split into two groups).

    Parameters
    ----------
    structure : ase.Atoms
        ASE Atoms object containing the crystal structure.

    Returns
    -------
    crystal_system : str
        Crystal system name, one of: 'Cubic', 'Hexagonal', 'Trigonal I',
        'Trigonal II', 'Tetragonal I', 'Tetragonal II', 'Orthorhombic',
        'Monoclinic', or 'Triclinic'.
    """

    spacegroup = check_symmetry(structure).number

    if 195 <= spacegroup <= 230:
        return 'Cubic'
    elif 168 <= spacegroup <= 194:
        return 'Hexagonal'
    elif 149 <= spacegroup <= 167:
        return 'Trigonal I'
    elif 143 <= spacegroup <= 148:
        return 'Trigonal II'
    elif 89 <= spacegroup <= 142:
        return 'Tetragonal I'
    elif 75 <= spacegroup <= 88:
        return 'Tetragonal II'
    elif 16 <= spacegroup <= 74:
        return 'Orthorhombic'
    elif 3 <= spacegroup <= 15:
        return 'Monoclinic'
    else:
        return 'Triclinic'


def write_elastic_3d(elastic_3d):
    """Print and write the full 3D elastic tensor to Elastic.dat.

    Parameters
    ----------
    elastic_3d : np.ndarray, shape (6, 6)
        3D elastic tensor in Voigt notation (GPa).
    """

    C = elastic_3d  # shorthand for formatting

    header = (
        "# Elastic tensor(GPa)\n"
        "#     C11         C12         C13         C14         C15         C16\n"
        "#     C12         C22         C23         C24         C25         C26\n"
        "#     C13         C23         C33         C34         C35         C36\n"
        "#     C14         C24         C34         C44         C45         C46\n"
        "#     C15         C25         C35         C45         C55         C56\n"
        "#     C16         C26         C36         C46         C56         C66\n\n"
    )
    rows = (
        f"   {C[0, 0]:>11.4f} {C[0, 1]:>11.4f} {C[0, 2]:>11.4f} {C[0, 3]:>11.4f} {C[0, 4]:>11.4f} {C[0, 5]:>11.4f}\n"
        f"   {C[1, 0]:>11.4f} {C[1, 1]:>11.4f} {C[1, 2]:>11.4f} {C[1, 3]:>11.4f} {C[1, 4]:>11.4f} {C[1, 5]:>11.4f}\n"
        f"   {C[2, 0]:>11.4f} {C[2, 1]:>11.4f} {C[2, 2]:>11.4f} {C[2, 3]:>11.4f} {C[2, 4]:>11.4f} {C[2, 5]:>11.4f}\n"
        f"   {C[3, 0]:>11.4f} {C[3, 1]:>11.4f} {C[3, 2]:>11.4f} {C[3, 3]:>11.4f} {C[3, 4]:>11.4f} {C[3, 5]:>11.4f}\n"
        f"   {C[4, 0]:>11.4f} {C[4, 1]:>11.4f} {C[4, 2]:>11.4f} {C[4, 3]:>11.4f} {C[4, 4]:>11.4f} {C[4, 5]:>11.4f}\n"
        f"   {C[5, 0]:>11.4f} {C[5, 1]:>11.4f} {C[5, 2]:>11.4f} {C[5, 3]:>11.4f} {C[5, 4]:>11.4f} {C[5, 5]:>11.4f}\n"
    )

    with open('Elastic.dat', 'w') as o:
        o.write(header + rows)

    print("\n# Elastic tensor(GPa)\n"
          "#     C11         C12         C13         C14         C15         C16\n"
          "#     C12         C22         C23         C24         C25         C26\n"
          "#     C13         C23         C33         C34         C35         C36\n"
          "#     C14         C24         C34         C44         C45         C46\n"
          "#     C15         C25         C35         C45         C55         C56\n"
          "#     C16         C26         C36         C46         C56         C66\n\n"
          f"   {C[0, 0]:>11.4f} {C[0, 1]:>11.4f} {C[0, 2]:>11.4f} {C[0, 3]:>11.4f} {C[0, 4]:>11.4f} {C[0, 5]:>11.4f}\n"
          f"   {C[1, 0]:>11.4f} {C[1, 1]:>11.4f} {C[1, 2]:>11.4f} {C[1, 3]:>11.4f} {C[1, 4]:>11.4f} {C[1, 5]:>11.4f}\n"
          f"   {C[2, 0]:>11.4f} {C[2, 1]:>11.4f} {C[2, 2]:>11.4f} {C[2, 3]:>11.4f} {C[2, 4]:>11.4f} {C[2, 5]:>11.4f}\n"
          f"   {C[3, 0]:>11.4f} {C[3, 1]:>11.4f} {C[3, 2]:>11.4f} {C[3, 3]:>11.4f} {C[3, 4]:>11.4f} {C[3, 5]:>11.4f}\n"
          f"   {C[4, 0]:>11.4f} {C[4, 1]:>11.4f} {C[4, 2]:>11.4f} {C[4, 3]:>11.4f} {C[4, 4]:>11.4f} {C[4, 5]:>11.4f}\n"
          f"   {C[5, 0]:>11.4f} {C[5, 1]:>11.4f} {C[5, 2]:>11.4f} {C[5, 3]:>11.4f} {C[5, 4]:>11.4f} {C[5, 5]:>11.4f}\n")


def check_stability_3d(elastic_3d):
    """Check mechanical stability of a 3D material via eigenvalue criterion.

    A 3D material is mechanically stable if all eigenvalues of the
    elastic tensor are positive. Exits if the material is unstable.

    Parameters
    ----------
    elastic_3d : np.ndarray, shape (6, 6)
        3D elastic tensor in Voigt notation (GPa).
    """

    if np.all(np.linalg.eigvalsh(elastic_3d) > 1e-5):
        print("This material is mechanically stable.")
    else:
        print("This material is mechanically unstable!!")
        exit(0)


def compute_mechanical_properties_3d(elastic_3d, structure):
    """Compute bulk mechanical properties of a 3D material.

    Calculates Voigt, Reuss, and Hill (VRH) averages for bulk and shear
    moduli, derived elastic properties (Young's modulus, Poisson's ratio,
    P-wave modulus, Lame parameter, Pugh's ratio), anisotropy indices,
    sound velocities, and the Debye temperature.

    Parameters
    ----------
    elastic_3d : np.ndarray, shape (6, 6)
        3D elastic tensor in Voigt notation (GPa).
    structure : ase.Atoms
        ASE Atoms object, used to obtain mass, volume, and atom count.

    Returns
    -------
    props : dict
        Dictionary containing all computed mechanical properties:
        bulk_voigt, bulk_reuss, shear_voigt, shear_reuss,
        bulk_modulus, shear_modulus, young_modulus, poisson_ratio,
        pwave_modulus, lame_parameter, pugh_ratio, v_m,
        debye_temperature, universal_anisotropy, bulk_anisotropy,
        shear_anisotropy, anisotropy_1, anisotropy_2, anisotropy_3.
    """

    C11 = elastic_3d[0, 0]; C22 = elastic_3d[1, 1]; C33 = elastic_3d[2, 2]
    C12 = elastic_3d[0, 1]; C13 = elastic_3d[0, 2]; C23 = elastic_3d[1, 2]
    C44 = elastic_3d[3, 3]; C55 = elastic_3d[4, 4]; C66 = elastic_3d[5, 5]

    compliance_3d = np.linalg.inv(elastic_3d)

    S11 = compliance_3d[0, 0]; S22 = compliance_3d[1, 1]; S33 = compliance_3d[2, 2]
    S12 = compliance_3d[0, 1]; S13 = compliance_3d[0, 2]; S23 = compliance_3d[1, 2]
    S44 = compliance_3d[3, 3]; S55 = compliance_3d[4, 4]; S66 = compliance_3d[5, 5]

    total_mass  = structure.get_masses().sum()  # amu
    volume      = structure.get_volume()         # Angstrom^3
    density     = total_mass / volume            # amu / Angstrom^3
    total_atoms = len(structure)

    # VRH averages
    bulk_voigt    = (C11 + C22 + C33 + 2 * (C12 + C23 + C13)) / 9
    shear_voigt   = (C11 + C22 + C33 - C12 - C23 - C13 + 3 * (C44 + C55 + C66)) / 15
    bulk_reuss    = 1 / (S11 + S22 + S33 + 2 * (S12 + S23 + S13))
    shear_reuss   = 15 / (4 * (S11 + S22 + S33 - S12 - S23 - S13) + 3 * (S44 + S55 + S66))
    bulk_modulus  = (bulk_voigt + bulk_reuss) / 2
    shear_modulus = (shear_voigt + shear_reuss) / 2

    young_modulus  = (9 * bulk_modulus * shear_modulus) / (3 * bulk_modulus + shear_modulus)
    pugh_ratio     = bulk_modulus / shear_modulus
    poisson_ratio  = (3 * bulk_modulus - 2 * shear_modulus) / (2 * (3 * bulk_modulus + shear_modulus))
    pwave_modulus  = bulk_modulus + 4 * shear_modulus / 3
    lame_parameter = bulk_modulus - 2 * shear_modulus / 3

    # Anisotropy indices
    universal_anisotropy = 5 * (shear_voigt / shear_reuss) + (bulk_voigt / bulk_reuss) - 6
    bulk_anisotropy      = (bulk_voigt - bulk_reuss) / (bulk_voigt + bulk_reuss)
    shear_anisotropy     = (shear_voigt - shear_reuss) / (shear_voigt + shear_reuss)
    anisotropy_1 = 4 * C44 / (C11 + C33 - 2 * C13)
    anisotropy_2 = 4 * C55 / (C22 + C33 - 2 * C23)
    anisotropy_3 = 4 * C66 / (C11 + C22 - 2 * C12)

    # Unit conversions
    unit_modulus = 1e9            # GPa to Pa
    unit_mass    = 1e-3 / Avogadro  # amu to kg
    unit_volume  = (1e-10)**3     # Angstrom^3 to m^3
    unit_density = unit_mass / unit_volume  # kg / m^3

    # Sound velocities
    v_t = np.sqrt((shear_modulus * unit_modulus) / (density * unit_density))                               # m/s
    v_l = np.sqrt(((3 * bulk_modulus + 4 * shear_modulus) * unit_modulus) / (3 * density * unit_density))  # m/s
    v_m = np.cbrt(((2 / v_t**3) + (1 / v_l**3)) / 3) ** -1                                                # m/s

    debye_temperature = (h / k) * np.cbrt((3 * total_atoms) / (4 * np.pi * volume * unit_volume)) * v_m   # K

    return {
        'bulk_voigt': bulk_voigt, 'bulk_reuss': bulk_reuss,
        'shear_voigt': shear_voigt, 'shear_reuss': shear_reuss,
        'bulk_modulus': bulk_modulus, 'shear_modulus': shear_modulus,
        'young_modulus': young_modulus, 'poisson_ratio': poisson_ratio,
        'pwave_modulus': pwave_modulus, 'lame_parameter': lame_parameter,
        'pugh_ratio': pugh_ratio, 'v_m': v_m, 'debye_temperature': debye_temperature,
        'universal_anisotropy': universal_anisotropy,
        'bulk_anisotropy': bulk_anisotropy, 'shear_anisotropy': shear_anisotropy,
        'anisotropy_1': anisotropy_1, 'anisotropy_2': anisotropy_2, 'anisotropy_3': anisotropy_3,
        }


def print_and_write_mechanical_properties_3d(props):
    """Print and write mechanical properties and anisotropy indices to output files.

    Prints all properties to stdout and writes them to two files:
    Mechanics.dat (bulk mechanical properties) and Anisotropy.dat
    (anisotropy indices).

    Parameters
    ----------
    props : dict
        Dictionary of mechanical properties as returned by
        compute_mechanical_properties_3d().
    """

    bv  = props['bulk_voigt']          ; br  = props['bulk_reuss']
    gv  = props['shear_voigt']         ; gr  = props['shear_reuss']
    B   = props['bulk_modulus']        ; G   = props['shear_modulus']
    E   = props['young_modulus']       ; v   = props['poisson_ratio']
    M   = props['pwave_modulus']       ; L   = props['lame_parameter']
    P   = props['pugh_ratio']          ; vm  = props['v_m']
    tD  = props['debye_temperature']
    A_B = props['bulk_anisotropy']     ; A_G = props['shear_anisotropy']
    A_U = props['universal_anisotropy']
    A1  = props['anisotropy_1']        ; A2  = props['anisotropy_2']; A3 = props['anisotropy_3']

    mech_str = (
        f"\n# Mechanical properties\n"
        f"Voigt's Bulk modulus          (B_V): {bv:>10.4f} GPa\n"
        f"Reuss' Bulk modulus           (B_R): {br:>10.4f} GPa\n"
        f"Voigt's Shear Modulus         (G_V): {gv:>10.4f} GPa\n"
        f"Reuss' Shear Modulus          (G_R): {gr:>10.4f} GPa\n"
        f"Bulk Modulus                  (B)  : {B:>10.4f} GPa\n"
        f"Shear Modulus                 (G)  : {G:>10.4f} GPa\n"
        f"Young's Modulus               (E)  : {E:>10.4f} GPa\n"
        f"Poisson's Ratio               (v)  : {v:>10.4f}\n"
        f"P-wave Modulus                (M)  : {M:>10.4f} GPa\n"
        f"Lam\u00e9's first parameter        (L)  : {L:>10.4f} GPa\n"
        f"Pugh's Ratio                  (P)  : {P:>10.4f}\n"
        f"Sound velocity                (v_m): {vm:>10.4f} m/s\n"
        f"Debye temperature             (\u03B8_D): {tD:>10.4f} K\n"
    )

    aniso_str = (
        f"\n# Anisotropic analysis\n"
        f"Bulk Anisotropy               (A_B): {A_B:>10.4f}\n"
        f"Shear Anisotropy              (A_G): {A_G:>10.4f}\n"
        f"Universal Anisotropy          (A_U): {A_U:>10.4f}\n"
        f"(100) Planar Shear Anisotropy (A_1): {A1:>10.4f}\n"
        f"(010) Planar Shear Anisotropy (A_2): {A2:>10.4f}\n"
        f"(001) Planar Shear Anisotropy (A_3): {A3:>10.4f}\n"
    )

    print(mech_str + aniso_str)

    with open('Mechanics.dat', 'w') as o:
        o.write("# Mechanical properties\n")
        o.write(f"Bulk Modulus                  (B)  : {B:>10.4f} GPa\n")
        o.write(f"Voigt's Bulk modulus          (B_V): {bv:>10.4f} GPa\n")
        o.write(f"Reuss' Bulk modulus           (B_R): {br:>10.4f} GPa\n")
        o.write(f"Voigt's Shear Modulus         (G_V): {gv:>10.4f} GPa\n")
        o.write(f"Reuss' Shear Modulus          (G_R): {gr:>10.4f} GPa\n")
        o.write(f"Shear Modulus                 (G)  : {G:>10.4f} GPa\n")
        o.write(f"Young's Modulus               (E)  : {E:>10.4f} GPa\n")
        o.write(f"Poisson's Ratio               (v)  : {v:>10.4f}\n")
        o.write(f"P-wave Modulus                (M)  : {M:>10.4f} GPa\n")
        o.write(f"Lam\u00e9's first parameter        (L)  : {L:>10.4f} GPa\n")
        o.write(f"Pugh's Ratio                  (P)  : {P:>10.4f}\n")
        o.write(f"Sound velocity                (v_m): {vm:>10.4f} m/s\n")
        o.write(f"Debye temperature             (\u03B8_D): {tD:>10.4f} K\n")

    with open('Anisotropy.dat', 'w') as o:
        o.write("# Anisotropic analysis\n")
        o.write(f"Bulk Anisotropy               (A_B): {A_B:>10.4f}\n")
        o.write(f"Shear Anisotropy              (A_G): {A_G:>10.4f}\n")
        o.write(f"Universal Anisotropy          (A_U): {A_U:>10.4f}\n")
        o.write(f"(100) Planar Shear Anisotropy (A_1): {A1:>10.4f}\n")
        o.write(f"(010) Planar Shear Anisotropy (A_2): {A2:>10.4f}\n")
        o.write(f"(001) Planar Shear Anisotropy (A_3): {A3:>10.4f}\n")


def run_3d(structure, elastic_coef):
    """Run the full 3D mechanical analysis pipeline.

    Writes the elastic tensor to file, identifies the crystal system,
    checks mechanical stability, then computes and writes all bulk
    mechanical properties and anisotropy indices.

    Parameters
    ----------
    structure : ase.Atoms
        ASE Atoms object containing the crystal structure.
    elastic_coef : np.ndarray, shape (6, 6)
        3D elastic tensor in Voigt notation (GPa).
    """

    elastic_3d = np.copy(elastic_coef)
    write_elastic_3d(elastic_3d)

    crystal_system = get_crystal_system(structure)
    print(f"This material is {crystal_system}")

    check_stability_3d(elastic_3d)

    props = compute_mechanical_properties_3d(elastic_3d, structure)
    print_and_write_mechanical_properties_3d(props)


def main():
    """Parse arguments, read inputs, and dispatch to 2D or 3D analysis."""

    if '-h' in argv or len(argv) != 3:
        usage()

    structure    = read_structure(argv[1])
    elastic_coef = read_elastic_tensor(argv[2])

    print("""Choices of type of material
1) 2D materials
2) bulk materials""")

    while True:
        input_type = input("Enter choice: ")
        if input_type == '1':
            run_2d(structure, elastic_coef)
            break
        elif input_type == '2':
            run_3d(structure, elastic_coef)
            break
        else:
            print("Warning! Wrong input")


if __name__ == '__main__':
    main()
