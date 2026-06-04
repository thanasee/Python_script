# VASP Python Utility Scripts

A collection of Python scripts for VASP output analysis and related tasks, developed for computational materials science research on the LANTA HPC cluster.

**Author:** [Thanasee Thanasarnsurapong](https://scholar.google.com/citations?user=4KHXv9gAAAAJ&hl=en)

---

## Overview

This repository is organized into six functional categories:

1. **Thermal transport analysis** — post-process force constants, reconnect phonon branches, generate phonon band plot boundaries, extract and analyze lattice thermal conductivity variables from Phono3py HDF5 output files and ShengBTE output files
2. **Structural analysis** — calculate structural properties (e.g., bond distances) and extract vibrational normal modes from VASP POSCAR/CONTCAR and output files
3. **Mechanical properties** — extract and plot elastic tensors, piezoelectric tensors, and related quantities from VASP output files
4. **Structure preparation** — generate and manipulate POSCAR files for various VASP calculations
5. **MLFF utilities** — monitor training errors, evaluate MLFF accuracy against DFT references, and convert or merge VASP `ML_AB` training data files
6. **Dielectric & polar properties** — extract dielectric tensors and Born effective charge tensors from VASP DFPT output files

All scripts are standalone CLI tools written in Python using NumPy as the primary dependency. Each follows a consistent modular design with a `main()` entry point and NumPy-style docstrings.

---

## Requirements

- Python 3.8+
- NumPy
- h5py
- PyYAML
- matplotlib
- ASE — Atomic Simulation Environment
- hiPhive
- Phonopy/Phono3py

---

## Script Reference

### 1. Thermal Transport Analysis

Scripts in this category post-process force constants and read HDF5 output files from [Phono3py](https://phonopy.github.io/phono3py/) or output files from [ShengBTE](https://www.shengbte.org/) to analyze lattice thermal conductivity data.

---

#### `enforceIFC.py`

Enforces rotational sum rules (Huang and Born-Huang) on second-order interatomic force constants (IFC2) using [hiPhive](https://hiphive.materialsmodeling.org/), and writes the corrected IFC2 in Phonopy-compatible format. Reads `POSCAR` (primitive cell) and `SPOSCAR` (supercell) from the working directory. If no input IFC file is found, the script auto-generates one by calling Phonopy on any `vasprun.xml-*` displacement files present in the working directory.

```
Usage: enforceIFC.py <input FORCE_CONSTANTS> <output FORCE_CONSTANTS>
```

File format is auto-detected from the extension: `.hdf5` → HDF5; any other extension → Phonopy text format. Both input and output independently follow this rule. The cutoff radius for the hiPhive cluster space is set to the maximum cutoff supported by the supercell geometry minus a small margin (0.00001 Å).

**Defaults (when arguments are omitted):**
- `input FORCE_CONSTANTS` — `FORCE_CONSTANTS` (Phonopy text format)
- `output FORCE_CONSTANTS` — `<input_basename>_rot` (same format as input; e.g., `FORCE_CONSTANTS_rot` or `FORCE_CONSTANTS_rot.hdf5`)

**Auto-generation of IFC file (when `input FORCE_CONSTANTS` is absent):**
1. If `phonopy_params.yaml` is not present, runs `phonopy --fz vasprun.xml-sposcar <vasprun.xml-*> --sp` to generate it
2. Runs `phonopy-load phonopy_params.yaml --writefc --full-fc` to write `FORCE_CONSTANTS`

**Output:** One IFC2 file with rotational sum rules enforced, ready for use with Phonopy or Phono3py.

---

#### `calRMS.py`

Computes the RMS of each 3×3 IFC block from a Phonopy `FORCE_CONSTANTS` file and pairs it with the corresponding minimum-image interatomic distance from a POSCAR. Useful for visualizing how IFC strength decays with distance for each element pair.

```
Usage: calRMS.py <POSCAR> <FORCE_CONSTANTS>
```

All distances are computed under periodic boundary conditions using the minimum-image convention. Results are written as one file per unique element pair, sorted by ascending distance.

**Output:** `RMS_A-B.dat` per element pair — columns: distance (Å), IFC RMS (eV/Å<sup>2</sup>).

---

#### `compareIFCs.py`

Compares interatomic force constants (IFCs) between DFT and MLFF calculations by reading Phono3py HDF5 files and writing the residual (MLFF − DFT) to `.dat` files. Auto-detects whether the file contains 2nd-order (`force_constants`) or 3rd-order (`fc3`) IFCs.

```
Usage: compareIFCs.py <DFT's force constants HDF5 input> <MLFF's force constants HDF5 input>
```

**Output files:**
- `2ndIFCs.dat` — 2nd-order IFC comparison in eV/Å<sup>2</sup>
- `3rdIFCs.dat` — 3rd-order IFC comparison in eV/Å<sup>3</sup>

---

#### `reorderPhonopy.py`

Reconnects phonon branches across path segment boundaries in a Phonopy `band.yaml` file. Phonopy's built-in band connection operates only within each segment; this script extends it across segment boundaries to ensure globally consistent branch labeling throughout the full band path.

```
Usage: reorderPhonopy.py <input band.yaml> <output band.yaml>
```

The input file must contain eigenvectors and segment information. The recommended way to generate it is:

```
phonopy-load --band "<band path>" --band-connection --eigenvectors
```

The reordered data is written to a new `band.yaml` in the same format as the Phonopy output, ready for plotting with `phonopy-bandplot`.

---

#### `getQPATH.py`

Reads the high-symmetry q-point path positions from a `band.dat` file produced by `phonopy-bandplot --gnuplot` and writes `QLINES.dat` — a boundary-line file in the same format as `KLINES.dat` from VASPKIT, suitable for overlaying q-path tick marks and the frequency window on a phonon band structure plot in xmgrace or gnuplot.

```
Usage: getQPATH.py <input band.dat>
```

Q-point path distances (1/Å) are read from the second line of the input file. The frequency range is determined automatically as floor(f_min) to ceil(f_max) from all frequency values in the file. For each interior high-symmetry q-point, three coordinate pairs are written to trace a vertical tick from `fmin` to `fmax` and back. The outer box boundaries and the zero-frequency axis are appended at the end.

**Output:** `QLINES.dat` — columns: q-path distance (1/Å), frequency boundary (THz).

---

#### `convergePhono3py.py`

Checks the convergence of lattice thermal conductivity (κ) as a function of q-mesh density by reading multiple `kappa-mXXX.hdf5` files from the current directory.

```
Usage: convergePhono3py.py
```

Automatically scans for all `kappa-m*.hdf5` files, sorts them by mesh number, and writes convergence data. Supports all Phono3py calculation modes: `--br`, `--lbte`, `--wigner`, and their combinations (`kappa`, `kappa_RTA`, `kappa_C`, `kappa_P_RTA`, `kappa_TOT_RTA`, `kappa_P_exact`, `kappa_TOT_exact`).

**2D renormalization:** After loading the HDF5 files, the script interactively prompts for dimensionality (1 = 3D, 2 = 2D). For 2D materials, the vacuum direction is assumed to be c. A dimensionless renormalization factor derived from the c-axis length is applied to all κ values, correcting Phono3py's bulk-convention κ to the 2D-referenced value. Units remain W/(m·K) throughout.

---

#### `analyzePhono3py.py`

Extracts mode-resolved thermal transport properties from a single Phono3py `kappa-mXXX.hdf5` file and writes output files suitable for plotting in xmgrace or matplotlib. Supports all Phono3py calculation modes (`--br`, `--lbte`, `--wigner`). If a Grüneisen HDF5 file is provided, Grüneisen parameters and group velocities are also extracted.

```
Usage: analyzePhono3py.py <kappa HDF5 file> <gruneisen HDF5 file (optional)>
```

Output filenames follow the pattern `<tag>-mXXXXXX.dat`, where the mesh token is preserved from the input filename (e.g., `kappa-m111111.hdf5` → `KappaVsT-m111111.dat`). All κ tensor components are written in Voigt notation (xx, yy, zz, yz, xz, xy) in W/(m·K).

**2D renormalization:** After loading the HDF5 file, the script interactively prompts for dimensionality (1 = 3D, 2 = 2D). For 2D materials, the vacuum direction is assumed to be c. A dimensionless renormalization factor derived from the c-axis length is applied to all κ arrays before any output is written, correcting Phono3py's bulk-convention κ to the 2D-referenced value. Units remain W/(m·K) throughout. The renormalization applies to all output file groups below.

**Temperature-dependent files** (one value per temperature row, written to the working directory):
- `KappaVsT.dat` / `Kappa_bandVsT.dat` — total κ tensor and band decomposition (3 acoustic + 1 summed optical) vs. temperature
- `ContributeKappaVsT.dat` — per-mode percentage contribution to total κ vs. temperature
- `CvVsT.dat` — total heat capacity Cv (eV/K) vs. temperature
- `Tau_CRTAVsT.dat` / `Tau_AvgVsT.dat` — CRTA and average phonon lifetime τ (ps) vs. temperature
- `Kappa_RTAVsT.dat` / `Kappa_RTA_bandVsT.dat` — RTA κ tensor and band decomposition vs. temperature *(--lbte only)*
- `Kappa_C*VsT.dat` — wave-like (coherence) Wigner κ tensor and band decomposition vs. temperature *(--wigner only)*
- `Kappa_P_RTA*VsT.dat` / `Kappa_TOT_RTA*VsT.dat` — particle-like and total Wigner κ (RTA) vs. temperature *(--wigner --br only)*
- `Kappa_P_exact*VsT.dat` / `Kappa_TOT_exact*VsT.dat` — particle-like and total Wigner κ (exact) vs. temperature *(--wigner --lbte only)*

**Temperature-independent files** (written to the working directory):
- `GvVsFrequency.dat` / `Gv_amplitudeVsFrequency.dat` — group velocity vector (vx, vy, vz) and amplitude |v| vs. frequency (THz)
- `GruneisenVsFrequency.dat` — Grüneisen parameter vs. frequency (THz) *(if Grüneisen HDF5 provided)*
- `Gamma_isotopeVsFrequency.dat` — isotope scattering rate vs. frequency *(if available)*

**Per-temperature spectral files** (one file per temperature, written to subdirectories `T<value>K/`):
- `KappaVsFrequency.dat` / `KappaVsMfp.dat` — mode κ vs. phonon frequency (THz) and vs. mean free path (Å)
- `cumulative_KappaVsFrequency.dat` / `cumulative_KappaVsMfp.dat` — cumulative κ sorted by ascending frequency and MFP
- `derivative_KappaVsFrequency.dat` / `derivative_KappaVsMfp.dat` — spectral κ density d(κ)/d(frequency) and d(κ)/d(MFP)

---

#### `poscar2control.py`

Converts a VASP POSCAR file into a CONTROL input file for the [ShengBTE](https://www.shengbte.org/) lattice thermal conductivity code (Fortran BTE solver).

```
Usage: poscar2control.py <POSCAR>
```

Interactively prompts for supercell matrix and phonon process order (3-phonon or 4-phonon, with CPU/GPU branching). Sets `lfactor=0.1` (Å → nm) as required by ShengBTE. **Output:** `CONTROL.initial`

---

#### `analyzeShengBTE.py`

Extracts thermal transport properties from [ShengBTE](https://www.shengbte.org/) output files and writes output files suitable for plotting in xmgrace or matplotlib. If `4ph` is specified, all [FourPhonon](https://github.com/FourPhonon/FourPhonon) four-phonon scattering quantities are also extracted. Run from the ShengBTE output directory.

```
Usage: analyzeShengBTE.py <3ph/4ph>
```

Temperature subdirectories (`T<value>K/`) are detected automatically from the working directory. All κ tensor components are written as the full 3×3 tensor (xx, xy, xz, yx, yy, yz, zx, zy, zz) in W/(m·K). All scattering rate and lifetime files are written per phonon branch. Phonon lifetimes are computed as τ = 1/(2 × 2π × Γ) (ps); modes with Γ ≤ 0 are assigned τ = 0.

**Temperature-dependent files** (one value per temperature row, written to the working directory):
- `Kappa_*VsT.dat` — total κ tensor vs. temperature, RTA and iterative (CONV) solutions
- `Kappa_bandVsT.dat` — κ tensor decomposed into 3 acoustic branches + 1 summed optical branch vs. temperature
- `HeatCapacityVsT.dat` — total heat capacity Cv (J/(m<sup>3</sup>·K)) vs. temperature

**Temperature-independent files** (written to the working directory):
- `GroupVelocityVsFrequency.dat` / `GroupVelocityAmplitudeVsFrequency.dat` — group velocity vector (vx, vy, vz) and amplitude |v| in km/s vs. frequency (THz)
- `GruneisenVsFrequency.dat` — Grüneisen parameter vs. frequency (THz)
- `ScatteringRate_IsotopicVsFrequency.dat` / `Lifetime_IsotopicVsFrequency.dat` — isotope Γ (ps<sup>-1</sup>) and τ (ps) vs. frequency
- `P3*VsFrequency.dat` — total, absorption (+), and emission (−) 3-phonon phase space vs. frequency; each header records the corresponding scalar total
- `P4*VsFrequency.dat` — same set for 4-phonon phase space (total, recombination ++, redistribution +-, splitting −−) *[FourPhonon only]*

**Per-temperature files** (written into each `T<value>K/` subdirectory):
- `CumulativeKappaVsMFP.dat` / `CumulativeKappaVsFrequency.dat` — cumulative κ tensor vs. mean free path (Å) and vs. frequency (THz)
- `ScatteringRate_3ph*.dat` / `Lifetime_3ph*.dat` — 3ph scattering rate Γ and lifetime τ vs. frequency; process variants: total, `_Adsorption`(+), `_Emission`(-)
- `WeightedPhaseSpace_3ph*.dat` — weighted 3-phonon phase space vs. frequency; process variants: total, `_Adsorption`(+), `_Emission`(-)
- `ScatteringRateVsFrequency.dat` / `LifetimeVsFrequency.dat` — total combined (3ph + isotope) Γ and τ vs. frequency
- `ScatteringRateFinalVsFrequency.dat` / `LifetimeFinalVsFrequency.dat` — final iterative Γ and τ vs. frequency
- `ScatteringRate_4ph*.dat` / `Lifetime_4ph*.dat` — 4ph Γ and τ vs. frequency; process variants: total, `_Recombination`(++), `_Redistribution`(+-), `_Splitting`(--) *[FourPhonon only]*
- `WeightedPhaseSpace_4ph*.dat` — weighted 4-phonon phase space vs. frequency; process variants: total, `_Recombination`(++), `_Redistribution`(+-), `_Splitting`(--) *[FourPhonon only]*

---

### 2. Structural Analysis

Scripts that read VASP POSCAR/CONTCAR structure files and compute structural properties.

---

#### `calDistance.py`
**Inspired by:** [Jiraroj T-Thienprasert](https://scholar.google.com/citations?user=_U_cXy0AAAAJ&hl=en)

Computes interatomic distances from a VASP POSCAR/CONTCAR file under periodic boundary conditions using the minimum-image convention. Four calculation modes are available interactively.

```
Usage: calDistance.py <POSCAR>
```

**Mode 1 — one atom to all:** Distances from a selected atom to every other atom in the cell, written in POSCAR order and sorted by ascending distance.
- **Output:** `distance-unsorted.dat`, `distance-sorted.dat`

**Mode 2 — atom pairs:** Distances between user-specified pairs of atoms.
- **Output:** `distance-atom-atom.dat`

**Mode 3 — atom to molecule:** Distance from a selected atom to the geometric centroid of a user-defined group of atoms.
- **Output:** `distance-atom-molecule.dat`

**Mode 4 — z-axis separation:** Separation along the z-axis between the highest atom in a substrate group and the lowest atom in an adsorbent group. Useful for measuring adsorption height or slab thickness. Output printed to stdout only.

All modes support free-format atom selection by index, range (e.g., `1-4`), element symbol, or `all`.

---

#### `vaspVibration.py`

Extracts vibrational normal modes from a VASP `OUTCAR` or Phonopy `YAML` file and writes each mode as an XSF file for visualization in VESTA or XCrySDen.

```
Usage: vaspVibration.py <structure file> <OUTCAR or phonopy YAML> [scaling factor]
```

- VASP OUTCAR: modes written in descending frequency order (VASP convention)
- Phonopy YAML: modes written in ascending frequency order (Phonopy convention)

**Output:** One `.xsf` file per normal mode.

---

### 3. Mechanical Properties

Scripts for extracting, computing, and visualizing elastic and piezoelectric properties from VASP output files.

---

#### `vaspMechanics.py`

Reads a VASP `POSCAR` and `OUTCAR` to compute mechanical properties for either 2D or 3D materials.

```
Usage: vaspMechanics.py <POSCAR> <OUTCAR>
```

**2D mode** (N/m units):
- Detects lattice type: hexagonal, square, rectangular, or oblique
- Computes angle-dependent Young's modulus E(θ), Poisson's ratio ν(θ), and shear modulus G(θ) using compliance tensor rotation
- **Output:** `Elastic.dat`, `Young.dat`, `Poisson.dat`, `Shear.dat`

**3D mode** (GPa units):
- Identifies crystal system from spacegroup number via ASE
- Computes Voigt, Reuss, and Hill (VRH) averages for bulk and shear moduli
- Derives Young's modulus, Poisson's ratio, P-wave modulus, Lamé parameter, Pugh's ratio
- Computes sound velocities (transverse, longitudinal, mean) and Debye temperature
- Computes anisotropy indices: universal (A<sub>U</sub>), bulk (A<sub>B</sub>), shear (A<sub>G</sub>), and planar (A<sub>1</sub>, A<sub>2</sub>, A<sub>3</sub>)
- **Output:** `Elastic.dat`, `Mechanics.dat`, `Anisotropy.dat`

---

#### `ElasticTensor2D.py`

Calculates the 2D elastic tensor from VASP DFT calculations using the strain-energy method, with two operating modes.

```
Usage: ElasticTensor2D.py pre  <structure file>   # Generate strained POSCARs
       ElasticTensor2D.py post                    # Fit energies and extract constants
```

**`pre` mode:** Applies a set of strain tensors to the input structure and writes strained POSCAR files to individual directories. Detects crystal system (oblique vs. non-oblique) and applies the appropriate strain set.

**`post` mode:** Reads total energies from each strain directory's `OUTCAR`, fits energy vs. strain to a quadratic, extracts elastic constants (C11, C22, C12, C66, and C16/C26 for oblique), checks mechanical stability via eigenvalue positivity, and computes angle-dependent mechanical properties.

**Output:** `Elastic.dat`, `Young.dat`, `Poisson.dat`, `Shear.dat`

---

#### `vaspPiezoelectric.py`

Extracts the piezoelectric stress tensor (e, C/m<sup>2</sup>) and elastic stiffness tensor (C, GPa or N/m) from a VASP `OUTCAR` and computes the piezoelectric strain tensor (**d** = **e**·**S**, pm/V) via the compliance tensor **S** = **C**<sup>-1</sup>.

```
Usage: vaspPiezoelectric.py <OUTCAR>
```

Supports both 2D materials (with vacuum-layer thickness correction) and 3D bulk materials. Uses a three-level fallback chain for the elastic tensor (ionic + electronic → total → user input).

**Output:** Piezoelectric tensor files in Voigt notation.

---

#### `plotMechanics.py`

Plots polar diagrams of angle-dependent mechanical properties (Young's modulus, Poisson's ratio, Shear modulus) for up to 6 materials simultaneously for visual comparison.

```
Usage: plotMechanics.py <file1> [file2 ... file6]
```

Input files are the `.dat` output files from `vaspMechanics.py` or `ElasticTensor2D.py`. Auxetic materials (negative Poisson's ratio) are handled by plotting |ν| as a dashed envelope. Output is saved as a 300 dpi PNG.

---

### 4. Structure Preparation

Scripts for generating, transforming, and manipulating VASP POSCAR files for various DFT calculations.

---

#### `vaspReformat.py`

Converts a VASP POSCAR/CONTCAR to a standardized VASP5 format with Direct coordinates. Handles VASP4 (no element line), VASP5, and VASP6 (with Hash code) input formats, PAW/GGA suffix stripping, duplicate element reordering, and all scaling factor conventions.

```
Usage: vaspReformat.py <POSCAR> <output POSCAR>
```

Supports optional Selective Dynamics and writes per-atom label comments (e.g., `Mo001`, `S002`) for identification in VESTA or XCrySDen.

---

#### `vaspDefect.py`

Applies point defects to a VASP POSCAR and writes the modified structure. Atoms are reordered into contiguous element blocks before and after defect application. Four defect types are available interactively. Free-format atom selection (index, range e.g. `1-4`, element symbol, `all`) is used throughout.

```
Usage: vaspDefect.py <input POSCAR> <output POSCAR>
```

**Defect type 1 — Vacancy:** Removes selected atoms. Multiple atoms can be removed in a single operation.

**Defect type 2 — Substitution:** Replaces selected atoms with a new element symbol. Element blocks are regrouped automatically after substitution.

**Defect type 3 — Interstitial:** Inserts one new atom at a user-defined site. Two site modes: (1) mean fractional coordinate of ≥2 selected atoms with PBC unwrapping; (2) manual fractional (a, b, c) input. The new atom is inserted after the last atom of matching species, or appended if the element is new. Default Selective Dynamics flag: T T T.

**Defect type 4 — Displacement:** Moves selected atoms to a new XY position while preserving their fractional c coordinate. Atoms eligible for displacement are auto-detected as those above a threshold Z, where the threshold is the midpoint of the largest gap in sorted Cartesian Z values. The target XY is defined by: (1) centroid of selected reference atoms with PBC unwrapping (1 atom = on-top, 2+ atoms = center); or (2) manual fractional (a, b) input. A single displaced atom is moved directly to the target XY; multiple atoms receive a rigid XY shift so their centroid lands at the target.

---

#### `vaspShift.py`

Shifts atomic positions in a VASP POSCAR to a standardized reference frame. The script interactively prompts for a shifting mode based on material dimensionality.

```
Usage: vaspShift.py <POSCAR> <output POSCAR>
```

All modes first unwrap atoms across periodic boundaries to compute a geometrically correct centroid before shifting.

- **Mode 0 — 0D molecule:** centroid shifted to cell center (0.5, 0.5, 0.5)
- **Mode 1 — 1D nanowire:** extend direction shifted to origin; transverse directions centered at 0.5. User selects the extend direction (x/y/z)
- **Mode 2 — 2D sheet:** vacuum direction centered at 0.5; periodic directions shifted to origin. User selects the vacuum direction (x/y/z)
- **Mode 3 — 3D bulk:** all atoms shifted so atom[0] lands at the cell origin
- **Mode 4 — Adsorbate:** selected adsorbate group centered in XY at (0.5, 0.5); z-coordinates of all atoms left unchanged. Adsorbate selection supports free-format input (index, range, element symbol, `all`)

---

#### `vaspMirror.py`

Reflects all atomic positions in a VASP POSCAR across a chosen Cartesian plane (XY, XZ, or YZ) by negating the perpendicular coordinate component. The lattice matrix is unchanged.

```
Usage: vaspMirror.py <POSCAR> <output POSCAR>
```

---

#### `vaspRotate.py`

Rotates atoms in a VASP POSCAR/CONTCAR file about a user-specified pivot point and axis using Rodrigues' rotation formula.

```
Usage: vaspRotate.py <POSCAR> <output POSCAR>
```

Supports rotation about arbitrary axes; pivot can be set to a specific atom, the center of mass, or a custom Cartesian point.

---

#### `vaspFix.py`

Applies Selective Dynamics constraints to a VASP POSCAR, fixing atoms in specified Cartesian directions.

```
Usage: vaspFix.py <POSCAR> <output POSCAR>
```

Three atom-selection modes: by index/label, by cutoff radius (PBC-aware), or from an existing `SELECTED_FIX_ATOMS_LIST` file. Writes a `SELECTED_FIX_ATOMS_LIST` log for reference and reuse.

---

#### `vaspStrain.py`

Applies a strain tensor to a crystal structure POSCAR for DFT elastic constant calculations.

```
Usage: vaspStrain.py <POSCAR> <output POSCAR>
```

Accepts 3 values (diagonal strain) or 9 values (full 3×3 tensor). Off-diagonal inputs are symmetrized. Applies the deformation gradient **F** = **I** + **ε** to the lattice matrix, preserving fractional atomic coordinates.

---

#### `vaspSupercell.py`

Generates a supercell POSCAR from a unit cell input using an expansion matrix. Accepts 3 values (diagonal expansion) or 9 values (full 3×3 matrix).

```
Usage: vaspSupercell.py <POSCAR> <output POSCAR>
```

Uses an integer-exact (adjugate-matrix-based) grid point generation to avoid floating-point rounding errors. Supports anisotropic scale factors and Selective Dynamics.

---

#### `vaspStack.py`

Generates a set of bilayer POSCAR files from a single input monolayer POSCAR by stacking two copies along the c-axis with systematically varied interlayer shifts and orientations.

```
Usage: vaspStack.py <POSCAR>
```

The script detects the 2D Bravais lattice type (hexagonal, square, rectangular, oblique) and generates a grid of high-symmetry stacking configurations appropriate for that lattice. Each configuration applies a fractional interlayer shift to the top layer, optionally with a mirror-flip. A summary `STACK_LIST.txt` is written listing all generated POSCAR filenames and their corresponding shift vectors.

---

#### `vaspTwist.py`
**Inspired by:** [CellMatch](https://doi.org/10.1016/j.cpc.2015.08.038)

Generates moiré twisted bilayer POSCAR files by searching for commensurate supercells across twist angles. The workflow is split into two modes run sequentially.

```
Usage: vaspTwist.py match    <bottom POSCAR> [top POSCAR]   # Step 1: search & write TWIST_LIST.dat
       vaspTwist.py generate <bottom POSCAR> [top POSCAR]   # Step 2: read TWIST_LIST.dat & write POSCARs
```

**`match` mode:** Searches twist angles from 0° to 180° (step: 0.1°) for commensurate supercell pairs using the CellMatch symmetric relative distance metric. Results are written to `TWIST_LIST.dat`, sorted by (θ, strain). No POSCAR is written at this stage.

**`generate` mode:** Reads `TWIST_LIST.dat`, displays the candidate table interactively, and writes one POSCAR per selected stacking configuration. The post-search prompt accepts individual indices, `'all'`, or `'none'`.

Providing a single POSCAR uses it for both layers (homobilayer). Providing two different POSCARs enables heterobilayer mode. Key parameters: `MAX_ATOMS = 200` (hard cap on total supercell atoms); `N_MAX` derived as `ceil(sqrt(MAX_ATOMS / primitive_atoms))`; `THETA_STEP = 0.1°` (suitable for angles above ~2°; use `0.01°` for magic-angle regimes below ~2°). Lattice-type detection and high-symmetry stacking shift grids are implemented for hexagonal, square, rectangular, and oblique supercells.

---

#### `vaspAdsorb.py`
**Inspired by:** [Aroon Ananchuensook](https://scholar.google.com/citations?user=6mmg4mUAAAAJ&hl=en)

Combines a substrate and an adsorbent POSCAR into a single POSCAR for adsorption DFT calculations. Supports placing multiple adsorbent copies simultaneously. The user specifies the number of copies and the vertical separation distance (Å) before selecting a placement mode.

```
Usage: vaspAdsorb.py <substrate POSCAR> <adsorbent POSCAR> <output POSCAR>
```

**Mode 1 — on top of a specific site:** Places each adsorbent copy above a user-defined target point on the substrate. For each copy, the user selects: (a) the substrate reference height (highest atom, a selected atom, or average height); (b) the adsorbent anchor point (centroid or a specific/lowest atom); (c) the target xy position (by atom selection with free-format input, or custom fractional coordinates). The adsorbent is translated so its anchor lands at the target at the specified vertical distance.

**Mode 2 — ring around a target atom:** Places N copies evenly distributed at angular intervals of 2π/N around a chosen substrate atom using Rodrigues z-axis rotation. The user selects the target atom and the initial adsorption direction (by atom selection or custom fractional coordinates). The radial distance from the target atom is set by the distance input.

If either POSCAR has Selective Dynamics, flags are merged, and the other structure defaults to all-T. If neither has Selective Dynamics, the user is prompted after placement to optionally add constraints, with free-format atom and direction selection. A summary table of atom counts per element (substrate/adsorbent/total) is printed at the end.

---

### 5. MLFF Utilities

Scripts for working with VASP Machine Learning Force Fields (MLFF).

---

#### `mlError.py`

Extracts Bayesian Error Estimation in Forces (BEEF) and Root Mean Square Errors (RMSE) from a VASP `ML_LOGFILE` during MLFF on-the-fly training.

```
Usage: mlError.py <ML_LOGFILE>
```

**Output:** `BEEF.dat`, `ERR.dat` (formatted for xmgrace)

---

#### `mlRegression.py`

Evaluates MLFF accuracy against DFT reference data from the VASP `ML_REG` file, computing RMSE, MAE, and R-square for energies (meV/atom), forces (eV/Å), and stresses (kbar).

```
Usage: mlRegression.py <ML_REG>
```

**Output:** `Energy.dat`, `Force.dat`, `Stress.dat`, `ERROR.dat`

---

#### `mlab2extxyz.py`
**Inspired by:** [utf/pymlff](https://github.com/utf/pymlff)

Converts VASP's `ML_AB` binary training data file to extended XYZ (`.extxyz`) format for use with external MLFF training frameworks such as MACE, NequIP, and GPUMD.

```
Usage: mlab2extxyz.py <ML_AB input> <output.extxyz>
```

Each configuration block is mapped to one extxyz frame with lattice, positions, energy, forces, and stress. Stress is converted from kbar to eV/Å<sup>3</sup>.

---

#### `mergeMLAB.py`
**Inspired by:** [utf/pymlff](https://github.com/utf/pymlff)

Merges multiple VASP `ML_AB` training data files into a single unified file, unifying element lists, basis sets, and renumbering configurations.

```
Usage: mergeMLAB.py <ML_AB 1> <ML_AB 2> [ML_AB 3 ...] <output ML_AB>
```

Resolves header metadata conflicts (reference energies, atomic masses) by first-file-wins with a warning for mismatches.

---

### 6. Dielectric & Polar Properties

Scripts for extracting dielectric and Born effective charge tensors from VASP DFPT output files.

---

#### `vaspBorn.py`

Extracts the ion-clamped (electronic) dielectric tensor and Born effective charge tensors from a VASP DFPT calculation (`LEPSILON=.TRUE.`) and writes them to `INCAR.LR`. Accepts either `OUTCAR` or `vasprun.xml`; file type is auto-detected by filename with content-based fallback.

```
Usage: vaspBorn.py <OUTCAR or vasprun.xml>
```

**Output:** `INCAR.LR` — `PHON_DIELECTRIC` and `PHON_BORN_CHARGES` tags in backslash-continuation format.

---

## Design Conventions

All scripts follow the same conventions modeled on `vaspSupercell.py`:

- Single-responsibility functions with NumPy-style docstrings (`Parameters`, `Returns`, inline notes for unit conversions and formulas)
- `main()` entry point with `if __name__ == '__main__'` guard
- Interactive input loops with validation and retry on invalid input
- Handles both VASP4 (no element line), VASP5, and VASP6 (with Hash code) POSCAR formats
- Handles Selective Dynamics, anisotropic scale factors, and non-orthogonal cells
- Atom-label comments (e.g., `Mo001`, `S002`) in output POSCARs for VESTA/XCrySDen identification
- Output files formatted for xmgrace (`.dat` with column headers) unless otherwise noted

---

## License

Developed by [Thanasee Thanasarnsurapong](https://scholar.google.com/citations?user=4KHXv9gAAAAJ&hl=en) and [Claude](https://claude.ai/). For research and academic use.
