# VASP Python Utility Scripts

A collection of Python scripts for VASP output analysis and related tasks, developed for computational materials science research on the LANTA HPC cluster.

**Author:** [Thanasee Thanasarnsurapong](https://scholar.google.com/citations?user=4KHXv9gAAAAJ&hl=en)

---

## Overview

This repository is organized into eight functional categories:

1. **Thermal transport analysis** — post-process force constants, reconnect phonon branches, generate phonon band plot boundaries, extract and analyze lattice thermal conductivity variables from Phono3py HDF5 output files and ShengBTE output files
2. **Structural analysis** — calculate structural properties (e.g., bond distances) and extract vibrational normal modes from VASP POSCAR/CONTCAR and output files
3. **Mechanical properties** — extract and plot elastic tensors, piezoelectric tensors, and related quantities from VASP output files
4. **Structure preparation** — generate and manipulate POSCAR files for various VASP calculations
5. **MLFF utilities** — monitor training errors, evaluate MLFF accuracy against DFT references, and convert or merge VASP `ML_AB` training data files
6. **NEB calculations** — generate interpolated image directories and analyze completed NEB runs
7. **Charge density analysis** — analyze and transform VASP CHGCAR volumetric data (slices, averages, STM simulation, supercells, charge-density differences), and plot/export VASPKIT `.grd` grid files
8. **Calculation setup** — build VASP/follow-on calculation-control tags (initial MAGMOM, dielectric/Born effective charges) from POSCAR or DFPT output

All scripts are standalone Python tools operated via command-line arguments and interactive prompts. Each follows a consistent modular design with a `main()` entry point and NumPy-style docstrings.

---

## Requirements

- Python 3.8+
- NumPy
- h5py
- PyYAML
- matplotlib
- ASE — Atomic Simulation Environment
- SciPy
- hiPhive
- Phonopy/Phono3py

---

## Script Reference

### 1. Thermal Transport Analysis

Scripts in this category post-process force constants and read HDF5 output files from [Phono3py](https://phonopy.github.io/phono3py/) or output files from [ShengBTE](https://www.shengbte.org/) to analyze lattice thermal conductivity data.

---

#### `enforceIFC.py`

Enforces rotational sum rules (Huang and Born-Huang) on second-order interatomic force constants (IFC2) using [hiPhive](https://hiphive.materialsmodeling.org/), and writes the corrected IFC2 in Phonopy-compatible format. Reads `POSCAR` (primitive cell) and `SPOSCAR` (supercell) from the working directory. If no input IFC file is given and `FORCE_CONSTANTS` is not present, the script auto-generates one by calling Phonopy on the `vaspout-*.h5` displacement files in the working directory. An input IFC file that is given explicitly but does not exist is an error.

```
Usage: enforceIFC.py <input FORCE_CONSTANTS> <output FORCE_CONSTANTS>
```

File format is auto-detected from the extension: `.hdf5` → HDF5; any other extension → Phonopy text format. Both input and output independently follow this rule. The cutoff radius for the hiPhive cluster space is set to the maximum cutoff supported by the supercell geometry minus a small margin (10<sup>–5</sup> Å).

**Defaults (when arguments are omitted):**
- `input FORCE_CONSTANTS` — `FORCE_CONSTANTS` (Phonopy text format)
- `output FORCE_CONSTANTS` — `<input_basename>_rot` (same format as input; e.g., `FORCE_CONSTANTS_rot` or `FORCE_CONSTANTS_rot.hdf5`)

**Auto-generation of IFC file (when `input FORCE_CONSTANTS` is omitted and `FORCE_CONSTANTS` is absent; Phonopy 4.x CLI):**
1. If `phonopy_params.yaml` is not present, runs `phonopy-init --fz vaspout-sposcar.h5 $(printf "%s\n" vaspout-*.h5 | grep -v '^vaspout-sposcar.h5$') --sp` to generate it
2. Runs `phonopy phonopy_params.yaml --writefc --full-fc` to write `FORCE_CONSTANTS`

**Output:** One IFC2 file with rotational sum rules enforced, ready for use with Phonopy or Phono3py.

---

#### `calRMS.py`
**Inspired by:** [ACS Appl. Energy Mater. 2022, 5, 11, 14522–14530](https://doi.org/10.1021/acsaem.2c03141)

Computes the RMS of each 3×3 2nd IFC block from a Phonopy `FORCE_CONSTANTS` file and pairs it with the corresponding minimum-image interatomic distance from a POSCAR. Useful for visualizing how 2nd IFC strength decays with distance for each element pair.

```
Usage: calRMS.py <POSCAR> <FORCE_CONSTANTS>
```

All distances are computed under periodic boundary conditions using the minimum-image convention. Results are written as one file per unique element pair, sorted by ascending distance.

**Output:** `RMS_A-B.dat` per element pair — columns: distance (Å), 2nd IFC RMS (eV/Å<sup>2</sup>).

---

#### `compareIFCs.py`

Compares interatomic force constants (IFCs) between DFT and MLFF calculations by reading Phono3py HDF5 files and writing the residual (MLFF − DFT) to `.dat` files. Auto-detects whether the file contains 2nd (`force_constants`) or 3rd (`fc3`) IFCs.

```
Usage: compareIFCs.py <DFT's force constants HDF5 input> <MLFF's force constants HDF5 input>
```

Both files must contain the same IFC order with identical array shapes. Only one file is written per run; if a file contains both `force_constants` and `fc3`, the 2nd IFCs take precedence. The RMSE and maximum |residual| are printed to stdout.

**Output files** (columns: DFT value, MLFF − DFT residual):
- `2ndIFCs.dat` — 2nd IFC comparison in eV/Å<sup>2</sup>
- `3rdIFCs.dat` — 3rd IFC comparison in eV/Å<sup>3</sup>

---

#### `reorderPhonopy.py`
**Inspired by:** [raymond-yiqunwang/phonon_bandplot](https://github.com/raymond-yiqunwang/phonon_bandplot)

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

Reads the high-symmetry q-point path positions from a `band.dat` file produced by `phonopy-bandplot --gnuplot` and writes `QLINES.dat` — a boundary-line file in the same format as `KLINES.dat` from [VASPKIT](https://doi.org/10.1016/j.cpc.2021.108033), suitable for overlaying q-path tick marks and the frequency window on a phonon band structure plot in xmgrace or gnuplot.

```
Usage: getQPATH.py <input band.dat>
```

Q-point path distances (Å<sup>–1</sup>) are read from the second line of the input file. The frequency window is determined automatically from all frequency values in the file as fmin = min(5·floor(f_min/5), −5) and fmax = 5·ceil(f_max/5) + 5, i.e. rounded outward to multiples of 5, with fmin at most −5 and 5 THz of headroom above the highest frequency.

For each interior high-symmetry q-point, three coordinate pairs are written to trace a vertical tick from `fmin` to `fmax` and back. The outer box boundaries and the zero-frequency axis are appended at the end.

**Output:** `QLINES.dat` — columns: q-path distance (Å<sup>–1</sup>), frequency boundary (THz).

---

#### `convergePhono3py.py`

Checks the convergence of lattice thermal conductivity (κ<sub>L</sub>) as a function of q-mesh density by reading multiple `kappa-mXXX.hdf5` files from the current directory.

```
Usage: convergePhono3py.py
```

Automatically scans for all `kappa-m*.hdf5` files, sorts them by mesh number, and writes convergence data. Supports the Phono3py `--br` and `--lbte` calculation modes. The target temperature is prompted once and reused for every file with the same temperature grid.

**2D renormalization:** Before the HDF5 files are read, the script interactively prompts for dimensionality (1 = 3D, 2 = 2D). For 2D materials, all κL values are multiplied by c_proj / t_eff, where c_proj is the projection of c onto the ab-plane normal and t_eff is the atomic span along that normal plus the Alvarez vdW radii of the outermost atoms. Reads `POSCAR` from the working directory, or asks for a path if it is missing (falls back to 3D if the path does not exist). Units remain W/(m·K) throughout.

**Output files** (columns: Q_x, Q_y, Q_z, xx, yy, zz, yz, xz, xy):
- `KappaVsMesh.dat` — κ<sub>L</sub> vs. q-mesh
- `Kappa_RTAVsMesh.dat` — RTA κ<sub>L</sub> vs. q-mesh (--lbte only)
-  `Kappa_intraVsMesh.dat` / `Kappa_interVsMesh.dat` — intra-band and inter-band κ<sub>L</sub> vs. q-mesh (--br --tt njc23/ibdb19/smm19 only)
- `Kappa_intra_exactVsMesh.dat` / `Kappa_intra_RTAVsMesh.dat` / `Kappa_interVsMesh.dat` — intra-band (exact and RTA) and inter-band κ<sub>L</sub> vs. q-mesh (--lbte --tt njc23/ibdb19/smm19 only); `KappaVsMesh.dat` holds the total (intra exact + inter) and `Kappa_RTAVsMesh.dat` is not written

---

#### `analyzePhono3py.py`

Extracts mode-resolved thermal transport properties from a single Phono3py (4.x) `kappa-mXXX.hdf5` file and writes output files suitable for plotting in xmgrace or matplotlib. Supports `--br` and `--lbte` kappa files, including `--tt smm19/njc23/ibdb19`. Group velocities are read from the kappa file.

```
Usage: analyzePhono3py.py <kappa HDF5 file> <gruneisen HDF5 file (optional)>
```

Output filenames follow the pattern `<tag>-mXXXXXX.dat`, where the mesh token is preserved from the input filename (e.g., `kappa-m111111.hdf5` → `KappaVsT-m111111.dat`). All κ<sub>L</sub> tensor components are written in Voigt notation (xx, yy, zz, yz, xz, xy) in W/(m·K).

**Grüneisen parameter:** If no Grüneisen file is given (or it does not exist), the script asks whether to run `phono3py --mesh <mesh> --gruneisen` in the current directory, using the mesh of the kappa file. The resulting `gruneisen.hdf5` is renamed to `gruneisen-m<mesh>.hdf5`. A Grüneisen file on a different mesh from the kappa file is skipped with a warning.

**2D renormalization:** After loading the HDF5 file, the script interactively prompts for dimensionality (1 = 3D, 2 = 2D). For 2D materials, `POSCAR` is read from the working directory (a path is asked for if it is missing) and all κ<sub>L</sub> arrays are multiplied by c_proj / t_eff before any output is written, where c_proj is the projection of c onto the ab-plane normal and t_eff is the atomic span along that normal plus the Alvarez vdW radii of the two outermost atoms. Units remain W/(m·K) throughout. The factor is recorded in `Renorm2D_info.dat`.

**Temperature-dependent files** (one value per temperature row, written to the working directory):
- `KappaVsT.dat` / `Kappa_bandVsT.dat` — total κL tensor and band decomposition (3 acoustic + 1 summed optical) vs. temperature
- `ContributeKappaVsT.dat` — per-mode percentage contribution to total κ<sub>L</sub> vs. temperature
- `CvVsT.dat` — heat capacity C<sub>v</sub> per unit cell (eV/K) vs. temperature
- `Tau_CRTAVsT.dat` / `Tau_AvgVsT.dat` — CRTA lifetime tensor and mode-averaged phonon lifetime τ (ps) vs. temperature
- `Kappa_RTAVsT.dat` / `Kappa_RTA_bandVsT.dat` — RTA κ<sub>L</sub> tensor and band decomposition vs. temperature (--lbte only)
- `Kappa_intraVsT.dat` / `Kappa_interVsT.dat` — `kappa_intra` (particle-like) and `kappa_inter` (coherence) κ<sub>L</sub> vs. temperature (--br --tt)
- `Kappa_intra_RTAVsT.dat` / `Kappa_intra_exactVsT.dat` / `Kappa_interVsT.dat` — `kappa_intra_RTA`, `kappa_intra_exact` and `kappa_inter` κ<sub>L</sub> vs. temperature (--lbte --tt)

**Temperature-independent files** (written to the working directory):
- `GvVsFrequency.dat` / `Gv_amplitudeVsFrequency.dat` — group velocity vector (v<sub>x</sub>, v<sub>y</sub>, v<sub>z</sub>) and amplitude |v| in km/s vs. frequency (THz)
- `GruneisenVsFrequency.dat` — Grüneisen parameter vs. frequency (THz) (if a Grüneisen file is given or calculated)
- `Gamma_isotopeVsFrequency.dat` / `Tau_isotopeVsFrequency.dat` — isotope scattering rate and lifetime vs. frequency (--isotope)
- `PqjVsFrequency.dat` — averaged phonon-phonon interaction strength (eV2) vs. frequency (--full-pp)
- `Renorm2D_info.dat` — 2D renormalization factor and thickness (2D only)

**Per-temperature spectral files** (one file per temperature, written to subdirectories `T<value>K/`):
- `GammaVsFrequency.dat` / `TauVsFrequency.dat` — scattering rate Γ (ps−1) and lifetime τ (ps) vs. frequency
- `KappaVsFrequency.dat` / `KappaVsMfp.dat` — mode κ<sub>L</sub> vs. phonon frequency (THz) and vs. mean free path (Å)
- `cumulative_KappaVsFrequency.dat` / `cumulative_KappaVsMfp.dat` — cumulative κ<sub>L</sub> sorted by ascending frequency and MFP
- `derivative_KappaVsFrequency.dat` / `derivative_KappaVsMfp.dat` — spectral κ<sub>L</sub> density d(κ<sub>L</sub>)/d(frequency) and d(κ<sub>L</sub>)/d(MFP)
- `Kappa_RTA*.dat` — the same six spectral files for the RTA solution (--lbte only)
- `Gamma_NVsFrequency.dat` / `Tau_NVsFrequency.dat` and `Gamma_UVsFrequency.dat` / `Tau_UVsFrequency.dat` — Normal and Umklapp Γ and τ vs. frequency (--nu)

**Special cases:**
- `--lbte --reducible-colmat` files store frequencies on all grid points but other quantities on irreducible q-points only, so only the κ<sub>L</sub> tensor vs. temperature files are written (with a warning).
- If `mode_kappa` does not add up to `kappa` (e.g. phono3py-wte `--br` files), a warning is printed and `ContributeKappaVsT.dat` and `Tau_CRTAVsT.dat` use the sum of `mode_kappa`; `KappaVsT.dat` still writes `kappa`.
- `--tt` files contain no `mode_kappa` or `gv_by_gv`. In `--br --tt` files `kappa` = `kappa_intra` + `kappa_inter`; in `--lbte --tt` files `kappa` = `kappa_intra_exact` only. Each dataset is written as stored, with no derived total.
- Outputs that need a missing dataset (`mode_kappa`, `mode_kappa_RTA`, `gv_by_gv`) are skipped with a warning.

---

#### `poscar2control.py`

Converts a VASP POSCAR file into a CONTROL input file for the [ShengBTE](https://www.shengbte.org/) lattice thermal conductivity code (Fortran BTE solver).

    Usage: poscar2control.py <POSCAR>

**Interactive prompts:**
1. 2D material (Y/N). If yes: vacuum direction (a/b/c). The sheet is recentred at 0.5 along the vacuum direction (periodic directions shifted to origin), then the vacuum lattice vector is shrunk so its projection on the layer normal equals t_eff (atomic span along the normal plus the Alvarez vdW radii of the two outermost atoms), or a user-entered thickness (must be ≥ the atomic span).
2. q-point spacing (2π/Å) → `ngrid`: n_i = round(|b_i| / (2π × spacing)), minimum 1; the vacuum direction is set to 1.
3. Supercell matrix (3 positive integers) → `scell`; must match the supercell used for the force constants.
4. Phonon process order: 3 → `convergence=.true.`, `nthreads=-1`; 4 → CPU (`convergence` and `four_phonon_iteration` = `.true.`) or GPU (both `.false.`), with `four_phonon=.true.`.

Repeated element blocks in the POSCAR (e.g. S Mo S) are merged into unique elements; atom order is kept so `types` and `positions` match the force constant files.

**Hard-coded values:** `lfactor=0.1` (Å → nm), `T=300.`, `scalebroad=1.0`, `nonanalytic=.false.`, `isotopes=.true.`, `autoisotopes=.true.`, `nanowire=.false.`, `onlyharmonic=.false.` — edit `CONTROL.initial` to change them.

**Output:** `CONTROL.initial`

---

#### `analyzeShengBTE.py`

Extracts thermal transport properties from [ShengBTE](https://www.shengbte.org/) output files and writes output files suitable for plotting in xmgrace or matplotlib. If `4ph` is specified, all [FourPhonon](https://github.com/FourPhonon/FourPhonon) four-phonon scattering quantities are also extracted. Run from the ShengBTE output directory.

```
Usage: analyzeShengBTE.py <3ph/4ph>
```

Temperature subdirectories (`T<value>K/`) are detected automatically from the working directory. Only `BTE.omega` and the `T<value>K/` directories are required; any other missing input (e.g. `BTE.KappaTensorVsT_CONV` when `convergence = .false.`) is skipped with a warning. All κ<sub>L</sub> tensor components are written as the full 3×3 tensor (xx, xy, xz, yx, yy, yz, zx, zy, zz) in W/(m·K). All scattering rate and lifetime files are written per phonon branch. ShengBTE's scattering rate Γ is 1/τ (ps−1), so phonon lifetimes are computed as τ = 1/Γ (ps); modes with Γ ≤ 0 are assigned τ = 0.

**Temperature-dependent files** (one value per temperature row, written to the working directory):
- `Kappa_*VsT.dat` — total κ<sub>L</sub> tensor vs. temperature, RTA and iterative (CONV) solutions
- `Kappa_bandVsT.dat` — κ<sub>L</sub> tensor decomposed into 3 acoustic branches + 1 summed optical branch vs. temperature
- `HeatCapacityVsT.dat` — total heat capacity C<sub>v</sub> (J/(m<sup>3</sup>·K)) vs. temperature

**Temperature-independent files** (written to the working directory):
- `GroupVelocityVsFrequency.dat` / `GroupVelocityAmplitudeVsFrequency.dat` — group velocity vector (v<sub>x</sub>, v<sub>y</sub>, v<sub>z</sub>) and amplitude |v| in km/s vs. frequency (THz)
- `GruneisenVsFrequency.dat` — Grüneisen parameter vs. frequency (THz)
- `ScatteringRate_IsotopicVsFrequency.dat` / `Lifetime_IsotopicVsFrequency.dat` — isotope Γ (ps<sup>–1</sup>) and τ (ps) vs. frequency
- `P3*VsFrequency.dat` — total, absorption (+), and emission (−) 3-phonon phase space vs. frequency; each header records the corresponding scalar total
- `P4*VsFrequency.dat` — same set for 4-phonon phase space (total, recombination ++, redistribution +-, splitting −−) [FourPhonon only]

**Per-temperature files** (written into each `T<value>K/` subdirectory):
- `CumulativeKappaVsMFP.dat` / `CumulativeKappaVsFrequency.dat` — cumulative κ<sub>L</sub> tensor vs. mean free path (nm) and vs. frequency (THz)
- `ScatteringRate_3ph*.dat` / `Lifetime_3ph*.dat` — 3ph scattering rate Γ and lifetime τ vs. frequency; process variants: total, `_Absorption`(+), `_Emission`(-)
- `WeightedPhaseSpace_3ph*.dat` — weighted 3-phonon phase space vs. frequency; process variants: total, `_Absorption`(+), `_Emission`(-)
- `ScatteringRateVsFrequency.dat` / `LifetimeVsFrequency.dat` — total combined (3ph + isotope) Γ and τ vs. frequency
- `ScatteringRateFinalVsFrequency.dat` / `LifetimeFinalVsFrequency.dat` — final iterative Γ and τ vs. frequency
- `ScatteringRate_4ph*.dat` / `Lifetime_4ph*.dat` — 4ph Γ and τ vs. frequency; process variants: total, `_Recombination`(++), `_Redistribution`(+-), `_Splitting`(--) [FourPhonon only]
- `WeightedPhaseSpace_4ph*.dat` — weighted 4-phonon phase space vs. frequency; process variants: total, `_Recombination`(++), `_Redistribution`(+-), `_Splitting`(--) [FourPhonon only]

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

**Mode 3 — atom to molecule:** Distance from a selected atom to the geometric centroid of a user-defined group of atoms. The group is unwrapped across periodic boundaries before averaging, so molecules split across the cell edge are handled correctly.

**Mode 4 — z-axis separation:** Separation along the z-axis between the highest atom in a substrate group and the lowest atom in an adsorbent group. Useful for measuring adsorption height or slab thickness. Output printed to stdout only.

All modes support free-format atom selection by index, range (e.g., `1-4`), element symbol, or `all`.Modes 1 and 2, and the reference atom in mode 3, take a single atom index. The molecule group in mode 3 and the substrate/adsorbent groups in mode 4 use free-format selection by index, range (e.g., `1-4`), or element symbol; `all` is accepted in mode 3 but not in mode 4.

---

#### `vaspVibration.py`
**Inspired by:** [QijingZheng/VaspVib2XSF](https://github.com/QijingZheng/VaspVib2XSF)

Extracts vibrational normal modes from a VASP `OUTCAR` or Phonopy `YAML` file and writes each mode as an XSF file for visualization in VESTA or XCrySDen.

```
Usage: vaspVibration.py <structure file> <OUTCAR or phonopy YAML> [scaling factor]
       vaspVibration.py <vaspout.h5> [scaling factor]
```

- VASP OUTCAR (Γ block): modes written in descending frequency order (VASP convention); NWRITE = 3 output is supported
- VASP phonon dispersion (LPHON_DISPERSION): read from OUTCAR (PHON_NWRITE = -3; the structure file is the supercell POSCAR of the run) or from vaspout.h5 (any PHON_NWRITE; the primitive cell is read from the file). If the OUTCAR has both the Γ block and the dispersion, the user is prompted to choose. Modes written in ascending frequency order
- Phonopy YAML: modes written in ascending frequency order (Phonopy convention); the file must contain eigenvectors (`--eigenvectors`)

**q-point selection (YAML and VASP dispersion):** Candidates are the high-symmetry q-points of the path (segment ends for band.yaml; path ends, segment ends and kinks for VASP, labeled from QPOINTS if present), or every q-point for mesh.yaml/qpoints.yaml. Duplicates are removed, and if more than one remains, the script prompts for one. Phonopy eigenvectors get the Bloch phase exp(2πi q·r); VASP eigenvectors already refer to each ion's stored cell, so ions outside the cell are wrapped with the matching phase. Eigenvectors are divided by √m, and a warning is printed if they are still complex (e.g. K or general q-points).

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
- **Output:** `Elastic.dat`, `Young.dat`, `Poisson.dat`, `Shear.dat`, `Extreme.dat` (max/min E, ν, G and their angles)

**3D mode** (GPa units):
- Identifies crystal system from spacegroup number via ASE
- Computes Voigt, Reuss, and Hill (VRH) averages for bulk and shear moduli
- Derives Young's modulus, Poisson's ratio, P-wave modulus, Lamé parameter, Pugh's ratio
- Computes sound velocities (transverse, longitudinal, mean) and Debye temperature
- Computes anisotropy indices: universal (A<sub>U</sub>), bulk (A<sub>B</sub>), shear (A<sub>G</sub>), and planar (A<sub>1</sub>, A<sub>2</sub>, A<sub>3</sub>)
- **Output:** `Elastic.dat`, `Mechanics.dat` (includes v<sub>t</sub>, v<sub>l</sub>, v<sub>m</sub> and θ<sub>D</sub>), `Anisotropy.dat`

---

#### `ElasticTensor2D.py`

Calculates the 2D elastic tensor from VASP DFT calculations by the energy-strain method, the stress-strain method, or both, with two operating modes.

```
Usage: ElasticTensor2D.py pre  <structure file>   # Generate strained POSCARs
       ElasticTensor2D.py post                    # Fit energies/stresses and extract constants
```

**Methods** (chosen interactively in `pre` and saved to a `METHOD` file, which `post` re-reads; `post` prompts again if `METHOD` is missing or invalid):
1. `energy` — energy-strain: energy per area vs. strain is fitted to a quadratic for each strain set
2. `stress` — stress-strain: 2D stress vs. strain is fitted linearly; each strain set gives one column of the elastic tensor (requires a converged ENCUT, ~1.3×ENMAX)
3. `both` — runs both methods and compares them

**`pre` mode:** Detects the 2D lattice type (square, rectangular, hexagonal, oblique) and writes the reference structure to `unstrain/POSCAR` plus one POSCAR per strain set and strain value to `<strain set>/strain<±δ>/POSCAR` — 9 strains from −2% to +2% in 0.5% steps (`strain-0.020` ... `strain+0.020`). Strain is applied as F = I + ε to the lattice; fractional coordinates are kept.
- Energy method: `C<sub>11</sub>`, `C<sub>22</sub>`, `C<sub>11</sub>×C<sub>22</sub>-2×C<sub>12</sub>`, `C<sub>66</sub>`; oblique cells add `C<sub>11</sub>×C<sub>66</sub>-2×C<sub>16</sub>` and `C<sub>22</sub>×C<sub>66</sub>-2×C<sub>26</sub>`
- Stress method: `C<sub>11</sub>`, `C<sub>22</sub>`, `C<sub>66</sub>` for every lattice type
- Both: the union of the two sets

**`post` mode:** Reads the final energy and stress from each `OUTCAR`. A strain set is skipped if any of its OUTCARs is missing or has not reached required accuracy; a method left with an incomplete tensor is reported and skipped without stopping the other method. For each method, the elastic tensor (C<sub>11</sub>, C<sub>22</sub>, C<sub>12</sub>, C<sub>66</sub>, plus C<sub>16</sub>/C<sub>26</sub> for oblique) is written, mechanical stability is checked via eigenvalue positivity, and angle-dependent Young's modulus, Poisson's ratio, and shear modulus are computed for stable materials. In the stress method, the raw tensor is symmetrized (asymmetry printed) and C<sub>16</sub>/C<sub>26</sub> are set to zero for non-oblique cells.

**Output:**
- `StrainVsEnergy_<strain set>.dat` — strain vs. energy per area (eV/Å<sup>2</sup>) (energy method)
- `StrainVsStress_<strain set>.dat` — strain vs. 2D stress σ<sub>xx</sub>, σ<sub>yy</sub>, σ<sub>xy</sub> (N/m) (stress method)
- `Elastic.dat`, `Young.dat`, `Poisson.dat`, `Shear.dat` — when one method is selected
- `Elastic_<method>.dat`, `Young_<method>.dat`, `Poisson_<method>.dat`, `Shear_<method>.dat` (`<method>` = `energy` or `stress`) — when both methods are selected
- `Elastic_compare.dat` — constants from both methods side by side with the relative difference (%) (both methods selected and both tensors assembled)

---

#### `vaspPiezoelectric.py`

Extracts the piezoelectric stress tensor (e, C/m<sup>2</sup>) and elastic stiffness tensor (C, GPa or N/m) from a VASP `OUTCAR` and computes the piezoelectric strain tensor (d = e·S, pm/V) via the compliance tensor S = C<sup>–1</sup>.

```
Usage: vaspPiezoelectric.py <POSCAR> <OUTCAR>
```

Supports both 2D materials (with vacuum-layer thickness correction) and 3D bulk materials.

**Piezoelectric stress tensor:** The electronic part (`PIEZOELECTRIC TENSOR (including local field effects)`) is always read. If the ionic part (`PIEZOELECTRIC TENSOR IONIC CONTR`) is also present, the script asks whether to include it: electronic + ionic (relaxed-ion) or electronic only (clamped-ion). Without the ionic part, e is clamped-ion. The choice is written in the output file headers.

**Elastic tensor:** Fallback chain OUTCAR (last `TOTAL ELASTIC MODULI` block, relaxed-ion when the ionic contribution is present) → `Elastic.dat` → manual input. A 6×6 tensor (GPa) is converted to N/m for 2D materials. `Elastic.dat` is not overwritten when it is used as input. A warning is printed if e and C are not both clamped-ion or both relaxed-ion.

**Output:** `Piezoelectric_Stress.dat`, `Piezoelectric_Strain.dat`, `Elastic.dat`

---

#### `plotMechanics.py`
Inspired by:** [Klichchupong Dabsamut](https://scholar.google.com/citations?user=KbqQhGkAAAAJ)

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

#### `vaspConvert.py`

Converts a VASP POSCAR/CONTCAR between Direct and Cartesian coordinate representations. Toggles the coordinate type of the input: Direct coordinates are converted to Cartesian, and Cartesian coordinates are converted to Direct. Handles VASP4/5/6 formats, all scaling factor conventions, Selective Dynamics, and duplicate element reordering.

```
Usage: vaspConvert.py <input POSCAR> <output POSCAR>
```

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
- **Mode 3 — 3D bulk:** selected atom shifted to origin
- **Mode 4 — Adsorbate:** selected adsorbate group centered in XY at (0.5, 0.5); z-coordinates of all atoms left unchanged. Adsorbate selection supports free-format input (index, range, element symbol, `all`)

---
 
#### `vaspVacuum.py`
 
Adds vacuum space to, or rebuilds it in, a VASP POSCAR along a chosen lattice direction (a, b, or c) — for adjusting the vacuum gap in slab or 2D-material cells.
 
```
Usage: vaspVacuum.py <POSCAR> <output POSCAR>
```
 
- **Mode 1 — Add vacuum:** Grows the chosen lattice vector's length by an entered thickness (Å), preserving its direction; the other two vectors are unchanged. Vacuum is added on top of whatever already exists.
 
- **Mode 2 — Rebuild vacuum:** Resets the total vacuum along the chosen direction to an entered thickness, discarding whatever vacuum already exists — the vector is resized to (slab thickness + entered vacuum).
 
**Output:** One VASP5-format POSCAR with Direct coordinates.
 
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

The rotation axis is x, y, z, or a custom Cartesian vector; the angle is entered in degrees.

- **Molecules:** all atoms are rotated about a pivot set to the geometric centre (unweighted mean of atomic positions), a specific atom, or a custom point entered in Direct (fractional) coordinates.
- **2D/3D materials:** only the selected atoms (free-format selection) are rotated about their own geometric centre; all other atoms stay fixed.

---

#### `vaspFix.py`

Applies Selective Dynamics constraints to a VASP POSCAR, fixing atoms in specified Cartesian directions.

```
Usage: vaspFix.py <POSCAR> <output POSCAR>
```

Three atom-selection modes: by index/label; by cutoff radius, which fixes the atoms outside the radius from the PBC-unwrapped centroid of selected reference atoms (minimum-image distances); or from an existing `FIXED_ATOMS_LIST` file. Writes a `FIXED_ATOMS_LIST` log (atoms in output POSCAR order) for reference and reuse.

---

#### `vaspStrain.py`

Applies a strain tensor to a crystal structure POSCAR for DFT elastic constant calculations.

```
Usage: vaspStrain.py <POSCAR> <output POSCAR>
```

Accepts 3 values (diagonal strain) or 9 values (full 3×3 tensor). Off-diagonal inputs are symmetrized. Applies the deformation gradient F = I + ε to the lattice matrix, preserving fractional atomic coordinates.

---

#### `vaspSupercell.py`

Generates a supercell POSCAR from a unit cell input using an expansion matrix. Accepts 3 values (diagonal expansion) or 9 values (full 3×3 matrix).

```
Usage: vaspSupercell.py <POSCAR> <output POSCAR>
```

Uses an integer-exact (adjugate-matrix-based) grid point generation to avoid floating-point rounding errors. Supports anisotropic scale factors and Selective Dynamics.

---

#### `vaspStack.py`

Generates a set of bilayer POSCAR files from a single input monolayer POSCAR by stacking two copies along the c-axis with systematically varied interlayer shifts, rotations, and flips of the top layer.

```
Usage: vaspStack.py <POSCAR>
```

The script detects the 2D Bravais lattice type (hexagonal, square, rectangular, oblique) from the in-plane lattice vectors and generates every flip × rotation × shift combination appropriate for that lattice:

- **Shifts:** hexagonal AA/AB/AB'; square AA/AB/AC; rectangular and oblique AA/AB<sub>x</sub>/AB<sub>y</sub>/AC
- **Rotations** (in-plane, about the layer centroid): hexagonal R<sub>0</sub>/R<sub>60</sub>/R<sub>120</sub>/R<sub>180</sub>; square R<sub>0</sub>/R<sub>90</sub>/R<sub>180</sub>; rectangular and oblique R<sub>0</sub>/R<sub>180</sub>
- **Flips:** hexagonal and oblique F<sub>0</sub>/F<sub>z</sub>; square F<sub>0</sub>/F<sub>z</sub>/F<sub>x</sub>/F<sub>xy</sub>; rectangular F<sub>0</sub>/F<sub>z</sub>/F<sub>x</sub>/F<sub>y</sub>/F<sub>xy</sub>

**Interlayer gap:** Computed from the sum of van der Waals radii of the two atoms facing each other across the interface (highest-z atom of the bottom layer, lowest-z atom of the top layer). The bilayer is centered at 0.5 along c. The element order of the combined bilayer is prompted once (empty input keeps the input order).

**Output:** One POSCAR per configuration in `<i>_<flip>_<rot>_<stack>/POSCAR` (index zero-padded, e.g. `01_F0_R0_AA`), plus `STACK_LIST.dat` listing each configuration's index, flip, rotation, stacking label, fractional shift, and directory path.

---

#### `vaspTwist.py`
**Inspired by:** [CellMatch](https://doi.org/10.1016/j.cpc.2015.08.038) and [vasp-grace-tensorpotential](https://github.com/Asif-Iqbal-Bhatti/vasp-grace-tensorpotential/blob/main/src/vasp_grace/moire.py)

Generates moiré twisted bilayer POSCAR files by searching for exact commensurate supercells. A single command performs the full workflow: search, candidate selection, and POSCAR generation.

```
Usage: vaspTwist.py <POSCAR1> [POSCAR2]
```

Providing one POSCAR builds a homobilayer (both layers identical); providing two builds a heterobilayer. Homobilayer/heterobilayer is auto-detected by comparing in-plane lattice vectors and composition.

**Search method (closed-form, no angle grid):**
- **Homobilayer:** Exact coincidence-site-lattice (CSL) angles are computed in closed form for hexagonal and square lattices. Strain is zero by construction. Every cell is reduced to its primitive coincidence cell, so θ and 60° − θ (90° − θ for square) appear as separate candidates. Falls back to the heterostrain search below for rectangular or oblique lattices, with a warning that prints a, b and γ.
- **Heterobilayer:** Closed-form vector matching finds integer-indexed lattice vector pairs whose rotation and stretch align within `MAX_STRAIN`, then combines pairs whose angles lie within the window allowed by `MAX_STRAIN` into full 2D supercell candidates via polar decomposition. For each twist angle (0.01°), the smallest supercell is kept, plus any larger supercell with lower strain. No angle-step parameter is needed.

**Candidate filtering:** `MAX_ATOMS = 500` (hard cap on total supercell atoms); `MAX_STRAIN = 0.05` (5%, heterobilayer only); search range `THETA_MIN`/`THETA_MAX = 0°`/`180°`.

**`TWIST_LIST.dat` reuse:** After a search, results are written to `TWIST_LIST.dat` along with a structure snapshot (`original.vasp` for homobilayer; `bottom.vasp` + `top.vasp` for heterobilayer). On a later run with the same input file(s), the snapshot is compared against the provided POSCAR(s) (tolerance 10−6 Å); if they match, the saved candidate list is reused directly — skipping the search — and the script goes straight to candidate selection. If they don't match, a fresh search runs automatically with a warning.

**Candidate selection and generation:** The candidate table is displayed interactively; selection accepts individual indices, `'all'`, or `'none'`. For each selected candidate, one POSCAR is written per high-symmetry stacking configuration (lattice-type-dependent shift grid: hexagonal, square, rectangular, or oblique), to `<index>_<theta>_<atoms>/<shift_no>_<stacking_label>/POSCAR`, where `<index>` is the candidate number in the table and `TWIST_LIST.dat` (zero-padded).

**Interlayer gap:** Computed from the sum of van der Waals radii of the two atoms facing each other across the interface (highest-z atom of the bottom layer, lowest-z atom of the top layer); falls back to 3.5 Å with a warning if either element's radius is unknown.

**Strain metric:** Lagrangian finite strain computed via metric-tensor Cholesky decomposition, used as a hard filter for heterobilayer candidates and a zero-sanity-check for homobilayer (CSL) candidates.

---

#### `vaspAdsorb.py`
**Inspired by:** [Aroon Ananchuensook](https://scholar.google.com/citations?user=6mmg4mUAAAAJ)

Combines a substrate and an adsorbent POSCAR into a single POSCAR for adsorption DFT calculations. Supports placing multiple adsorbent copies simultaneously. After selecting a placement mode, the user specifies the number of copies and the separation distance (Å).

```
Usage: vaspAdsorb.py <substrate POSCAR> <adsorbent POSCAR> <output POSCAR>
```

**Mode 1 — on top of a specific site:** Places each adsorbent copy above a user-defined target point on the substrate. For each copy, the user selects: (a) the placement side (top or bottom); (b) the substrate reference height (highest atom, a selected atom, or average height); (c) the adsorbent anchor point (centroid or a specific/lowest atom); (d) the target xy position (by atom selection with free-format input, or custom fractional coordinates). The adsorbent is translated so its anchor lands at the target at the specified vertical distance.

**Mode 2 — ring around a target atom:** Places N copies evenly distributed at angular intervals of 2π/N around each chosen substrate atom (e.g. a decorating ion) using Rodrigues z-axis rotation. Supports multiple target atoms; the user specifies the number of targets and selects each one sequentially. The placement side (top or bottom) is auto-detected from the target atom z position relative to the substrate mean z; if the target lies within 0.01 Å of the mean (e.g. a flat monolayer), the user is prompted to choose the side. For each target, the user selects the initial adsorption direction (by atom selection or custom fractional coordinates). The adsorbent's lowest atom (top) or highest atom (bottom) is aligned with the target atom's height, and the radial distance from the target atom is set by the distance input.

**Automatic distance:** A negative distance switches to an automatic distance equal to the sum of the van der Waals radii of the two facing atoms (3.0 Å for an element without a tabulated radius). In mode 1 these are the outermost substrate atom on the chosen side and the adsorbent atom facing it, and the result is used as the vertical distance. In mode 2 they are the target atom and the adsorbent atom facing it, and the result is used as the radial distance. The value is computed per copy (mode 1) or per target (mode 2) and printed.

If either POSCAR has Selective Dynamics, flags are merged, and the other structure defaults to all-T. If neither has Selective Dynamics, the user is prompted after placement to optionally add constraints, with free-format atom and direction selection. A summary table of atom counts per element (substrate/adsorbent/total) is printed at the end.

---

#### `vaspMove.py`

Moves selected atoms within a VASP POSCAR by displacement or to absolute coordinates. Two move modes are available interactively.

```
Usage: vaspMove.py <input POSCAR> <output POSCAR>
```

Atom selection uses free-format input (index, range e.g. `1-4`, element symbol, `all`). For each mode, the coordinate system (Cartesian in Å or Direct fractional) is selected independently.

**Mode 1 — Displace by vector:** Adds a (dx, dy, dz) displacement vector to all selected atoms.

**Mode 2 — Move to absolute coordinates:** Translates the centroid of the selected atoms to a target point; all selected atoms are shifted rigidly by the same offset.

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
Usage: mlRegression.py <ML_REG> [ML_ABN]
```

`ML_ABN` (optional) must be the one written together with `ML_REG`; it gives the number of atoms of each structure for per-atom energies. Without it, all structures are assumed to have the same number of atoms.

**Output:** `Energy.dat`, `Force.dat`, `Stress.dat`, `ERROR.dat`

---

#### `mlab2extxyz.py`
**Inspired by:** [utf/pymlff](https://github.com/utf/pymlff)

Converts VASP's `ML_AB` binary training data file to extended XYZ (`.extxyz`) format for use with external MLFF training frameworks such as MACE, NequIP, and GPUMD.

```
Usage: mlab2extxyz.py <ML_AB input> <output.extxyz>
```

Each configuration block is mapped to one extxyz frame with lattice, positions, energy, forces, and stress. Stress is converted from kbar to eV/Å<sup>3</sup> and sign-flipped, since VASP reports compressive stress as positive while the extxyz (ASE) convention reports it as negative.

---

#### `xml2mlab.py`

Converts VASP AIMD `vasprun.xml` trajectory files to VASP `ML_AB` training data format for MLFF training.

```
Usage: xml2mlab.py <vasprun1.xml> [vasprun2.xml ...]
```

Multiple files are concatenated into a single `ML_AB` output. Interactively prompts for equilibration skip (number of initial steps to discard) and stride (use every N-th step). A per-file and total configuration summary is printed before writing begins. CTIFOR is omitted (external training data convention); reference atomic energies are set to 0.0 and basis sets are written as dummy entries, both compatible with `ML_MODE = select`.

Element sets are verified to be consistent across files; mismatches trigger a warning and the union of all element types is used. Stress is passed through without sign flip, consistent with native VASP MLFF output.

**Output:** `ML_AB` written to the current directory.

---

#### `mergeMLAB.py`
**Inspired by:** [utf/pymlff](https://github.com/utf/pymlff)

Merges multiple VASP `ML_AB` training data files into a single unified file, unifying element lists, basis sets, and renumbering configurations.

```
Usage: mergeMLAB.py <ML_AB 1> <ML_AB 2> [ML_AB 3 ...] <output ML_AB>
```

Resolves header metadata conflicts (reference energies, atomic masses) by first-file-wins with a warning for mismatches.

---

### 6. NEB Calculations

Scripts for setting up and analyzing Nudged Elastic Band (NEB) calculations in VASP.

---

#### `vaspNEB.py`
**Inspired by:** [VTST scripts](https://theory.cm.utexas.edu/vtsttools/scripts.html)

Generates NEB image directories with interpolated POSCAR files between two endpoint structures. By default uses the [IDPP method](https://doi.org/10.1063/1.4878664) (Image Dependent Pair Potential), which produces smoother initial paths and reduces the number of NEB iterations. Linear interpolation is available via `-linear`.

```
Usage: vaspNEB.py <initial POSCAR> <final POSCAR> <N_images> [-linear]
```

`N_images` is the number of intermediate images excluding endpoints. The script validates that both POSCARs share the same element symbols and atom counts, and warns if lattice matrices differ (dynamic cell / SSNEB mode), in which case the lattice is also linearly interpolated. All interpolation uses the minimum-image convention so atoms always take the shortest path across periodic boundaries. Output directories are two-digit (`00`, `01`, ..., up to `99`), so `N_images` can be at most 98; each contains one `POSCAR`. Selective Dynamics flags are carried from the initial image.

IDPP requires SciPy; if SciPy is not available the script falls back to linear interpolation automatically.

---

#### `analyzeNEB.py`
**Inspired by:** [VTST scripts](https://theory.cm.utexas.edu/vtsttools/scripts.html)

Analyzes a completed VASP NEB run from the current directory. Automatically detects image directories matching the two-digit pattern (`00`, `01`, ...) and auto-detects SSNEB mode (LNEBCELL = .TRUE. in `01/OUTCAR`).

```
Usage: analyzeNEB.py [nj]
```

`nj` sets the number of cubic spline sub-steps per image interval (default: 20). All output is written to the working directory except per-image convergence files.

**Output files (working directory):**
- `neb.dat` — cumulative path distance (Å), relative energy (eV), and tangent force (eV/Å) per image
- `nebss.dat` — same quantities normalized per atom and per √N_ions (SSNEB mode only)
- `spline.dat` — dense cubic Hermite spline along the MEP interpolated from `neb.dat`
- `spliness.dat` — same spline from `nebss.dat` (SSNEB mode only)
- `exts.dat` — transition states and minima located analytically along the spline
- `extsss.dat` — extrema from `spliness.dat` (SSNEB mode only)
- `nebef.dat` — image-by-image table: max force (eV/Å), absolute energy (eV), relative energy (eV)
- `nebefs.dat` — extended table adding stress (kBar), volume (Å<sup>3</sup>), magnetic moment (μ<sub>B</sub>) (SSNEB mode only)
- `movie` — concatenated POSCAR-format trajectory from all image CONTCARs (or POSCARs)
- `movie.xyz` — same trajectory in XYZ format, with force and energy annotations per frame

**Per-image output:**
- `<img>/fe.dat` — force and energy convergence history over ionic steps for each interior image

---

### 7. Charge Density Analysis

Scripts for analyzing, transforming, and visualizing VASP volumetric charge-density data (CHGCAR/CHG/PARCHG) and VASPKIT-exported grids.

---

#### `vaspCharge.py`

Reads a CHGCAR-format file (CHG and PARCHG share the same layout) and performs one of ten charge-density analysis or transformation tasks, selected interactively. Non-collinear (LSORBIT/LNONCOLLINEAR) CHGCAR files are detected and rejected with an error (not yet supported).

```
Usage: vaspCharge.py <CHGCAR>
```

**Tasks (interactive menu):**
1. Slice of Charge Density — 2D cross-section perpendicular to a chosen lattice axis at a chosen fractional coordinate, linearly interpolated between grid planes. Output: `SLICE_X.grd`, `SLICE_Y.grd`, `CHGSLICE.grd`
2. Charge Density — total electron density, unmodified. Output: `CHGTOT.vasp`
3. Spin Density — magnetization ρ<sub>up</sub> − ρ<sub>down</sub> (ISPIN = 2 only). Output: `CHGMAG.vasp`
4. Spin-Up & Spin-Down Density — derived densities (tot ± mag) / 2 (ISPIN = 2 only). Output: `CHGUP.vasp`, `CHGDW.vasp`
5. Linear-Average Charge Density — density averaged over one axis, leaving a 2D map. Output: `LAVG_X.grd`, `LAVG_Y.grd`, `CHGLAVG.grd` — ready for `plotGrid.py`
6. Planar-Average Charge Density — 1D profile along a chosen axis. Output: `PLANAR_AVERAGE.dat`
7. Macroscopic-Average Charge Density — the planar-average profile further smoothed with a periodic moving-window average (FFT convolution), for work-function/band-alignment analysis; window length should normally equal one interplanar spacing. Output: `MACROSCOPIC_AVERAGE.dat`
8. Charge Density Along a Specified Path — trilinear-interpolated density along a line between two fractional points. Output: `CHG_LINE.dat`
9. Build Supercell of Charge Density — tiles the density grid and structure by integer (n<sub>1</sub>, n<sub>2</sub>, n<sub>3</sub>) multiples; tiled values are rescaled by n<sub>1</sub>×n<sub>2</sub>×n<sub>3</sub> since CHGCAR stores ρ(r)·V_cell. Output: `CHGCAR_SUPERCELL.vasp`
10. STM Simulation — Tersoff-Hamann-style image from the density (e.g. a PARCHG near E<sub>F</sub>), assuming the surface normal is axis C with vacuum on the +C side. Constant-height mode writes the density map ρ(x,y) at a fixed height; constant-current mode writes the height map z(x,y) where density first crosses a chosen threshold, scanning in from vacuum. Output: `STM_X.grd`, `STM_Y.grd`, and `STM_DENSITY.grd` (constant-height) or `STM_TOPOGRAPHY.grd` (constant-current)

**Output convention:** `.vasp` outputs are written in VASP's native ρ(r)·V<sub>cell</sub> convention with no PAW augmentation-occupancies block — valid for visualization (e.g. VESTA) but not for restarting VASP. `.grd` outputs are plain whitespace-delimited grids, readable by `plotGrid.py`.

---

#### `vaspChargeDiff.py`

Computes a charge-density difference Δρ = ρ<sub>AB</sub> − ρ<sub>A</sub> − ρ<sub>B</sub> from three CHGCAR files — a combined system and its two isolated fragments in the same cell — e.g. for analyzing electron transfer in an adsorbate + substrate system.

```
Usage: vaspChargeDiff.py <CHGCAR_AB> <CHGCAR_A> <CHGCAR_B>
```

All three files must share identical grid dimensions (hard error on mismatch). A lattice-vector mismatch between `CHGCAR_AB` and either fragment (tolerance 10<sup>–4</sup> Å) only warns, since the difference may still be informative. For spin-polarized (ISPIN = 2) inputs only the total density is differenced; the magnetization block is ignored, so `CHGDIFF.vasp` contains a single density block.

**Output:** `CHGDIFF.vasp` — same ρ(r)·V_cell convention as `vaspCharge.py`.

---

#### `plotGrid.py`

Reads a VASPKIT-generated 2D grid — `X.grd`, `Y.grd`, and a value grid such as `CHGLAVG.grd` (linear-averaged charge density) — and produces a filled-contour heatmap plus an ASCII export for external plotting.

```
Usage: plotGrid.py <X.grd> <Y.grd> <CHGLAVG.grd> [output prefix]
```

If [output prefix] is omitted, the value-grid filename (without extension) is used, e.g. `CHGLAVG.grd` → `CHGLAVG.png` / `CHGLAVG.dat`.

**Grid validation:** `X`, `Y`, and the value grid must share identical shape (hard error on mismatch, since a shape mismatch would silently truncate the arrays when they are flattened and zipped together for the ASCII export). Axis alignment — X constant along rows, Y constant along columns, i.e. a regular rectilinear mesh — is checked with `atol=2e-4`, twice VASPKIT's 4-decimal-place output resolution, and only warns rather than errors, since both the contour plot and the ASCII export handle a non-rectilinear grid correctly.

**Interactive axis labeling:** Before plotting, the script always prompts (mandatory, non-empty, restricted to X/Y/Z) for which physical axis each plot axis represents, and labels it in Å with mathtext formatting.

**Plot:** filled contour (viridis colormap, 100 levels, equal aspect ratio) with a colorbar labeled for charge density (e/Å<sup>3</sup> by default — edit the `CHGLAVG_LABEL` constant for other quantities), rendered in Times New Roman/serif per the toolkit's plot styling convention, saved at 300 dpi.

**Output:**
- `<prefix>.png` — filled-contour heatmap
- `<prefix>.dat` — plain three-column `X Y CHGLAVG` ASCII file, importable via Origin's Data > Convert to Matrix > XYZ Gridding, or any generic contour/3D plotting tool

---

### 8. Calculation Setup

Scripts for preparing calculation-control tags for VASP and follow-on codes — initial MAGMOM values, and dielectric/Born-effective-charge tags derived from DFPT output.

---

#### `vaspBorn.py`

Extracts the ion-clamped (electronic) dielectric tensor and Born effective charge tensors from a VASP DFPT calculation (`LEPSILON=.TRUE.`) and writes them to `INCAR.LR`. Accepts either `OUTCAR` or `vasprun.xml`; file type is auto-detected by filename with content-based fallback.

```
Usage: vaspBorn.py <OUTCAR or vasprun.xml>
```

**Output:** `INCAR.LR` — `PHON_DIELECTRIC` and `PHON_BORN_CHARGES` tags in backslash-continuation format.

---

#### `vaspMagmom.py`

Builds an initial `MAGMOM` tag from a POSCAR/CONTCAR's element composition and writes it into `INCAR`, using standard per-element default magnetic moments (a built-in periodic-table lookup table, non-zero for the 3d, 4d and 5d transition metals, the lanthanides Ce–Yb, and U; every other element gets the non-magnetic placeholder 0.6).

```
Usage: vaspMagmom.py <POSCAR>
```

Reads the structure via the toolkit's shared VASP4/5 POSCAR parser. For a VASP4-format file without the element-symbol line, element names are read from `POTCAR` in the same directory as the POSCAR; the user is prompted only if `POTCAR` is missing or its species count does not match. Per-atom moments are expanded in POSCAR atom order and run-length-encoded into VASP's `count*value` shorthand, grouped by (element, value) so different elements are never merged (e.g. `3*5.0 2*0.6`).

**Single-atom override:** If an element's bulk default is 0.6 and the structure contains only one atom of it (e.g. a single adsorbate or dopant), its free-atom unpaired-electron count (Hund's rule) is used instead.

**INCAR handling:** If `INCAR` already exists in the working directory, the `MAGMOM` tag is set/overwritten in place (other tags, comments, and ordering are preserved), `NUPDOWN` is removed if present (it fixes the total moment and conflicts with a per-atom guess), and `ISPIN` is forced to `2` if not already set to `2`. If no `INCAR` exists, a fresh one is written from a standard spin-polarized template.

**Output:** `INCAR` (created or updated in place).

---

## Design Conventions

All scripts follow the same conventions:

- Single-responsibility functions with NumPy-style docstrings (`Parameters`, `Returns`, inline notes for unit conversions and formulas)
- `main()` entry point with `if __name__ == '__main__'` guard
- Interactive input loops with validation and retry on invalid input; `readline` (stdlib) is imported in all scripts that use `input()` to enable arrow-key navigation and line editing at prompts
- Handles both VASP4 (no element line), VASP5, and VASP6 (with Hash code) POSCAR formats
- Handles Selective Dynamics, anisotropic scale factors, and non-orthogonal cells
- Atom-label comments (e.g., `Mo001`, `S002`) in output POSCARs for VESTA/XCrySDen identification
- Output files formatted for xmgrace (`.dat` with column headers) unless otherwise noted

---

## License

Developed by [Thanasee Thanasarnsurapong](https://scholar.google.com/citations?user=4KHXv9gAAAAJ) and [Claude](https://claude.ai/). For research and academic use.
