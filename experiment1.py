import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import cKDTree
from scipy.spatial.distance import pdist, squareform
from ase import Atoms
from ase.io import read, write
from ase.visualize import view
from ase.build import make_supercell, molecule
from ase.units import Ang
from ase.io.lammpsdata import write_lammps_data
from collections import Counter

# --- Configuration Parameters ---
MOF_FILENAME = 'C:/Users/Cameron/Documents/MOF files/UiO-66-NH2.cif'
SUPERCELL_MATRIX = [[2, 0, 0], [0, 2, 0], [0, 0, 2]]
PEO_SEGMENTS = 30
IL_MOLECULES = 20
NA_IONS = 10
CO2_MOLECULES = 10
BUFFER_SIZE = 5.0
PLACEMENT_RADIUS = 2.0 
MIN_SAFE_DISTANCE = 0.8  # Critical threshold for steric clash removal

# --- Helper Functions (PEO, IL, Pore Centers) ---
def make_peo_segment():
    symbols = ['C', 'C', 'O', 'H', 'H', 'H', 'H', 'H', 'H']
    positions = [
        [0.0, 0.0, 0.0], [1.5, 0.0, 0.0], [3.0, 0.0, 0.0],
        [-0.5, 0.9, 0.0], [-0.5, -0.9, 0.0],
        [1.5, 0.9, 0.0], [1.5, -0.9, 0.0],
        [3.5, 0.5, 0.0], [3.5, -0.5, 0.0]
    ]
    peo = Atoms(symbols=symbols, positions=positions)
    axis_vector = np.random.normal(size=3)
    axis_vector /= np.linalg.norm(axis_vector)
    angle = np.random.uniform(0, 360)
    peo.rotate(angle, v=axis_vector, center=peo.get_center_of_mass())
    return peo

def make_il_mol():
    symbols = ['C']*6 + ['N']*2 + ['H']*11
    positions = [
        [0.000, 0.000, 0.000], [1.400, 0.000, 0.000], [2.800, 0.000, 0.000],
        [-1.200, 0.000, 0.000], [-2.400, 0.000, 0.000], [0.000, 1.400, 0.000],
        [0.700, -1.200, 0.000], [-0.700, -1.200, 0.000],
        [3.200, 0.800, 0.000], [3.200, -0.800, 0.000],
        [-2.800, 0.800, 0.000], [-2.800, -0.800, 0.000],
        [1.400, 1.000, 0.800], [1.400, 1.000, -0.800],
        [-1.200, 1.000, 0.800], [-1.200, 1.000, -0.800],
        [0.000, 2.000, 0.800], [0.000, 2.000, -0.800],
        [0.000, -2.000, 0.000]
    ]
    il = Atoms(symbols=symbols, positions=positions)

    # Apply random rotation around a random 3D axis
    axis_vector = np.random.normal(size=3)
    axis_vector /= np.linalg.norm(axis_vector)
    angle = np.random.uniform(0, 360)
    il.rotate(angle, v=axis_vector, center=il.get_center_of_mass())

    return il

def get_pore_centers(mof, num_points=2000, min_distance=3.0):
    """Return candidate pore centers by sampling points far from MOF atoms."""
    cell = mof.get_cell()
    positions = mof.get_positions()
    tree = cKDTree(positions)

    pore_centers = []
    for _ in range(num_points * 5): 
        point = np.random.uniform([0, 0, 0], cell.lengths())
        dist, _ = tree.query(point)
        if dist > min_distance + 0.5: 
            pore_centers.append(point)
        if len(pore_centers) >= num_points:
            break
    return np.array(pore_centers)

def remove_clashes(system, min_dist=0.8):
    positions = system.get_positions()
    D = squareform(pdist(positions))
    np.fill_diagonal(D, np.inf)
    clashes = np.where(D < min_dist)
    to_remove = set(clashes[0]) | set(clashes[1])
    if to_remove:
        print(f"Removing {len(to_remove)} atoms due to residual clashes.")
        mask = np.ones(len(system), dtype=bool)
        mask[list(to_remove)] = False
        system = system[mask]
    return system

# --- 1. MOF Preparation ---
print("--- 1. MOF Preparation ---")
try:
    mof_primitive = read(MOF_FILENAME)
except FileNotFoundError:
    print(f"ERROR: MOF file not found at {MOF_FILENAME}.")
    exit()

mof = make_supercell(mof_primitive, SUPERCELL_MATRIX)
mof_cell_len = mof.get_cell().lengths()
print(f"MOF Supercell built. Atoms: {len(mof)}. Cell: {mof_cell_len} Å")

# Store initial MOF indices before combination
initial_mof_indices = set(range(len(mof)))

# --- Cavity-Aware Placement ---
print("--- Cavity-Aware Placement ---")
pore_centers = get_pore_centers(mof, num_points=2000)

num_components = PEO_SEGMENTS + IL_MOLECULES + NA_IONS + CO2_MOLECULES
if len(pore_centers) < num_components:
    print(f"Warning: Only {len(pore_centers)} suitable pore centers found. Overlap is likely.")
    
all_placement_centers = pore_centers[np.random.choice(len(pore_centers), num_components, replace=False)]

peo_centers = all_placement_centers[:PEO_SEGMENTS]
il_centers = all_placement_centers[PEO_SEGMENTS:PEO_SEGMENTS+IL_MOLECULES]
na_centers = all_placement_centers[PEO_SEGMENTS+IL_MOLECULES:PEO_SEGMENTS+IL_MOLECULES+NA_IONS]
co2_centers = all_placement_centers[PEO_SEGMENTS+IL_MOLECULES+NA_IONS:]

# --- 2. Build Shell/Gas Components (with random jitter) ---
print("--- 2. Building Shell/Gas Components ---")

polymer_shell = Atoms()
for pos in peo_centers:
    peo = make_peo_segment()
    jitter = np.random.uniform(low=-PLACEMENT_RADIUS, high=PLACEMENT_RADIUS, size=3)
    peo.translate(pos + jitter - peo.get_center_of_mass())
    polymer_shell += peo

ionic_liquid = Atoms()
for pos in il_centers:
    il = make_il_mol()
    jitter = np.random.uniform(low=-PLACEMENT_RADIUS, high=PLACEMENT_RADIUS, size=3)
    il.translate(pos + jitter - il.get_center_of_mass())
    ionic_liquid += il

na_ions = Atoms('Na' * NA_IONS, positions=na_centers)
na_ions.positions += np.random.uniform(low=-PLACEMENT_RADIUS, high=PLACEMENT_RADIUS, size=(NA_IONS, 3))

co2_molecules = []
for pos in co2_centers:
    co2 = molecule('CO2')
    jitter = np.random.uniform(low=-PLACEMENT_RADIUS, high=PLACEMENT_RADIUS, size=3)
    co2.translate(pos + jitter - co2.get_center_of_mass())
    co2_molecules.append(co2)
co2_gas = sum(co2_molecules, Atoms())

# --- 3. Combine and Define Box ---
system = mof + polymer_shell + ionic_liquid + na_ions + co2_gas
new_cell_len = mof_cell_len + 2 * BUFFER_SIZE
system.set_cell(new_cell_len, scale_atoms=False)
system.set_pbc(True)
system.center()
print(f"Final System built (pre-filter). Total Atoms: {len(system)}. Final Box: {system.get_cell().lengths()} Å")

# --- 4. LAMMPS Setup with Steric Clash Filtering ---
print("\n--- 4. LAMMPS Setup & Steric Clash Filtering ---")

# Perform an initial check for severe overlaps
initial_min_dist = np.min(pdist(system.get_positions()))
print(f"Initial minimum interatomic distance: {initial_min_dist:.3f} Å")

if initial_min_dist < MIN_SAFE_DISTANCE:
    print(f"**Steric Clash Warning**: Initial distance {initial_min_dist:.3f} Å is below safety threshold. Starting removal process.")
    
    # --- Iterative Removal of Clashing Atoms ---
    # Define indices of the MOF (these are fixed and should not be removed)
    mof_indices_in_system = set(range(len(mof)))
    
    atoms_to_keep = []
    
    # Build a KDTree of the MOF for fast checks against mobile atoms
    mof_tree = cKDTree(system.get_positions()[list(mof_indices_in_system)])
    
    # Now, iterate through all atoms in the system
    for i in range(len(system)):
        pos = system.get_positions()[i]
        
        # 1. Check MOF atoms: Always keep them
        if i in mof_indices_in_system:
            atoms_to_keep.append(i)
            continue
            
        # 2. Check Mobile atoms (PEO, IL, Na+, CO2):
        # Check against MOF (Fast check using the MOF tree)
        mof_dist, _ = mof_tree.query(pos)
        if mof_dist < MIN_SAFE_DISTANCE:
            # Atom is clashing with the MOF, remove it (skip adding to keep list)
            continue
        
        # Check against all previously KEPT atoms (slower, but necessary)
        # Note: We only need to check against other *mobile* atoms that were kept
        kept_positions = system.get_positions()[atoms_to_keep]
        
        if len(kept_positions) > 0:
            kept_tree = cKDTree(kept_positions)
            
            # The query returns the distance to the nearest KEPT atom
            kept_dist, _ = kept_tree.query(pos)
            
            if kept_dist < MIN_SAFE_DISTANCE:
                # Atom is clashing with another kept atom, remove it
                continue
        
        # If the atom passes all checks, keep it
        atoms_to_keep.append(i)

    # Rebuild the system using only the kept indices
    removed_count = len(system) - len(atoms_to_keep)
    system = system[atoms_to_keep]
    print(f"**Clash Removal Complete**: Removed {removed_count} mobile atoms (PEO/IL/Na+/CO2) to enforce minimum distance > {MIN_SAFE_DISTANCE} Å.")

# Re-calculate the minimum distance after filtering to confirm success
D = squareform(pdist(system.get_positions()))
np.fill_diagonal(D, np.inf)  # ignore self-distances
min_dist = np.min(D)
i, j = np.unravel_index(np.argmin(D), D.shape)
print(f"Final minimum interatomic distance: {min_dist:.3f} Å (Target: > 0.8 Å)")
print(f"Closest atoms: {i} and {j}, distance = {D[i, j]:.3f} Å")

# Iteratively clean the system
while True:
    D = squareform(pdist(system.get_positions()))
    np.fill_diagonal(D, np.inf)
    min_dist = np.min(D)
    if min_dist > 0.8:
        break
    i, j = np.unravel_index(np.argmin(D), D.shape)
    print(f"Clash detected: atoms {i} and {j}, distance = {D[i, j]:.3f} Å")
    system = remove_clashes(system, min_dist=0.8)

# Recalculate after final cleanup
D = squareform(pdist(system.get_positions()))
np.fill_diagonal(D, np.inf)
min_dist = np.min(D)
i, j = np.unravel_index(np.argmin(D), D.shape)
print(f"Final minimum interatomic distance: {min_dist:.3f} Å (Target: > 0.8 Å)")
print(f"Closest atoms: {i} and {j}, distance = {D[i, j]:.3f} Å")

# Assign Atom Types for LAMMPS (re-assign tags as atom indices have changed)
unique_elements = sorted(set(system.get_chemical_symbols()))
element_map = {el: i + 1 for i, el in enumerate(unique_elements)}
system.set_tags([element_map[symbol] for symbol in system.get_chemical_symbols()])

print(f"Atom Type Mapping (Element: Type ID): {element_map}")
print(f"Final Atom type counts: {Counter(system.get_tags())}")

# Write the LAMMPS data file
write_lammps_data('battery_system.data', system, atom_style='atomic') 
print("LAMMPS data file 'battery_system.data' written successfully.")

# --- 5. Metrics (Updated Component Offsets) ---
# Re-establish component indices/offsets AFTER filtering, as some atoms were removed
mof_len = len(mof) # The MOF is the first component, its length is fixed
polymer_shell_len = len(polymer_shell) - Counter(polymer_shell.get_chemical_symbols()).get('X', 0) # Placeholder to recalculate length
ionic_liquid_len = len(ionic_liquid)
na_ions_len = len(na_ions)
co2_gas_len = len(co2_gas)

# Since we lost track of which specific component an atom belonged to after filtering,
# we rely on the total count of each element to estimate the remaining mobile molecules.
# A simpler and safer way for the metrics section is to use the atom tags/elements:
total_mass = sum(system.get_masses())
total_volume = system.get_volume()
density_g_per_cm3 = (total_mass * Ang**3) / (total_volume * 1e-24) / 1000

print("\n--- System Metrics ---")
print(f"Total Mass: {total_mass:.2f} u")
print(f"Total Volume: {total_volume:.2f} Å³")
print(f"Approximate System Density: {density_g_per_cm3:.4f} g/cm³")

# Determine component counts based on element tags after filtering
# NOTE: This is an approximation as C, H, O are shared between PEO, IL, and MOF!
# The most reliable count is for Na+ and CO2 (if CO2 is the only source of pure C/O structure)
final_symbols = system.get_chemical_symbols()
na_count = final_symbols.count('Na')
# CO2 is C+O; PEO/IL also have C, O. Use the original component counts as a *target* baseline.
# For accurate metrics, you'd need to re-index the components after filtering.

print("\n--- Component Breakdown (Post-Filter Estimate) ---")
print(f"- MOF: {len(mof)} atoms (Fixed)")
print(f"- Sodium Ions (Na+): {na_count} atoms (Accurate)")
print(f"- Other Components (PEO/IL/CO2): {len(system) - len(mof) - na_count} atoms")


# --- Proximity Checks (Re-establish indices post-filter) ---
# NOTE: The MOF indices are the first ones in the new system (0 to len(mof)-1)
mof_indices = list(range(len(mof)))
mobile_indices = list(range(len(mof), len(system)))
na_indices = [i for i, symbol in enumerate(final_symbols) if symbol == 'Na']

# --- Na+ Proximity ---
print("\n--- Na+ Ion Proximity Check ---")
min_distances_to_mof = []
# We cannot reliably find PEO/IL indices after filtering, so we only check MOF proximity
for na_index in na_indices:
    # Need to handle the case where Na+ is the only mobile species left (unlikely)
    if len(mof_indices) > 0:
        dist_mof = system.get_distances(na_index, mof_indices, mic=True, vector=False)
        min_distances_to_mof.append(np.min(dist_mof))
    
if min_distances_to_mof:
    print(f"Average Minimum Na+ to MOF Distance: {np.mean(min_distances_to_mof):.3f} Å")
else:
    print("No Na+ ions or MOF atoms found for proximity check.")

# --- Visualization ---
print("\n--- Visualization ---")
# (Visualization code remains correct as it uses element symbols/atom indices)
positions = system.get_positions()
fig, ax = plt.subplots(figsize=(8, 8))

# Mask atoms based on current element symbols/indices
is_mof = np.arange(len(system)) < len(mof) # MOF is guaranteed to be the first block
is_na = np.array([symbol == 'Na' for symbol in final_symbols])
is_other = ~(is_mof | is_na) # PEO/IL/CO2

# Plot MOF atoms
ax.scatter(positions[is_mof, 0], positions[is_mof, 1], s=1, color='gray', alpha=0.3, label='MOF')
# Plot Na+ ions
ax.scatter(positions[is_na, 0], positions[is_na, 1], s=10, color='blue', alpha=0.8, label='Na+')
# Plot PEO/IL/CO2 atoms
ax.scatter(positions[is_other, 0], positions[is_other, 1], s=0.5, color='green', alpha=0.2, label='PEO/IL/CO2')

ax.set_title(f"Atom Distribution (XY-Plane Projection) - Final Atoms: {len(system)}")
ax.set_xlabel("X-coordinate (Å)")
ax.set_ylabel("Y-coordinate (Å)")
ax.set_aspect('equal', adjustable='box')
ax.legend(markerscale=5)
plt.savefig('system_distribution_colorcoded.png')
print("Color-coded atom distribution plot saved as 'system_distribution_colorcoded.png'.")

view(system)
