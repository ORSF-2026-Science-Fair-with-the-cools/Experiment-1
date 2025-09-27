import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import cKDTree
from scipy.spatial.distance import pdist
from ase import Atoms
from ase.io import read, write
from ase.visualize import view
from ase.build import make_supercell, molecule
from ase.units import _amu, Ang
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
# NEW: Define a randomization radius to prevent molecules from landing exactly on pore centers
PLACEMENT_RADIUS = 2.0 

# --- Helper Functions (No change here, using your definitions) ---
def make_peo_segment():
    symbols = ['C', 'C', 'O', 'H', 'H', 'H', 'H', 'H', 'H']
    positions = [
        [0.0, 0.0, 0.0], [1.5, 0.0, 0.0], [3.0, 0.0, 0.0],
        [-0.5, 0.9, 0.0], [-0.5, -0.9, 0.0],
        [1.5, 0.9, 0.0], [1.5, -0.9, 0.0],
        [3.5, 0.5, 0.0], [3.5, -0.5, 0.0]
    ]
    return Atoms(symbols=symbols, positions=positions)

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
    return Atoms(symbols=symbols, positions=positions)

def get_pore_centers(mof, num_points=2000, min_distance=3.0):
    """Return candidate pore centers by sampling points far from MOF atoms."""
    cell = mof.get_cell()
    positions = mof.get_positions()
    tree = cKDTree(positions)

    pore_centers = []
    # Oversample significantly to ensure good coverage
    for _ in range(num_points * 5): 
        point = np.random.uniform([0, 0, 0], cell.lengths())
        dist, _ = tree.query(point)
        # Use a slightly larger minimum distance for better initial clearance
        if dist > min_distance + 0.5: 
            pore_centers.append(point)
        if len(pore_centers) >= num_points:
            break
    # If not enough centers were found, return all we have
    return np.array(pore_centers)

# --- 1. MOF Preparation ---
print("--- 1. MOF Preparation ---")
mof_primitive = read(MOF_FILENAME)
mof = make_supercell(mof_primitive, SUPERCELL_MATRIX)
mof_cell_len = mof.get_cell().lengths()
print(f"MOF Supercell built. Atoms: {len(mof)}. Cell: {mof_cell_len} Å")

# --- Cavity-Aware Placement ---
print("--- Cavity-Aware Placement ---")
# Use more sampling points to ensure better choice selection
pore_centers = get_pore_centers(mof, num_points=2000)

if len(pore_centers) < PEO_SEGMENTS + IL_MOLECULES + NA_IONS + CO2_MOLECULES:
    print(f"Warning: Only {len(pore_centers)} suitable pore centers found, which is less than the number of components ({PEO_SEGMENTS + IL_MOLECULES + NA_IONS + CO2_MOLECULES}). Overlap is likely.")
    
# Randomly select enough unique pore centers for all components
all_placement_centers = pore_centers[np.random.choice(len(pore_centers), PEO_SEGMENTS + IL_MOLECULES + NA_IONS + CO2_MOLECULES, replace=False)]

# Split the selected centers for each component
peo_centers = all_placement_centers[:PEO_SEGMENTS]
il_centers = all_placement_centers[PEO_SEGMENTS:PEO_SEGMENTS+IL_MOLECULES]
na_centers = all_placement_centers[PEO_SEGMENTS+IL_MOLECULES:PEO_SEGMENTS+IL_MOLECULES+NA_IONS]
co2_centers = all_placement_centers[PEO_SEGMENTS+IL_MOLECULES+NA_IONS:]

# --- 2. Build Shell/Gas Components (with random jitter) ---
print("--- 2. Building Shell/Gas Components ---")

polymer_shell = Atoms()
for pos in peo_centers:
    peo = make_peo_segment()
    # Add a small random offset (jitter) to the center to avoid perfect alignment clashes
    jitter = np.random.uniform(low=-PLACEMENT_RADIUS, high=PLACEMENT_RADIUS, size=3)
    peo.translate(pos + jitter - peo.get_center_of_mass())
    polymer_shell += peo

ionic_liquid = Atoms()
for pos in il_centers:
    il = make_il_mol()
    jitter = np.random.uniform(low=-PLACEMENT_RADIUS, high=PLACEMENT_RADIUS, size=3)
    il.translate(pos + jitter - il.get_center_of_mass())
    ionic_liquid += il

# Na⁺ Ions with random jitter
na_ions = Atoms('Na' * NA_IONS, positions=na_centers)
na_ions.positions += np.random.uniform(low=-PLACEMENT_RADIUS, high=PLACEMENT_RADIUS, size=(NA_IONS, 3))

# CO₂ Molecules with random jitter
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
print(f"Final System built. Total Atoms: {len(system)}. Final Box: {system.get_cell().lengths()} Å")

# --- 4. LAMMPS Setup ---
unique_elements = sorted(set(system.get_chemical_symbols()))
element_map = {el: i + 1 for i, el in enumerate(unique_elements)}
system.set_tags([element_map[symbol] for symbol in system.get_chemical_symbols()])

print("\n--- LAMMPS Setup ---")
print(f"Atom Type Mapping (Element: Type ID): {element_map}")
print("Atom type counts:", Counter(system.get_tags()))
assert np.all(np.isfinite(system.get_positions())), "Non-finite atom positions detected!"

# ROBUST STERICS CHECK: Minimum distance calculation using pdist
min_dist = np.min(pdist(system.get_positions()))
# 0.8 Å is a commonly accepted absolute minimum for MD (less than any bond length)
MIN_SAFE_DISTANCE = 0.8 
print(f"Minimum interatomic distance: {min_dist:.3f} Å")
if min_dist < MIN_SAFE_DISTANCE:
    raise AssertionError(f"FATAL: Overlapping atoms detected (min distance: {min_dist:.3f} Å). Must be > {MIN_SAFE_DISTANCE} Å to run LAMMPS.")
elif min_dist < 1.5:
    print("Warning: Minimum distance is below the common VDW safety threshold of 1.5 Å. LAMMPS may require a short energy minimization.")

# Write the LAMMPS data file
# Using write_lammps_data and specifying 'atomic' style which is standard
write_lammps_data('battery_system.data', system, atom_style='atomic') 
print("LAMMPS data file 'battery_system.data' written successfully.")

# --- 5. Metrics (Updated Distance Calculations) ---
total_mass = sum(system.get_masses())
total_volume = system.get_volume()
# Recalculating density using ASE's Ang constant for clarity, though your method was correct
density_g_per_cm3 = (total_mass * Ang**3) / (total_volume * 1e-24) / 1000

print("\n--- System Metrics ---")
print(f"Total Mass: {total_mass:.2f} u")
print(f"Total Volume: {total_volume:.2f} Å³")
print(f"Approximate System Density: {density_g_per_cm3:.4f} g/cm³")

component_counts = {
    "MOF": len(mof),
    "Polymer (PEO)": len(polymer_shell),
    "Ionic Liquid (IL)": len(ionic_liquid),
    "Sodium Ions (Na+)": len(na_ions),
    "CO2 Gas": len(co2_gas)
}
print("\n--- Component Breakdown ---")
for name, count in component_counts.items():
    print(f"- {name}: {count} atoms")

# --- Index Offsets ---
# These are correct for isolating components in the combined 'system'
offset_mof = 0
offset_peo = offset_mof + len(mof)
offset_il  = offset_peo + len(polymer_shell)
offset_na  = offset_il + len(ionic_liquid)
offset_co2 = offset_na + len(na_ions)
final_index = offset_co2 + len(co2_gas)

mof_indices = list(range(offset_mof, offset_peo))
peo_indices = list(range(offset_peo, offset_il))
na_indices = list(range(offset_na, offset_co2))
co2_indices = list(range(offset_co2, final_index))

# --- Na+ Proximity (Corrected Indexing) ---
print("\n--- Na+ Ion Proximity Check ---")
min_distances_to_mof = []
min_distances_to_peo = []
# Iterate over each Na+ ion index
for na_index in na_indices:
    # We only need the distances from one Na+ atom to all MOF/PEO atoms
    dist_mof = system.get_distances(na_index, mof_indices, mic=True, vector=False)
    dist_peo = system.get_distances(na_index, peo_indices, mic=True, vector=False)
    min_distances_to_mof.append(np.min(dist_mof))
    min_distances_to_peo.append(np.min(dist_peo))
print(f"Average Minimum Na+ to MOF Distance: {np.mean(min_distances_to_mof):.3f} Å")
print(f"Average Minimum Na+ to PEO Distance: {np.mean(min_distances_to_peo):.3f} Å")

# --- CO2 Proximity (Corrected Indexing) ---
print("\n--- CO2 Molecule Proximity Check ---")
min_distances_co2_mof = []
# Iterate over each CO2 atom index
for co2_atom_index in co2_indices:
    dist_mof = system.get_distances(co2_atom_index, mof_indices, mic=True, vector=False)
    min_distances_co2_mof.append(np.min(dist_mof))
print(f"Average Minimum CO2 (atom) to MOF Distance: {np.mean(min_distances_co2_mof):.3f} Å")


# --- Visualization ---
print("\n--- Visualization ---")
# (Visualization code remains correct as it uses the same offsets/masks)
positions = system.get_positions()
fig, ax = plt.subplots(figsize=(8, 8))
is_mof = np.arange(len(system)) < offset_peo
is_na = (np.arange(len(system)) >= offset_na) & (np.arange(len(system)) < offset_co2)
is_co2 = np.arange(len(system)) >= offset_co2
is_other = ~(is_mof | is_na | is_co2)

# Plot MOF atoms
ax.scatter(positions[is_mof, 0], positions[is_mof, 1], s=1, color='gray', alpha=0.3, label='MOF')
# Plot Na+ ions (critical for conductivity)
ax.scatter(positions[is_na, 0], positions[is_na, 1], s=10, color='blue', alpha=0.8, label='Na+')
# Plot CO2 gas (critical for selectivity/uptake)
ax.scatter(positions[is_co2, 0], positions[is_co2, 1], s=5, color='red', alpha=0.5, label='CO2')
# Plot PEO/IL atoms (electrolyte)
ax.scatter(positions[is_other, 0], positions[is_other, 1], s=0.5, color='green', alpha=0.1, label='PEO/IL')

ax.set_title(f"Atom Distribution (XY-Plane Projection) - Total Atoms: {len(system)}")
ax.set_xlabel("X-coordinate (Å)")
ax.set_ylabel("Y-coordinate (Å)")
ax.set_aspect('equal', adjustable='box')
ax.legend(markerscale=5)
plt.savefig('system_distribution_colorcoded.png')
print("Color-coded atom distribution plot saved as 'system_distribution_colorcoded.png'.")

#view(system)
