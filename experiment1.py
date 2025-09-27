
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import cKDTree
from ase import Atoms
from ase.io import read, write
from ase.visualize import view
from ase.build import make_supercell, molecule
from ase.units import _amu
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

# --- Helper Functions ---
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

def get_pore_centers(mof, num_points=1000, min_distance=3.0):
    """Return candidate pore centers by sampling points far from MOF atoms."""
    cell = mof.get_cell()
    positions = mof.get_positions()
    tree = cKDTree(positions)

    pore_centers = []
    for _ in range(num_points * 2):  # oversample
        point = np.random.uniform([0, 0, 0], cell.lengths())
        dist, _ = tree.query(point)
        if dist > min_distance:
            pore_centers.append(point)
        if len(pore_centers) >= num_points:
            break
    return np.array(pore_centers)

# --- 1. MOF Preparation ---
print("--- 1. MOF Preparation ---")
mof_primitive = read(MOF_FILENAME)
mof = make_supercell(mof_primitive, SUPERCELL_MATRIX)
mof_cell_len = mof.get_cell().lengths()
print(f"MOF Supercell built. Atoms: {len(mof)}. Cell: {mof_cell_len} Å")

# --- Cavity-Aware Placement ---
print("--- Cavity-Aware Placement ---")
pore_centers = get_pore_centers(mof, num_points=1000)

min_coords = np.array([0, 0, 0])
max_coords = mof_cell_len

# --- 2. Build Shell/Gas Components ---
print("--- 2. Building Shell/Gas Components ---")
peo_positions = pore_centers[np.random.choice(len(pore_centers), PEO_SEGMENTS, replace=False)]

polymer_shell = Atoms()
for pos in peo_positions:
    peo = make_peo_segment()
    peo.translate(pos - peo.get_center_of_mass())
    polymer_shell += peo

il_positions = pore_centers[np.random.choice(len(pore_centers), IL_MOLECULES, replace=False)]

ionic_liquid = Atoms()
for pos in il_positions:
    il = make_il_mol()
    il.translate(pos - il.get_center_of_mass())
    ionic_liquid += il

# Na⁺ Ions near pore centers
na_positions = pore_centers[np.random.choice(len(pore_centers), NA_IONS, replace=False)]
na_ions = Atoms('Na' * NA_IONS, positions=na_positions)

# CO₂ Molecules near pore centers
co2_molecules = []
for pos in pore_centers[np.random.choice(len(pore_centers), CO2_MOLECULES, replace=False)]:
    co2 = molecule('CO2')
    co2.translate(pos - co2.get_center_of_mass())
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

from scipy.spatial.distance import pdist
min_dist = np.min(pdist(system.get_positions()))
print(f"\nMinimum interatomic distance: {min_dist:.3f} Å")
assert min_dist > 1.5, "Overlapping atoms detected. Aborting export..."

write_lammps_data('battery_system.data', system, atom_style='atomic')
print("LAMMPS data file 'battery_system.data' written successfully.")

# --- 5. Metrics ---
total_mass = sum(system.get_masses())
total_volume = system.get_volume()
density_g_per_cm3 = (total_mass * 1.66053906660e-24) / (total_volume * 1e-24)
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
offset_mof = 0
offset_peo = offset_mof + len(mof)
offset_il  = offset_peo + len(polymer_shell)
offset_na  = offset_il + len(ionic_liquid)
offset_co2 = offset_na + len(na_ions)

# --- Na+ Proximity ---
print("\n--- Na+ Ion Proximity Check ---")
min_distances_to_mof = []
min_distances_to_peo = []
for i in range(NA_IONS):
    na_index = offset_na + i
    mof_indices = list(range(offset_mof, offset_peo))
    peo_indices = list(range(offset_peo, offset_il))
    dist_mof = system.get_distances(na_index, mof_indices, mic=True)
    dist_peo = system.get_distances(na_index, peo_indices, mic=True)
    min_distances_to_mof.append(np.min(dist_mof))
    min_distances_to_peo.append(np.min(dist_peo))
print(f"Average Minimum Na+ to MOF Distance: {np.mean(min_distances_to_mof):.3f} Å")
print(f"Average Minimum Na+ to PEO Distance: {np.mean(min_distances_to_peo):.3f} Å")

# --- CO2 Proximity ---
print("\n--- CO2 Molecule Proximity Check ---")
min_distances_co2_mof = []
for i in range(len(co2_gas)):
    co2_index = offset_co2 + i
    mof_indices = list(range(offset_mof, offset_peo))
    dist = system.get_distances(co2_index, mof_indices, mic=True)
    min_distances_co2_mof.append(np.min(dist))
print(f"Average Minimum CO2 to MOF Distance: {np.mean(min_distances_co2_mof):.3f} Å")

# --- Visualization ---
print("\n--- Visualization ---")
print("Generating color-coded distribution plot...")
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

#view(system) # Uncomment this to view the final structure interactively
