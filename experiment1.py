import numpy as np
import matplotlib.pyplot as plt
from ase.io import read, write
from ase.visualize import view
from ase.build import make_supercell, molecule, bulk
from ase.geometry import analysis
from ase import Atoms
from ase.units import Ang

# --- Configuration Parameters ---
MOF_FILENAME = 'UiO-66-NH2.cif'
SUPERCELL_MATRIX = [[2, 0, 0], [0, 2, 0], [0, 0, 2]]
PEO_SEGMENTS = 30   # Number of simplified C2H4O segments
IL_MOLECULES = 20   # Number of simplified EMIM molecules (C6H11N2)
NA_IONS = 10       # Number of Na+ ions
CO2_MOLECULES = 10 # Number of CO2 gas molecules
BUFFER_SIZE = 5.0  # 5 Å buffer on all sides of the MOF for the simulation box

# --- 1. MOF Component Preparation ---
print("--- 1. MOF Preparation ---")
try:
    mof_primitive = read(MOF_FILENAME)
except FileNotFoundError:
    print(f"ERROR: {MOF_FILENAME} not found. Ensure it is in the same directory.")
    exit()

# Build the supercell
mof = make_supercell(mof_primitive, SUPERCELL_MATRIX)
mof_cell = mof.get_cell()
mof_cell_len = mof_cell.lengths()
print(f"MOF Supercell built. Atoms: {len(mof)}. Cell: {mof_cell_len} Å")

# Define a bounding box for random placement based on the MOF supercell
min_coords = np.array([0, 0, 0])
max_coords = mof_cell_len

# --- 2. Generate Other Components (PEO, IL, Na+, CO2) ---
print("--- 2. Building Shell/Gas Components ---")

# Polymer Coating (PEO segments)
polymer_segments = []
for _ in range(PEO_SEGMENTS):
    peo = molecule('C2H4O')
    peo.translate(np.random.uniform(low=min_coords, high=max_coords, size=3))
    polymer_segments.append(peo)
polymer_shell = sum(polymer_segments)

# Ionic Liquid Molecules (EMIM placeholder)
il_molecules = []
for _ in range(IL_MOLECULES):
    emim = molecule('C6H11N2')
    emim.translate(np.random.uniform(low=min_coords, high=max_coords, size=3))
    il_molecules.append(emim)
ionic_liquid = sum(il_molecules)

# Na+ Ions
na_positions = np.random.uniform(low=min_coords, high=max_coords, size=(NA_IONS, 3))
na_ions = Atoms('Na' * len(na_positions), positions=na_positions)

# CO₂ Molecules
co2_molecules = []
for _ in range(CO2_MOLECULES):
    co2 = molecule('CO2')
    co2.translate(np.random.uniform(low=min_coords, high=max_coords, size=3))
    co2_molecules.append(co2)
co2_gas = sum(co2_molecules)

# --- 3. Combine Components and Define Box ---

# Combine All Components
system = mof + polymer_shell + ionic_liquid + na_ions + co2_gas

# Set the cell of the combined system with a buffer
new_cell_len = mof_cell_len + 2 * BUFFER_SIZE
system.set_cell(new_cell_len, scale_atoms=False)
system.set_pbc(True) # Set periodic boundary conditions

# Center the MOF within the new, larger box
system.center()
print(f"Final System built. Total Atoms: {len(system)}. Final Box: {system.get_cell().lengths()} Å")

# --- 4. LAMMPS Preparation and File Output ---

# Assign Atom Types for LAMMPS
unique_elements = sorted(set(system.get_chemical_symbols()))
element_map = {el: i + 1 for i, el in enumerate(unique_elements)}
system.set_tags([element_map[symbol] for symbol in system.get_chemical_symbols()])

print("\n--- LAMMPS Setup ---")
print(f"Atom Type Mapping (Element: Type ID): {element_map}")

# Write LAMMPS data file
try:
    write('battery_system.data', system, format='lammps-data')
    print("LAMMPS data file 'battery_system.data' written successfully.")
except Exception as e:
    print(f"Error writing LAMMPS file: {e}")

# --- 5. New Metrics and Visualization ---

# Calculate System Density
total_mass = sum(system.get_masses()) # Mass in atomic mass units (u)
total_volume = system.get_volume() # Volume in Å³
# Density in g/cm³
density_g_per_cm3 = (total_mass * Ang**3) / (total_volume * 1e-24) / 1000

print("\n--- System Metrics ---")
print(f"Total Mass: {total_mass:.2f} u")
print(f"Total Volume: {total_volume:.2f} Å³")
print(f"Approximate System Density: {density_g_per_cm3:.4f} g/cm³")

# Component Atom Count Metric
component_counts = {
    "MOF": len(mof),
    "Polymer (PEO)": len(polymer_shell),
    "Ionic Liquid (IL)": len(ionic_liquid),
    "Sodium Ions (Na+)": len(na_ions),
    "CO2 Gas": len(co2_gas)
}

print("\n--- Component Breakdown ---")
for component, count in component_counts.items():
    print(f"- {component}: {count} atoms")

# --- Na+ Ion Proximity Metric (Hypothesis Focus: Ion Coordination) ---
print("\n--- Na+ Ion Proximity Check ---")

na_positions = na_ions.get_positions()
mof_positions = mof.get_positions()
polymer_positions = polymer_shell.get_positions()

# 1. Na+ to MOF distance
min_distances_to_mof = []
for pos in na_positions:
    distances = system.get_distances(
        a=[atom.index for atom in na_ions if np.allclose(atom.position, pos)], 
        b=[atom.index for atom in mof], 
        mic=True
    )[0]
    min_distances_to_mof.append(np.min(distances))

avg_na_mof_dist = np.mean(min_distances_to_mof)
print(f"Average Minimum Na+ to MOF Distance: {avg_na_mof_dist:.3f} Å")

# 2. Na+ to PEO (Polymer) distance
min_distances_to_peo = []
if len(polymer_positions) > 0:
    for pos in na_positions:
        distances = system.get_distances(
            a=[atom.index for atom in na_ions if np.allclose(atom.position, pos)],
            b=[atom.index for atom in polymer_shell],
            mic=True
        )[0]
        min_distances_to_peo.append(np.min(distances))
    
    avg_na_peo_dist = np.mean(min_distances_to_peo)
    print(f"Average Minimum Na+ to PEO Distance: {avg_na_peo_dist:.3f} Å")

# --- CO2 Proximity Metric (Hypothesis Focus: Adsorption) ---
print("\n--- CO2 Molecule Proximity Check ---")

co2_positions = co2_gas.get_positions()

min_distances_co2_mof = []
for pos in co2_positions:
    distances = system.get_distances(
        a=[atom.index for atom in co2_gas if np.allclose(atom.position, pos)],
        b=[atom.index for atom in mof],
        mic=True
    )[0]
    min_distances_co2_mof.append(np.min(distances))

avg_co2_mof_dist = np.mean(min_distances_co2_mof)
print(f"Average Minimum CO2 to MOF Distance: {avg_co2_mof_dist:.3f} Å")


# --- Visualization Plot (Sanity Check) ---
print("\n--- Visualization ---")
print("Generating color-coded distribution plot...")

fig, ax = plt.subplots(figsize=(8, 8))
positions = system.get_positions()
symbols = system.get_chemical_symbols()

# Create masks for the main components
mof_indices = set(mof.indices)
na_indices = set(na_ions.indices)
co2_indices = set(co2_gas.indices)

# Assign components based on index sets
is_mof = np.array([i in mof_indices for i in system.indices])
is_na = np.array([i in na_indices for i in system.indices])
is_co2 = np.array([i in co2_indices for i in system.indices])
# Everything else is PEO/IL
is_other = ~is_mof & ~is_na & ~is_co2

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

# view(system) # Uncomment this to view the final structure interactively
