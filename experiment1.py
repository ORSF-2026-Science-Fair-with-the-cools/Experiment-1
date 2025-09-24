from ase.io import read, write
from ase.visualize import view
from ase.build import make_supercell, molecule
from ase.geometry import analysis
from ase import Atoms
import numpy as np

# Read the .cif file to get the desired MOF
atoms = read('UiO-66-NH2.cif')  # Replace with your actual CIF filename

# Measuring info
print("Number of atoms:", len(atoms))
print("Cell parameters (Å):", atoms.cell.lengths())
print("Cell angles (°):", atoms.cell.angles())
print("Chemical symbols:", atoms.get_chemical_symbols())

mof = read('UiO-66-NH2.cif')
mof = make_supercell(mof, [[2, 0, 0], [0, 2, 0], [0, 0, 2]])

# Generate Polymer Coating (PEO segments)
polymer_segments = []
for i in range(30):
    peo = molecule('C2H4O')  # Simplified PEO repeat unit
    peo.translate(np.random.uniform(low=0, high=50, size=3))
    polymer_segments.append(peo)
polymer_shell = sum(polymer_segments)

# Add Ionic Liquid Molecules (EMIM placeholder)
il_molecules = []
for i in range(20):
    emim = molecule('C6H11N2')  # Placeholder for EMIM
    emim.translate(np.random.uniform(low=0, high=50, size=3))
    il_molecules.append(emim)
ionic_liquid = sum(il_molecules)

# Add Na⁺ Ions
na_positions = np.random.uniform(low=0, high=50, size=(10, 3))
na_ions = Atoms('Na' * len(na_positions), positions=na_positions)

# Add CO₂ Molecules
co2_molecules = []
for i in range(10):
    co2 = molecule('CO2')
    co2.translate(np.random.uniform(low=0, high=50, size=3))
    co2_molecules.append(co2)
co2_gas = sum(co2_molecules)

# Combine All Components
system = mof + polymer_shell + ionic_liquid + na_ions + co2_gas

# Assign Atom Types for LAMMPS
unique_elements = sorted(set(system.get_chemical_symbols()))
element_map = {el: i + 1 for i, el in enumerate(unique_elements)}
system.set_tags([element_map[symbol] for symbol in system.get_chemical_symbols()])
print("LAMMPS atom types:", element_map)

# Write up a file for LAMMPS
write('battery.data', system, format='lammps-data')

# Look at our little guy
view(system)
