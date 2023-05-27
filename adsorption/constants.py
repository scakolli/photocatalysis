from ase import Atoms
from ase.constraints import Hookean

### Define relaxed adsorbates as ASE Atoms situated at the origin
OH_POSITIONS = [(0., 0., 0.), (0., 0., 0.96348978)]
O_POSITIONS = [(0., 0., 0.)]
OOH_POSITIONS= [(0., 0., 0.), (0., 0., 1.30156180e+00), (-9.67021158e-17, 9.35524949e-01, 1.60833957e+00)]

OH = Atoms(symbols=['O', 'H'], positions=OH_POSITIONS)
O = Atoms(symbols=['O'], positions=O_POSITIONS)
OOH = Atoms(symbols=['O', 'O', 'H'], positions=OOH_POSITIONS)

### Define adsorbate constraints, for preserving their identity upon relaxation
# Hookean restoring forces conserve total energy
HOOKEAN_OH = Hookean(-len(OH), -len(OH)+1, rt=1.4, k=5) #constraint for O-H
HOOKEAN_OOH_A = Hookean(-len(OOH), -len(OOH)+1, rt=1.79, k=5) #constraint for O-O bond in OOH (C-O bond params used)
HOOKEAN_OOH_B = Hookean(-len(OOH)+1, -len(OOH)+2, rt=1.4, k=5) #constraint for O-H bond in OOH