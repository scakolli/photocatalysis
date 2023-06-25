from adsorption.tools import ase2rdkit_valencies
from osc_discovery.descriptor_calculation.conformers import *

def get_conformers_rdkit_ase(asemol, n_cpu=1, max_conformers=-1, rmsd_threshold=0.35, ff='mmff94', pool_multiplier=1):
    """ Wrapper for ConformerGenerator_custom
    Implements conformer generation using the ideas laid out in
    https://pubs-acs-org.eaccess.ub.tum.de/doi/pdf/10.1021/ci2004658
    """
    mol = ase2rdkit_valencies(asemol)
    n_rot_bonds = CalcNumRotatableBonds(mol)
    print('Number of rotatable bonds: {}'.format(n_rot_bonds))

    # determine number of rotatable bonds using rule stated in article above
    if max_conformers == -1:
        if n_rot_bonds <= 7:
            max_conformers = 50
        elif n_rot_bonds >= 8 and n_rot_bonds <= 12:
            max_conformers = 200
        else:
            max_conformers = 300

    cg = ConformerGenerator(max_conformers=max_conformers, n_cpu=n_cpu,
                            rmsd_threshold=rmsd_threshold, force_field=ff,
                            pool_multiplier=pool_multiplier)
    mol = cg(mol)
    atoms_list = []
    for i in range(mol.GetNumConformers()):
        atoms_list.append(rdkit2ase(mol, confId=i))

    #    ff_energies=cg.get_conformer_energies(mol)
    print('Number of conformers found: {}'.format(len(atoms_list)))
    return atoms_list