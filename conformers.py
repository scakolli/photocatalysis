""" Conformer generation. """

__author__ = "Steven Kearnes"
__copyright__ = "Copyright 2014, Stanford University"
__license__ = "3-clause BSD"

# Changes: Christian Kunkel
# The used conformer generator is part of the deepchem project, Copyright 2017 PandeLab

import time,os
import numpy as np
import multiprocessing as mp
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem.rdMolDescriptors import CalcNumRotatableBonds
from osc_discovery.cheminformatics.cheminformatics_misc import rdkit2ase


def _getrmsd(tuple_proc):
    ''' Helper for multiprocessing '''
    mol = tuple_proc[2]
    return [tuple_proc[0][0], tuple_proc[0][1], 
                AllChem.GetBestRMS(mol, mol, tuple_proc[1][0], tuple_proc[1][1]) ]


class ConformerGenerator(object):
  """
  Generate molecule conformers.

  Procedure
  ---------
  1. Generate a pool of conformers.
  2. Minimize conformers.
  3. Prune conformers using an RMSD threshold.

  Note that pruning is done _after_ minimization, which differs from the
  protocol described in the references.

  References
  ----------
  * http://rdkit.org/docs/GettingStartedInPython.html
    #working-with-3d-molecules
  * http://pubs.acs.org/doi/full/10.1021/ci2004658

  Parameters
  ----------
  max_conformers : int, optional (default 1)
      Maximum number of conformers to generate (after pruning).
  rmsd_threshold : float, optional (default 0.5)
      RMSD threshold for pruning conformers. If None or negative, no
      pruning is performed.
  force_field : str, optional (default 'uff')
      Force field to use for conformer energy calculation and
      minimization. Options are 'uff', 'mmff94', and 'mmff94s'.
  pool_multiplier : int, optional (default 10)
      Factor to multiply by max_conformers to generate the initial
      conformer pool. Since conformers are pruned after energy
      minimization, increasing the size of the pool increases the chance
      of identifying max_conformers unique conformers.
  """

  def __init__(self,
               max_conformers=1,
               rmsd_threshold=0.5,
               n_cpu=1,
               force_field='uff',
               pool_multiplier=10,
               print_output=True):
    self.max_conformers = max_conformers
    if rmsd_threshold is None or rmsd_threshold < 0:
      rmsd_threshold = -1.
    self.rmsd_threshold = rmsd_threshold
    self.force_field = force_field
    self.pool_multiplier = pool_multiplier
    self.n_cpu = n_cpu
    self.print_output = print_output

  def __call__(self, mol):
    """
    Generate conformers for a molecule.

    Parameters
    ----------
    mol : RDKit Mol
        Molecule.
    """
    return self.generate_conformers(mol)

  def generate_conformers(self, mol):
    """
    Generate conformers for a molecule.

    This function returns a copy of the original molecule with embedded
    conformers.

    Parameters
    ----------
    mol : RDKit Mol
        Molecule.
    """
    
    # initial embedding
    time_start=time.time()
    if self.print_output: print('1) Embedding conformers ({}). Using {} processes'.\
                     format(self.max_conformers * self.pool_multiplier, self.n_cpu))
    mol = self.embed_molecule(mol, userandom=False)

    if not mol.GetNumConformers():
      # For large molecules, default can fail: https://github.com/rdkit/rdkit/issues/1593
      # resorting to random
      msg = 'No conformers generated for molecule (useRandomCoords=False)'
      if self.print_output: print('No conformers generated for molecule, useRandomCoords=False, now setting it to True')
      os.system('touch using_random_embedding.txt')

      mol = self.embed_molecule(mol, userandom=True)
      if not mol.GetNumConformers():
          msg = 'No conformers generated for molecule (useRandomCoords=True)'
          if mol.HasProp('_Name'):
            name = mol.GetProp('_Name')
            msg += ' "{}".'.format(name)
          else:
            msg += '.'
          raise RuntimeError(msg)
    if self.print_output: print('Took: {} s'.format(time.time()-time_start))
    
    # minimization and pruning
    time_start=time.time()
    if self.print_output: print('2) Minimizing {} conformers ({})'.format(mol.GetNumConformers() ,self.force_field))
    self.minimize_conformers(mol)
    if self.print_output: print('Took: {} s'.format(time.time()-time_start))

    time_start=time.time()
    if self.print_output: print('3) Pruning conformers (RMSD threshold: {} Ang)'.format(self.rmsd_threshold))
    mol = self.prune_conformers(mol)
    if self.print_output: print('Took: {} s'.format(time.time()-time_start))

    return mol

  def embed_molecule(self, mol, userandom=False):
    """
    Generate conformers, possibly with pruning.

    Parameters
    ----------
    mol : RDKit Mol
        Molecule.
    """
    from rdkit import Chem
    from rdkit.Chem import AllChem
    mol = Chem.AddHs(mol)  # add hydrogens
    n_confs = self.max_conformers * self.pool_multiplier
    # ETKDG was default, see also https://github.com/rdkit/UGM_2015/blob/master/Presentations/ETKDG.SereinaRiniker.pdf
    # good intro also: https://github.com/rdkit/rdkit/blob/6d1266615bec4c18fe60c7cc2d8673324a5c1d6f/Docs/Book/GettingStartedInC%2B%2B.md#ring-information
    AllChem.EmbedMultipleConfs(mol, numConfs=n_confs, pruneRmsThresh=-1., numThreads=self.n_cpu, randomSeed=42, useRandomCoords=userandom )
    return mol

  def get_molecule_force_field(self, mol, conf_id=None, **kwargs):
    """
    Get a force field for a molecule.

    Parameters
    ----------
    mol : RDKit Mol
        Molecule.
    conf_id : int, optional
        ID of the conformer to associate with the force field.
    kwargs : dict, optional
        Keyword arguments for force field constructor.
    """
    from rdkit.Chem import AllChem
    if self.force_field == 'uff':
      ff = AllChem.UFFGetMoleculeForceField(mol, confId=conf_id, **kwargs)
    elif self.force_field.startswith('mmff'):
      AllChem.MMFFSanitizeMolecule(mol)
      mmff_props = AllChem.MMFFGetMoleculeProperties(
          mol, mmffVariant=self.force_field)
      ff = AllChem.MMFFGetMoleculeForceField(
          mol, mmff_props, confId=conf_id, **kwargs)
    else:
      raise ValueError("Invalid force_field " +
                       "'{}'.".format(self.force_field))
    return ff

  def minimize_conformers(self, mol):
    """
    Minimize molecule conformers.

    Parameters
    ----------
    mol : RDKit Mol
        Molecule.
    """
    for conf in mol.GetConformers():
      ff = self.get_molecule_force_field(mol, conf_id=conf.GetId())
      ff.Minimize()

  def get_conformer_energies(self, mol):
    """
    Calculate conformer energies.

    Parameters
    ----------
    mol : RDKit Mol
        Molecule.

    Returns
    -------
    energies : array_like
        Minimized conformer energies.
    """
    energies = []
    for conf in mol.GetConformers():
      ff = self.get_molecule_force_field(mol, conf_id=conf.GetId())
      energy = ff.CalcEnergy()
      energies.append(energy)
    energies = np.asarray(energies, dtype=float)
    return energies

  def prune_conformers(self, mol):
    """
    Prune conformers from a molecule using an RMSD threshold, starting
    with the lowest energy conformer.

    Parameters
    ----------
    mol : RDKit Mol
        Molecule.

    Returns
    -------
    A new RDKit Mol containing the chosen conformers, sorted by
    increasing energy.
    """
    if self.rmsd_threshold < 0 or mol.GetNumConformers() <= 1:
      return mol
    energies = self.get_conformer_energies(mol)
    if self.n_cpu<=1: rmsd = self.get_conformer_rmsd(mol)
    else: rmsd = self.get_conformer_rmsd_multiproc(mol, n_cpu=self.n_cpu)

    sort = np.argsort(energies)  # sort by increasing energy
    keep = []  # always keep lowest-energy conformer
    discard = []
    for i in sort:
      # always keep lowest-energy conformer
      if len(keep) == 0:
        keep.append(i)
        continue

      # discard conformers after max_conformers is reached
      if len(keep) >= self.max_conformers:
        discard.append(i)
        continue

      # get RMSD to selected conformers
      this_rmsd = rmsd[i][np.asarray(keep, dtype=int)]

      # discard conformers within the RMSD threshold
      if np.all(this_rmsd >= self.rmsd_threshold):
        keep.append(i)
      else:
        discard.append(i)

    # create a new molecule to hold the chosen conformers
    # this ensures proper conformer IDs and energy-based ordering
    from rdkit import Chem
    new = Chem.Mol(mol)
    new.RemoveAllConformers()
    conf_ids = [conf.GetId() for conf in mol.GetConformers()]
    for i in keep:
      conf = mol.GetConformer(conf_ids[i])
      new.AddConformer(conf, assignId=True)
    return new

  @staticmethod
  def get_conformer_rmsd(mol):
    """
    Calculate conformer-conformer RMSD.

    Parameters
    ----------
    mol : RDKit Mol
        Molecule.
    """
    from rdkit.Chem import AllChem
    rmsd = np.zeros(
        (mol.GetNumConformers(), mol.GetNumConformers()), dtype=float)
    for i, ref_conf in enumerate(mol.GetConformers()):
      for j, fit_conf in enumerate(mol.GetConformers()):
        if i >= j:
          continue
        rmsd[i, j] = AllChem.GetBestRMS(mol, mol, ref_conf.GetId(),
                                        fit_conf.GetId())
        rmsd[j, i] = rmsd[i, j]
    return rmsd

  @staticmethod
  def get_conformer_rmsd_multiproc(mol, n_cpu=1):
    """
    Calculate conformer-conformer RMSD using multiprocessing.

    Parameters
    ----------
    mol : RDKit Mol
        Molecule.
    """
    from rdkit.Chem import AllChem

    rmsd = np.zeros(
        (mol.GetNumConformers(), mol.GetNumConformers()), dtype=float)

    tuples_proc =[]

    for i, ref_conf in enumerate(mol.GetConformers()):
        for j, fit_conf in enumerate(mol.GetConformers()):
            if i >= j: continue
            tuples_proc.append( [ [i,j], [ref_conf.GetId(),fit_conf.GetId()], mol ] )

    pool = mp.Pool(processes=n_cpu)
    results = pool.map(_getrmsd, tuples_proc)

    for tup in results:
        rmsd[tup[0], tup[1]] = tup[2]
        rmsd[tup[1], tup[0]] = tup[2]

    return rmsd


def get_conformers_rdkit(smi, n_cpu=1, max_conformers=-1, rmsd_threshold=0.35, ff='mmff94', pool_multiplier=1, print_output=True):
    """ Wrapper for ConformerGenerator_custom
    Implements conformer generation using the ideas laid out in
    https://pubs-acs-org.eaccess.ub.tum.de/doi/pdf/10.1021/ci2004658
    """
    
    mol = Chem.AddHs(Chem.MolFromSmiles(smi))
    n_rot_bonds=CalcNumRotatableBonds(mol)
    if print_output: print('Number of rotatable bonds: {}, {}'.format(n_rot_bonds, smi))
    
    # determine number of rotatable bonds using rule stated in article above
    if max_conformers==-1:
        if n_rot_bonds<=7: max_conformers=50
        elif n_rot_bonds>=8 and n_rot_bonds<=12: max_conformers=200
        else: max_conformers=300
    
    cg = ConformerGenerator(max_conformers=max_conformers, n_cpu=n_cpu, 
                            rmsd_threshold=rmsd_threshold, force_field=ff, 
                            pool_multiplier=pool_multiplier,
                            print_output=print_output)
    mol = cg(mol)
    
    atoms_list=[]
    for i in range(mol.GetNumConformers()):
        atoms_list.append(rdkit2ase(mol, confId=i))
        
    #    ff_energies=cg.get_conformer_energies(mol)
    if print_output: print('Number of conformers found: {}'.format(len(atoms_list)))
    return atoms_list
