import numpy as np
from ase.units import Hartree, Bohr
from ase.io import read, Trajectory
from glob import glob
from itertools import takewhile, dropwhile
import logging
import sys

MAX_MULTI_PROCESS = 42 # Maximum number of cores allowable for multiprocessing

def parse_wall_cpu_time(string):
    walltime_flag = False
    cputime_flag = False
    for line in string.splitlines():
        if 'wall-time' in line and not walltime_flag:
            # e = float(line.split()[-3]) * Hartree
            walltime = float(line.split()[-2])
            walltime_flag = True
        if 'cpu-time' in line and not cputime_flag:
            # e = float(line.split()[-3]) * Hartree
            cputime = float(line.split()[-2])
            cputime_flag = True
    return walltime, cputime

def parse_energies(string):
    energy_parsed = False
    for line in string.splitlines():
        if 'TOTAL' in line and not energy_parsed:
            # Total Energy in eV
            e = float(line.split()[-3]) * Hartree
            energy_parsed = True
    return e

def parse_ipea_homolumo(string):
    """Parse IP/EA aswell as HOMO/LUMO for comparison
    First HOMO/LUMO instance in stdout corresponds to neutral molecule as calculated with IPEA parametetized GFN1-xTB.
    HOMO/LUMO with regular GFN1-xTB differs but since IPEA parameterized version is better suited for IP/EA,
    it might correspondingly better descirbe HOMO/LUMO"""
    homo_parsed, lumo_parsed = False, False
    for line in string.splitlines():
        if 'delta SCC IP (eV)' in line:
            # ionization potential in eV
            ip = float(line.split()[-1])
        elif 'delta SCC EA (eV)' in line:
            # electron affinity in eV
            ea = float(line.split()[-1])
        elif '(HOMO)' in line and not homo_parsed:
            # KS Homo in eV
            homo = float(line.split()[-2])
            homo_parsed = True
        elif '(LUMO)' in line and not lumo_parsed:
            # KS Lumo in eV
            lumo = float(line.split()[-2])
            lumo_parsed = True

    return ip, ea, homo, lumo

def parse_hess(string):
    for line in string.splitlines():
        if 'zero point energy' in line:
            # Zero point energy (in eV)
            zpe = float(line.split()[4]) * Hartree
        elif 'TOTAL ENERGY' in line:
            # Total electronic energy E (in eV)
            energy = float(line.split()[3]) * Hartree
        elif 'TOTAL ENTHALPY' in line:
            # E + ZPE + U(T) + pV = E + ZPE + H(T)
            total_enthalpy = float(line.split()[3]) * Hartree
            enthalpy = total_enthalpy - energy - zpe
        elif 'TOTAL FREE ENERGY' in line:
            # E + ZPE + H(T) - TS(T)
            total_free_energy = float(line.split()[4]) * Hartree
            entropy = total_enthalpy - total_free_energy

            # Equivalently 'G(RRHO) w/o ZPVE' in stdout
            free_energy = enthalpy - entropy

    return zpe, enthalpy, entropy, free_energy

def parse_charges():
    # Look in directory for the 'charges' file
    with open('charges') as f:
        charges = [float(chg) for chg in f.readlines()]
    return charges

def parse_fukui_indices(string):
    # Parse fukui indices for each atom
    assert 1 > 2, "Debug Fukui parsing"
    lines = string.splitlines()
    fukui_printout_start = dropwhile(lambda line: "#        f(+)     f(-)     f(0)" not in line, lines)
    fki = [float(l.split()[-1]) for l in list(takewhile(lambda line: "-----" not in line, lines))[1:]]
    return fki

def parse_stdoutput(xtb_output, runtype):
    # Standard output from xtb call is parsed according to runtype
    d = dict()
    if runtype == 'sp' or 'opt' in runtype:
        d['energy'] = parse_energies(xtb_output)
    elif runtype == 'hess':
        d['zpe'], d['enthalpy'], d['entropy'], d['free_energy'] = parse_hess(xtb_output)
    elif 'ohess' in runtype:
        d['energy'] = parse_energies(xtb_output)
        d['zpe'], d['enthalpy'], d['entropy'], d['free_energy'] = parse_hess(xtb_output)
    elif runtype == 'vipea':
        d['ip'], d['ea'], d['ehomo'], d['elumo'] = parse_ipea_homolumo(xtb_output)
    elif runtype == 'vfukui':
        d['fukui'] = parse_fukui_indices(xtb_output)
    
    d['walltime'], d['cputime'] = parse_wall_cpu_time(xtb_output)

    return d

def xtboptlog_to_ase_trajectory(optimization_log_path, trajectory_log_path):
    # Convert an optimization log generated by 'xtb --opt' to an ASE trajectory .traj file
    # for easy viewing with 'ase gui *.traj' in commandline
    opt_steps = read(optimization_log_path, format='xyz', index=':')
    # trajectory_log_path = optimization_log_path.replace('.log', '.traj')

    with Trajectory(trajectory_log_path, 'w') as t:
        for step in opt_steps:
            t.write(step)

def create_trajectories_from_logs(path):
    # Look for xtbopt.log files in 'path' and create .traj files
    # Very messy... creates opt{job_num}.traj one directory back
    path_of_optlogs = glob(f"{path}/**/xtbopt.log", recursive=True) # absolute paths
    parsing_indices = [(p.find('run_'), p[(p.find('run_')+4):(p.find('/xtbopt.log'))]) for p in path_of_optlogs]

    for i, p in zip(parsing_indices, path_of_optlogs):
        traj_path = p[:i[0]] + 'opt' +  i[1] + '.traj'
        xtboptlog_to_ase_trajectory(p, traj_path)

def get_logger():
    logger_ = logging.getLogger()
    logger_.setLevel(logging.INFO)
    if not logger_.handlers:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(logging.Formatter('%(asctime)s | %(levelname)s: %(message)s'))
        logger_.addHandler(console_handler)
    return logger_

def explicitly_broadcast_to(shape, *gs_expres):
    # Fillout arrays
    out = []
    for g in gs_expres:
        out.append(np.broadcast_to(g, shape))
    return tuple(out)

def get_batches(lst, num_batches):
    return [lst[i:i + num_batches] for i in range(0, len(lst), num_batches)]

def get_multi_process_cores(num_jobs, multi_process):
    # Returns number_of_cores
    if multi_process == -1:
        # Use maximum avaliable cores
        if num_jobs < MAX_MULTI_PROCESS:
            # dont commit more cores than jobs... adds overhead
            multi_process = num_jobs
        else:
            # use user selected max number of cores
            multi_process = MAX_MULTI_PROCESS
    else:
        pass

    return multi_process