from ase.units import Hartree, Bohr

def parse_energies(string):
    for line in string.splitlines():
        if 'TOTAL' in line:
            # Total Energy in eV
            e = float(line.split()[-3]) * Hartree

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
        if 'delta SCC EA (eV)' in line:
            # electron affinity in eV
            ea = float(line.split()[-1])
        if '(HOMO)' in line and not homo_parsed:
            # KS Homo in eV
            homo = float(line.split()[-2])
            homo_parsed = True
        if '(LUMO)' in line and not lumo_parsed:
            # KS Lumo in eV
            lumo = float(line.split()[-2])
            lumo_parsed = True

    return ip, ea, homo, lumo

def parse_zpe(string):
    for line in string.splitlines():
        if 'zero point energy' in line:
            zpe_ = float(line.split()[4]) * Hartree # in eV
            return zpe_

def parse_stdoutput(xtb_output, runtype):
    # Standard output from xtb call is parsed according to runtype
    d = dict()
    if runtype == 'sp' or 'opt' in runtype:
        d['energy'] = parse_energies(xtb_output)
    elif runtype == 'hess':
        d['zpe'] = parse_zpe(xtb_output)
        # Thermo parsing
    elif runtype == 'vipea':
        d['ip'], d['ea'], d['ehomo'], d['elumo'] = parse_ipea_homolumo(xtb_output)

    return d