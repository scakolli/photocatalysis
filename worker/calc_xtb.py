import sys
import os

# sys.path.insert(1, '/home/btpq/bt308495/Thesis/')
from photocatalysis.evaluate import evaluate_substrate

CALC_PARAMS = {'gfn':2, 'acc':0.2, 'etemp':298.15, 'gbsa':'water'}
SCRATCH_DIR = '/home/btpq/bt308495/Thesis/scratch'

if __name__ == '__main__':

    # mol_xxx.smi file as input
    system_to_calculate = sys.argv[1]
    with open(system_to_calculate) as out:
        smi = out.readlines()[0]

    ip, rdg, asites, rds, essi = evaluate_substrate(smi, CALC_PARAMS, scratch_dir=SCRATCH_DIR)

    os.system('echo "{} {} {}" >> results.txt'.format(smi, ip, rdg, asites, rds, essi))
    os.system('echo "{} {} {}" >> ../results_calculations.txt'.format(smi, ip, rdg, asites, rds, essi))