"""Thermodynamic Quantities of Interest (in eV)"""

"""E°(H2O/O2) Standard redox potential (in V relative to SHE)"""

WATER_OXIDATION_POTENTIAL = 1.23

"""Energy of SHE relative to vacuum, where electrons are at rest
E(H+/H2, in H2O @ STP 298.15K/1.01325bar)(SHE) = 0 V
E(H+/H2, in H2O @ STP 298.15K/1.01325bar)(absolute) = 4.44 V ± 0.02 

Reference: The absolute electrode potential: an explanatory note (Recommendations 1986)
S. Trasatti J. Electroanal. Chem. Interfacial Electrochem., 1986, 209 , 417 —428"""

SHE_VACUUM_POTENTIAL = 4.44

"""Static Free energy expression corrections (in eV)"""
dG1_CORR, dG2_CORR, dG3_CORR, dG4_CORR = 124.6464158599693, -13.44012642161362, 124.6464158599693, -230.93270529832492

"""DFT derived E and ZPE 
GFN2-xTB, fmax=0.005 eV/Angstrom, accuracy=0.2"""
# Energy
# E_H2 = -26.74025284322724
# E_H2O = -137.9765422815829

# ZPE DFT
# ZPE_H2_DFT = 0.2326865538281221
# ZPE_H2O_DFT = 0.5475951786796269

"""Referenced ZPE and S
Origin of the Overpotential for Oxygen Reduction at a Fuel-Cell Cathode (https://doi.org/10.1021/jp047349j)
H2(g) at 298.15k and 1bar (STP)
H2O(g) at 298.15K and 0.035bar (at equilibrium with liquid water at the same temp.)
So long as both these quanties are quoted w.r.t. same reference (i.e. STP), there shouldn't be problems when integrating
with DFT calculated energies, as we are only interested in taking differences of these quantites"""

# ZPE
# ZPE_H2 = 0.27
# ZPE_H2O = 0.56

# Gas Phase Entropies
# TS_H2_g = 0.41
# TS_H2O_g = 0.67

"""Experimental water splitting free energy change
Used to avoid problematic calculation of Oxygen gas free energy
G_O2g = 2 * dG_ws + 2 * (G_H2O_l - G_H2_g), where the free energy of G_H2O_l = G_H2O_g at 298.15K and 0.035bar
This as to avoid an inaccurate estimation of liquid free energy (hydrogen bond underestimation)"""

# dG_WS = 2.46 # per water molecule

"""Free Energy Constants
Since these variables are unchanging, the later half of each expression is constant, thus each reaction enthalpy (Δ(E + ZPE)) 
is simply modified by a constant + dG_corr

#dG1 = (E_OH + ZPE_OH) - (E_s + ZPE_s) + 0.5 * (E_H2 + ZPE_H2) - (E_H2O + ZPE_H2O) - (0.5 * TS_H2_g - TS_H2O_g)
#dG2 = (E_O + ZPE_O) - (E_OH + ZPE_OH) + 0.5 * (E_H2 + ZPE_H2) - 0.5 * TS_H2_g
#dG3 = (E_OOH + ZPE_OOH) - (E_O + ZPE_O) + 0.5 * (E_H2 + ZPE_H2) - (E_H2O + ZPE_H2O) - (0.5 * TS_H2_g - TS_H2O_g)
#dG4 = (E_s + ZPE_s) - (E_OOH + ZPE_OOH) + 2 * dG_WS + 2 * (E_H2O + ZPE_H2O) - 3/2 * (E_H2 + ZPE_H2) - (2 * TS_H2O_g - 3/2 * TS_H2_g)"""

# dG1_CORR = 0.5 * (E_H2 + ZPE_H2) - (E_H2O + ZPE_H2O) - (0.5 * TS_H2_g - TS_H2O_g)
# dG2_CORR = 0.5 * (E_H2 + ZPE_H2) - 0.5 * TS_H2_g
# dG3_CORR = 0.5 * (E_H2 + ZPE_H2) - (E_H2O + ZPE_H2O) - (0.5 * TS_H2_g - TS_H2O_g)
# dG4_CORR = 2 * dG_ws + 2 * (E_H2O + ZPE_H2O) - 3/2 * (E_H2 + ZPE_H2) - (2 * TS_H2O_g - 3/2 * TS_H2_g)