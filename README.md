# Photocatalysis of the Oxygen Evolution Reaction

## Problem:
The efficient generation of clean hydrogen via water splitting is limited by the oxygen evolution reaction (OER), which suffers from sluggish adsorption kinetics and high energy barriers. Traditional inorganic catalysts often rely on scarce or environmentally harmful materials, motivating the search for organic photocatalysts that can both absorb light and catalyze the reaction more sustainably. However, experimentally discovering such materials is an enormous challenge due to the vast size of the chemical space.

## Methodolgy:
To address this, my research employed a computational screening approach that integrates first-principles electronic structure calculations and machine learning (ML)-driven search strategies. Using Density Functional Theory (DFT) and semi-empirical methods, we assessed the key thermodynamic and kinetic descriptors necessary for OER activity, specifically ionization potential (IP) and adsorption free energy (GRD). We then applied an active learning (AML) framework to efficiently explore a chemical space of p-type organic semiconductors, identifying high-performing photocatalysts while significantly reducing computational cost. Additionally, we utilized a generative ML model (Variational Autoencoder, VAE) to propose novel molecular structures without predefined design constraints.
<br />
<br />
The AML approach was inspired by Christian Kunkel's paper:
<br />
"Active discovery of organic semiconductors"
<br />
https://www.nature.com/articles/s41467-021-22611-4
<br />
<br />
The Generative ML approach was inspired by Rafael Gómez-Bombarelli paper:
<br />
"Automatic Chemical Design Using a Data-Driven Continuous Representation of Molecules"
<br />
https://pubs.acs.org/doi/10.1021/acscentsci.7b00572

## Results:
My approach proved highly effective: the AML strategy identified 39% of top-performing candidates while sampling only 1.8% of the total space, demonstrating its scalability and precision. Meanwhile, the VAE-based inverse design method successfully generated new organic photocatalysts, some with predicted properties surpassing their parent molecules. These results highlight the power of combining machine learning with computational chemistry to accelerate the discovery of sustainable photocatalysts for renewable energy applications.
