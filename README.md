# rgc_simulation_multielectrode

# Overview 
This is a simulation setup for epiretinal RGC stimulation developed as part of the Stanford Artificial Retina Project (contact: vilkhu@stanford.edu). Last updated: Aug 2024. Associated with the following paper: Ramandeep Vilkhu, et. al., "Understanding responses to multi-electrode epiretinal stimulation using a biophysical model" 

## Setup
1. On a linux machine, `git clone https://github.com/ramanvilkhu/rgc_simulation.git`
2. As a prerequisite, ensure you have Python (>=3.6) and pip
3. Install NEURON: `pip install neuron`
4. Navigate to the nrn directory: `cd rgc_simulation/nrn`
5. Once in the dir, compile the custom membrane mechanisms: `nrnivmodl`
6. Confirm this worked by checking to see if a `x86_64` directory was created.
7. Try stepping through `src/intro_script.ipynb` to ensure setup is complete.
