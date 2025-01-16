import multiprocessing as mp
import numpy as np
from neuron import h
# import pandas as pd
import helperFuncs as hf
import simulation as sim
import pickle as pk
import os
# import sys


def generateERF(electrode):
    print('electrode: ', electrode, ' started.')
    
    ### --- setup biophysical simulaiton environment
    # setup cell
    RGC = sim.Local_Cell()
    filename = 'cell_param_files/params_35_v2.csv'
    RGC.build_cell(filename,'mammalian_spike_35')

    ### --- setup stimulus
    Stim = {}

    Stim['pulseShape'] = 'triphasic'
    Stim['dt'] = .005
    Stim['delay'] = 10
    Stim['dur'] = 0.05
    Stim['stop'] = 80
    Stim['amp'] = 2 # units: [uA]

    Stim['initDur'] = 50
    Stim['initDt'] = 1
    Stim['vInit'] = -70 # units: [mV]

    Stim['electrodes'] = [electrode]
    Stim['electrode_type'] = 'disk'
    Stim['rhoExt'] = 1000 # Ohm*cm
    Stim['elecDiam'] = 15 # um
    Stim['polarity'] = -1 # anodic == 1; cathodic == -1 

    Stim['pulseRatioTri'] = [-2/3,1,-1/3]
    Stim['pulseRatioBi'] = [1,-1]
    Stim['frequency'] = 10 # units: [Hz]

    # # define transfer impedances 
    # hf.setupTransferImpedance(Stim, RGC)

    # run simulation
    thr, t_vec, vRec = hf.lowerThreshold_v2(Stim, RGC, lower=0, upper=5, num=25, epsilon=0.02)

    ### --- output and save results
    print('ERF for electrode position: ', str(electrode), ' is: ', thr)
    
    filename = '/Volumes/Scratch/Users/vilkhu/sim-data/ERF/triphasic/fixed/cathodic_'+\
                str(electrode[0])+'_'+\
                str(electrode[1])+'_'+str(electrode[2])+'_erf.pkl'
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    data = np.asarray([thr, t_vec, vRec], dtype=object)

    # write compressed pickle file
    try: 
        hf.compressed_pickle(filename, data)
        print('Data saved.')
    except:
        print('Error writing file: ', filename)

def main(): 
    ### --- setup parallel processing
    pool = mp.Pool(mp.cpu_count() - 10)

    ### --- create a list of electrode locations
    electrodes = []

    # # -------- comment out elecMap for testing ---------
    # # load in electrode map for 30um array 
    # elecMap = hf.generate_electrode_map(519)
    # # define some fixed z location for the electrodes [um]
    # z = 79
    # for elec in elecMap:
    #     electrodes.append([float(elec[0]), float(elec[1]), float(z)])
    # # -------- comment out elecMap for testing ---------

    # -------- test electrodes ---------
    for x in range(31):
        electrodes.append([-1000, 0, 45+(x*0.5)])

    ### --- run parallel simulations for each electrode location
    result = pool.map(generateERF, electrodes)


if __name__ == "__main__":
    main()
