import multiprocessing as mp
import numpy as np
import neuron as h
import helperFuncs as hf
import simulation as sim
import os
from tqdm import tqdm

def expPillars(electrode):
    print('Testing electrode loc: ', electrode)
    
    ### --- setup biophysical simulaiton environment
    # setup cell
    RGC = sim.Local_Cell()
    filename = 'cell_param_files/params_35.csv'
    RGC.build_cell(filename,'mammalian_spike_35')

    ### --- setup stimulus
    Stim = {}

    Stim['pulseShape'] = 'biphasic'
    Stim['dt'] = .005 # units: [ms]
    Stim['delay'] = 10 # units: [ms]
    Stim['dur'] = 0.05 # units: [ms]
    Stim['stop'] = 80 # units: [ms]

    Stim['initDur'] = 50 # units: [ms]
    Stim['initDt'] = 1 # units: [ms]
    Stim['vInit'] = -70 # units: [mV]

    Stim['electrodes'] = [electrode]
    Stim['electrode_type'] = 'disk' # TODO: Change to 'disk' or 'point'
    Stim['rhoExt'] = 1000 # Ohm*cm
    Stim['elecDiam'] = 15 # um
    Stim['polarity'] = -1 # anodic == 1; cathodic == -1 

    Stim['pulseRatioTri'] = [-2/3,1,-1/3]
    Stim['pulseRatioBi'] = [1,-1]
    Stim['frequency'] = 10 # units: [Hz]

    # run simulation
    thr,t_vec,_ = hf.lowerThreshold_v2(Stim, RGC, lower=0, upper=5, num=20, epsilon=0.02)

    ### --- output and save results
    print('Threshold for electrode at position: ',\
            str(Stim['electrodes']),' is: ', thr)
    filename = '/Volumes/Scratch/Users/vilkhu/sim-data/pillar/biphasic/'+\
                str(Stim['electrodes'][0][0])+'_'+\
                str(Stim['electrodes'][0][1])+'_'+\
                str(Stim['electrodes'][0][2])+'.pkl'
    
    os.makedirs(os.path.dirname(filename), exist_ok=True)

    data = np.asarray([thr, t_vec], dtype=object)

    # write compressed pickle file
    try: 
        hf.compressed_pickle(filename, data)
        print('Data saved.')
    except:
        print('Error writing file: ', filename)

def main(): 
    ### --- setup parallel processing
    pool = mp.Pool(mp.cpu_count() - 10)

    ### --- create list of electrodes to test
    ### Note: Axon fiber layer, z = 40
    ###       Soma center location, x = -12, y = 0, z = 10
    ###       Distal axon extends in negative x direction
    ### We test x = -1000 --> distal axon
    ###          x = -65 --> SOCB
    ###          x = 0 --> Soma
    electrodes = []
    for x in [-1000,-65,0]:
        for y in [0]:
            for z in np.arange(10,60,5):
                electrodes.append([x,y,z])

    # # Collect data for single planer array
    # electrodes = []
    # for x in [-1000,-65,0]:
    #     for y in [0]:
    #         for z in [57]:
    #             electrodes.append([x,y,z])

    ### --- run parallel simulations for each electrode location
    result = list(tqdm(pool.imap(expPillars, electrodes), total=len(electrodes)))


if __name__ == "__main__":
    main()
