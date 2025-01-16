import multiprocessing as mp
import numpy as np
import neuron as h
import helperFuncs as hf
import simulation as sim
import os
from tqdm import tqdm

def expStrengthDuration(PW):
    print('Duration: ', PW, 'ms started.')
    
    ### --- setup biophysical simulaiton environment
    # setup cell
    RGC = sim.Local_Cell()
    filename = 'cell_param_files/params_35.csv'
    RGC.build_cell(filename,'mammalian_spike_35')

    # Shift cell to be centered at 0,0,0
    x_shift = -1000 + 997.8304687738416
    RGC.shift_cell_x_y_z(0+x_shift,0,0)

    ### --- setup stimulus
    Stim = {}

    Stim['pulseShape'] = 'biphasic'
    Stim['dt'] = .0005 # TODO: Changed from .005
    Stim['delay'] = 10
    Stim['dur'] = float(PW)
    Stim['stop'] = 80

    Stim['initDur'] = 50
    Stim['initDt'] = 1
    Stim['vInit'] = -70 # units: [mV]

    Stim['electrodes'] = [[-1000, 0, 42]] #[[-200, 0, 55]]
    Stim['electrode_type'] = 'disk'
    Stim['rhoExt'] = 1000 # Ohm*cm
    Stim['elecDiam'] = 10 # um
    Stim['polarity'] = -1 # anodic == 1; cathodic == -1 

    Stim['pulseRatioTri'] = [-2/3,1,-1/3]
    Stim['pulseRatioBi'] = [1,-1]
    Stim['frequency'] = 10 # units: [Hz]

    # define transfer impedances 
    hf.setupTransferImpedance(Stim, RGC)

    # run simulation
    # thr, t_vec, vRec = hf.lowerThreshold(Stim, RGC, 0, 20, 25, 0.02)
    thr, t_vec, vRec = hf.lowerThreshold(Stim, RGC, 0, 5, 25, 0.02)

    ### --- output and save results
    print('Threshold for ', str(Stim['elecDiam']), 'um electrode -- Position: ',\
            str(Stim['electrodes'][0]), ' PW: ', str(PW), ' is: ', thr)
    filename = '/Volumes/Lab/Users/vilkhu/data-backup/strength_duration/biphasic/paul_0.001ms/cathodicFirst_axonal/'+\
                str(PW)+'ms_'+\
                str(Stim['electrodes'][0][0])+'_'+\
                str(Stim['electrodes'][0][1])+'_'+\
                str(Stim['electrodes'][0][2])+'.pkl'
    
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
    pool = mp.Pool(20)

    ### --- biphasic/triphasic
    # PW = [0.05, 0.1, 0.2, 0.3, 0.5, 0.8, 1, 2, 3, 5, 8, 10]
    PW = [0.01,0.02,0.03,0.04,0.05,0.06,0.07,0.08,0.09,0.1,0.2,0.3,0.4,0.5]

    # # ### --- monophasic
    # # ### --- create a list of pulse widths to test; units: [ms]
    # PW = [0.1, 0.2, 0.5, 1, 2, 5, 10, 20, 50]

    ### --- run parallel simulations for each electrode location
    result = list(tqdm(pool.imap(expStrengthDuration, PW), total=len(PW)))


if __name__ == "__main__":
    main()
