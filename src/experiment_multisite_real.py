import multiprocessing as mp
import numpy as np
from neuron import h
# import pandas as pd
import helperFuncs as hf
import simulation as sim
import pickle as pk
import os
import tqdm  


def expMultisite(currents):
    ### --- setup biophysical simulaiton environment
    # setup cell
    RGC = sim.Local_Cell()
    filename = 'cell_param_files/params_35_v2.csv'
    RGC.build_cell(filename,'mammalian_spike_35')

    # Shift cell to be centered at 0,0,0
    x_shift = -1000 + 997.8304687738416
    RGC.shift_cell_x_y_z(0+x_shift,0,0)
    # RGC.shift_cell_x_y_z(0,0,0)

    ### --- setup stimulus
    Stim = {}

    Stim['pulseShape'] = 'monophasic'
    Stim['dt'] = .005 # TODO: Changed from .005
    Stim['delay'] = 10
    Stim['dur'] = 0.05 # [ms]
    Stim['stop'] = 70
    Stim['amp'] = [currents[0],currents[1]]

    Stim['initDur'] = 50
    Stim['initDt'] = 1
    Stim['vInit'] = -70 # units: [mV]

    Stim['electrodes'] = [[-1030,0,42],[-1000,0,42]]
    Stim['electrode_type'] = 'disk'
    Stim['rhoExt'] = 1000 # Ohm*cm
    Stim['elecDiam'] = 15 # um
    Stim['polarity'] = -1 # anodic == 1; cathodic == -1 

    Stim['pulseRatioTri'] = [-2/3,1,-1/3]
    Stim['pulseRatioBi'] = [1,-1]
    Stim['frequency'] = 10 # units: [Hz]


    # define transfer impedances 
    active_x = [-1015] # Old code, this doesn't do anything anymore
    hf.setupTransferImpedance_v3(Stim, RGC, active_x, limit_sites=False)

    ## -- set up recording vectors
    APC_axon = h.APCount(1, sec = RGC.axon)
    APC_axon.thresh = 0 

    # --- Track spike at axon ---
    vRec_axon = []
    for seg in RGC.axon:
        vRec_axon.append(h.Vector().record(RGC.axon(seg.x)._ref_v))
    t = h.Vector().record(h._ref_t)

    # # --- Track spike at socb ---
    # vRec_socb = []
    # for seg in RGC.SOCB:
    #     vRec_socb.append(h.Vector().record(RGC.SOCB(seg.x)._ref_v))
    # t = h.Vector().record(h._ref_t)

    ## -- run simulation
    tsvec, isvec = hf.setupStimulus_unit(Stim)
    isvec.play(h._ref_is_xtra, tsvec, 1) 

    h.init()
    h.tstop = Stim['initDur']
    h.dt = Stim['initDt']
    h.finitialize(Stim['vInit'])
    h.continuerun(h.tstop)

    h.tstop = Stim['initDur'] + Stim['stop']
    h.steps_per_ms = int(1/Stim['dt'])
    h.dt = Stim['dt']
    h.continuerun(h.tstop)

    ## -- if spike, increment spike count and record initSeg
    spike = 0
    min_peaks_inds = []
    init_time = np.inf
    if APC_axon.n > 0:
        spike_timing = APC_axon.time
        if spike_timing > 50 and spike_timing < 70:
            spike = 1
            print('SPIKE: Currents: ', Stim['amp'], 'uA')
            # Determine where it started
            ### ---- Logic to record spike initiation segment ----
            t_vec = np.array(t)
            # --- axon ---
            vRecord = np.array(vRec_axon)
            # # --- socb ---
            # vRecord = np.array(vRec_socb)
            # First, truncate out first 50 time samples (initDur)
            vRecord_trunc = vRecord[:,int(Stim['initDur']/Stim['initDt']):]
            t_trunc = t_vec[int(Stim['initDur']/Stim['initDt']):]
            ## Second, truncate initial delay period (delay)
            # Figure out offset, according to stim_pulse length
            if Stim['pulseShape'] == 'triphasic':
                vRecord_trunc = vRecord_trunc[:,int((Stim['delay']+Stim['dur']*3)/Stim['dt']):]
                t_trunc = t_trunc[int((Stim['delay']+Stim['dur']*3)/Stim['dt']):]
            elif Stim['pulseShape'] == 'monophasic':
                vRecord_trunc = vRecord_trunc[:,int((Stim['delay']+Stim['dur'])/Stim['dt']):]
                t_trunc = t_trunc[int((Stim['delay']+Stim['dur'])/Stim['dt']):]
            elif Stim['pulseShape'] == 'biphasic':
                vRecord_trunc = vRecord_trunc[:,int((Stim['delay']+Stim['dur']*2)/Stim['dt']):]
                t_trunc = t_trunc[int((Stim['delay']+Stim['dur']*2)/Stim['dt']):]
            
            peaks_list = []
            for ii in range(vRecord_trunc.shape[0]):
                vec = vRecord_trunc[ii,:]
                # Find the peaks of the vec waveform 
                # peaks = hf.findPeaks(vec,height=-10,widthSamples=15)
                peaks = hf.findPeaks(vec,height=-10,widthSamples=10) #TODO: Changed
                if len(peaks) == 0:
                    peaks_list.append(np.inf)
                else:
                    peaks_list.append(peaks[0])
            
            min_peaks = np.min(peaks_list)
            min_peaks_inds = [i for i, x in enumerate(peaks_list) if x == min_peaks]
            init_time = t_trunc[peaks_list[min_peaks_inds[0]]]
            #### ---------
        else:
            print('SPIKE TIME NOT MET: Currents: ', Stim['amp'], 'uA')
    else:
        print('NO SPIKE: Currents: ', Stim['amp'], 'uA')

    ### --- output and save results
    dir = '/Volumes/Scratch/Users/vilkhu/sim-data/multisite/fixed/monophasic/fig3_y0/005Dt/'

    # First, save electrode information 
    filename_elecs = dir + 'electrodes.pkl'
    os.makedirs(os.path.dirname(filename_elecs), exist_ok=True)
    data = np.asarray(Stim['electrodes'], dtype=object)
    try: 
        hf.compressed_pickle(filename_elecs, data)
    except:
        print('Error writing file: ', filename_elecs)

    # Second, save spiking information (spike, init compartment)
    filename = dir+str(currents[0])+'_'+str(currents[1])+'.pkl'
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    data = np.asarray([spike,min_peaks_inds,init_time], dtype=object)
    try: 
        hf.compressed_pickle(filename, data)
    except:
        print('Error writing file: ', filename)

def main(): 
    # ### --- setup parallel processing
    # pool = mp.Pool(mp.cpu_count() - 10)

    # Print out a message at the start for debugging
    pool = mp.Pool(30)
    print('Testing fixed monophasic bielec for fig3_y0 with 0.005 dt')

    # ## --- create a list of currents to test; units: [uA] (2-electrode, full)
    currents = []
    current_range = np.arange(-5,5.1,0.2)
    current_range = np.round(current_range, 2)
    for i1 in current_range:
        for i2 in current_range:
            # # Store [i1,i2] in currents, rount to 1 sig fig
            # if ((i1 + i2) >= 0.6 and (i1 + i2) <= 1.4):
            #     currents.append([round(i1,2),round(i2,2)])
            currents.append([round(i1,2),round(i2,2)])


    # ### --- create a list of currents to test; units: [uA] (1-electrode)
    # currents = []
    # current_range = np.arange(-5,5,0.05)
    # current_range = np.round(current_range, 2)
    # i2 = 0
    # for i1 in current_range:
    #     currents.append([round(i1,2),round(i2,2)])

    ### --- run parallel simulations for each electrode location
    # result = pool.map(expSigmoid, currents)
    result = list(tqdm.tqdm(pool.imap(expMultisite, currents), total=len(currents)))

if __name__ == "__main__":
    main()
