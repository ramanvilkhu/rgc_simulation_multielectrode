import multiprocessing as mp
import numpy as np
from neuron import h
# import pandas as pd
import helperFuncs as hf
import simulation as sim
import pickle as pk
import os
import tqdm  


def expMultisite(sample):
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

    Stim['pulseShape'] = 'triphasic'
    Stim['dt'] = .005 # TODO: Changed from .005
    Stim['delay'] = 10
    Stim['dur'] = 0.05 # [ms]
    Stim['stop'] = 70

    Stim['initDur'] = 50
    Stim['initDt'] = 1
    Stim['vInit'] = -70 # units: [mV]

    Stim['electrodes'] = [[-1030,15,42],[-1015,-15,42],[-1000,15,42]]
    Stim['electrode_type'] = 'disk'
    Stim['rhoExt'] = 1000 # Ohm*cm
    Stim['elecDiam'] = 15 # um
    Stim['polarity'] = -1 # anodic == 1; cathodic == -1 

    Stim['pulseRatioTri'] = [-2/3,1,-1/3]
    Stim['pulseRatioBi'] = [1,-1]
    Stim['frequency'] = 10 # units: [Hz]

    ## Define direction vectors
        # Endpoints of the line
    # r0 = np.array([-4, 4, 0])
    # r1 = np.array([4, -4, 0])
    # r0 = np.array([3, -4, 3])
    # r1 = np.array([-3, 4, -2])
    r0 = np.array([7.5, -9.5, 7])
    r1 = np.array([-7.5, 9.5, -7])

    # Direction vector
    d = r1 - r0

    # Normalize direction vector
    d_unit = d / np.linalg.norm(d)

    # Find two perpendicular vectors using Gram-Schmidt process
    u = np.cross(d_unit, np.array([1, 0, 0]))
    if np.linalg.norm(u) == 0:  # if d_unit is parallel to [1, 0, 0]
        u = np.cross(d_unit, np.array([0, 1, 0]))
    u = u / np.linalg.norm(u)
    v = np.cross(d_unit, u)

    ## Extract sample params
    r_t = sample[0]
    theta = sample[1]

    ## Implement a bisection search to find the optimal scaling factor
    epsilon = 0.05
    lowBound = 0.1
    upBound = 5
    stop = False
    scale_factor = np.nan

    while not stop:
        scaling_factor = lowBound + ((upBound - lowBound) / 2)
        currents = r_t + scaling_factor * (np.cos(theta) * u + np.sin(theta) * v)
        Stim['amp'] = currents

        # Run simulation
        active_x = [-1015] # Old code, this doesn't do anything anymore
        hf.setupTransferImpedance_v3(Stim, RGC, active_x, limit_sites=False)
        # --- Track spike at axon ---
        vRec = []
        for seg in RGC.axon:
            vRec.append(h.Vector().record(RGC.axon(seg.x)._ref_v))
        t = h.Vector().record(h._ref_t)

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

        # Detect whether spike happened and update bounds
        if (max(vRec[0]) > 0):
            upBound = scaling_factor
            precision = upBound - lowBound
            vRecord = np.array(vRec)
            scale_factor = scaling_factor
        else:
            lowBound = scaling_factor
            precision = upBound - lowBound

        # Figure out if I should escape
        if precision < epsilon:
            stop = True
    
    if scale_factor == np.nan:
        print('No spiking within range')
    else:
        ## -- run one last sim at the threshold and record spike init
        # Use APCount to detect if an AP occured
        APC_axon = h.APCount(1, sec = RGC.axon)
        APC_axon.thresh = 0 

        currents = r_t + scale_factor * (np.cos(theta) * u + np.sin(theta) * v)
        Stim['amp'] = currents

        # Run simulation
        active_x = [-1015] # Old code, this doesn't do anything anymore
        hf.setupTransferImpedance_v3(Stim, RGC, active_x, limit_sites=False)
        # --- Track spike at axon ---
        vRec = []
        for seg in RGC.axon:
            vRec.append(h.Vector().record(RGC.axon(seg.x)._ref_v))
        t = h.Vector().record(h._ref_t)

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
                vRecord = np.array(vRec)
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
        dir = '/Volumes/Scratch/Users/vilkhu/sim-data/multisite/fixed/triphasic/radial_v2/fig5_onAxonal_largeSpace/005Dt/'

        # First, save electrode information 
        filename_elecs = dir + 'electrodes.pkl'
        os.makedirs(os.path.dirname(filename_elecs), exist_ok=True)
        data = np.asarray(Stim['electrodes'], dtype=object)
        try: 
            hf.compressed_pickle(filename_elecs, data)
        except:
            print('Error writing file: ', filename_elecs)

        # Second, save spiking information (spike, init compartment)
        filename = dir+str(currents[0])+'_'+str(currents[1])+'_'+str(currents[2])+'.pkl'
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
    print('Testing fixed triphasic radial_v2 sampling for onAxonal_v2 large space with 0.005 dt')

    # # Endpoints of the line
    # r0 = np.array([-4, 4, 0])
    # r1 = np.array([4, -4, 0])
    r0 = np.array([7.5, -9.5, 7])
    r1 = np.array([-7.5, 9.5, -7])

    # Direction vector
    d = r1 - r0

    # Normalize direction vector
    d_unit = d / np.linalg.norm(d)

    # Find two perpendicular vectors using Gram-Schmidt process
    u = np.cross(d_unit, np.array([1, 0, 0]))
    if np.linalg.norm(u) == 0:  # if d_unit is parallel to [1, 0, 0]
        u = np.cross(d_unit, np.array([0, 1, 0]))
    u = u / np.linalg.norm(u)
    v = np.cross(d_unit, u)

    # Parameters for sampling
    # num_samples_along_line = 30
    # num_radial_samples = 40
    num_samples_along_line = 50
    num_radial_samples = 20
    

    # Generate samples
    samples = []
    t_values = np.linspace(0, 1, num_samples_along_line)
    theta_values = np.linspace(0, 2 * np.pi, num_radial_samples)

    # Create array of (r_t, theta) tuples
    samples = []
    for t in t_values:
        r_t = (1 - t) * r0 + t * r1
        for theta in theta_values:
            samples.append([r_t, theta])

    ### --- run parallel simulations for each electrode location
    # result = pool.map(expSigmoid, currents)
    result = list(tqdm.tqdm(pool.imap(expMultisite, samples), total=len(samples)))

if __name__ == "__main__":
    main()
