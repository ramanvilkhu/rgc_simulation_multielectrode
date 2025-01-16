import multiprocessing as mp
import numpy as np
from neuron import h
import helperFuncs as hf
import os
from tqdm import tqdm
import simulation as sim


def expMultisiteThreshold(primary_elec_curr):
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

    Stim['electrodes'] = [[-1030,0,72],[-1000,0,72]]
    Stim['electrode_type'] = 'disk'
    Stim['rhoExt'] = 1000 # Ohm*cm
    Stim['elecDiam'] = 15 # um
    Stim['polarity'] = -1 # anodic == 1; cathodic == -1 

    Stim['pulseRatioTri'] = [-2/3,1,-1/3]
    Stim['pulseRatioBi'] = [1,-1]
    Stim['frequency'] = 10 # units: [Hz]


    # Do the threshold search 
    current = 0.05
    # currents = [-1*current,primary_elec_curr] # TODO: Changed 
    currents = [primary_elec_curr, current]
    precision = .4 # %
    upperLim = 12

    firstAP = 0
    stop = False
    while not stop:
        Stim['amp'] = currents

        # define transfer impedances 
        active_x = [-1015] # Old code, this doesn't do anything anymore
        hf.setupTransferImpedance_v3(Stim, RGC, active_x, limit_sites=False)
        
        ## -- set up recording vectors
        APC_axon = h.APCount(1, sec = RGC.axon)
        APC_axon.thresh = 0 

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
        
        # Check to see if we see a spike 
        if firstAP == 0:
            if (max(vRec[0]) < 0):
                current = current * 1.5
            else:
                firstAP = 1
                stepCurr = current / 4
                current = current - stepCurr
                # vRecord = np.array(vRec)
        else:
            if (max(vRec[0]) > 0 and stepCurr <= current*precision/100):
                stop = True
                vRecord = np.array(vRec)
            if (max(vRec[0]) > 0 and stepCurr > current*precision/100):
                stepCurr = stepCurr / 2
                current = current - stepCurr 
                vRecord = np.array(vRec)
            if (max(vRec[0]) < 0 and stepCurr > current*precision/100):
                stepCurr = stepCurr / 2
                current = current + stepCurr
            if (max(vRec[0]) < 0 and stepCurr <= current*precision/100):
                current = current + stepCurr
        
        if current > upperLim:
            stop = True
        
        # currents = [-1*current,primary_elec_curr] # TODO: Changed 
        currents = [primary_elec_curr, current]

    if current < upperLim:
        # if primary_elec_curr <= -0.34:
        if False:
            threshold = 0 
        else:
            # threshold = currents[0] # TODO: Changed
            threshold = currents[1]
    else:
        threshold = np.nan
    
    if threshold == np.nan:
        return 
    else:
        print(primary_elec_curr, threshold)

        ### ---- Logic to record spike initiation segment ----
        t_vec = np.array(t)
        # --- axon ---
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

        ### --- output and save results
        dir = '/Volumes/Scratch/Users/vilkhu/sim-data/multisite/fixed/triphasic/fig3_y40_threshold/005Dt/'

        # First, save electrode information 
        filename_elecs = dir + 'electrodes.pkl'
        os.makedirs(os.path.dirname(filename_elecs), exist_ok=True)
        data = np.asarray(Stim['electrodes'], dtype=object)
        try: 
            hf.compressed_pickle(filename_elecs, data)
        except:
            print('Error writing file: ', filename_elecs)

        # Second, save spiking information (spike, init compartment)
        # currents = [threshold,primary_elec_curr] # TODO: Changed
        currents = [primary_elec_curr, threshold]
        filename = dir+str(currents[0])+'_'+str(currents[1])+'.pkl'
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        data = np.asarray([1,min_peaks_inds,init_time], dtype=object)
        try: 
            hf.compressed_pickle(filename, data)
        except:
            print('Error writing file: ', filename)

def main(): 
    ### --- setup parallel processing
    pool = mp.Pool(mp.cpu_count() - 10)

    # ### --- monophasic
    # ### --- create a list of currents for elec1 to test; units: [ms]
    curr = np.linspace(3,7.5,61) 
    # curr = np.linspace(-2,2,101) 

    ### --- run parallel simulations for each electrode location
    result = list(tqdm(pool.imap(expMultisiteThreshold, curr), total=len(curr)))


if __name__ == "__main__":
    main()
