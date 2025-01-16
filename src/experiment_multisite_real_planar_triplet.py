import multiprocessing as mp
import numpy as np
from neuron import h
# import pandas as pd
import helperFuncs as hf
import simulation as sim
import pickle as pk
import os
import tqdm  

def define_plane_and_sample(points):
    # Unpack points
    P1, P2, P3, P4 = points

    # Define vectors
    v1 = P2 - P1
    v2 = P3 - P1

    # Normal vector of the plane
    n = np.cross(v1, v2)
    n = n / np.linalg.norm(n)  # Normalize

    # Define the plane equation n . (P - P1) = 0, which simplifies to Ax + By + Cz + D = 0
    A, B, C = n
    D = -np.dot(n, P1)

    # Define the parameterization of the plane
    def parameterize(s, t):
        return P1 + s * v1 + t * v2

    # Sample points evenly
    s_values = np.linspace(0, 1, 30)
    t_values = np.linspace(0, 1, 30)
    sampled_points = [parameterize(s, t) for s in s_values for t in t_values]

    return np.array(sampled_points), n


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

    # Stim['electrodes'] = [[-1030,45,42],[-1015,15,42],[-1000,45,42]]
    Stim['electrodes'] = [[-5,25,40],[10,-5,40],[25,25,40]]
    Stim['electrode_type'] = 'disk'
    Stim['rhoExt'] = 1000 # Ohm*cm
    Stim['elecDiam'] = 10 # um
    Stim['polarity'] = -1 # anodic == 1; cathodic == -1 

    Stim['pulseRatioTri'] = [-2/3,1,-1/3]
    Stim['pulseRatioBi'] = [1,-1]
    Stim['frequency'] = 10 # units: [Hz]

    ## Define direction vectors
    # Define the four bounding points
    # P1 = np.array([-4, 2.32, -4])
    # P2 = np.array([-4, 1.47, 4])
    # P3 = np.array([4, -2.74, -4])
    # P4 = np.array([4, -4, 4])
    # points = [P1, P2, P3, P4]
    P1 = np.array([-4, 3, -4])
    P2 = np.array([-4, 1.47, 4])
    P3 = np.array([4, -2.74, -4])
    P4 = np.array([4, -4, 4])
    points = [P1, P2, P3, P4]

    # Get the sampled points
    _, n = define_plane_and_sample(points)

    # Extract coordinates for Plotly

    ## Implement a bisection search to find the optimal scaling factor
    epsilon = 0.05
    lowBound = 0.2
    upBound = 4
    stop = False
    scale_factor = np.nan

    while not stop:
        scaling_factor = lowBound + ((upBound - lowBound) / 2)
        currents = sample + (n*scaling_factor)
        # currents = sample + (-1*n*scaling_factor)
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
    
    if scale_factor is np.nan:
        print('No spiking within range')
    else:
        ## -- run one last sim at the threshold and record spike init
        # Use APCount to detect if an AP occured
        APC_axon = h.APCount(1, sec = RGC.axon)
        APC_axon.thresh = 0 

        currents = sample + (n*scale_factor)
        # currents = sample + (-1*n*scale_factor)
        Stim['amp'] = currents

        # Run simulation
        active_x = [-1015] # Old code, this doesn't do anything anymore
        hf.setupTransferImpedance_v3(Stim, RGC, active_x, limit_sites=False)
        
        # # --- Track spike at axon ---
        # vRec = []
        # for seg in RGC.axon:
        #     vRec.append(h.Vector().record(RGC.axon(seg.x)._ref_v))
        # t = h.Vector().record(h._ref_t)
        # --- Track spike at SOCB --- # TODO: CHANGED THIS
        vRec = []
        for seg in RGC.SOCB:
            vRec.append(h.Vector().record(RGC.SOCB(seg.x)._ref_v))
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
        dir = '/Volumes/Scratch/Users/vilkhu/sim-data/multisite/fixed/triphasic/planar/fig6_somatic_v2/005Dt/'

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
    print('Testing fixed triphasic planar sampling for somatic triplet_v2 with 0.005 dt')

    # Define the four bounding points
    # P1 = np.array([-4, 0.5, -4])
    # P2 = np.array([-4, 0.5, 4])
    # P3 = np.array([4, -0.5, -4])
    # P4 = np.array([4, -0.5, 4])
    # points = [P1, P2, P3, P4]
    P1 = np.array([-4, 3, -4])
    P2 = np.array([-4, 1.47, 4])
    P3 = np.array([4, -2.74, -4])
    P4 = np.array([4, -4, 4])
    points = [P1, P2, P3, P4]

    # Get the sampled points
    samples, _ = define_plane_and_sample(points)

    ### --- run parallel simulations for each electrode location
    # result = pool.map(expSigmoid, currents)
    result = list(tqdm.tqdm(pool.imap(expMultisite, samples), total=len(samples)))

if __name__ == "__main__":
    main()
