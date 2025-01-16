import multiprocessing as mp
import numpy as np
from neuron import h
# import pandas as pd
import helperFuncs as hf
import simulation as sim
import pickle as pk
import os
import tqdm
import itertools

def expSigmoid(args):
    electrode, current = args

    print('Calculating probability for: ',\
          electrode, ': ', current, 'uA.')
    
    ### --- setup biophysical simulaiton environment
    # setup cell
    RGC = sim.Local_Cell()
    filename = 'cell_param_files/params_35.csv'
    RGC.build_cell(filename,'mammalian_spike_35')

    ### --- setup stimulus
    Stim = {}

    Stim['pulseShape'] = 'biphasic'
    Stim['dt'] = .005 
    Stim['delay'] = 10
    Stim['dur'] = 0.05 # [ms]
    Stim['stop'] = 80

    Stim['initDur'] = 50
    Stim['initDt'] = 1
    Stim['vInit'] = -70 # units: [mV]

    Stim['electrodes'] = [electrode]
    Stim['electrode_type'] = 'point'
    Stim['rhoExt'] = 1000 # Ohm*cm
    Stim['elecDiam'] = 15 # um
    Stim['polarity'] = -1 # anodic == 1; cathodic == -1 

    Stim['pulseRatioTri'] = [-2/3,1,-1/3]
    Stim['pulseRatioBi'] = [1,-1]
    Stim['frequency'] = 10 # units: [Hz]

    # define transfer impedances
    Stim['amp'] = [Stim['polarity']*float(current)]
    hf.setupTransferImpedance_v2(Stim, RGC)
    
    ### --- inject Gaussian noise into axon
    tstop = 80
    dt = 0.005
    random_stream_offset_ = (tstop+1000)*1/dt * np.floor(current*(1e-3)*10)
    r = h.Random()
    trials = 12
    knoise = 0.00025*(1e3)
    spikes = 0
    for k in range(trials):
        # ## --- Inject noise into distal axon
        # objs = [h.InGauss(seg.x,sec=RGC.axon) for seg in RGC.axon]
        # for obj in objs:
        #     r.MCellRan4(random_stream_offset_*k) #TODO: change back to +k
        #     segment = obj.get_segment()

        #     obj.delay = 0
        #     obj.dur = tstop
        #     obj.mean = 0
        #     # TODO: change whether axonal or somatic
        #     obj.stdev = knoise*(np.sqrt(segment.area()*1e-8*1 \
        #                     * segment.gnabar_mammalian_spike*1e3))
        #     obj.noiseFromRandom(r)
            
        # h.setpointers()

        ## --- Inject noise into sodium channel band
        objs = [h.InGauss(seg.x,sec=RGC.SOCB) for seg in RGC.SOCB]
        objs_axon = [h.InGauss(seg.x,sec=RGC.axon) for seg in RGC.axon]
        for obj in objs:
            r.MCellRan4(random_stream_offset_*k) #TODO: change back to +k
            segment = obj.get_segment()

            obj.delay = 0
            obj.dur = tstop
            obj.mean = 0
            obj.stdev = knoise*(np.sqrt(segment.area()*1e-8*1 \
                            *segment.gnabar_mammalian_spike_35*1e3))
            obj.noiseFromRandom(r)

        for obj in objs_axon:
            r.MCellRan4(random_stream_offset_*k) #TODO: change back to +k
            segment = obj.get_segment()

            obj.delay = 0
            obj.dur = tstop
            obj.mean = 0
            obj.stdev = knoise*(np.sqrt(segment.area()*1e-8*1 \
                            * segment.gnabar_mammalian_spike_35*1e3))
            obj.noiseFromRandom(r)
            
        h.setpointers()

        ## -- set up recording vectors
        APC_axon = h.APCount(1, sec = RGC.axon)
        APC_axon.thresh = 0 

        vRec_axon = []
        for seg in RGC.axon:
            vRec_axon.append(h.Vector().record(RGC.axon(seg.x)._ref_v))

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

        ## -- if spike, increment spike count
        if APC_axon.n > 0:
            spike_timing = APC_axon.time
            if spike_timing > 50 and spike_timing < 70:
                spikes += 1
                print('SPIKE: Current: ', current, 'uA, Spikes: ', spikes, '/', k)
            else:
                print('SPIKE TIME NOT MET: Current: ', current, 'uA, Spikes: ', spikes, '/', k)
                continue
        else:
            print('NO SPIKE: Current: ', current, 'uA, Spikes: ', spikes, '/', k)
    
    spike_prob = float(spikes)/float(trials)

    ### --- output and save results
    print('Probability for  Position: ',\
            str(Stim['electrodes'][0]), ' Current: ', str(current), ' is: ', spike_prob)
    filename = '/Volumes/Scratch/Users/vilkhu/sim-data/sigmoid/pillars/knoise2.5/'+\
                str(Stim['electrodes'][0][0])+'_'+\
                str(Stim['electrodes'][0][1])+'_'+\
                str(Stim['electrodes'][0][2])+'_'+\
                str(current)+'uA'+'.pkl'
    
    os.makedirs(os.path.dirname(filename), exist_ok=True)

    data = np.asarray([spike_prob], dtype=object)

    # write compressed pickle file
    try: 
        hf.compressed_pickle(filename, data)
        print('Data saved.')
    except:
        print('Error writing file: ', filename)

def main(): 
    ### --- setup parallel processing
    pool = mp.Pool(mp.cpu_count() - 10)

    ### --- create a list of currents to test; units: [uA]
    currents = np.linspace(0, 4, 30)
    ### --- create list of electrodes to test
    electrodes = [[-5,0,10],[-5,0,15],[-5,0,20],\
                  [-5,0,25],[-5,0,30],[-5,0,35],\
                  [-65,0,10],[-65,0,15],[-65,0,20],\
                  [-65,0,25],[-65,0,30],[-65,0,35]]

    ### --- create list of electrode-current pairs
    combs = itertools.product(electrodes, currents)
    total_len = len(currents)*len(electrodes)

    ### --- run parallel simulations for each electrode location
    # result = pool.map(expSigmoid, currents)
    result = list(tqdm.tqdm(pool.imap(expSigmoid, combs), total=total_len))

if __name__ == "__main__":
    main()
