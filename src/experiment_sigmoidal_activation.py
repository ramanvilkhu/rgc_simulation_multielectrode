import multiprocessing as mp
import numpy as np
from neuron import h
# import pandas as pd
import helperFuncs as hf
import simulation as sim
import pickle as pk
import os
import tqdm

def expSigmoid(current):
    print('Calculating probability for: ', current, 'uA.')
    
    ### --- setup biophysical simulaiton environment
    # setup cell
    RGC = sim.Local_Cell()
    filename = 'cell_param_files/params_35_v2.csv'
    RGC.build_cell(filename,'mammalian_spike_35')

    # Rotate cell to match real EI
    degrees = 70
    radians = np.deg2rad(degrees)
    RGC._rotateZ(radians)
    RGC.shift_cell_x_y_z(80,-20,0)

    ### --- setup stimulus
    Stim = {}

    Stim['pulseShape'] = 'triphasic'
    Stim['dt'] = .005 # TODO: Changed from .005
    Stim['delay'] = 10
    Stim['dur'] = 0.05 # [ms]
    Stim['stop'] = 80

    Stim['initDur'] = 50
    Stim['initDt'] = 1
    Stim['vInit'] = -70 # units: [mV]

    # Stim['electrodes'] = [[90,75,42]]
    # Stim['electrodes'] = [[-30,-375,42]] # Axonal
    Stim['electrodes'] = [[60,-90,46]] # Somatic
    Stim['electrode_type'] = 'disk'
    Stim['rhoExt'] = 1000 # Ohm*cm
    Stim['elecDiam'] = 15 # um
    Stim['polarity'] = -1 # anodic == 1; cathodic == -1 

    Stim['pulseRatioTri'] = [-2/3,1,-1/3]
    Stim['pulseRatioBi'] = [1,-1]
    Stim['frequency'] = 10 # units: [Hz]

    # define transfer impedances 
    hf.setupTransferImpedance(Stim, RGC)

    ### --- inject Gaussian noise into axon
    tstop = 80
    dt = 0.005
    random_stream_offset_ = (tstop+1000)*1/dt * np.floor(current*(1e-3)*10)
    r = h.Random()
    trials = 20
    knoise = 0.00060*(1e3)
    spikes = 0
    for k in range(trials):
        ## --- Inject noise into all compartments
        for sec_name in RGC.section_list:
            objs = [h.InGauss(seg.x,sec=sec_name) for seg in sec_name]
            for obj in objs:
                r.MCellRan4(random_stream_offset_*k) #TODO: change back to +k
                segment = obj.get_segment()

                obj.delay = 0
                obj.dur = tstop
                obj.mean = 0
                # TODO: change whether axonal or somatic
                obj.stdev = knoise*(np.sqrt(segment.area()*1e-8*1 \
                                * segment.gnabar_mammalian_spike_35*1e3))
                obj.noiseFromRandom(r)

                # print('axon noise std: ', obj.stdev)
                
            h.setpointers()
        ## --- Inject noise into distal axon
        # objs = [h.InGauss(seg.x,sec=RGC.axon) for seg in RGC.axon]
        # for obj in objs:
        #     r.MCellRan4(random_stream_offset_*k) #TODO: change back to +k
        #     segment = obj.get_segment()

        #     obj.delay = 0
        #     obj.dur = tstop
        #     obj.mean = 0
        #     # TODO: change whether axonal or somatic
        #     obj.stdev = knoise*(np.sqrt(segment.area()*1e-8*1 \
        #                     * segment.gnabar_mammalian_spike_35*1e3))
        #     obj.noiseFromRandom(r)
            
        # h.setpointers()

        # ## --- Inject noise into sodium channel band
        # objs_socb = [h.InGauss(seg.x,sec=RGC.SOCB) for seg in RGC.SOCB]
        # for obj in objs_socb:
        #     r.MCellRan4(random_stream_offset_*k) #TODO: change back to +k
        #     segment = obj.get_segment()

        #     obj.delay = 0
        #     obj.dur = tstop
        #     obj.mean = 0
        #     # TODO: change whether axonal or somatic
        #     obj.stdev = knoise*(np.sqrt(segment.area()*1e-8*1 \
        #                     *segment.gnabar_mammalian_spike_35*1e3))
        #     obj.noiseFromRandom(r)
        
        # h.setpointers()

        # objs_ah = [h.InGauss(seg.x,sec=RGC.AH) for seg in RGC.AH]
        # for obj in objs_ah:
        #     r.MCellRan4(random_stream_offset_*k) #TODO: change back to +k
        #     segment = obj.get_segment()

        #     obj.delay = 0
        #     obj.dur = tstop
        #     obj.mean = 0
        #     # TODO: change whether axonal or somatic
        #     obj.stdev = knoise*(np.sqrt(segment.area()*1e-8*1 \
        #                     * segment.gnabar_mammalian_spike_35*1e3))
        #     obj.noiseFromRandom(r)
        
        # h.setpointers()

        # objs_soma = [h.InGauss(seg.x,sec=RGC.cell.soma) for seg in RGC.cell.soma]
        # for obj in objs_soma:
        #     r.MCellRan4(random_stream_offset_*k) #TODO: change back to +k
        #     segment = obj.get_segment()

        #     obj.delay = 0
        #     obj.dur = tstop
        #     obj.mean = 0
        #     # TODO: change whether axonal or somatic
        #     obj.stdev = knoise*(np.sqrt(segment.area()*1e-8*1 \
        #                     * segment.gnabar_mammalian_spike_35*1e3))
        #     obj.noiseFromRandom(r)
            
        # h.setpointers()


        ## -- set up recording vectors
        APC_axon = h.APCount(1, sec = RGC.axon)
        APC_axon.thresh = 0 

        vRec_axon = []
        for seg in RGC.axon:
            vRec_axon.append(h.Vector().record(RGC.axon(seg.x)._ref_v))

        ## -- run simulation
        Stim['amp'] = Stim['polarity']*float(current)
        tsvec, isvec = hf.setupStimulus(Stim)
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
                print('SPIKE: Current: ', current, 'uA, Spikes: ', spikes, '/', k+1)
            else:
                print('SPIKE TIME NOT MET: Current: ', current, 'uA, Spikes: ', spikes, '/', k+1)
                continue
        else:
            print('NO SPIKE: Current: ', current, 'uA, Spikes: ', spikes, '/', k+1)
    
    spike_prob = float(spikes)/float(trials)

    ### --- output and save results
    print('Probability for  Position: ',\
            str(Stim['electrodes'][0]), ' Current: ', str(current), ' is: ', spike_prob)
    filename = '/Volumes/Scratch/Users/vilkhu/sim-data/sigmoid/fixed/somatic_real/knoise_00060/'+\
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
    print('Test dendrite')
    # ### --- setup parallel processing
    # pool = mp.Pool(mp.cpu_count() - 10)
    pool = mp.Pool(30)

    ### --- create a list of currents to test; units: [uA]
    currents = np.linspace(0, 2.5, 100)

    ### --- run parallel simulations for each electrode location
    # result = pool.map(expSigmoid, currents)
    result = list(tqdm.tqdm(pool.imap(expSigmoid, currents), total=len(currents)))

if __name__ == "__main__":
    main()
