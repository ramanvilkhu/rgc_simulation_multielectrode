import multiprocessing as mp
import numpy as np
from neuron import h
import helperFuncs as hf
import os
from tqdm import tqdm
import sys

if sys.platform.startswith('linux'): # LINUX
    h.nrn_load_dll('../nrn/x86_64/.libs/libnrnmech.so') 

h.load_file('stdrun.hoc')
h.load_file('interpxyz.hoc')
h.load_file('setpointers.hoc')

for sec in h.allsec():
   h.delete_section(sec=sec)

def setTransferResistance(Stim,axon):
    # First let's decide the number of segments to make active in the simulation
    num_seg_active = 500

    # Determine index of center segment, given axon.nseg is even
    center_seg = int(axon.nseg/2)

    # Create list of segment indices to activate given num_seg_active
    segs = []
    while len(segs) < num_seg_active:
        if len(segs) == 0:
            segs.append(center_seg)
        if len(segs) < num_seg_active:
            segs.append(min(segs)-1)
        if len(segs) < num_seg_active:
            segs.append(max(segs)+1)

    # Order segs list
    segs = sorted(segs)

    # Set up transfer resistances, only for active segments
    for idx,seg in enumerate(axon):
        rx = 0
        for i in range(len(Stim['electrodes'])):
            # Current amplitude on elec scaling factor 
            I = Stim['amp'][i]

            if Stim['electrode_type'] == 'disk':
                # Distance from electrode to center of each segment [cm]
                r = 1e-4*\
                    np.sqrt((seg.xtra.x - (Stim['electrodes'][i][0]))**2\
                            + (seg.xtra.y - (Stim['electrodes'][i][1]))**2)
                # Vertical distance [cm]
                z = 1e-4*(seg.xtra.z-Stim['electrodes'][i][2])

                # Transfer impedance [MOhm]  
                rDisk = 1e-4*Stim['elecDiam']/2             
                rx += I * (1e-6*(2.0*Stim['rhoExt'])/(4.0*np.pi*rDisk) \
                    * np.arcsin((2*rDisk) / \
                    (np.sqrt((r-rDisk)**2+z**2)+np.sqrt((r+rDisk)**2+z**2))))
            elif Stim['electrode_type'] == 'point':
                dist = 1e-4*\
                    np.sqrt((seg.xtra.x - (Stim['electrodes'][i][0]))**2 + \
                            (seg.xtra.y - (Stim['electrodes'][i][1]))**2 + \
                            (seg.xtra.z - (Stim['electrodes'][i][2]))**2)
                rx += I * (1e-6*Stim['rhoExt']/(4.0*np.pi*dist))

        # Set transfer impedance for each segment
        if idx in segs:
            seg.xtra.rx  = rx  # in MOhm
        else:
            seg.xtra.rx  = 0
                            
        # Map references of extracellular and membrane current with 
        #  xtra mechanism
        h.setpointer(seg._ref_e_extracellular,'ex',seg.xtra)
        h.setpointer(seg._ref_i_membrane,'im',seg.xtra)

def expMultisite(primary_elec_curr):
    ### --- setup biophysical simulaiton environment

    # Axon parameters 
    L = 1000 # length[um]
    d = 2 # diamter [um]
    dx = 2 # segment length [um]

    # Setup axon geometry
    axon = h.Section(name='axon')
    h.pt3dadd(-L/2, 0, 0, d, sec=axon)
    h.pt3dadd(L/2, 0, 0, d, sec=axon)
    axon.nseg = int(L/dx)

    # Setup axon biophysics 
    axon.insert('pas')
    axon.insert('mammalian_spike_35')
    axon.insert('cad')
    axon.insert('extracellular')
    axon.insert('xtra')

    axon.Ra = 143.2 # [Ohm*cm]
    axon.cm = 1.0 # [uF/cm^2]
    axon.ena = 60.6 # [mV]
    axon.ek = -101.34 # [mV]
    axon.e_pas = -70 # [mV]

    axon.gnabar_mammalian_spike_35 = 0.42 # [S/cm^2]
    axon.gkbar_mammalian_spike_35 = 0.050 # [S/cm^2]
    axon.gkcbar_mammalian_spike_35 = 0.00031 # [S/cm^2]
    axon.gcabar_mammalian_spike_35 = 0.0 # [S/cm^2]
    axon.g_pas = 0.0002 # [S/cm^2]
    # ------ distal axon below ------------
    # axon.gnabar_mammalian_spike_35 = 0.08 # [S/cm^2]
    # axon.gkbar_mammalian_spike_35 = 0.050 # [S/cm^2]
    # axon.gkcbar_mammalian_spike_35 = 0.0002 # [S/cm^2]
    # axon.gcabar_mammalian_spike_35 = 0.00075 # [S/cm^2]
    # axon.g_pas = 0.0001 # [S/cm^2]

    h.setpointers()

    # Compute segment centers and allocate memory 
    n3d = int(h.n3d(sec=axon))
    xx = h.Vector(n3d)
    yy = h.Vector(n3d)
    zz = h.Vector(n3d)
    dd = h.Vector(n3d)
    ll = h.Vector(n3d)

    # Get xyz coordinates, diameters, and lengths along each segment 
    for i in range(n3d):
        xx.x[i] = h.x3d(i, sec=axon)
        yy.x[i] = h.y3d(i, sec=axon)
        zz.x[i] = h.z3d(i, sec=axon)
        dd.x[i] = h.diam3d(i, sec=axon)
        ll.x[i] = h.arc3d(i, sec=axon)

    # Interpolate the xyz coordinates to the center of each segment
    ll.div(axon.L)
    rint = h.Vector(axon.nseg+2)
    rint.indgen(1./axon.nseg)
    rint.sub(1./(2.*axon.nseg))
    rint.x[0] = 0
    rint.x[axon.nseg+1] = 1

    xint = h.Vector(axon.nseg+2)
    yint = h.Vector(axon.nseg+2)
    zint = h.Vector(axon.nseg+2)
    dint = h.Vector(axon.nseg+2)
    xint.interpolate(rint,ll,xx)
    yint.interpolate(rint,ll,yy)
    zint.interpolate(rint,ll,zz)
    dint.interpolate(rint,ll,dd)

    # Set centers in xtra mechanism
    for i in range(1,axon.nseg+1):
        axon(rint.x[i]).x_xtra = xint.x[i]
        axon(rint.x[i]).y_xtra = yint.x[i]
        axon(rint.x[i]).z_xtra = zint.x[i]

    # Setup stimulus 
    ### Most features of the stimulus are set up a struct (dict) called Stim 
    Stim = {}

    # temporal features of the sitmulus 
    Stim['pulseShape'] = 'triphasic' # TODO: old 'monophasic'
    Stim['dt'] = .005 # units: [ms]
    Stim['delay'] = 10 # units: [ms]
    Stim['dur'] = 0.1 # units: [ms]
    Stim['stop'] = 50 # units: [ms]

    # calibration phase of the stimulus 
    Stim['initDur'] = 50 # units: [ms]
    Stim['initDt'] = 1 # units: [ms]
    Stim['vInit'] = -70 # units: [mV] #TODO: Old is -70

    # spatial features of the stimulus 
    # --> specifically defining the electrode geometry 
    Stim['electrodes'] = [[-15,50,20],[15,50,20]] # units: [um] 
    Stim['electrode_type'] = 'disk' # todo: 'disk' or 'point'
    Stim['rhoExt'] = 1000 # Ohm*cm
    Stim['elecDiam'] = 15 # um #TODO: Changed from 10 to 15
    Stim['polarity'] = -1 # anodic == 1; cathodic == -1 

    Stim['pulseRatioTri'] = [-2/3,1,-1/3]
    Stim['pulseRatioBi'] = [1,-1]
    Stim['frequency'] = 10 # units: [Hz]


    # Do the threshold search 
    current = 0.05
    currents = [primary_elec_curr,-1*current]
    precision = .4 # %
    upperLim = 10

    firstAP = 0
    stop = False
    while not stop:
        Stim['amp'] = currents
        setTransferResistance(Stim,axon)
        
        # Setup recording vectors 
        # Capture membrane voltage at all cell compartments
        vRec = []
        for i in range(axon.nseg):
                vRec.append(h.Vector().record(axon(1/axon.nseg+i/axon.nseg)._ref_v))
        t = h.Vector().record(h._ref_t)

        # Play the stimulus into the cell, setting it up for stimulation
        tsvec, isvec = hf.setupStimulus_unit(Stim)
        isvec.play(h._ref_is_xtra, tsvec, 1) 

        # Initialization phase of the simulation 
        h.init()
        h.tstop = Stim['initDur']
        h.dt = Stim['initDt']
        h.finitialize(Stim['vInit'])
        h.continuerun(h.tstop)

        # Run the simulation 
        h.tstop = Stim['initDur'] + Stim['stop']
        h.steps_per_ms = int(1/Stim['dt'])
        h.dt = Stim['dt']
        h.continuerun(h.tstop)
        
        # Check to see if we see a spike 
        if firstAP == 0:
            if (max(vRec[0]) < 0):
                current = current * 2
            else:
                firstAP = 1
                stepCurr = current / 4
                current = current - stepCurr
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
        
        currents = [primary_elec_curr,-1*current]

    if current < upperLim:
        # if primary_elec_curr <= -0.34:
        if False:
            threshold = 0 
        else:
            threshold = currents[1]
    else:
        threshold = np.nan

    ### ---- Logic to record spike initiation segment ----
    t_vec = np.array(t)
    # First, truncate out first 50 time samples (initDur)
    vRecord_trunc = vRecord[:,int(Stim['initDur']/Stim['initDt']):]
    t_trunc = t_vec[int(Stim['initDur']/Stim['initDt']):]
    # Second, truncate initial delay period (delay)
    vRecord_trunc = vRecord_trunc[:,int((Stim['delay']+0.15)/Stim['dt']):]
    t_trunc = t_trunc[int((Stim['delay']+0.15)/Stim['dt']):]
    # find x and y index of first vRecord_trunc > 0
    x,y = np.where(vRecord_trunc > 0)
    init_seg = x[np.argmin(y)]
    init_time = t_trunc[y[np.argmin(y)]]
    #### ---------

    ### --- output and save results
    print('Threshold for ', Stim['electrodes'], ' is \nPrimary: ', primary_elec_curr, \
          ' uA \nSecondary: ', threshold, ' uA')
    filename = '/Volumes/Scratch/Users/vilkhu/sim-data/multisite/triphasic/allSites/socb/spikeInit/disk/'+\
                str(primary_elec_curr)+'uA_'+\
                str(Stim['electrodes'][0][0])+'_'+\
                str(Stim['electrodes'][0][1])+'_'+\
                str(Stim['electrodes'][0][2])+'.pkl'
    
    os.makedirs(os.path.dirname(filename), exist_ok=True)

    data = np.asarray([primary_elec_curr,threshold,init_seg,init_time], dtype=object)

    # write compressed pickle file
    try: 
        hf.compressed_pickle(filename, data)
        print('Data saved.')
    except:
        print('Error writing file: ', filename)

def main(): 
    ### --- setup parallel processing
    pool = mp.Pool(mp.cpu_count() - 10)

    # ### --- monophasic
    # ### --- create a list of currents for elec1 to test; units: [ms]
    curr = np.linspace(0,-4,60) # y=80um

    ### --- run parallel simulations for each electrode location
    result = list(tqdm(pool.imap(expMultisite, curr), total=len(curr)))


if __name__ == "__main__":
    main()
