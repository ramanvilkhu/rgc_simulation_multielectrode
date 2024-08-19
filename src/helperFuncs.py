import numpy as np
from neuron import h
import bz2
import pickle
import _pickle as cPickle
import constants
from scipy.signal import find_peaks

# Peak finding function
def findPeaks(vec,height,widthSamples):
    peaks, _ = find_peaks(vec, height, width=widthSamples)
    return peaks

# Create a version that accounts for multi-electrode stimulation
# Notably, this assumes a "unit" stimulus current pulse and scales Rx instead
def setupTransferImpedance_v3(Stim, RGC, active_x, limit_sites=True):
    # Figure out which sites to limit
    if limit_sites:

        # Next, figure out x,y,z of all segs
        seg_positions = {}
        for sec in RGC.section_list:
            # If sec name has 'dend', skip
            if 'dend' in sec.name():
                continue
            else:
                seg_positions[sec.name()] = []
                for seg in sec:
                    seg_positions[sec.name()].append([seg.xtra.x,seg.xtra.y,seg.xtra.z])
        
        # Next, figure out closest segs to active_x
        axon_segs_to_keep = []
        for x in active_x:
            print('X location being kept active: ', x)
            dist_to_elec = np.inf
            closest_seg = [] 
            for sec in seg_positions:
                for ii,seg in enumerate(seg_positions[sec]):
                    dist = np.sqrt((seg[0]-x)**2 + (seg[1])**2 + (seg[2])**2)
                    if dist < dist_to_elec:
                        dist_to_elec = dist
                        seg_id = ii
                        closest_seg = [sec,seg_id]   
        
            # Keep closest seg and its two neighbors
            min_seg_size = 5
            for kk in range(int(np.floor(min_seg_size/2)+1)):
                axon_segs_to_keep.append(closest_seg[1]-kk)           
                axon_segs_to_keep.append(closest_seg[1]+kk)
            axon_segs_to_keep = np.unique(axon_segs_to_keep)
             
            print('Axon segs being kept active: ',axon_segs_to_keep)


    # Loop over all sections
    for sec in RGC.section_list:
        # Loop over all segments in section
        for ii, seg in enumerate(sec):
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
            seg.xtra.rx  = rx  # in MOhm

            # Do some limited activation site stuff if needed
            if limit_sites:
                if 'axon' in sec.name():
                    if ii not in axon_segs_to_keep:
                        seg.xtra.rx = 0
                                
            # Map references of extracellular and membrane current with 
            #  xtra mechanism
            h.setpointer(seg._ref_e_extracellular,'ex',seg.xtra)
            h.setpointer(seg._ref_i_membrane,'im',seg.xtra)


# Create a version that accounts for multi-electrode stimulation
# Notably, this assumes a "unit" stimulus current pulse and scales Rx instead
def setupTransferImpedance_v2(Stim, RGC):
    # Loop over all sections
    for sec in RGC.section_list:
        # Loop over all segments in section
        for ii, seg in enumerate(sec):
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
            seg.xtra.rx  = rx  # in MOhm
                                
            # Map references of extracellular and membrane current with 
            #  xtra mechanism
            h.setpointer(seg._ref_e_extracellular,'ex',seg.xtra)
            h.setpointer(seg._ref_i_membrane,'im',seg.xtra)


def setupTransferImpedance(Stim, RGC):
    # Loop over all sections
    for sec in RGC.section_list:

        # Loop over all segments in section
        for seg in sec:
            rx = 0
            for i in range(len(Stim['electrodes'])):
                if Stim['electrode_type'] == 'disk':
                    # Distance from electrode to center of each segment [cm]
                    r = 1e-4*\
                        np.sqrt((seg.xtra.x - (Stim['electrodes'][i][0]))**2\
                             + (seg.xtra.y - (Stim['electrodes'][i][1]))**2)
                    # Vertical distance [cm]
                    z = 1e-4*(seg.xtra.z-Stim['electrodes'][i][2])

                    # Transfer impedance [MOhm]  
                    rDisk = 1e-4*Stim['elecDiam']/2             
                    rx += 1e-6*(2.0*Stim['rhoExt'])/(4.0*np.pi*rDisk) \
                        * np.arcsin((2*rDisk) / \
                        (np.sqrt((r-rDisk)**2+z**2)+np.sqrt((r+rDisk)**2+z**2)))
                elif Stim['electrode_type'] == 'point':
                    dist = dist = 1e-4*\
                        np.sqrt((seg.xtra.x - (Stim['electrodes'][i][0]))**2 + \
                                (seg.xtra.y - (Stim['electrodes'][i][1]))**2 + \
                                (seg.xtra.z - (Stim['electrodes'][i][2]))**2)
                    rx += 1e-6*Stim['rhoExt']/(4.0*np.pi*dist)

            # Set transfer impedance for each segment
            seg.xtra.rx  = rx  # in MOhm
                                
            # Map references of extracellular and membrane current with 
            #  xtra mechanism
            h.setpointer(seg._ref_e_extracellular,'ex',seg.xtra)
            h.setpointer(seg._ref_i_membrane,'im',seg.xtra)


def setupStimulus_unit(Stim):
    # Helper variables
    initDur = Stim['initDur']    
    dt = Stim['dt']
    delay = Stim['delay']
    dur = Stim['dur']
    stop = Stim['stop']
    pulseShape = Stim['pulseShape']
    freq = Stim['frequency']
    I = 1

    ### SET UP PULSE AND PLAY INTO XTRA
    timeSteps = int((stop) / dt)
    tsvec = np.concatenate(([0,0],initDur+np.linspace(0,stop,int(stop/dt)+1)))
    isvec = np.concatenate(([0,0],np.zeros(int(stop/dt)+1)))

    # Triphasic
    if pulseShape == 'triphasic':
        # calucalte time between stimuli at given frequency
        timeStepsPerStim = int(dur*3 / dt)
        indxFirstStimWindow = int(delay/dt)
        indxLastStimWindow = int(timeSteps - timeStepsPerStim)
        timeStepsBetweenStim = int((1/freq) * (1e3) / dt)

        # assemble stimulus pulse at given frequency 
        pulseIndx = indxFirstStimWindow

        while pulseIndx < indxLastStimWindow:
            isvec[2+pulseIndx:2+pulseIndx+int((dur)/dt)+1] = \
                    I*Stim['pulseRatioTri'][0]
            isvec[2+pulseIndx+int((dur)/dt):2+pulseIndx+int((dur*2)/dt)+1] = \
                    I*Stim['pulseRatioTri'][1]
            isvec[2+pulseIndx+int((dur*2)/dt):2+pulseIndx+int((dur*3)/dt)+1] = \
                    I*Stim['pulseRatioTri'][2]
            pulseIndx = pulseIndx + timeStepsBetweenStim

    # Biphasic
    elif pulseShape == 'biphasic':
        # calucalte time between stimuli at given frequency
        timeStepsPerStim = int(dur*2 / dt)
        indxFirstStimWindow = int(delay/dt)
        indxLastStimWindow = int(timeSteps - timeStepsPerStim)
        timeStepsBetweenStim = int((1/freq) * (1e3) / dt)

        # assemble stimulus pulse at given frequency 
        pulseIndx = indxFirstStimWindow

        while pulseIndx < indxLastStimWindow:
            isvec[2+pulseIndx:2+pulseIndx+int((dur)/dt)+1] = \
                    I*Stim['pulseRatioBi'][0]
            isvec[2+pulseIndx+int((dur)/dt):2+pulseIndx+int((dur*2)/dt)+1] = \
                    I*Stim['pulseRatioBi'][1]
            pulseIndx = pulseIndx + timeStepsBetweenStim

    # Monophasic
    elif pulseShape == 'monophasic':
        # calucalte time between stimuli at given frequency
        timeStepsPerStim = int(dur / dt)
        indxFirstStimWindow = int(delay/dt)
        indxLastStimWindow = int(timeSteps - timeStepsPerStim)
        timeStepsBetweenStim = int((1/freq) * (1e3) / dt)

        # assemble stimulus pulse at given frequency 
        pulseIndx = indxFirstStimWindow

        while pulseIndx < indxLastStimWindow:
            isvec[2+pulseIndx:2+pulseIndx+int((dur)/dt)+1] = I
            pulseIndx = pulseIndx + timeStepsBetweenStim

    return h.Vector(tsvec), h.Vector(isvec)*1e-3

def setupStimulus(Stim):
    # Helper variables
    initDur = Stim['initDur']    
    dt = Stim['dt']
    delay = Stim['delay']
    dur = Stim['dur']
    stop = Stim['stop']
    pulseShape = Stim['pulseShape']
    freq = Stim['frequency']
    I = Stim['amp']

    ### SET UP PULSE AND PLAY INTO XTRA
    timeSteps = int((stop) / dt)
    tsvec = np.concatenate(([0,0],initDur+np.linspace(0,stop,int(stop/dt)+1)))
    isvec = np.concatenate(([0,0],np.zeros(int(stop/dt)+1)))

    # Triphasic
    if pulseShape == 'triphasic':
        # calucalte time between stimuli at given frequency
        timeStepsPerStim = int(dur*3 / dt)
        indxFirstStimWindow = int(delay/dt)
        indxLastStimWindow = int(timeSteps - timeStepsPerStim)
        timeStepsBetweenStim = int((1/freq) * (1e3) / dt)

        # assemble stimulus pulse at given frequency 
        pulseIndx = indxFirstStimWindow

        while pulseIndx < indxLastStimWindow:
            isvec[2+pulseIndx:2+pulseIndx+int((dur)/dt)+1] = \
                    I*Stim['pulseRatioTri'][0]
            isvec[2+pulseIndx+int((dur)/dt):2+pulseIndx+int((dur*2)/dt)+1] = \
                    I*Stim['pulseRatioTri'][1]
            isvec[2+pulseIndx+int((dur*2)/dt):2+pulseIndx+int((dur*3)/dt)+1] = \
                    I*Stim['pulseRatioTri'][2]
            pulseIndx = pulseIndx + timeStepsBetweenStim

    # Biphasic
    elif pulseShape == 'biphasic':
        # calucalte time between stimuli at given frequency
        timeStepsPerStim = int(dur*2 / dt)
        indxFirstStimWindow = int(delay/dt)
        indxLastStimWindow = int(timeSteps - timeStepsPerStim)
        timeStepsBetweenStim = int((1/freq) * (1e3) / dt)

        # assemble stimulus pulse at given frequency 
        pulseIndx = indxFirstStimWindow

        while pulseIndx < indxLastStimWindow:
            isvec[2+pulseIndx:2+pulseIndx+int((dur)/dt)+1] = \
                    I*Stim['pulseRatioBi'][0]
            isvec[2+pulseIndx+int((dur)/dt):2+pulseIndx+int((dur*2)/dt)+1] = \
                    I*Stim['pulseRatioBi'][1]
            pulseIndx = pulseIndx + timeStepsBetweenStim

    # Monophasic
    elif pulseShape == 'monophasic':
        # calucalte time between stimuli at given frequency
        timeStepsPerStim = int(dur / dt)
        indxFirstStimWindow = int(delay/dt)
        indxLastStimWindow = int(timeSteps - timeStepsPerStim)
        timeStepsBetweenStim = int((1/freq) * (1e3) / dt)

        # assemble stimulus pulse at given frequency 
        pulseIndx = indxFirstStimWindow

        while pulseIndx < indxLastStimWindow:
            isvec[2+pulseIndx:2+pulseIndx+int((dur)/dt)+1] = I
            pulseIndx = pulseIndx + timeStepsBetweenStim

    return h.Vector(tsvec), h.Vector(isvec)*1e-3

# Apply a bisection algorithm in order to find the threshold of a certain neuron 
def findthreshold(low, high, epsilon, Stim, RGC, detect='axon'):
    lowBound = low 
    upBound = high 
    error = upBound-lowBound 
    ap_number = 1
    thresholds = []
    vRec = []
    t_vec = h.Vector().record(h._ref_t)

    if detect == 'axon':
        print('detecting AP at the distal axon')
        APC = h.APCount(1, sec = RGC.axon)
        APC.thresh = 0 
    elif detect == 'soma':
        print('detecting AP at the soma')
        APC = h.APCount(0.5, sec = RGC.cell.soma)
        APC.thresh = 0 

    while error > epsilon: 

        vRec_cell1_axon = []
        for seg in RGC.axon:
            vRec_cell1_axon.append(h.Vector().record(RGC.axon(seg.x)._ref_v))
 
        scaling_var = lowBound+((upBound-lowBound)/2) 

        print("Starting trial with",str(scaling_var),"upper bound =",\
                str(upBound),"lower bound =",str(lowBound))
        
        Stim['amp'] = Stim['polarity']*scaling_var
        tsvec, isvec = setupStimulus(Stim)
        isvec.play(h._ref_is_xtra, tsvec, 1) 

        # run sim
        # Init
        h.init()
        h.tstop = Stim['initDur']
        h.dt = Stim['initDt']
        h.finitialize(Stim['vInit'])
        h.continuerun(h.tstop)

        h.tstop = Stim['initDur'] + Stim['stop']
        h.steps_per_ms = int(1/Stim['dt'])
        h.dt = Stim['dt']
        h.continuerun(h.tstop)
        
        # determine timing of detected AP, has to be after stim
        if (APC.time <= Stim['initDur'] + Stim['delay'] + 2*Stim['dur']):
            print('AP timing detected was: ',APC.time,'ms')
            APC_count = 0
        else:
            APC_count = APC.n

        if APC_count < ap_number: 
            lowBound = scaling_var 
        
        else:  
            upBound = scaling_var 
            thresholds.append(scaling_var)
            vRec.append(vRec_cell1_axon)

        error = upBound-lowBound 
    
    # Populate return variables if no threshold was found
    if not thresholds: 
        print('No threshold found')
        thresholds.append(np.inf)
        vRec.append(np.inf)
    
    print('--------- Finished threshold finding. -------')

    return(thresholds[-1], t_vec, vRec[-1])

def findthreshold_fixedSteps(Stim, RGC, detect='axon'):
    currents = np.logspace(0,5,num=50,base=2)-1

    lower_threshold = np.inf
    upper_threshold = np.inf

    ap_number = 1
    vRec = []
    t_vec = h.Vector().record(h._ref_t)

    for current in currents:
        if detect == 'axon':
            APC = h.APCount(1, sec = RGC.axon)
            APC.thresh = 0 
        elif detect == 'soma':
            APC = h.APCount(1, sec = RGC.cell.soma)
            APC.thresh = 0 
        
        Stim['amp'] = Stim['polarity']*float(current)
        tsvec, isvec = setupStimulus(Stim)
        isvec.play(h._ref_is_xtra, tsvec, 1) 

        vRec_axon = []
        for seg in RGC.axon:
            vRec_axon.append(h.Vector().record(RGC.axon(seg.x)._ref_v))

        # run sim
        h.init()
        h.tstop = Stim['initDur']
        h.dt = Stim['initDt']
        h.finitialize(Stim['vInit'])
        h.continuerun(h.tstop)

        h.tstop = Stim['initDur'] + Stim['stop']
        h.steps_per_ms = int(1/Stim['dt'])
        h.dt = Stim['dt']
        h.continuerun(h.tstop)

        # determine timing of detected AP, has to be after stim
        if (APC.time <= Stim['initDur'] + Stim['delay'] + 2*Stim['dur']):
            APC_count = 0
        else:
            APC_count = APC.n
        
        if (APC_count == ap_number) and (current < lower_threshold):  
            lower_threshold = current
            vRec.append(vRec_axon)
            print('Threshold found at: ', str(current), ' uA')
            vRec.append(vRec_axon)
        elif (APC_count < ap_number) and (current > lower_threshold):  
            upper_threshold = current            
            print('Upper threshold found at: ', str(current), ' uA')
            vRec.append(vRec_axon)
            return lower_threshold, upper_threshold, t_vec, vRec

    return lower_threshold, upper_threshold, t_vec, vRec

def findthreshold_fixedSteps_lowerOnly(Stim, RGC):
    currents = np.logspace(0,5,num=50,base=2)-1

    lower_threshold = np.inf

    ap_number = 1
    vRec = []
    t_vec = h.Vector().record(h._ref_t)

    for current in currents:
        APC_axon = h.APCount(1, sec = RGC.axon)
        APC_axon.thresh = 0 
        
        Stim['amp'] = Stim['polarity']*float(current)
        tsvec, isvec = setupStimulus(Stim)
        isvec.play(h._ref_is_xtra, tsvec, 1) 

        vRec_axon = []
        for seg in RGC.axon:
            vRec_axon.append(h.Vector().record(RGC.axon(seg.x)._ref_v))

        # run sim
        h.init()
        h.tstop = Stim['initDur']
        h.dt = Stim['initDt']
        h.finitialize(Stim['vInit'])
        h.continuerun(h.tstop)

        h.tstop = Stim['initDur'] + Stim['stop']
        h.steps_per_ms = int(1/Stim['dt'])
        h.dt = Stim['dt']
        h.continuerun(h.tstop)
        
        if (APC_axon.n >= ap_number) and (current < lower_threshold):  
            lower_threshold = current
            vRec.append(vRec_axon)
            # print('Threshold found at: ', str(current), ' uA')
            # if threshold found, finish and return threshold
            return lower_threshold, t_vec, vRec

    return lower_threshold, t_vec, vRec

'''
Find lower threshold of activation for a given cell. The searching procedure 
starts by using fixed steps to determine the relative range of currents that
can activate the cell. Then, it selects that finding as an upper-bound for 
the bisection search algorithm. It then iteratively finds the lowest current
required to activate the cell. 

Notably, this method mitigates confounds from upper-threshold effects. 
'''
def lowerThreshold(Stim, RGC, lower, upper, num, epsilon):
    currents = np.logspace(lower,upper,num,base=2)-1

    lower_threshold = np.inf
    vRec = []
    ap_number = 1
    t_vec = h.Vector().record(h._ref_t)

    for ii, current in enumerate(currents):
        APC_axon = h.APCount(1, sec = RGC.axon)
        APC_axon.thresh = 0 
        
        Stim['amp'] = Stim['polarity']*float(current)
        tsvec, isvec = setupStimulus(Stim)
        isvec.play(h._ref_is_xtra, tsvec, 1) 

        vRec_axon = []
        for seg in RGC.axon:
            vRec_axon.append(h.Vector().record(RGC.axon(seg.x)._ref_v))

        # run sim
        h.init()
        h.tstop = Stim['initDur']
        h.dt = Stim['initDt']
        h.finitialize(Stim['vInit'])
        h.continuerun(h.tstop)

        h.tstop = Stim['initDur'] + Stim['stop']
        h.steps_per_ms = int(1/Stim['dt'])
        h.dt = Stim['dt']
        h.continuerun(h.tstop)
        
        if (APC_axon.n >= ap_number): 
            vRec = np.array(vRec_axon)
            lower_threshold = current
            # Now that we have found one current that causes a spike, apply
            #  the bisection method to narrow down exact threshold
            upBound = current
            lowBound = currents[ii-1]
            error = upBound-lowBound 

            while error > epsilon: 
                scaling_var = lowBound+((upBound-lowBound)/2) 

                print("Starting trial with",str(scaling_var),"upper bound =",\
                        str(upBound),"lower bound =",str(lowBound))
                
                Stim['amp'] = Stim['polarity']*scaling_var
                tsvec, isvec = setupStimulus(Stim)
                isvec.play(h._ref_is_xtra, tsvec, 1) 

                vRec_axon_1 = []
                for seg in RGC.axon:
                    vRec_axon_1.append(h.Vector().record(RGC.axon(seg.x)._ref_v))

                # run sim
                # Init
                h.init()
                h.tstop = Stim['initDur']
                h.dt = Stim['initDt']
                h.finitialize(Stim['vInit'])
                h.continuerun(h.tstop)

                h.tstop = Stim['initDur'] + Stim['stop']
                h.steps_per_ms = int(1/Stim['dt'])
                h.dt = Stim['dt']
                h.continuerun(h.tstop)
                
                if (vRec_axon_1[-1].max() > 0):
                    upBound = scaling_var 
                    lower_threshold = scaling_var
                    vRec = np.array(vRec_axon_1) # cast copy to save
                else:
                    lowBound = scaling_var

                error = upBound-lowBound

            print('Lower threshold found at: ', str(lower_threshold), ' uA')
            return lower_threshold, t_vec, vRec

    # -- Commented out for debugging, uncomment for final version
    print('No threshold found in current range.')
    return lower_threshold, t_vec, vRec


'''
Find lower threshold of activation for a given cell. The searching procedure 
starts by using fixed steps to determine the relative range of currents that
can activate the cell. Then, it selects that finding as an upper-bound for 
the bisection search algorithm. It then iteratively finds the lowest current
required to activate the cell. 

Notably, this method mitigates confounds from upper-threshold effects. 
'''
def lowerThreshold_v2(Stim, RGC, lower, upper, num, epsilon):
    currents = np.logspace(lower,upper,num,base=2)-1

    lower_threshold = np.inf
    vRec = []
    ap_number = 1
    t_vec = h.Vector().record(h._ref_t) 

    for ii, current in enumerate(currents):
        APC_axon = h.APCount(1, sec = RGC.axon)
        APC_axon.thresh = 0 
        
        Stim['amp'] = [Stim['polarity']*float(current)]
        setupTransferImpedance_v2(Stim, RGC)
        tsvec, isvec = setupStimulus_unit(Stim)
        isvec.play(h._ref_is_xtra, tsvec, 1)

        vRec_axon = []
        for seg in RGC.axon:
            vRec_axon.append(h.Vector().record(RGC.axon(seg.x)._ref_v))

        print('Testing current: ',Stim['amp'])

        # run sim
        h.init()
        h.tstop = Stim['initDur']
        h.dt = Stim['initDt']
        h.finitialize(Stim['vInit'])
        h.continuerun(h.tstop)

        h.tstop = Stim['initDur'] + Stim['stop']
        h.steps_per_ms = int(1/Stim['dt'])
        h.dt = Stim['dt']
        h.continuerun(h.tstop)
        
        if (APC_axon.n >= ap_number): 
            vRec = np.array(vRec_axon)
            lower_threshold = current
            # Now that we have found one current that causes a spike, apply
            #  the bisection method to narrow down exact threshold
            upBound = current
            lowBound = currents[ii-1]
            error = upBound-lowBound 

            while error > epsilon: 
                scaling_var = lowBound+((upBound-lowBound)/2) 

                print("Starting trial with",str(scaling_var),"upper bound =",\
                        str(upBound),"lower bound =",str(lowBound))
                
                Stim['amp'] = [Stim['polarity']*scaling_var]
                setupTransferImpedance_v2(Stim, RGC) 
                tsvec, isvec = setupStimulus_unit(Stim)
                isvec.play(h._ref_is_xtra, tsvec, 1)

                vRec_axon_1 = []
                for seg in RGC.axon:
                    vRec_axon_1.append(h.Vector().record(RGC.axon(seg.x)._ref_v))

                # run sim
                # Init
                h.init()
                h.tstop = Stim['initDur']
                h.dt = Stim['initDt']
                h.finitialize(Stim['vInit'])
                h.continuerun(h.tstop)

                h.tstop = Stim['initDur'] + Stim['stop']
                h.steps_per_ms = int(1/Stim['dt'])
                h.dt = Stim['dt']
                h.continuerun(h.tstop)
                
                if (vRec_axon_1[-1].max() > 0):
                    upBound = scaling_var 
                    lower_threshold = scaling_var
                    vRec = vRec_axon_1.copy() # cast copy to save
                else:
                    lowBound = scaling_var

                error = upBound-lowBound

            print('Lower threshold found at: ', str(lower_threshold), ' uA')
            return lower_threshold, t_vec, vRec

    # -- Commented out for debugging, uncomment for final version
    print('No threshold found in current range.')
    return lower_threshold, t_vec, vRec


def lowerThreshold_v3(active_x, Stim, RGC, lower, upper, num, epsilon):
    currents = np.logspace(lower,upper,num,base=2)-1

    lower_threshold = np.inf
    vRec = []
    ap_number = 1
    t_vec = h.Vector().record(h._ref_t) 

    for ii, current in enumerate(currents):
        APC_axon = h.APCount(1, sec = RGC.axon)
        APC_axon.thresh = 0 
        
        Stim['amp'] = [Stim['polarity']*float(current)]
        print('Testing current: ', Stim['polarity']*float(current))

        setupTransferImpedance_v3(Stim, RGC, active_x)
        tsvec, isvec = setupStimulus_unit(Stim)
        isvec.play(h._ref_is_xtra, tsvec, 1)

        vRec_axon = []
        for seg in RGC.axon:
            vRec_axon.append(h.Vector().record(RGC.axon(seg.x)._ref_v))

        # run sim
        h.init()
        h.tstop = Stim['initDur']
        h.dt = Stim['initDt']
        h.finitialize(Stim['vInit'])
        h.continuerun(h.tstop)

        h.tstop = Stim['initDur'] + Stim['stop']
        h.steps_per_ms = int(1/Stim['dt'])
        h.dt = Stim['dt']
        h.continuerun(h.tstop)
        
        if (APC_axon.n >= ap_number): 
            vRec = np.array(vRec_axon)
            lower_threshold = current
            # Now that we have found one current that causes a spike, apply
            #  the bisection method to narrow down exact threshold
            upBound = current
            lowBound = currents[ii-1]
            error = upBound-lowBound 

            while error > epsilon: 
                scaling_var = lowBound+((upBound-lowBound)/2) 

                print("Starting trial with",str(scaling_var),"upper bound =",\
                        str(upBound),"lower bound =",str(lowBound))
                
                Stim['amp'] = [Stim['polarity']*scaling_var]
                setupTransferImpedance_v2(Stim, RGC) 
                tsvec, isvec = setupStimulus_unit(Stim)
                isvec.play(h._ref_is_xtra, tsvec, 1)

                vRec_axon_1 = []
                for seg in RGC.axon:
                    vRec_axon_1.append(h.Vector().record(RGC.axon(seg.x)._ref_v))

                # run sim
                # Init
                h.init()
                h.tstop = Stim['initDur']
                h.dt = Stim['initDt']
                h.finitialize(Stim['vInit'])
                h.continuerun(h.tstop)

                h.tstop = Stim['initDur'] + Stim['stop']
                h.steps_per_ms = int(1/Stim['dt'])
                h.dt = Stim['dt']
                h.continuerun(h.tstop)
                
                if (vRec_axon_1[-1].max() > 0):
                    upBound = scaling_var 
                    lower_threshold = scaling_var
                    vRec = vRec_axon_1.copy() # cast copy to save
                else:
                    lowBound = scaling_var

                error = upBound-lowBound

            print('Lower threshold found at: ', str(lower_threshold), ' uA')
            return lower_threshold, t_vec, vRec

    # -- Commented out for debugging, uncomment for final version
    print('No threshold found in current range.')
    return lower_threshold, t_vec, vRec

'''
Find if a multi-elec stimuli caused a spike in the cell of interest
'''
def spiking_multi(Stim, RGC):
    spike = 0
    vRec = []
    ap_number = 1
    t_vec = h.Vector().record(h._ref_t)

    APC_axon = h.APCount(1, sec = RGC.axon)
    APC_axon.thresh = 0 
    
    tsvec, isvec = setupStimulus_unit(Stim)
    isvec.play(h._ref_is_xtra, tsvec, 1) 

    vRec_axon = []
    for seg in RGC.axon:
        vRec_axon.append(h.Vector().record(RGC.axon(seg.x)._ref_v))

    # run sim
    h.init()
    h.tstop = Stim['initDur']
    h.dt = Stim['initDt']
    h.finitialize(Stim['vInit'])
    h.continuerun(h.tstop)

    h.tstop = Stim['initDur'] + Stim['stop']
    h.steps_per_ms = int(1/Stim['dt'])
    h.dt = Stim['dt']
    h.continuerun(h.tstop)
        
    if (APC_axon.n >= ap_number): 
        spike = 1
        print('Amplitudes: ', Stim['amp'], ' yielded a SPIKE!')
        return spike, t_vec, vRec
    else:
        print('Amplitudes: ', Stim['amp'], ' did not yield a spike.')
        return spike, t_vec, vRec

# return electrode locations for 30um or 60um array
def generate_electrode_map(array):
    if array == 512:
        return constants.LITKE_512_ARRAY_MAP
    elif array == 519:
        return constants.LITKE_519_ARRAY_MAP
    
# Pickle a file and then compress it into a file with extension 
def compressed_pickle(title, data):
    with bz2.BZ2File(title + '.pbz2', 'w') as f: 
        cPickle.dump(data, f)

# Load any compressed pickle file
def decompress_pickle(file):
    data = bz2.BZ2File(file+'.pbz2', 'rb')
    data = cPickle.load(data)
    return data
