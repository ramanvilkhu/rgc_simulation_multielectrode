import numpy as np 
from neuron import h

# Foundational function that applies extracellular voltage to the neuron through time
def integrate(tstop, dt, pulse_shape, voltages, scaling_var, RGC):
    h.load_file("stdrun.hoc")
    h.init() # re-initialize cell at the start of every integration
    h.v_init = -63.9 # resting membrane potential # -70 -> -64
    h.finitialize(h.v_init) # initialize the starting voltage throughout the cell
    h.fcurrent() # initialize the membrane currents  
    h.cvode_active(0)
    h.tstop = tstop # total stimulation time (msec)
    h.dt = dt
    t = 0 # time indexing variable
    while round(h.t,4) < h.tstop:
        s = 0 # space indexing variable
        # for sec in h.allsec():
        for sec in RGC.cell.soma.wholetree():
            for seg in sec:
                seg.e_extracellular = pulse_shape[t]*voltages[s]*scaling_var
                s = s+1
        t = t+1
        h.fadvance()

# Adapted from Paul
def integrate_general(tstop, dt, pulse_shape, voltages_list, \
                        scaling_var_list, list_RGCs):
    # Extract number of electrodes 
    num_elecs = len(voltages_list)

    h.load_file("stdrun.hoc")
    h.init() # re-initialize cell at the start of every integration
    h.v_init = -63.9 # resting membrane potential
    h.finitialize(h.v_init) # initialize the starting voltage throughout cell
    h.fcurrent() # initialize the membrane currents  
    h.cvode_active(0) # use normal integration method, not CVode
    h.tstop = tstop # total stimulation time (msec)
    h.dt = dt
    t = 0 # time indexing variable
    while round(h.t,4) < h.tstop:
        s = 0 # space indexing variable
        for RGC in list_RGCs:
            for sec in RGC.section_list:
                for seg in sec:
                    for i in range(num_elecs):
                        voltages = voltages_list[i]
                        scaling_var = scaling_var_list[i]
                        seg.e_extracellular += pulse_shape[t]*voltages[s]*scaling_var
                    s = s+1
        t = t+1
        h.fadvance()

# Apply a bisection algorithm in order to find the threshold of a certain neuron 
def findthreshold(sv_min,sv_max,epsilon,pulse_shape,voltages,tstop,dt,ap_number):
    '''
    sv_min is the lowest current (scaling variable * 1 A) that we want to test
    sv_max is the highest current (scaling variable * 1 A) that we want to test
    epsilon is the accuracy with which we want to know the threshold (i.e. within 1 uA)
    simulation_length is the length of each simulation (msec)
    '''
    lowBound = sv_min # the lower bound is initialized as the desired minimum scaling variable called by the function
    upBound = sv_max # the upper bound is initialized as the desired maximum scaling variable called by the function
    error = upBound-lowBound # the error is range in which we know the threshold
    
    while error > epsilon: # while the range that we are looking at is larger than the accurary we want
 
        scaling_var = lowBound+((upBound-lowBound)/2) # scaling variable is halfway between the max and min
        
        APC_axon = h.APCount(1, sec = h.axon) # initialize variable to count the action potentials # TODO: 0.5 -> 1
        APC_axon.thresh = 0 # cutoff to count as an action potential (0 mV)

        print("Starting trial with",str(scaling_var),"upper bound =",str(upBound),"lower bound =",str(lowBound))
        integrate(tstop, dt, pulse_shape, voltages, scaling_var) # run the simulation
        
        if APC_axon.n < ap_number: # if there is less than one action potential; the cell doesn't fire
            lowBound = scaling_var # the lower bound becomes the previous scaling variable
        
        else:  # if there is at least one action potential; the cell does fire
            upBound = scaling_var # the upper bound becomes the previous scaling variable
            
        error = upBound-lowBound # the error is the range between the max and min
    
    return(scaling_var) # returns the final scaling variable (threshold) once the range is sufficently small

# Apply a bisection algorithm in order to find the threshold of a certain neuron for paired pulse
def findthreshold_PP(sv_min,sv_max,epsilon,pulse_shape,voltages,tstop,dt,ap_number):
    '''
    sv_min is the lowest current (scaling variable * 1 A) that we want to test
    sv_max is the highest current (scaling variable * 1 A) that we want to test
    epsilon is the accuracy with which we want to know the threshold (i.e. within 1 uA)
    simulation_length is the length of each simulation (msec)
    '''
    lowBound = sv_min # the lower bound is initialized as the desired minimum scaling variable called by the function
    upBound = sv_max # the upper bound is initialized as the desired maximum scaling variable called by the function
    error = upBound-lowBound # the error is range in which we know the threshold

    # determine starting and ending points of applied paired pulse 
    pulse_start = np.nonzero(pulse_shape)[0][0] * dt # msec
    pulse_end = np.nonzero(pulse_shape)[0][-1] * dt # msec

    # create list of tested amplitudes that worked 
    working_amps = []

    while error > epsilon: # while the range that we are looking at is larger than the accurary we want
 
        scaling_var = lowBound+((upBound-lowBound)/2) # scaling variable is halfway between the max and min
        
        APC_axon = h.APCount(1, sec = h.axon) # initialize variable to count the action potentials # TODO: 0.5 -> 1
        APC_axon.thresh = 0 # cutoff to count as an action potential (0 mV)

        print("Starting trial with",str(scaling_var),"upper bound =",str(upBound),"lower bound =",str(lowBound))
        integrate(tstop, dt, pulse_shape, voltages, scaling_var) # run the simulation

        spike_timing = APC_axon.time # time of the last action potential reaching end of axon

        print('Debugging: spike timing variable -- ', spike_timing)
        
        # # If 2 spikes, ensure spike timing of last one is within 5ms of last 
        # #  the last pulse and first spike is within 5ms of the first pulse
        # # If 1 spike, ensure spike timing is within 5ms of last pulse 
        if APC_axon.n == 0:
            lowBound = scaling_var
        elif APC_axon.n == 1:
            if ap_number == 1:
                # If only spike is after 2nd pulse, try lower currents
                if spike_timing < (pulse_end + 5) and spike_timing > (pulse_end - 0.15):
                    working_amps.append(scaling_var)
                    upBound = scaling_var
                # If only spike is after 1st pulse, try lower currents (need summation)
                elif spike_timing > pulse_start and spike_timing < (pulse_end - 0.15):
                    upBound = scaling_var
                else:
                    lowBound = scaling_var
            elif ap_number == 2:
                # If you require 2 spikes and only get one, need to increase current
                lowBound = scaling_var
        elif APC_axon.n == 2:
            print('Saw two spikes, spike timing: ', spike_timing)
            if ap_number == 1:
                # If you require 1 spike and get 2, need to decrease current
                upBound = scaling_var
            elif ap_number == 2:
                # Need to ensure the second spike happens after the second pulse
                if spike_timing > pulse_end + 5:
                    lowBound = scaling_var
                else:
                    working_amps.append(scaling_var)
                    upBound = scaling_var
        else:
            print('Error: saw more than 2 spikes')
            upBound = scaling_var
            # return np.inf, [np.inf]
            
        error = upBound-lowBound # the error is the range between the max and min
    
    return(scaling_var, working_amps) # returns the final scaling variable (threshold) once the range is sufficently small


def findthreshold_multi(sv_min,sv_max,epsilon,pulse_shape_list,voltages_list,tstop,dt,ap_number):
    '''
    sv_min is the lowest current (scaling variable * 1 A) that we want to test
    sv_max is the highest current (scaling variable * 1 A) that we want to test
    epsilon is the accuracy with which we want to know the threshold (i.e. within 1 uA)
    simulation_length is the length of each simulation (msec)
    '''
    lowBound = sv_min # the lower bound is initialized as the desired minimum scaling variable called by the function
    upBound = sv_max # the upper bound is initialized as the desired maximum scaling variable called by the function
    error = upBound-lowBound # the error is range in which we know the threshold
    
    while error > epsilon: # while the range that we are looking at is larger than the accurary we want
 
        scaling_var = lowBound+((upBound-lowBound)/2) # scaling variable is halfway between the max and min

        scaling_var_list = []
        scaling_var_list.append(scaling_var)
        scaling_var_list.append(scaling_var)
        
        APC_axon = h.APCount(0.5, sec = h.axon) # initialize variable to count the action potentials
        APC_axon.thresh = 0 # cutoff to count as an action potential (0 mV)

        print("Starting trial with",str(scaling_var),"upper bound =",str(upBound),"lower bound =",str(lowBound))
        integrate_multi(tstop, dt, pulse_shape_list, voltages_list, scaling_var_list) # run the simulation
        
        if APC_axon.n < ap_number: # if there is less than one action potential; the cell doesn't fire
            lowBound = scaling_var # the lower bound becomes the previous scaling variable
        
        else:  # if there is at least one action potential; the cell does fire
            upBound = scaling_var # the upper bound becomes the previous scaling variable
            
        error = upBound-lowBound # the error is the range between the max and min
    
    return(scaling_var) # returns the final scaling variable (threshold) once the range is sufficently small

# Load coordinate and voltage data from a .txt file
def load_txt_file(path_to_obj,dtype):
	output = open(path_to_obj,'r')
	out_file = list()
	
	for line in output:
		tmp = line.split()
		tmp1 = list()
		for ii in tmp:
			if dtype == 'float':
				tmp1.append(float(ii))
			elif dtype == 'int':
				tmp1.append(int(ii))
			else:
				tmp1.append(ii)
		out_file.append(tmp1)
	output.close()
	
	return out_file

# Find the neural compartment in which the action potential initiates
def find_initiation_point(AP_recording_dict, cell_length):
    times={}
    for i in np.arange(10):
        name = 'soma_'+str(i)+'_AP'
        times[name] = AP_recording_dict[name].time   
    for i in np.arange(40):
        name = 'AH_'+str(i)+'_AP'
        times[name] = AP_recording_dict[name].time    
    for i in np.arange(40):
        name = 'SOCB_'+str(i)+'_AP'
        times[name] = AP_recording_dict[name].time    
    for i in np.arange(90):
        name = 'NR_'+str(i)+'_AP'
        times[name] = AP_recording_dict[name].time    
    for i in np.arange(cell_length-181):
        name = 'axon_'+str(i)+'_AP'
        times[name] = AP_recording_dict[name].time
    key_min = min(times.keys(), key=(lambda k: times[k]))
    return key_min