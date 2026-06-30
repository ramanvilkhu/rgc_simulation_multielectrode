import numpy as np

def find_nearest(array, value):
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return idx

def stimulus_tri(PW,frequency,tstop,dt,latency): 
    # PW: msec; frequency: Hz; total_time, dt: msec
    time=np.arange(0,tstop,dt) # time vector is in msec
    time=np.around(time,decimals=6)
    time_between_stimuli=(np.float(1)/np.float(frequency))*(1e3) # convert frequency (Hz) to time between stimuli (msec)
    
    pulse_start_times=np.arange(latency,tstop,time_between_stimuli) # in msec
    pulse_phase1_times=np.arange(latency+PW,tstop,time_between_stimuli) # in msec
    pulse_phase2_times=np.arange(latency+(2*PW),tstop,time_between_stimuli) # in msec
    pulse_phase3_times=np.arange(latency+(3*PW),tstop,time_between_stimuli) # in msec
    number_of_pulses=len(pulse_start_times)
    
    amplitudes=[]
    for i in np.arange(len(time)):
        amplitudes.append(0)
    
    start_indices=[]
    pulse1_indices=[]
    pulse2_indices=[]
    pulse3_indices=[]
    for j in np.arange(number_of_pulses):
        s=find_nearest(time, pulse_start_times[j])
        start_indices.append(s)
        p1=find_nearest(time, pulse_phase1_times[j])
        pulse1_indices.append(p1)
        p2=find_nearest(time,pulse_phase2_times[j])
        pulse2_indices.append(p2)
        p3=find_nearest(time,pulse_phase3_times[j])
        pulse3_indices.append(p3)
    
    first_phase_indices=[]
    second_phase_indices=[]
    third_phase_indices=[]
    for k in np.arange(number_of_pulses):
        p1=np.arange(start_indices[k],pulse1_indices[k],1)
        first_phase_indices.append(p1)
        p2=np.arange(pulse1_indices[k],pulse2_indices[k],1)
        second_phase_indices.append(p2)
        p3=np.arange(pulse2_indices[k],pulse3_indices[k],1)
        third_phase_indices.append(p3)
    first_phase_indices_fixed=np.concatenate(first_phase_indices,axis=0)
    second_phase_indices_fixed=np.concatenate(second_phase_indices,axis=0)
    third_phase_indices_fixed=np.concatenate(third_phase_indices,axis=0)

    for (index, amplitude) in zip(first_phase_indices_fixed, amplitudes):
        amplitudes[index]= 2/3 # sign and relative amplitude of first-phase of stimulus pulse
    for (index, amplitude) in zip(second_phase_indices_fixed, amplitudes):
        amplitudes[index]= -1  # sign and relative amplitude of second-phase of stimulus pulse
    for (index, amplitude) in zip(third_phase_indices_fixed, amplitudes):
        amplitudes[index]= 1/3  # sign and relative amplitude of second-phase of stimulus pulse
    amplitudes[0]=0
                
    return(amplitudes)

def stimulus_pulse(PW,frequency,tstop,dt): 
    # PW: msec; frequency: Hz; total_time, dt: msec
    time=np.arange(0,tstop,dt) # time vector is in msec
    time=np.around(time,decimals=6)
    time_between_stimuli=(np.float(1)/np.float(frequency))*(1e3) # convert frequency (Hz) to time between stimuli (msec)
    
    pulse_start_times=np.arange(2,tstop,time_between_stimuli) # in msec
    pulse_stop_times=np.arange(2+PW,tstop,time_between_stimuli) # in msec
    number_of_pulses=len(pulse_start_times)
    
    amplitudes=[]
    for i in np.arange(len(time)):
        amplitudes.append(0)
    
    start_indices=[]
    pulse1_indices=[]
    for j in np.arange(number_of_pulses):
        s=find_nearest(time, pulse_start_times[j])
        start_indices.append(s)
        p1=find_nearest(time, pulse_stop_times[j])
        pulse1_indices.append(p1)
    
    first_phase_indices=[]
    for k in np.arange(number_of_pulses):
        p1=np.arange(start_indices[k],pulse1_indices[k],1)
        first_phase_indices.append(p1)
        
    first_phase_indices_fixed=np.concatenate(first_phase_indices,axis=0)

    for (index, amplitude) in zip(first_phase_indices_fixed, amplitudes):
        amplitudes[index]= 1 # sign and relative amplitude of first-phase of stimulus pulse
    amplitudes[0]=0
                
    return(amplitudes)

def stimulus_biphasic(PW,frequency,tstop,dt, latency): 
    # PW: msec; frequency: Hz; total_time, dt: msec
    time=np.arange(0,tstop,dt) # time vector is in msec
    time=np.around(time,decimals=6)
    time_between_stimuli=(np.float(1)/np.float(frequency))*(1e3) # convert frequency (Hz) to time between stimuli (msec)
    
    pulse_start_times=np.arange(latency,tstop,time_between_stimuli) # in msec
    pulse_middle_times=np.arange(latency+(1*PW),tstop,time_between_stimuli) # in msec
    pulse_end_times=np.arange(latency+(2*PW),tstop,time_between_stimuli) # in msec
    number_of_pulses=len(pulse_start_times)
    
    amplitudes=[]
    for i in np.arange(len(time)):
        amplitudes.append(0)
    
    start_indices=[]
    middle_indices=[]
    end_indices=[]
    for j in np.arange(number_of_pulses):
        s=find_nearest(time, pulse_start_times[j])
        start_indices.append(s)
        m=find_nearest(time, pulse_middle_times[j])
        middle_indices.append(m)
        e=find_nearest(time,pulse_end_times[j])
        end_indices.append(e)
    
    first_phase_indices=[]
    second_phase_indices=[]
    for k in np.arange(number_of_pulses):
        n=np.arange(start_indices[k],middle_indices[k],1)
        first_phase_indices.append(n)
        p=np.arange(middle_indices[k],end_indices[k],1)
        second_phase_indices.append(p)
    first_phase_indices_fixed=np.concatenate(first_phase_indices,axis=0)
    second_phase_indices_fixed=np.concatenate(second_phase_indices,axis=0)

    for (index, amplitude) in zip(first_phase_indices_fixed, amplitudes):
        amplitudes[index]= -1 # sign and relative amplitude of first-phase of stimulus pulse
    for (index, amplitude) in zip(second_phase_indices_fixed, amplitudes):
        amplitudes[index]= 1 # sign and relative amplitude of second-phase of stimulus pulse
    amplitudes[0]=0
                
    return(amplitudes)

def stimulus_tri_dpp(PW,frequency,tstop,dt,latency,gap): 
    # PW: msec; frequency: Hz; total_time, dt: msec
    time=np.arange(0,tstop,dt) # time vector is in msec
    time=np.around(time,decimals=6)
    time_between_stimuli=(np.float(1)/np.float(frequency))*(1e3) # convert frequency (Hz) to time between stimuli (msec)
    
    pulse_start_times=np.arange(latency,tstop,time_between_stimuli) # in msec
    pulse_phase1_times=np.arange(latency+PW,tstop,time_between_stimuli) # in msec
    pulse_phase2_times=np.arange(latency+(2*PW),tstop,time_between_stimuli) # in msec
    pulse_phase3_times=np.arange(latency+(3*PW),tstop,time_between_stimuli) # in msec
    pulse_start_times_2=np.arange(latency+(3*PW)+gap,tstop,time_between_stimuli) # in msec
    pulse_phase1_times_2=np.arange(latency+(3*PW)+gap+PW,tstop,time_between_stimuli) # in msec
    pulse_phase2_times_2=np.arange(latency+(3*PW)+gap+(2*PW),tstop,time_between_stimuli) # in msec
    pulse_phase3_times_2=np.arange(latency+(3*PW)+gap+(3*PW),tstop,time_between_stimuli) # in msec
    number_of_pulses=len(pulse_start_times)
    
    amplitudes=[]
    for i in np.arange(len(time)):
        amplitudes.append(0)
    
    start_indices=[]
    pulse1_indices=[]
    pulse2_indices=[]
    pulse3_indices=[]
    start_indices_2=[]
    pulse1_indices_2=[]
    pulse2_indices_2=[]
    pulse3_indices_2=[]
    for j in np.arange(number_of_pulses):
        s=find_nearest(time, pulse_start_times[j])
        start_indices.append(s)
        p1=find_nearest(time, pulse_phase1_times[j])
        pulse1_indices.append(p1)
        p2=find_nearest(time,pulse_phase2_times[j])
        pulse2_indices.append(p2)
        p3=find_nearest(time,pulse_phase3_times[j])
        pulse3_indices.append(p3)
        s=find_nearest(time, pulse_start_times_2[j])
        start_indices_2.append(s)
        p1=find_nearest(time, pulse_phase1_times_2[j])
        pulse1_indices_2.append(p1)
        p2=find_nearest(time,pulse_phase2_times_2[j])
        pulse2_indices_2.append(p2)
        p3=find_nearest(time,pulse_phase3_times_2[j])
        pulse3_indices_2.append(p3)
    
    first_phase_indices=[]
    second_phase_indices=[]
    third_phase_indices=[]
    first_phase_indices_2=[]
    second_phase_indices_2=[]
    third_phase_indices_2=[]
    for k in np.arange(number_of_pulses):
        p1=np.arange(start_indices[k],pulse1_indices[k],1)
        first_phase_indices.append(p1)
        p2=np.arange(pulse1_indices[k],pulse2_indices[k],1)
        second_phase_indices.append(p2)
        p3=np.arange(pulse2_indices[k],pulse3_indices[k],1)
        third_phase_indices.append(p3)
        p1=np.arange(start_indices_2[k],pulse1_indices_2[k],1)
        first_phase_indices_2.append(p1)
        p2=np.arange(pulse1_indices_2[k],pulse2_indices_2[k],1)
        second_phase_indices_2.append(p2)
        p3=np.arange(pulse2_indices_2[k],pulse3_indices_2[k],1)
        third_phase_indices_2.append(p3)

    first_phase_indices_fixed=np.concatenate(first_phase_indices,axis=0)
    second_phase_indices_fixed=np.concatenate(second_phase_indices,axis=0)
    third_phase_indices_fixed=np.concatenate(third_phase_indices,axis=0)
    first_phase_indices_fixed_2=np.concatenate(first_phase_indices_2,axis=0)
    second_phase_indices_fixed_2=np.concatenate(second_phase_indices_2,axis=0)
    third_phase_indices_fixed_2=np.concatenate(third_phase_indices_2,axis=0)

    for (index, amplitude) in zip(first_phase_indices_fixed, amplitudes):
        amplitudes[index]= 2/3 # sign and relative amplitude of first-phase of stimulus pulse
    for (index, amplitude) in zip(second_phase_indices_fixed, amplitudes):
        amplitudes[index]= -1  # sign and relative amplitude of second-phase of stimulus pulse
    for (index, amplitude) in zip(third_phase_indices_fixed, amplitudes):
        amplitudes[index]= 1/3  # sign and relative amplitude of second-phase of stimulus pulse
    for (index, amplitude) in zip(first_phase_indices_fixed_2, amplitudes):
        amplitudes[index]= 2/3 # sign and relative amplitude of first-phase of stimulus pulse
    for (index, amplitude) in zip(second_phase_indices_fixed_2, amplitudes):
        amplitudes[index]= -1  # sign and relative amplitude of second-phase of stimulus pulse
    for (index, amplitude) in zip(third_phase_indices_fixed_2, amplitudes):
        amplitudes[index]= 1/3  # sign and relative amplitude of second-phase of stimulus pulse
    amplitudes[0]=0
                
    return(amplitudes)