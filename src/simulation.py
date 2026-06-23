import numpy as np
from axon_ellipse import Build_Single_RGC
from neuron import h 
import sys
import pandas as pd

# set working dir
h.chdir("/Volumes/Lab/Users/vilkhu/workspace/rgc_simulaton/src")

# Load NEURON dll
if sys.platform.startswith('linux'): # LINUX
    h.nrn_load_dll('../nrn/x86_64/.libs/libnrnmech.so') 
elif sys.platform.startswith('win32'): # WINDOWS
    h.nrn_load_dll('../nrn/nrnmech.dll') 
elif sys.platform.startswith('darwin'): # MACOS
    h.nrn_load_dll('../nrn/x86_64/libnrnmech.dylib') 

#  Initialize cell morphology
class Local_Cell:
    def __repr__(self):
        return str(self.cell)
    
    def __init__(self):
        h.load_file('stdrun.hoc')
        h.load_file('interpxyz.hoc')
        h.load_file('setpointers.hoc')
        self.file = 'RGC_morph_4.hoc'
        # self.file = 'RGC_morph_4_15um.hoc'
        self.load_cell(self.file)
        self.Stim = {}
    
    def load_cell(self, file_path):
        h.load_file(file_path)
        self.cell = h.Cell()

    def build_subsets(self):
        self.all = h.SectionList()
        self.section_list = self.cell.soma.wholetree()
        self.soma_center = [self.cell.soma.x3d(1),self.cell.soma.y3d(1),\
                                self.cell.soma.z3d(1)]
    
    def build_cell(self, params_file, spike_biophysics):
        # load in params file 
        self.load_parameters(params_file, spike_biophysics)
        # build dendrite-somatic part of cell 
        self.build_morphology_dendrites_soma()
        self.build_biophysics_dendrites()
        self.build_biophysics_soma()
        # build axonal part of the cell
        self.create_sections()
        self.build_biophysics_axon_hillock()
        self.build_morphology_axon_hillock()
        self.build_biophysics_socb()
        self.build_morphology_socb()
        self.build_biophysics_nr()
        self.build_morphology_nr()
        self.build_biophysics_distal_axon()
        self.build_morphology_distal_axon()
        # define subset list, useful for looping over sections
        self.build_subsets()
        # set pointers
        h.setpointers()
    
    def shift_cell_x_y_z(self, new_x, new_y, new_z):
        for sec in self.section_list:
            for i in range(int(sec.n3d())):
                sec.pt3dchange(
                    i, 
                    new_x - self.soma_center[0] + sec.x3d(i), 
                    new_y - self.soma_center[1] + sec.y3d(i), 
                    new_z - self.soma_center[2] + sec.z3d(i), 
                    sec.diam3d(i))

        # update hoc pointers, important to have correct x_xtra,y_xtra,z_xtra
        h.setpointers()
        # re-calculate soma center and update
        self.soma_center = [self.cell.soma.x3d(1),self.cell.soma.y3d(1),\
                                self.cell.soma.z3d(1)]
    
    ### Rotate cell around x axis
    def _rotateX(self, theta):
        for section in self.section_list:
            for i in range(section.n3d()):
                x = section.x3d(i)
                y = section.y3d(i)
                z = section.z3d(i)
                c = np.cos(theta)
                s = np.sin(theta)
                yprime = y * c - z * s
                zprime = y * s + z * c
                section.pt3dchange(i, x, yprime, zprime, section.diam3d(i))
        
        h.setpointers()

    ### Rotate cell around y axis
    def _rotateY(self, theta):
        for section in self.section_list:
            for i in range(section.n3d()):
                x = section.x3d(i)
                y = section.y3d(i)
                z = section.z3d(i)
                c = np.cos(theta)
                s = np.sin(theta)
                xprime = x * c + z * s
                zprime = -x * s + z * c
                section.pt3dchange(i, xprime, y, zprime, section.diam3d(i))
        
        h.setpointers()
    
    ### Rotate cell around z axis
    def _rotateZ(self, theta):
        for section in self.section_list:
            for i in range(section.n3d()):
                x = section.x3d(i)
                y = section.y3d(i)
                z = section.z3d(i)
                c = np.cos(theta)
                s = np.sin(theta)
                xprime = x * c - y * s
                yprime = x * s + y * c
                section.pt3dchange(i, xprime, yprime, z, section.diam3d(i))

        h.setpointers()
    
    def load_parameters(self,params_file, spike_biophysics):
        # load in params file (csv) using pandas
        self.spike_biophysics = str(spike_biophysics) # define which temp to use
        self.params = pd.read_csv(params_file)
    
    def create_sections(self):
        # axon hillock
        self.AH = h.Section(name='AH', cell=self)
        # sodium channel band
        self.SOCB = h.Section(name='SOCB', cell=self)
        # axon narrow region
        self.NR = h.Section(name='NR', cell=self)
        # distal axon
        self.axon = h.Section(name='axon', cell=self)

    def build_morphology_dendrites_soma(self):
        # top compartment of the soma (mm)
        self.StartingCoordinates = (-.01194,0,.020) 
        # self.StartingCoordinates = (-.01194,0,.015) #TODO: I changed this
        # 100 um left and 20 um above (mm)
        self.NFLEntryCoordinates = (-0.11194,0,.040)
        # self.NFLEntryCoordinates = (-0.11194,0,.035)
        self.Length = 3051 # um
        self.axon_coords = Build_Single_RGC(self.StartingCoordinates,\
                                                    self.NFLEntryCoordinates,\
                                                    self.Length)

    # This function is very hard-coded, as is, needs to be updated
    def build_biophysics_dendrites(self):
        # load params for dendrites
        param = self.params.iloc[0]
        # set dendrite biophysics
        for i in np.arange(int(param['nseg'])):
            self.cell.dend[i].nseg = int(np.round((self.cell.dend[i].L)/10)+1)
            self.cell.dend[i].cm = float(param['cap_membrane'])
            self.cell.dend[i].Ra = float(param['axial_resistance']) 
            self.cell.dend[i].insert('pas') 
            self.cell.dend[i].e_pas = float(param['pass_mem_potential'])
            self.cell.dend[i].g_pas = float(param['leakage_conductance']) 
            self.cell.dend[i].insert(self.spike_biophysics) 
            self.cell.dend[i].ena = float(param['eNa']) 
            self.cell.dend[i].ek = float(param['eK']) 
            if self.spike_biophysics == 'mammalian_spike_35':
                self.cell.dend[i].gnabar_mammalian_spike_35 = float(param['gNa']) 
                self.cell.dend[i].gkbar_mammalian_spike_35 = float(param['gK'])
                self.cell.dend[i].gcabar_mammalian_spike_35 = float(param['gCa'])
                self.cell.dend[i].gkcbar_mammalian_spike_35 = float(param['gKc'])
            else:
                self.cell.dend[i].gnabar_mammalian_spike = float(param['gNa']) 
                self.cell.dend[i].gkbar_mammalian_spike = float(param['gK'])
                self.cell.dend[i].gcabar_mammalian_spike = float(param['gCa'])
                self.cell.dend[i].gkcbar_mammalian_spike = float(param['gKc'])
            self.cell.dend[i].insert('cad') 
            self.cell.dend[i].depth_cad = float(param['depth_cad'])
            self.cell.dend[i].taur_cad = float(param['taur_cad'])
            self.cell.dend[i].insert('extracellular') 
            self.cell.dend[i].xg[0] = float(param['mem_conductance']) 
            self.cell.dend[i].e_extracellular = float(param['e_ext']) 
            self.cell.dend[i].insert('xtra')
    
    def build_biophysics_soma(self):
        # load params for soma
        param = self.params.iloc[1]
        # set soma biophysics
        self.cell.soma.nseg = int(param['nseg'])
        self.cell.soma.cm = float(param['cap_membrane'])
        self.cell.soma.diam = float(param['diameter'])
        self.cell.soma.Ra = float(param['axial_resistance']) 
        self.cell.soma.insert('pas') 
        self.cell.soma.e_pas = float(param['pass_mem_potential'])
        self.cell.soma.g_pas = float(param['leakage_conductance']) 
        self.cell.soma.insert(self.spike_biophysics) 
        self.cell.soma.ena = float(param['eNa']) 
        self.cell.soma.ek = float(param['eK']) 
        if self.spike_biophysics == 'mammalian_spike_35':
            self.cell.soma.gnabar_mammalian_spike_35 = float(param['gNa']) 
            self.cell.soma.gkbar_mammalian_spike_35 = float(param['gK'])
            self.cell.soma.gcabar_mammalian_spike_35 = float(param['gCa'])
            self.cell.soma.gkcbar_mammalian_spike_35 = float(param['gKc'])
        else:
            self.cell.soma.gnabar_mammalian_spike = float(param['gNa']) 
            self.cell.soma.gkbar_mammalian_spike = float(param['gK'])
            self.cell.soma.gcabar_mammalian_spike = float(param['gCa'])
            self.cell.soma.gkcbar_mammalian_spike = float(param['gKc'])
        self.cell.soma.insert('cad') 
        self.cell.soma.depth_cad = float(param['depth_cad'])
        self.cell.soma.taur_cad = float(param['taur_cad'])
        self.cell.soma.insert('extracellular') 
        self.cell.soma.xg[0] = float(param['mem_conductance']) 
        self.cell.soma.e_extracellular = float(param['e_ext']) 
        self.cell.soma.insert('xtra')
    
    def build_biophysics_axon_hillock(self):
        # load params for axon hillock
        param = self.params.iloc[2]
        # set axon hillock biophysics
        self.AH.nseg = int(param['nseg'])
        self.AH.cm = float(param['cap_membrane'])
        self.AH.Ra = float(param['axial_resistance']) 
        self.AH.insert('pas') 
        self.AH.e_pas = float(param['pass_mem_potential'])
        self.AH.g_pas = float(param['leakage_conductance']) 
        self.AH.insert(self.spike_biophysics) 
        self.AH.ena = float(param['eNa']) 
        self.AH.ek = float(param['eK']) 
        if self.spike_biophysics == 'mammalian_spike_35':
            self.AH.gnabar_mammalian_spike_35 = float(param['gNa']) 
            self.AH.gkbar_mammalian_spike_35 = float(param['gK'])
            self.AH.gcabar_mammalian_spike_35 = float(param['gCa'])
            self.AH.gkcbar_mammalian_spike_35 = float(param['gKc'])
        else:
            self.AH.gnabar_mammalian_spike = float(param['gNa']) 
            self.AH.gkbar_mammalian_spike = float(param['gK'])
            self.AH.gcabar_mammalian_spike = float(param['gCa'])
            self.AH.gkcbar_mammalian_spike = float(param['gKc'])
        self.AH.insert('cad') 
        self.AH.depth_cad = float(param['depth_cad'])
        self.AH.taur_cad = float(param['taur_cad'])
        self.AH.insert('extracellular') 
        self.AH.xg[0] = float(param['mem_conductance']) 
        self.AH.e_extracellular = float(param['e_ext']) 
        self.AH.insert('xtra')
    
    def build_morphology_axon_hillock(self):
        # load params for axon hillock
        param = self.params.iloc[2]
        # extract axon geometry
        xpts_axon = self.axon_coords[:,0]*(1000)
        ypts_axon = self.axon_coords[:,1]*(1000)
        zpts_axon = self.axon_coords[:,2]*(1000)
         # set axon hillock morphology 
        self.AH.connect(self.cell.soma)
        self.AH.pt3dclear()
        for i in range(int(param['length'])+1):
            self.AH.pt3dadd(xpts_axon[i], ypts_axon[i], zpts_axon[i], \
                         float(param['diameter']))
    
    def build_biophysics_socb(self):
        # load params for socb
        param = self.params.iloc[3]
        # set socb biophysics
        self.SOCB.nseg = int(param['nseg'])
        self.SOCB.cm = float(param['cap_membrane'])
        self.SOCB.Ra = float(param['axial_resistance']) 
        self.SOCB.insert('pas') 
        self.SOCB.e_pas = float(param['pass_mem_potential'])
        self.SOCB.g_pas = float(param['leakage_conductance']) 
        self.SOCB.insert(self.spike_biophysics) 
        self.SOCB.ena = float(param['eNa']) 
        self.SOCB.ek = float(param['eK']) 
        if self.spike_biophysics == 'mammalian_spike_35':
            self.SOCB.gnabar_mammalian_spike_35 = float(param['gNa']) 
            self.SOCB.gkbar_mammalian_spike_35 = float(param['gK'])
            self.SOCB.gcabar_mammalian_spike_35 = float(param['gCa'])
            self.SOCB.gkcbar_mammalian_spike_35 = float(param['gKc'])
        else:
            self.SOCB.gnabar_mammalian_spike = float(param['gNa']) 
            self.SOCB.gkbar_mammalian_spike = float(param['gK'])
            self.SOCB.gcabar_mammalian_spike = float(param['gCa'])
            self.SOCB.gkcbar_mammalian_spike = float(param['gKc'])
        self.SOCB.insert('cad') 
        self.SOCB.depth_cad = float(param['depth_cad'])
        self.SOCB.taur_cad = float(param['taur_cad'])
        self.SOCB.insert('extracellular') 
        self.SOCB.xg[0] = float(param['mem_conductance']) 
        self.SOCB.e_extracellular = float(param['e_ext']) 
        self.SOCB.insert('xtra')
    
    def build_morphology_socb(self):
        # load params for socb
        param = self.params.iloc[3]
        # extract axon geometry
        xpts_axon = self.axon_coords[:,0]*(1000)
        ypts_axon = self.axon_coords[:,1]*(1000)
        zpts_axon = self.axon_coords[:,2]*(1000)
         # set socb morphology 
        self.SOCB.connect(self.AH)
        self.SOCB.pt3dclear()
        length = int(param['length'])
        for i in range(length+1):
            # Taper diameter from AH to NR
            AH_diam = float(self.params.iloc[2]['diameter'])
            NR_diam = float(self.params.iloc[4]['diameter'])
            scaling_factor = (AH_diam - NR_diam)/length
            SOCB_diameter = AH_diam - (scaling_factor*i)  

            AH_length = int(self.params.iloc[2]['length'])
            self.SOCB.pt3dadd(xpts_axon[i+AH_length],ypts_axon[i+AH_length],\
                           zpts_axon[i+AH_length],SOCB_diameter)
            
    def build_biophysics_nr(self):
        # load params for narrow region
        param = self.params.iloc[4]
        # set narrow region biophysics
        self.NR.nseg = int(param['nseg'])
        self.NR.cm = float(param['cap_membrane'])
        self.NR.Ra = float(param['axial_resistance']) 
        self.NR.insert('pas') 
        self.NR.e_pas = float(param['pass_mem_potential'])
        self.NR.g_pas = float(param['leakage_conductance']) 
        self.NR.insert(self.spike_biophysics) 
        self.NR.ena = float(param['eNa']) 
        self.NR.ek = float(param['eK']) 
        if self.spike_biophysics == 'mammalian_spike_35':
            self.NR.gnabar_mammalian_spike_35 = float(param['gNa']) 
            self.NR.gkbar_mammalian_spike_35 = float(param['gK'])
            self.NR.gcabar_mammalian_spike_35 = float(param['gCa'])
            self.NR.gkcbar_mammalian_spike_35 = float(param['gKc'])
        else:
            self.NR.gnabar_mammalian_spike = float(param['gNa']) 
            self.NR.gkbar_mammalian_spike = float(param['gK'])
            self.NR.gcabar_mammalian_spike = float(param['gCa'])
            self.NR.gkcbar_mammalian_spike = float(param['gKc'])
        self.NR.insert('cad') 
        self.NR.depth_cad = float(param['depth_cad'])
        self.NR.taur_cad = float(param['taur_cad'])
        self.NR.insert('extracellular') 
        self.NR.xg[0] = float(param['mem_conductance']) 
        self.NR.e_extracellular = float(param['e_ext']) 
        self.NR.insert('xtra')
    
    def build_morphology_nr(self):
        # load params for NR
        param = self.params.iloc[4]
        # extract axon geometry
        xpts_axon = self.axon_coords[:,0]*(1000)
        ypts_axon = self.axon_coords[:,1]*(1000)
        zpts_axon = self.axon_coords[:,2]*(1000)
         # set NR morphology 
        self.NR.connect(self.SOCB)
        self.NR.pt3dclear()
        length = int(param['length'])
        for i in range(length+1):
            AH_length = int(self.params.iloc[2]['length'])
            SOCB_length = int(self.params.iloc[3]['length'])
            offset = AH_length + SOCB_length
            self.NR.pt3dadd(xpts_axon[i+offset],ypts_axon[i+offset],\
                           zpts_axon[i+offset],float(param['diameter']))
            
    def build_biophysics_distal_axon(self):
        # load params for distal axon
        param = self.params.iloc[5]
        # set distal axon biophysics
        self.axon.nseg = int(param['nseg'])
        self.axon.cm = float(param['cap_membrane'])
        self.axon.Ra = float(param['axial_resistance']) 
        self.axon.insert('pas') 
        self.axon.e_pas = float(param['pass_mem_potential'])
        self.axon.g_pas = float(param['leakage_conductance']) 
        self.axon.insert(self.spike_biophysics) 
        self.axon.ena = float(param['eNa']) 
        self.axon.ek = float(param['eK']) 
        if self.spike_biophysics == 'mammalian_spike_35':
            self.axon.gnabar_mammalian_spike_35 = float(param['gNa']) 
            self.axon.gkbar_mammalian_spike_35 = float(param['gK'])
            self.axon.gcabar_mammalian_spike_35 = float(param['gCa'])
            self.axon.gkcbar_mammalian_spike_35 = float(param['gKc'])
        else:
            self.axon.gnabar_mammalian_spike = float(param['gNa']) 
            self.axon.gkbar_mammalian_spike = float(param['gK'])
            self.axon.gcabar_mammalian_spike = float(param['gCa'])
            self.axon.gkcbar_mammalian_spike = float(param['gKc'])
        self.axon.insert('cad') 
        self.axon.depth_cad = float(param['depth_cad'])
        self.axon.taur_cad = float(param['taur_cad'])
        self.axon.insert('extracellular') 
        self.axon.xg[0] = float(param['mem_conductance']) 
        self.axon.e_extracellular = float(param['e_ext']) 
        self.axon.insert('xtra')
    
    def build_morphology_distal_axon(self):
        # load params for distal axon
        param = self.params.iloc[5]
        # extract axon geometry
        xpts_axon = self.axon_coords[:,0]*(1000)
        ypts_axon = self.axon_coords[:,1]*(1000)
        zpts_axon = self.axon_coords[:,2]*(1000)
         # set NR morphology 
        self.axon.connect(self.NR)
        self.axon.pt3dclear()
        length = int(param['length'])
        for i in range(length+1):
            AH_length = int(self.params.iloc[2]['length'])
            SOCB_length = int(self.params.iloc[3]['length'])
            NR_length = int(self.params.iloc[4]['length'])
            offset = AH_length + SOCB_length + NR_length
            self.axon.pt3dadd(xpts_axon[i+offset],ypts_axon[i+offset],\
                           zpts_axon[i+offset],float(param['diameter']))
        
        








    

        






