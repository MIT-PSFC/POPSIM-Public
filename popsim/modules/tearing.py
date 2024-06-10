import chex
import os
from popsim import ModuleBase
from popsim import PACKAGE_ROOT
from jaxtyping import Array
import jax.numpy as jnp
from popsim import interp
import numpy as np

"""
An example template to copy and paste when creating a new module.
"""


@chex.dataclass
class Tearing(ModuleBase):
    @chex.dataclass
    class Config:
        # Define the data that configures the module and will be static during the simulation.

        magx_time: Array # deg, a time array on which to output the data
        thincurr_file: str # path and filename of txt file defining the ThinCurr transfer functions
        ods_file: str # path to the ODS object 

    @chex.dataclass
    class State:
        # Define the differential state variables that will be integrated during the simulation.
        # If a variable is defined in here, then the module must output its time derivative in the __call__ method.
        dWdt: Array # m/s, Nxm 
        dFdt: Array # Hz/s
        dt: float # s
        
        # NEED SOMETHING LIKE A LOCKING STATE HERE

    @chex.dataclass
    class Output:
        # Define the output variables that will be returned by the module.
        # below m refers to the number of modes
        # island_phases: Array # deg, Nxm 
        # island_widths: Array # m, Nxm 
        # island_freqs: Array # Hz
        # onset_time: float # s
        # lowmn_bp: Array # T, Nx16
        # lowmn_br: Array # T, Nx16
        test: str # just a test output

    @chex.dataclass
    class Params:
        # Define the, possibly time dependent, parameters that will be passed to the module.
        trigger_time: float 
        q2_rot_freq: float 
        rot_dur: float  
        locking_dur: float 
        survival_time: float 
        disrupt_time: float 
        dur_tq_to_spike: float 
        dur_cq: float 
        sim_dt: float # s, constant time-step for this simulation
        

    config: Config
    phi_probes: Array  # deg, toroidal locations of probes
    phi_sens: Array # deg, toroidal locations of sensors

    def __init__(self, config):
        self.config = config
        
        
        default_path = os.path.join(PACKAGE_ROOT, "data", config.thincurr_file)
        out = np.loadtxt(default_path, skiprows=1)
        
        print(jnp.shape(out))
        
        freq = out[:,0]
        bppA = out[:,1] # Bp (poloidal field) per Amp of tearing mode current
        brpA = out[:,2] # Br (radial field) per Amp of tearing mode current
        
        
        self.freqTransf = freq 
        self.bppA = bppA
        self.brpA = brpA
        
       
        # we also want to build interpolated functions that can be called for 
        # any frequency in order to get the br and bp
        # if we can't find a interp1d equivalent, we might need to fit the 
        # ThinCurr data with polynomials and simply hard-code those polynomials
        # here. 
        self.fbppA = interp.interp(freq, bppA) # name stands for "function, bp per A"
        self.fbrpA = interp.interp(freq, brpA) # name stands for "function, br per A"
        
        self.magx_time = config.magx_time

    def __call__(self, state: State, params: Params) -> State | Output:
        
        # MAGXTimeIndex = jnp.argmin(jnp.abs(state.time - self.magx_time))
       
        # initialize all state variables
        dWdt = jnp.array([0., 0., 0.])
        dFdt = jnp.array([0., 0., 0.])
        


        # is the mode triggered?
        if state.time > params.trigger_time:
           
            '''
            Here is where we will step the island widths via the Modified 
            Rutherford Eqn.
            '''
            # growth rates are hard-coded throughout this module. 
            # we should anticipate something more intelligent in the future. 
            dWdt = jnp.array([4e-2/0.5, 10e-2/0.5, 3e-2/0.5]) # m/s  
            
            
            # we will need a different island growth rate during the TQ. Is
            # this the TQ? 
            if (state.time - params.disruptTime > 0. and 
                state.time - (params.disruptTime + params.dur_tq_to_spike) < 0):
                
                dWdt = jnp.array([4e-2/0.002, 10e-2/0.002, 3e-2/0.002]) # m/s
                
            # we will need another growth rate (or decay rate) during the CQ.
            # Is this the CQ?
            if state.time - (params.disruptTime + params.dur_tq_to_spike) > 0:
                dWdt = jnp.array([-4e-2/0.005, -10e-2/0.005, -3e-2/0.005]) # m/s            
                       
                                    
           
            # here we map from island width to current
            # curPerW = 1e3/1e-2 # 1 kA/cm
            # dIdt = dWdt*curPerW
           
            
            # now we step the island frequency                       
            if (state.time - params.onset_time > 0 and  
                state.time - params.onset_time < params.rot_dur):
                # this mode is in the 'rotating' state
                dFdt = -(params.q2_rot_freq/2.)/(params.rot_dur) #Hz/s
            
            elif (state.time - params.onset_time > params.rot_dur and 
                  state.time - params.onset_time - params.rot_dur < params.locking_dur):
                # this mode is in the 'decelerating' state                     
                dFdt = -(params.q2_rot_freq/2.)/(params.locking_dur) #Hz/s
                
            else:
                # this mode is locked
                dFdt = 0.
                

            # self.update_MAGX_Measurement(SimTime, SimTimeIndex, PrevTime, MAGXTimeIndex)
        
        
        # Make a state_dot.
        state_dot = Tearing.State(dWdt=dWdt, dFdt=dFdt, dt=params.sim_dt)
        # Make an output.
        out = Tearing.Output(test='Good day')
        return state_dot, out
    
    
