"""
Integrate-and-Fire Neuron Model Visualization

This script simulates and visualizes the leaky integrate-and-fire (LIF) neuron model:
- dV(t)/dt = -(V(t) - V_rest) / lambda + I
- When V(t) >= V_threshold, the neuron fires and V resets to V_rest
- A refractory period T_ref can be included
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

# Model parameters
V_rest = -70.0      # Resting potential (mV)
V_threshold = -55.0  # Threshold potential (mV)
lambda_const = 10.0  # Membrane time constant (ms)
T_ref = 2.0         # Refractory period (ms)
dt = 0.1            # Time step (ms)
t_max = 500.0       # Total simulation time (ms)

# Input current (can be constant or time-varying)
def input_current(t):
    """Define input current as a function of time"""
    # Constant current
    #if 20 <= t <= 180:
    #    return 2.5  # Sufficient to cause spiking
    #else:
    #    return 0.0
    
    #Alternative: Step currents
    if 20 <= t < 60:
         return 1.5
    elif 80 <= t < 120:
         return 3.0
    elif 140 <= t < 180:
         return 2.0
    elif 200 <= t < 240:
            return 2.5
    elif 240 <= t < 280:
            return 2.0
    elif 300 <= t < 340:
            return 3.0
    else:
         return 0.0


def simulate_lif_neuron(V_rest, V_threshold, lambda_const, T_ref, I_func, dt, t_max):
    """
    Simulate the leaky integrate-and-fire neuron
    
    Parameters:
    -----------
    V_rest : float
        Resting membrane potential
    V_threshold : float
        Threshold potential for spike generation
    lambda_const : float
        Membrane time constant
    T_ref : float
        Refractory period
    I_func : function
        Input current as a function of time
    dt : float
        Time step for simulation
    t_max : float
        Total simulation time
        
    Returns:
    --------
    t : array
        Time points
    V : array
        Membrane potential over time
    spikes : array
        Spike times
    I : array
        Input current over time
    """
    # Initialize arrays
    n_steps = int(t_max / dt)
    t = np.linspace(0, t_max, n_steps)
    V = np.zeros(n_steps)
    I = np.zeros(n_steps)
    V[0] = V_rest
    
    # Track spikes and refractory period
    spikes = []
    ref_time_remaining = 0
    
    # Simulation loop
    for i in range(1, n_steps):
        I[i] = I_func(t[i])
        
        # Check if in refractory period
        if ref_time_remaining > 0:
            V[i] = V_rest
            ref_time_remaining -= dt
        else:
            # Integrate-and-fire dynamics
            # dV/dt = -(V - V_rest) / lambda + I
            dV = (-(V[i-1] - V_rest) / lambda_const + I[i]) * dt
            V[i] = V[i-1] + dV
            
            # Check for threshold crossing (spike)
            if V[i] >= V_threshold:
                spikes.append(t[i])
                V[i] = V_rest  # Reset to resting potential
                ref_time_remaining = T_ref  # Enter refractory period
    
    return t, V, np.array(spikes), I


def plot_lif_simulation(t, V, spikes, I, V_rest, V_threshold):
    """
    Create a comprehensive visualization of the LIF neuron simulation
    """
    fig = plt.figure(figsize=(14, 10))
    gs = GridSpec(3, 2, figure=fig, hspace=0.3, wspace=0.3)
    
    # Main plot: Membrane potential over time
    ax1 = fig.add_subplot(gs[0, :])
    ax1.plot(t, V, 'b-', linewidth=1.5, label='Membrane Potential')
    ax1.axhline(V_threshold, color='r', linestyle='--', linewidth=1.5, 
                label=f'Threshold ({V_threshold} mV)')
    ax1.axhline(V_rest, color='g', linestyle='--', linewidth=1.5, 
                label=f'Rest ({V_rest} mV)')
    
    # Mark spikes
    if len(spikes) > 0:
        spike_indices = [np.argmin(np.abs(t - spike)) for spike in spikes]
        ax1.plot(spikes, [V_threshold] * len(spikes), 'r^', 
                markersize=10, label=f'Spikes (n={len(spikes)})')
    
    ax1.set_xlabel('Time (ms)', fontsize=12)
    ax1.set_ylabel('Membrane Potential (mV)', fontsize=12)
    ax1.set_title('Leaky Integrate-and-Fire Neuron Model', fontsize=14, fontweight='bold')
    ax1.legend(loc='upper right')
    ax1.grid(True, alpha=0.3)
    
    # Input current
    ax2 = fig.add_subplot(gs[1, :])
    ax2.plot(t, I, 'purple', linewidth=2)
    ax2.set_xlabel('Time (ms)', fontsize=12)
    ax2.set_ylabel('Input Current (I)', fontsize=12)
    ax2.set_title('Input Current', fontsize=12, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    
    # Inter-spike interval histogram
    ax3 = fig.add_subplot(gs[2, 0])
    if len(spikes) > 1:
        isi = np.diff(spikes)
        ax3.hist(isi, bins=20, color='orange', alpha=0.7, edgecolor='black')
        ax3.set_xlabel('Inter-Spike Interval (ms)', fontsize=11)
        ax3.set_ylabel('Count', fontsize=11)
        ax3.set_title('Inter-Spike Interval Distribution', fontsize=12, fontweight='bold')
        ax3.axvline(np.mean(isi), color='r', linestyle='--', linewidth=2, 
                   label=f'Mean: {np.mean(isi):.2f} ms')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
    else:
        ax3.text(0.5, 0.5, 'Not enough spikes\nfor ISI analysis', 
                ha='center', va='center', transform=ax3.transAxes, fontsize=12)
        ax3.set_title('Inter-Spike Interval Distribution', fontsize=12, fontweight='bold')
    
    # Spike raster and firing rate
    ax4 = fig.add_subplot(gs[2, 1])
    if len(spikes) > 0:
        # Spike raster
        ax4.eventplot([spikes], lineoffsets=1, linelengths=0.8, 
                     colors='black', linewidths=2)
        ax4.set_xlabel('Time (ms)', fontsize=11)
        ax4.set_ylabel('Spike Train', fontsize=11)
        ax4.set_ylim([0.5, 1.5])
        ax4.set_yticks([1])
        ax4.set_yticklabels(['Neuron'])
        
        # Calculate and display firing rate
        firing_rate = len(spikes) / (t[-1] / 1000.0)  # Convert to Hz
        ax4.set_title(f'Spike Train (Firing Rate: {firing_rate:.2f} Hz)', 
                     fontsize=12, fontweight='bold')
        ax4.grid(True, alpha=0.3, axis='x')
    else:
        ax4.text(0.5, 0.5, 'No spikes detected', 
                ha='center', va='center', transform=ax4.transAxes, fontsize=12)
        ax4.set_title('Spike Train', fontsize=12, fontweight='bold')
    
    plt.tight_layout()
    return fig


# Run simulation
print("Simulating Leaky Integrate-and-Fire Neuron Model...")
print(f"Parameters:")
print(f"  V_rest = {V_rest} mV")
print(f"  V_threshold = {V_threshold} mV")
print(f"  λ (membrane time constant) = {lambda_const} ms")
print(f"  T_ref (refractory period) = {T_ref} ms")
print(f"  Simulation time = {t_max} ms")
print(f"  Time step = {dt} ms")

t, V, spikes, I = simulate_lif_neuron(V_rest, V_threshold, lambda_const, 
                                       T_ref, input_current, dt, t_max)

print(f"\nResults:")
print(f"  Total spikes: {len(spikes)}")
if len(spikes) > 0:
    print(f"  First spike at: {spikes[0]:.2f} ms")
    print(f"  Last spike at: {spikes[-1]:.2f} ms")
    print(f"  Firing rate: {len(spikes) / (t_max / 1000.0):.2f} Hz")
    if len(spikes) > 1:
        isi = np.diff(spikes)
        print(f"  Mean ISI: {np.mean(isi):.2f} ms")
        print(f"  ISI std: {np.std(isi):.2f} ms")

# Create visualizations
print("\nGenerating visualizations...")
fig1 = plot_lif_simulation(t, V, spikes, I, V_rest, V_threshold)

plt.show()

print("\nVisualization complete!")
