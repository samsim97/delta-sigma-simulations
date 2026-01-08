"""Example: Basic delta-sigma DAC simulation."""

import numpy as np
import matplotlib.pyplot as plt
from delta_sigma import DeltaSigmaDAC, plot_signals


def main():
    # Simulation parameters
    sampling_rate = 1e6  # 1 MHz
    duration = 5e-4  # 0.5 ms
    signal_freq = 1e3  # 1 kHz input signal
    order = 2  # 2nd order modulator
    
    # Generate time array
    t = np.linspace(0, duration, int(sampling_rate * duration))
    
    # Generate input signal (sine wave)
    input_signal = 0.7 * np.sin(2 * np.pi * signal_freq * t)
    
    # Create and run simulator
    print(f"Simulating {order}-order Delta-Sigma DAC...")
    dac = DeltaSigmaDAC(order=order, sampling_rate=sampling_rate)
    output = dac.modulate(input_signal)
    
    # Plot time domain (show only first 200 samples for clarity)
    samples_to_plot = 200
    fig = plot_signals(t[:samples_to_plot], 
                      input_signal[:samples_to_plot], 
                      output[:samples_to_plot],
                      title=f"Delta-Sigma DAC (Order {order})")
    plt.savefig('basic_simulation.png', dpi=100, bbox_inches='tight')
    plt.show()
    
    print("Done! Check 'basic_simulation.png' for results.")


if __name__ == "__main__":
    main()
