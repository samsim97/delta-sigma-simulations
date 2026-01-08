"""Example: Compare different delta-sigma DAC orders."""

import numpy as np
import matplotlib.pyplot as plt
from delta_sigma import DeltaSigmaDAC, plot_signals, compare_orders


def main():
    # Simulation parameters
    sampling_rate = 1e6  # 1 MHz
    duration = 1e-3  # 1 ms
    signal_freq = 1e3  # 1 kHz input signal
    
    # Generate time array
    t = np.linspace(0, duration, int(sampling_rate * duration))
    
    # Generate input signal (sine wave)
    input_signal = 0.5 * np.sin(2 * np.pi * signal_freq * t)
    
    # Simulate different orders
    orders = [1, 2, 3]
    results = {}
    
    print("Simulating Delta-Sigma DAC at different orders...")
    for order in orders:
        print(f"  Order {order}...")
        dac = DeltaSigmaDAC(order=order, sampling_rate=sampling_rate)
        output = dac.modulate(input_signal)
        frequencies, psd = dac.compute_psd(output)
        results[order] = (frequencies, psd)
        
        # Plot time domain for first order as example
        if order == 1:
            # Show only first 100 samples for clarity
            samples_to_plot = 100
            fig = plot_signals(t[:samples_to_plot], 
                             input_signal[:samples_to_plot], 
                             output[:samples_to_plot],
                             title=f"Delta-Sigma Modulation (Order {order})")
            plt.savefig(f'order_{order}_time_domain.png', dpi=100, bbox_inches='tight')
            plt.close()
    
    # Compare all orders in frequency domain
    print("Generating comparison plot...")
    fig = compare_orders(results, max_freq=sampling_rate/4)
    plt.savefig('order_comparison.png', dpi=100, bbox_inches='tight')
    plt.show()
    
    print("Done! Check 'order_comparison.png' for results.")


if __name__ == "__main__":
    main()
