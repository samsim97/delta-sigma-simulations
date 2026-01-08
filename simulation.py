"""Example: Basic delta-sigma DAC simulation."""

import numpy as np
import matplotlib.pyplot as plt
from delta_sigma import DeltaSigmaDAC, plot_signals


def main():
  sampling_rate = 1_000_000
  duration = 5e-4  # 0.5 ms
  signal_freq = 1_000
  order = 1
  
  t = np.linspace(0, duration, int(sampling_rate * duration))
  
  input_signals = [0.5 * np.sin(2 * np.pi * signal_freq * t),
                                           0.7 * np.sin(2 * np.pi * signal_freq * t),
                                           0.9 * np.sin(2 * np.pi * signal_freq * t)]
  
  
  print(f"Simulating {order}-order Delta-Sigma DAC...")
  dac = DeltaSigmaDAC(order=order, sampling_rate=sampling_rate)
  outputs = [dac.modulate(input_signal) for input_signal in input_signals]
  
  samples_to_plot = 200
  fig = plot_signals(t[:samples_to_plot], 
                             input_signals[1][:samples_to_plot], 
                             outputs[1][:samples_to_plot],
                             title=f"Delta-Sigma DAC (Order {order})")
  # plt.savefig('simulations_out/basic_simulation.png', dpi=100, bbox_inches='tight')
  plt.show()
  
  print("Done! Check 'simulations_out/basic_simulation.png' for results.")


if __name__ == "__main__":
  main()
