"""Example: Basic delta-sigma DAC simulation."""

import numpy as np
import matplotlib.pyplot as plt
from delta_sigma import DeltaSigmaDAC, plot_signals, generate_pcm_sine


def main():
  sampling_rate = 1_000_000
  digital_sample_rate = 50_000
  duration = 5e-4  # 0.5 ms
  sin_signal_freq = 1_000
  order = 1
  bits = 8  # PCM word length
  
  if sampling_rate % digital_sample_rate != 0:
    raise ValueError("sampling_rate must be an integer multiple of digital_sample_rate")
  oversampling_ratio = sampling_rate // digital_sample_rate
  
  input_signals = [
    generate_pcm_sine(
      amplitude=a,
      analog_frequency=sin_signal_freq,
      duration=duration,
      bits=bits,
      digital_sample_rate=digital_sample_rate,
      oversampling_ratio=oversampling_ratio,
    )
    for a in (0.5, 0.7, 0.9)
  ]
  modulator_time = np.arange(len(input_signals[0])) / sampling_rate
  
  print(f"Simulating {order}-order Delta-Sigma DAC with {bits}-bit PCM input...")
  dac = DeltaSigmaDAC(order=order, sampling_rate=sampling_rate)
  outputs = [dac.modulate(input_signal) for input_signal in input_signals]
  
  samples_to_plot = 200
  fig = plot_signals(modulator_time[:samples_to_plot], 
                             input_signals[1][:samples_to_plot], 
                             outputs[1][:samples_to_plot],
                             title=f"Delta-Sigma DAC (Order {order})")
  plt.savefig('simulations_out/basic_simulation.png', dpi=100, bbox_inches='tight')
  plt.show()
  
  print("Done! Check 'simulations_out/basic_simulation.png' for results.")


if __name__ == "__main__":
  main()
