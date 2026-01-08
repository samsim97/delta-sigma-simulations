"""Example: Basic delta-sigma DAC simulation."""

import numpy as np
import matplotlib.pyplot as plt
from delta_sigma import (
  DeltaSigmaDAC,
  plot_signals,
  generate_pcm_sine,
  generate_pcm_sine_with_trace,
)


def main():
  sampling_rate = 20_000_000
  digital_sample_rate = 20_000_000
  duration = 0.001
  sin_signal_freq = 10_000
  order = 1
  bits = 8  # PCM word length
  amplitudes = (0.5,)
  
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
    for a in amplitudes
  ]

  # Generate trace for visualization (use mid amplitude)
  trace_amp = amplitudes[0]
  t_pcm, analog_sine, pcm_upsampled = generate_pcm_sine_with_trace(
    amplitude=trace_amp,
    analog_frequency=sin_signal_freq,
    duration=duration,
    bits=bits,
    digital_sample_rate=digital_sample_rate,
    oversampling_ratio=oversampling_ratio,
  )
  modulator_time = np.arange(len(pcm_upsampled)) / sampling_rate
  
  print(f"Simulating {order}-order Delta-Sigma DAC with {bits}-bit PCM input...")
  dac = DeltaSigmaDAC(order=order, sampling_rate=sampling_rate)
  outputs = [dac.modulate(input_signal) for input_signal in input_signals]
  
  # Plot analog vs digitalized (PCM) signal over a few sine cycles
  fig_inputs, (ax_a, ax_d) = plt.subplots(2, 1, figsize=(12, 6))
  ax_a.plot(t_pcm, analog_sine, 'b-', linewidth=1.2)
  ax_a.set_title('Analog Sine (pre-quantization)')
  ax_a.set_ylabel('Amplitude')
  ax_a.grid(True, alpha=0.3)
  
  ax_d.step(modulator_time, pcm_upsampled, where='post', color='g', linewidth=1.0)
  ax_d.set_title('PCM Quantized + Upsampled (digital input to modulator)')
  ax_d.set_xlabel('Time (s)')
  ax_d.set_ylabel('Amplitude')
  ax_d.grid(True, alpha=0.3)
  plt.tight_layout()
  plt.savefig('simulations_out/analog_vs_pcm.png', dpi=100, bbox_inches='tight')
  plt.show()
  
  # Plot modulator input/output for mid amplitude over several sine cycles
  fig = plot_signals(modulator_time, 
                             input_signals[0], 
                             outputs[0],
                             title=f"Delta-Sigma DAC (Order {order})")  
  plt.savefig('simulations_out/basic_simulation.png', dpi=100, bbox_inches='tight')
  plt.show()
  
  print("Done! Check 'simulations_out/basic_simulation.png' for results.")


if __name__ == "__main__":
  main()
