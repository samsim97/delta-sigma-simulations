"""Example: Compare different delta-sigma DAC orders."""

import numpy as np
import matplotlib.pyplot as plt
from delta_sigma import DeltaSigmaDAC, compare_orders, generate_pcm_sine_with_trace, snr
from delta_sigma import reconstruct_analog_from_dsm_bitstream
from delta_sigma.plotter import plot_signals


def main():
  modulator_sample_rate: int = 400_000_000  # Delta-sigma modulator clock rate (e.g., 20 MHz), the output signal has this frequency
  pcm_sample_rate: int = 200_000        # Input PCM digital signal sample rate
  duration: float = 0.01
  # sin_signal_freq: int = 1_000_000
  sin_signal_freq = 10_000
  bits: int = 8  # PCM word length
  amplitudes = (0.5,)
    
  use_log_scale = True
  display_points = False
  show_sdm_dac_temporal_response = True
  
  if modulator_sample_rate % pcm_sample_rate != 0:
    raise ValueError("modulator_sample_rate must be an integer multiple of pcm_sample_rate")
  oversampling_ratio: int = modulator_sample_rate // pcm_sample_rate
  
  # Generate PCM signal with trace for visualization
  t_pcm, analog_sine, input_signal = generate_pcm_sine_with_trace(
    amplitude=amplitudes,
    analog_frequency=sin_signal_freq,
    duration=duration,
    bits=bits,
    digital_sample_rate=pcm_sample_rate,
    oversampling_ratio=oversampling_ratio,
  )
  modulator_time = np.arange(len(input_signal)) / modulator_sample_rate
  
  print("Plotting digitalized sine input...")
  
  fig_input, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 6))
  ax1.plot(t_pcm, analog_sine, 'b-', linewidth=1.2)
  ax1.set_title('Analog Sine (pre-quantization)')
  ax1.set_ylabel('Amplitude')
  ax1.grid(True, alpha=0.3)
  
  ax2.step(modulator_time, input_signal, where='post', color='g', linewidth=1.0)
  ax2.set_title(f'PCM Digitalized Input ({bits}-bit, upsampled to modulator rate)')
  ax2.set_xlabel('Time (s)')
  ax2.set_ylabel('Amplitude')
  ax2.grid(True, alpha=0.3)
  plt.tight_layout()
  plt.savefig('simulations_out/input_signal.png', dpi=100, bbox_inches='tight')
  plt.show()
  
  orders: list[int] = [1, 2, 3]
  results: dict[int, tuple[np.ndarray[np.float64], np.ndarray[np.float64]]] = {}
  outputs: dict[int, np.ndarray[np.float64]] = {}
  
  print("Simulating Delta-Sigma DAC at different orders...")
  for order in orders:
    print(f"  Order {order}...")
    dac = DeltaSigmaDAC(order=order, sampling_rate=modulator_sample_rate)
    outputs[order] = dac.modulate(input_signal)
    # Use larger FFT size for better low-frequency resolution
    frequencies, psd = dac.compute_psd(outputs[order], fft_size=65536)
    results[order] = (frequencies, psd)
    
  if show_sdm_dac_temporal_response:        
    print("Generating output plots for each order...")
    for order in orders:     
      analog_output = reconstruct_analog_from_dsm_bitstream(outputs[order])
      fig = plot_signals(modulator_time, 
                                 input_signal, 
                                 analog_output,
                                 title=f"Delta-Sigma DAC (Order {order})")
      plt.savefig(f'simulations_out/output_order_{order}.png', dpi=100, bbox_inches='tight')
      plt.show()
  
  print("Computing SNR for each order...")
  for order in orders:
    frequencies, psd = results[order]
    snr_value = snr.compute_snr(frequencies, psd, sin_signal_freq)
    print(f"  Order {order}: SNR = {snr_value:.2f} dB")  
  
  # Compare all orders in frequency domain
  print("Generating comparison plot...")
  fig = compare_orders(results, use_log_scale=use_log_scale, display_points=display_points)#, max_freq=sampling_rate / 4)
  plt.savefig('simulations_out/order_comparison.png', dpi=100, bbox_inches='tight')
  plt.show()
  
  print("Plots saved in 'simulations_out/'.")


if __name__ == "__main__":
  main()
