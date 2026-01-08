"""Plotting utilities for delta-sigma simulations."""

import matplotlib.pyplot as plt
import numpy as np


def plot_signals(time, input_signal, output_signal, title="Delta-Sigma Modulation"):
  """
  Plot input and output signals in time domain.
  
  Args:
    time: Time array
    input_signal: Input signal array
    output_signal: Output modulated signal array
    title: Plot title
  """
  fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 6))
  
  # Plot input signal
  ax1.plot(time, input_signal, 'b-', linewidth=1.5)
  ax1.set_ylabel('Amplitude')
  ax1.set_title(f'{title} - Input Signal')
  ax1.grid(True, alpha=0.3)
  
  # Plot output signal
  ax2.plot(time, output_signal, 'r-', linewidth=0.8)
  ax2.set_xlabel('Time (s)')
  ax2.set_ylabel('Amplitude')
  ax2.set_title(f'{title} - Output Signal')
  ax2.grid(True, alpha=0.3)
  
  plt.tight_layout()
  return fig


def plot_psd(frequencies, psd, title="Power Spectral Density", max_freq=None):
  """
  Plot power spectral density.
  
  Args:
    frequencies: Frequency array
    psd: Power spectral density in dB
    title: Plot title
    max_freq: Maximum frequency to display (Hz)
  """
  fig, ax = plt.subplots(figsize=(12, 6))
  
  if max_freq:
    mask = frequencies <= max_freq
    frequencies = frequencies[mask]
    psd = psd[mask]

  ax.plot(frequencies, psd, 'b-', linewidth=1)
  ax.set_xlabel('Frequency (Hz)')
  ax.set_ylabel('Power Spectral Density (dB)')
  ax.set_title(title)
  ax.grid(True, alpha=0.3)
  
  plt.tight_layout()
  return fig


def compare_orders(orders_data, max_freq=None):
  """
  Compare PSDs of different delta-sigma orders.
  
  Args:
    orders_data: Dictionary with order as key and (frequencies, psd) as value
    max_freq: Maximum frequency to display (Hz)
  """
  fig, ax = plt.subplots(figsize=(12, 6))
  
  colors = ['b', 'r', 'g', 'm', 'c', 'y']
  
  for i, (order, (frequencies, psd)) in enumerate(sorted(orders_data.items())):
    if max_freq:
      mask = frequencies <= max_freq
      freq_plot = frequencies[mask]
      psd_plot = psd[mask]
    else:
      freq_plot = frequencies
      psd_plot = psd
  
    color = colors[i % len(colors)]
    ax.plot(freq_plot, psd_plot, color=color, linewidth=1.5, 
            label=f'Order {order}', alpha=0.8)
  
  ax.set_xlabel('Frequency (Hz)')
  ax.set_ylabel('Power Spectral Density (dB)')
  ax.set_title('Delta-Sigma DAC Order Comparison')
  ax.legend()
  ax.grid(True, alpha=0.3)
  
  plt.tight_layout()
  return fig
