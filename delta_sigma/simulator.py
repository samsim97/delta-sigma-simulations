"""Delta-Sigma DAC Simulator."""

import numpy as np


class DeltaSigmaDAC:
  """Simulate a Delta-Sigma DAC at various orders."""
  
  def __init__(self, order=1, sampling_rate=1e6, accumulator_bits=32):
    """
    Initialize Delta-Sigma DAC simulator.
    
    Args:
      order: Order of the delta-sigma modulator (1, 2, 3, etc.)
      sampling_rate: Sampling rate in Hz
      accumulator_bits: Bit width of integrator accumulators (mimics hardware fixed-point)
    """
    self.order = order
    self.sampling_rate = sampling_rate
    self.integrators = np.zeros(order)
    self.accumulator_bits = accumulator_bits
    
    # Calculate saturation limits (like hardware accumulators)
    # In VHDL: signal accumulator : signed(accumulator_bits-1 downto 0)
    self.accumulator_max = 2.0 ** (accumulator_bits - 1) - 1
    self.accumulator_min = -(2.0 ** (accumulator_bits - 1))
      
  def modulate(self, input_signal):
    """
    Perform delta-sigma modulation on input signal.
    
    Args:
      input_signal: Input signal array (values between -1 and 1)
        
    Returns:
      output: Modulated output signal (1-bit)
    """
    output = np.zeros(len(input_signal))
    self.integrators = np.zeros(self.order)
    
    prev_y = 0.0
    for i, sample in enumerate(input_signal):
      # CIFB-style loop: each integrator integrates previous stage output
      # with negative feedback from the previous 1-bit output (prev_y).
      # Stage 0 integrates input error; higher stages integrate their
      # predecessor's output minus feedback.
      e0 = sample - prev_y
      self.integrators[0] += e0
      self.integrators[0] = np.clip(self.integrators[0], self.accumulator_min, self.accumulator_max)

      integrator_out = self.integrators[0]
      for j in range(1, self.order):
        self.integrators[j] += (integrator_out - prev_y)
        self.integrators[j] = np.clip(self.integrators[j], self.accumulator_min, self.accumulator_max)
        integrator_out = self.integrators[j]

      # Quantizer
      y = 1.0 if integrator_out >= 0 else -1.0
      output[i] = y
      prev_y = y
        
    return output
  
  def compute_psd(self, signal, fft_size=8192) -> tuple[np.ndarray[np.float64], np.ndarray[np.float64]]:
    """
    Compute Power Spectral Density of a signal.
    
    Args:
      signal: Input signal array
      fft_size: FFT size for spectral analysis
        
    Returns:
      frequencies: Frequency array
      psd: Power spectral density in dB
    """
    # Apply window and compute FFT
    if len(signal) < fft_size:
      window: np.ndarray[np.float64] = np.hanning(len(signal))
      fft_data: np.ndarray[np.complex128] = np.fft.fft(signal * window, fft_size)
    else:
      window: np.ndarray[np.float64] = np.hanning(fft_size)
      fft_data: np.ndarray[np.complex128] = np.fft.fft(signal[:fft_size] * window, fft_size)
    
    psd: np.ndarray[np.float64] = 20 * np.log10(np.abs(fft_data) + 1e-10)
    frequencies: np.ndarray[np.float64] = np.fft.fftfreq(fft_size, 1 / self.sampling_rate)
    
    # Return only positive frequencies
    positive_idx: np.ndarray[np.bool_] = frequencies >= 0
    return frequencies[positive_idx], psd[positive_idx]
