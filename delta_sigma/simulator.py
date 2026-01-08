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
    
    for i, sample in enumerate(input_signal):
      # Negative feedback: subtract previous output from input
      # In VHDL: error <= input - feedback
      feedback = output[i-1] if i > 0 else 0
      error = sample - feedback
      
      # First integrator (accumulator)
      # In VHDL: accumulator <= accumulator + error (with saturation)
      self.integrators[0] += error
      # Simulate hardware saturation (like fixed-point arithmetic)
      self.integrators[0] = np.clip(self.integrators[0], 
                                     self.accumulator_min, 
                                     self.accumulator_max)
      integrator_out = self.integrators[0]
      
      # Higher order integrators (cascaded chain)
      for j in range(1, self.order):
        # In VHDL: accumulator(j) <= accumulator(j) + accumulator(j-1)
        self.integrators[j] += integrator_out
        # Saturation on each stage (hardware behavior)
        self.integrators[j] = np.clip(self.integrators[j],
                                       self.accumulator_min,
                                       self.accumulator_max)
        integrator_out = self.integrators[j]
      
      # 1-bit quantizer (comparator)
      # In VHDL: output <= '1' when accumulator >= 0 else '0'
      output[i] = 1.0 if integrator_out >= 0 else -1.0
        
    return output
  
  def compute_psd(self, signal, fft_size=8192):
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
      window = np.hanning(len(signal))
      fft_data = np.fft.fft(signal * window, fft_size)
    else:
      window = np.hanning(fft_size)
      fft_data = np.fft.fft(signal[:fft_size] * window, fft_size)
    
    psd = 20 * np.log10(np.abs(fft_data) + 1e-10)
    frequencies = np.fft.fftfreq(fft_size, 1 / self.sampling_rate)
    
    # Return only positive frequencies
    positive_idx = frequencies >= 0
    return frequencies[positive_idx], psd[positive_idx]
