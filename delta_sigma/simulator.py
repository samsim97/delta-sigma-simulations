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
    
    # Loop filter coefficients - scale integrators to prevent saturation
    # These coefficients ensure stability across different OSRs
    # Using optimized coefficients for noise-shaping transfer function
    if order == 1:
      self.coefficients = np.array([1.0])
    elif order == 2:
      # Second-order: scale each stage to balance SNR and stability
      self.coefficients = np.array([0.5, 0.5])
    elif order == 3:
      # Third-order: more aggressive scaling to prevent instability
      self.coefficients = np.array([0.3, 0.3, 0.4])
    else:
      # Generic scaling for higher orders
      self.coefficients = np.ones(order) / order
      
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
      # Error signal (input minus feedback)
      error = sample - prev_y
      
      # First integrator: accumulate scaled error
      self.integrators[0] += error * self.coefficients[0]
      self.integrators[0] = np.clip(self.integrators[0], self.accumulator_min, self.accumulator_max)
      
      # Higher-order integrators: cascade with local feedback
      for j in range(1, self.order):
        # Each stage integrates the previous stage's output minus local feedback
        stage_input = self.integrators[j-1] - prev_y * self.coefficients[j]
        self.integrators[j] += stage_input * self.coefficients[j]
        self.integrators[j] = np.clip(self.integrators[j], self.accumulator_min, self.accumulator_max)
      
      # Quantizer: uses the last integrator's output
      y = 1.0 if self.integrators[-1] >= 0 else -1.0
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
