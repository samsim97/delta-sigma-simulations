"""Digital stimulus generation utilities for delta-sigma simulations."""

import numpy as np
from scipy import signal


def generate_pcm_sine(amplitude, analog_frequency, duration, bits, digital_sample_rate, oversampling_ratio):
  """
  Create a quantized PCM sine wave and upsample it to the modulator rate.

  Args:
    amplitude: Peak amplitude of the analog sine (0..1).
    analog_frequency: Sine frequency in Hz.
    duration: Signal duration in seconds.
    bits: PCM word length.
    digital_sample_rate: Sample rate for PCM generation in Hz.
    oversampling_ratio: Repetition factor to reach the modulator rate.

  Returns:
    np.ndarray: Normalized PCM samples repeated to the modulator rate.
  """
  _, _, pcm_upsampled = generate_pcm_sine_with_trace(
    amplitude=amplitude,
    analog_frequency=analog_frequency,
    duration=duration,
    bits=bits,
    digital_sample_rate=digital_sample_rate,
    oversampling_ratio=oversampling_ratio,
  )
  return pcm_upsampled


def generate_pcm_sine_with_trace(amplitude, 
                                 analog_frequency, 
                                 duration, 
                                 bits, 
                                 digital_sample_rate, 
                                 oversampling_ratio) -> tuple[np.ndarray[np.float64], np.ndarray[np.float64], np.ndarray[np.float64]]:
  """
  Create a quantized PCM sine wave and upsample it, returning both analog and digital versions.

  Returns:
    t_pcm: Time axis for the analog/digital sample rate.
    analog_sine: Continuous-time sine sampled at digital_sample_rate (pre-quantization).
    pcm_upsampled: Quantized PCM samples repeated to the modulator rate.
  """
  t_pcm: np.ndarray[np.float64] = np.arange(0, duration, 1 / digital_sample_rate)
  analog_sine: np.ndarray[np.float64] = amplitude * np.sin(2 * np.pi * analog_frequency * t_pcm)
  max_code: int = 2 ** (bits - 1) - 1
  pcm_codes: np.ndarray[np.int64] = np.round(np.clip(analog_sine * max_code, -max_code, max_code)).astype(int)
  normalized_pcm: np.ndarray[np.float64] = pcm_codes / max_code
  pcm_upsampled: np.ndarray[np.float64] = np.repeat(normalized_pcm, oversampling_ratio)
  return t_pcm, analog_sine, pcm_upsampled


def apply_reconstruction_filter(bitstream, filter_order=4, cutoff_ratio=0.1):
  """
  Apply a low-pass filter to recover the analog signal from a 1-bit delta-sigma bitstream.
  
  This simulates the analog reconstruction filter that would follow the DAC output stage.
  
  Args:
    bitstream: 1-bit modulator output (±1 values at modulator rate).
    filter_order: Order of the Butterworth low-pass filter.
    cutoff_ratio: Cutoff frequency as a fraction of Nyquist (e.g., 0.1 = fs/20).
  
  Returns:
    np.ndarray: Filtered analog-like signal.
  """  
  # Design Butterworth low-pass filter
  # Normalized frequency: 0 to 1, where 1 is Nyquist (half the sampling rate)
  sos = signal.butter(filter_order, cutoff_ratio, btype='low', output='sos')
  
  # Apply zero-phase filtering (forward-backward to avoid phase distortion)
  filtered = signal.sosfiltfilt(sos, bitstream)
  
  return filtered


def reconstruct_analog_from_dsm_bitstream(bitstream, filter_order=4, cutoff_ratio=0.1):
  """
  Reconstruct the analog waveform from a 1-bit delta-sigma modulator output.
  
  This function takes the high-frequency 1-bit switching signal and recovers
  the encoded analog signal by low-pass filtering.
  
  Args:
    bitstream: 1-bit modulator output signal (values: -1 or +1).
    filter_order: Order of the reconstruction filter (higher = sharper cutoff).
    cutoff_ratio: Filter cutoff as fraction of Nyquist frequency (lower = more smoothing).
  
  Returns:
    np.ndarray: Reconstructed analog signal.
  """
  return apply_reconstruction_filter(bitstream, filter_order, cutoff_ratio)