"""Digital stimulus generation utilities for delta-sigma simulations."""

import numpy as np


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
  t_pcm = np.arange(0, duration, 1 / digital_sample_rate)
  analog_sine = amplitude * np.sin(2 * np.pi * analog_frequency * t_pcm)
  max_code = 2 ** (bits - 1) - 1
  pcm_codes = np.round(np.clip(analog_sine * max_code, -max_code, max_code)).astype(int)
  normalized_pcm = pcm_codes / max_code
  return np.repeat(normalized_pcm, oversampling_ratio)
