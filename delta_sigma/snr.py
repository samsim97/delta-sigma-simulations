import numpy as np


def compute_snr(frequencies: np.ndarray[np.float64],
                psd: np.ndarray[np.float64],
                input_frequency: float) -> float:
  # Identify signal and noise bins
  signal_bin: np.int64 = np.argmin(np.abs(frequencies - input_frequency))
  signal_power: np.float64 = 10 ** (psd[signal_bin] / 10)
  noise_power: np.float64 = np.sum(10 ** (psd / 10)) - signal_power
  snr: np.float64 = 10 * np.log10(signal_power / noise_power)  
  return snr