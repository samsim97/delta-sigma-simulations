"""Delta-Sigma DAC Simulator."""

import numpy as np


class DeltaSigmaDAC:
    """Simulate a Delta-Sigma DAC at various orders."""
    
    def __init__(self, order=1, sampling_rate=1e6):
        """
        Initialize Delta-Sigma DAC simulator.
        
        Args:
            order: Order of the delta-sigma modulator (1, 2, 3, etc.)
            sampling_rate: Sampling rate in Hz
        """
        self.order = order
        self.sampling_rate = sampling_rate
        self.integrators = np.zeros(order)
        
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
            # Feedback loop
            feedback = output[i-1] if i > 0 else 0
            error = sample - feedback
            
            # Pass through integrators (cascaded for higher orders)
            integrator_out = error
            for j in range(self.order):
                self.integrators[j] += integrator_out
                integrator_out = self.integrators[j]
            
            # Quantizer (1-bit)
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
        # Use Welch's method for better spectral estimation
        window = np.hanning(min(fft_size, len(signal)))
        n_segments = len(signal) // fft_size
        
        if n_segments == 0:
            fft_data = np.fft.fft(signal * np.hanning(len(signal)), fft_size)
        else:
            fft_data = np.fft.fft(signal[:fft_size] * window, fft_size)
        
        psd = 20 * np.log10(np.abs(fft_data) + 1e-10)
        frequencies = np.fft.fftfreq(fft_size, 1/self.sampling_rate)
        
        # Return only positive frequencies
        positive_idx = frequencies >= 0
        return frequencies[positive_idx], psd[positive_idx]
