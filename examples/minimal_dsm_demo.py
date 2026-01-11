"""Minimal delta-sigma modulator demo (with detailed comments)

This script demonstrates a minimal workflow to:
 1. Create a digital sinewave (sampled digital signal).
 2. Design a simple delta-sigma modulator (NTF synthesis).
 3. Realize the NTF into a state-space model (ABCD matrix form).
 4. Scale the realization so internal signals fit the quantizer range.
 5. Simulate the time-domain modulator output (1-bit stream).
 6. Convert the 1-bit digital stream to an "analog" waveform using a
    simple reconstruction filter (moving average) and save plots.

This example is intentionally simple and uses a boxcar filter for
reconstruction so you can see the overall flow. For a production DAC
you would use a dedicated analog reconstruction filter or a much
better digital filter prior to a DAC.
"""

import os
import sys
import numpy as np
import matplotlib

# Use a non-interactive backend so figures are saved to files even when
# no display is attached (headless/server environments).
# matplotlib.use('Agg')
import matplotlib.pyplot as plt

# Ensure the project root is on sys.path so the local `deltasigma` package
# (the one in this repo) is importable by the example script.
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from deltasigma import synthesizeNTF, realizeNTF, stuffABCD, scaleABCD, simulateDSM


# ------------------------
# Design parameters
# ------------------------
order = 3         # Modulator order (number of integrators). Higher -> stronger noise shaping.
osr = 64          # Oversampling ratio (how many times faster than Nyquist)
nlev = 2          # Number of quantizer levels (2 -> 1-bit quantizer)
f0 = 0.0          # Input tone frequency, normalized to sampling rate (0..0.5)
Hinf = 1.5        # Loop gain parameter used during NTF synthesis (stability tuning)
form = 'CRFB'     # Realization topology (choose one supported by the library)


# ------------------------
# 1) Synthesize the NTF
# ------------------------
# The noise transfer function (NTF) shapes quantization noise away from
# the signal band. The library function returns a tuple representing the
# NTF (zeros, poles, gain) suitable for later realization.
ntf = synthesizeNTF(order, osr, 2, Hinf, f0)


# ------------------------
# 2) Realize NTF -> ABCD
# ------------------------
# Convert the transfer function into a state-space realization (A,B,C,D)
# arranged in the library's ABCD matrix layout. This gives a direct map to
# a discrete-time implementation (integrators, gains, summers).
a, g, b, c = realizeNTF(ntf, form)

# `b` may contain multiple feed-in vectors; here we collapse them to a
# single primary feed-in (this mimics the higher-level example behavior).
b = np.hstack((np.atleast_1d(b[0]), np.zeros((b.shape[0] - 1,))))
ABCD = stuffABCD(a, g, b, c, form)


# ------------------------
# 3) Scale ABCD for quantizer range
# ------------------------
# Floating point simulations can tolerate any amplitude, but real hardware
# has limited range. `scaleABCD` finds a scaling so internal states fit the
# quantizer dynamic range determined by `nlev`. The returned `umax` is the
# maximum allowed input amplitude (peak) after scaling.
ABCD_scaled, umax, S = scaleABCD(ABCD.copy(), nlev, f0)
print(f"Scaled umax = {umax:.6f}")


# ------------------------
# 4) Create the digital input signal
# ------------------------
# We make a cosine tone sampled at the modulator sampling rate. The
# frequency `f_sig` is expressed relative to the sampling rate (normalized).
N = 16384                    # number of samples to simulate
t = np.arange(N)
f_sig = 1.0 / (2 * osr)      # choose an in-band test tone frequency
amp = 0.9 * umax             # pick amplitude slightly below allowed maximum
u = amp * np.cos(2 * np.pi * f_sig * t)


# ------------------------
# 5) Simulate the DSM (time-domain)
# ------------------------
# `simulateDSM` runs a time-domain simulation of the modulator described by
# `ABCD_scaled`. Different distributions of the package may return different
# shapes from `simulateDSM` (some return the output only, others return a
# tuple like (v, xn, xmax, y)). We handle both cases below.
out = simulateDSM(u, ABCD_scaled, nlev)
if isinstance(out, tuple) and len(out) >= 1:
    v = out[0]   # modulator output (quantized levels)
else:
    v = out
v = np.asarray(v).reshape(-1)


# ------------------------
# 6) Reconstruct analog signal (simple example)
# ------------------------
# A practical DAC or reconstructor will have a proper analog filter. Here
# we use a simple moving average (boxcar FIR) to smooth the 1-bit stream so
# you can visually see the recovered sine. This is not high quality but
# suffices for illustration and teaching.
lp_len = osr
h = np.ones(lp_len) / lp_len
analog = np.convolve(v, h, mode='same')


# ------------------------
# 7) Save time-domain plots
# ------------------------
# Save PNG files so plots appear even without an interactive GUI.
plt.figure(figsize=(10, 6))
plt.subplot(3, 1, 1)
plt.plot(t[:1000], u[:1000], label='input (digital sine)')
plt.ylabel('Amplitude')
plt.legend()

plt.subplot(3, 1, 2)
plt.plot(t[:1000], v[:1000], '.', markersize=1, label='DSM output (quantized)')
plt.ylabel('Quantizer output')
plt.legend()

plt.subplot(3, 1, 3)
plt.plot(t[:1000], analog[:1000], label='reconstructed analog (boxcar)')
plt.ylabel('Analog (smoothed)')
plt.xlabel('Sample index')
plt.legend()

plt.tight_layout()
plt.show()
plt.savefig('minimal_dsm_demo_time.png', dpi=150)
print('Wrote minimal_dsm_demo_time.png')


# ------------------------
# 8) Frequency-domain view of the reconstructed analog signal
# ------------------------
from numpy.fft import fft
Nfft = 16384
X = fft(analog * np.hanning(len(analog)), n=Nfft)
freq = np.arange(Nfft) / Nfft
mag_db = 20 * np.log10(np.abs(X) + 1e-12)

plt.figure(figsize=(8, 4))
plt.plot(freq[:Nfft//2], mag_db[:Nfft//2])
plt.title('Spectrum of reconstructed analog')
plt.xlabel('Normalized frequency')
plt.ylabel('Magnitude (dB)')
plt.grid(True)
plt.show()
plt.savefig('minimal_dsm_demo_spectrum.png', dpi=150)
print('Wrote minimal_dsm_demo_spectrum.png')

print('Done.')
