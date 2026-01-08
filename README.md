# Delta-Sigma DAC Simulations

A minimal Python project for simulating and comparing delta-sigma Digital-to-Analog Converters (DACs) at different orders.

## Features

- Simulate delta-sigma DACs at various orders (1st, 2nd, 3rd, etc.)
- Generate time-domain and frequency-domain plots
- Compare performance across different orders
- Quick and lightweight for rapid testing

## Installation

Install dependencies:
```bash
pip install -r requirements.txt
```

Or install as a package:
```bash
pip install -e .
```

## Usage

### Basic Simulation

```python
from delta_sigma import DeltaSigmaDAC, plot_signals
import numpy as np
import matplotlib.pyplot as plt

# Create a 2nd-order delta-sigma DAC
dac = DeltaSigmaDAC(order=2, sampling_rate=1e6)

# Generate input signal
t = np.linspace(0, 1e-3, 1000)
input_signal = 0.5 * np.sin(2 * np.pi * 1e3 * t)

# Modulate
output = dac.modulate(input_signal)

# Plot results
plot_signals(t[:100], input_signal[:100], output[:100])
plt.show()
```

### Compare Different Orders

```bash
cd examples
python compare_orders.py
```

This will generate a comparison plot showing the power spectral density for 1st, 2nd, and 3rd order modulators.

### Basic Example

```bash
cd examples
python basic_simulation.py
```

## Project Structure

```
delta-sigma-simulations/
├── delta_sigma/           # Main package
│   ├── __init__.py       # Package initialization
│   ├── simulator.py      # Delta-sigma DAC simulator
│   └── plotter.py        # Plotting utilities
├── examples/             # Example scripts
│   ├── basic_simulation.py
│   └── compare_orders.py
├── requirements.txt      # Dependencies
└── README.md            # This file
```

## How It Works

Delta-sigma modulation uses negative feedback and oversampling to achieve high-resolution digital-to-analog conversion with simple 1-bit quantization. Higher-order modulators push quantization noise to higher frequencies, improving signal-to-noise ratio in the band of interest.