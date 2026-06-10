# Red Pitaya Ultrasound Distance Calculation

When transitioning from the PicoScope to the Red Pitaya STEMlab 125-14, the timebase changes fundamentally due to the hardware's native clock speed. This document explains how to map the raw sample indices from the Red Pitaya's ADC to physical depth (distance) in centimeters.

## 1. The Timebase: Why 8ns?

The Red Pitaya's Analog-to-Digital Converter (ADC) operates at **125 Megahertz (MS/s)**.

$$ 1 \text{ second} / 125,000,000 \text{ cycles} = 0.000000008 \text{ seconds} $$

This means the hardware takes exactly one measurement every **8 nanoseconds (8ns)** when decimation is set to 1 (`RP_DEC_1`).

## 2. The Physics of Ultrasound

To calculate the distance to a target, we rely on the speed of sound through the medium being scanned (typically assumed to be human tissue or water).

*   **Speed of Sound in Tissue:** ~1540 meters per second ($1.54 \times 10^5 \text{ cm/s}$)
*   **The Round Trip:** The ultrasound pulse must travel *to* the target and bounce *back* to the transducer. Therefore, the total distance the sound travels is twice the actual depth of the target.

### The Universal Distance Formula

$$ \text{Depth} = \frac{\text{Time} \times \text{Speed of Sound}}{2} $$

## 3. Applying the Formula to the Red Pitaya

We can plug our hardware specifications into the universal formula to find a single constant multiplier that converts a sample index directly into centimeters of depth.

*   **Time ($t$):** `Sample Index` $\times (8 \times 10^{-9} \text{ seconds})$
*   **Speed ($c$):** $154,000 \text{ cm/s}$

$$ \text{Depth at Index } x = \frac{x \times (8 \times 10^{-9}) \times 154000}{2} $$
$$ \text{Depth at Index } x = x \times 0.000616 \text{ cm} $$

*Every single sample recorded by the Red Pitaya represents a slice of tissue exactly 0.000616 cm deep.*

## 4. Python Implementation for PyQtGraph

When updating the visualization scripts (e.g., `live_orientation_visualization_hybrid.py`), replace the legacy PicoScope distance calculation with this physics-based approach.

```python
import numpy as np

def calculate_depth_array(num_samples=7500, speed_of_sound_m_s=1540):
    """
    Calculates the physical depth in cm for each sample index from a Red Pitaya.
    
    Args:
        num_samples (int): The number of samples in the capture buffer.
        speed_of_sound_m_s (float): Speed of sound in the medium (m/s). Default is tissue (1540).
        
    Returns:
        np.ndarray: An array of distances in centimeters corresponding to each sample.
    """
    # 1. Create an array of sample indices [0, 1, 2, ..., num_samples - 1]
    indices = np.arange(num_samples)
    
    # 2. Convert speed of sound to cm/s
    speed_cm_s = speed_of_sound_m_s * 100
    
    # 3. Calculate the constant multiplier based on the 8ns timebase and round-trip
    # (8e-9 seconds * speed_cm_s) / 2
    cm_per_sample = (8e-9 * speed_cm_s) / 2.0
    
    # 4. Return the final distance array
    return indices * cm_per_sample

# Example usage in the visualization script:
# self.distance = calculate_depth_array(7500)
```

### Depth Reference Table
| Sample Index | Time Elapsed | Physical Depth (Tissue) |
| :--- | :--- | :--- |
| 0 | 0 ns | 0.000 cm |
| 1000 | 8 µs | 0.616 cm |
| 3750 | 30 µs | 2.310 cm |
| 7500 | 60 µs | 4.620 cm |
