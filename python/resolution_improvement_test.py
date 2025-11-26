import sensor_sim
import numpy as np

# 1. High Resolution Setup
# 256 Window Size @ 100Hz = ~0.39 Hz resolution
# (Old was 64 Window Size = ~1.5 Hz resolution)
WINDOW_SIZE = 256 
SAMPLE_RATE = 100.0

sim = sensor_sim.IMUSim(SAMPLE_RATE, "walking")
extractor = sensor_sim.FeatureExtractor(WINDOW_SIZE, SAMPLE_RATE)

print(f"--- Testing High-Res FFT (Window={WINDOW_SIZE}) ---")
print(f"Expected Resolution: {SAMPLE_RATE/WINDOW_SIZE:.4f} Hz")

# Warm up buffer
for _ in range(300):
    sim.update()
    extractor.add_sample(sim.get_acceleration(), sim.get_gyroscope())

# Check Features
feats = extractor.compute_features()
freq = feats[2] # Index 2 is Freq

print(f"\nTarget Frequency: 2.0 Hz")
print(f"Detected Frequency: {freq:.4f} Hz")

error = abs(2.0 - freq)
if error < 0.4:
    print("✅ SUCCESS: High precision detected!")
else:
    print("❌ FAIL: Still low resolution.")