import sensor_sim
import numpy as np

print("successfully imported C++ module")
# create a simulator instance
sim = sensor_sim.IMUSim(sample_rate_hz=100.0, activityType="walking")
print("IMUSim instance created.")

print("Running sim for 10 timesteps, for walking activity:")
for i in range(10):
    sim.update()
    # below come in as numpy, ready to use!
    accel = sim.get_acceleration() 
    gyro = sim.get_gyroscope()     
    
    print(f"  Time: {sim.get_current_time():.2f}s | Accel: {accel} | Gyro: {gyro}")

print("Test activity change to sitting")
sim.set_curr_activity("sitting")

for i in range(5):
    sim.update()
    accel = sim.get_acceleration()
    gyro = sim.get_gyroscope()
    print(f"  Time: {sim.get_current_time():.2f}s | Accel: {accel} | Gyro: {gyro}")

def run_test_segment(sim, extractor, activity_name, steps=100):
    print(f"\n--- Testing Activity: {activity_name.upper()} ---")
    sim.set_curr_activity(activity_name)
    # prep buffer with full window for current activity
    print(f"Simulating {steps} steps...")
    last_features = []
    for _ in range(steps):
        sim.update()
        # grab raw data, extract features
        acc = sim.get_acceleration()
        gyro = sim.get_gyroscope()
        extractor.add_sample(acc, gyro)
        last_features = extractor.compute_features()
    # conv to numpy array
    feats = np.array(last_features)
    # C++ original feature order: 
    # [Mean_AccX, Mean_AccY, Mean_AccZ, Var_AccX, Var_AccY, Var_AccZ]
    print("Final Feature Vector:")
    print(f"  Mean Accel (X, Y, Z):  [{feats[0]:.4f}, {feats[1]:.4f}, {feats[2]:.4f}]")
    print(f"  Var  Accel (X, Y, Z):  [{feats[3]:.4f}, {feats[4]:.4f}, {feats[5]:.4f}]")
    
    return feats

# setup new sim at 100 Hz, window of 0.5 sec
sim = sensor_sim.IMUSim(100.0, "sitting") 
extractor = sensor_sim.FeatureExtractor(50) 
# try sitting, walking, classifier check
feat_sit = run_test_segment(sim, extractor, "sitting")
feat_walk = run_test_segment(sim, extractor, "walking")
z_var_sit = feat_sit[5]
z_var_walk = feat_walk[5]

print("\n--- Check ---")
print(f"Sitting Z-Variance: {z_var_sit:.5f}")
print(f"Walking Z-Variance: {z_var_walk:.5f}")

if z_var_walk > (z_var_sit * 10):
    print("S: Walking variance is significantly higher!")
    print("ML can distinguish later on")
else:
    print("F: Signals are too similar.")