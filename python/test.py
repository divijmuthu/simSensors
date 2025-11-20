import sensor_sim
import time

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