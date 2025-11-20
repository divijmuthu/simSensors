import sensor_sim
import matplotlib.pyplot as plt
import numpy as np

def collect_data(sim, steps):
    accel_data = []
    gyro_data = []
    timestamps = []
    
    for _ in range(steps):
        sim.update()
        accel_data.append(sim.get_acceleration())
        gyro_data.append(sim.get_gyroscope())
        timestamps.append(sim.get_current_time())
        
    return np.array(timestamps), np.array(accel_data), np.array(gyro_data)

# setup
sample_rate = 100.0 # Hz
sim = sensor_sim.IMUSim(sample_rate, "walking")

# get 3 seconds of data, walking
print("Generating walking data...")
t_walk, acc_walk, gyro_walk = collect_data(sim, steps=300)

# switch to sitting, 2 seconds
print("Generating sitting data...")
sim.set_curr_activity("sitting")
t_sit, acc_sit, gyro_sit = collect_data(sim, steps=200)

# merge data
t_total = np.concatenate([t_walk, t_sit])
acc_total = np.vstack([acc_walk, acc_sit])
gyro_total = np.vstack([gyro_walk, gyro_sit])

# plot
fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True, figsize=(10, 8))
# accel
ax1.plot(t_total, acc_total[:, 0], label='X', alpha=0.7)
ax1.plot(t_total, acc_total[:, 1], label='Y', alpha=0.7)
ax1.plot(t_total, acc_total[:, 2], label='Z (Gravity)', color='green')
ax1.set_title('Accelerometer Data (Walking -> Sitting)')
ax1.set_ylabel('m/s^2')
ax1.legend()
ax1.grid(True)
# gyro
ax2.plot(t_total, gyro_total[:, 0], label='X', alpha=0.7)
ax2.plot(t_total, gyro_total[:, 1], label='Y (Leg Swing)', color='orange')
ax2.plot(t_total, gyro_total[:, 2], label='Z', alpha=0.7)
ax2.set_title('Gyroscope Data (Walking -> Sitting)')
ax2.set_ylabel('deg/s')
ax2.set_xlabel('Time (s)')
ax2.legend()
ax2.grid(True)

print("Displaying plot...")
plt.show()