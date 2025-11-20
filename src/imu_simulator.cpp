#include "imu_simulator.h"
#include <cmath> // For M_PI and sin()

// implement constructor
IMUSim::IMUSim(double sample_rate_hz, const std::string& activityType)
    : m_current_time(0.0),
      m_sample_rate(sample_rate_hz),
      m_time_step(1.0 / sample_rate_hz),
      m_activity_type(activityType),
      m_latest_accel(Eigen::Vector3d::Zero()),
      m_latest_gyro(Eigen::Vector3d::Zero()),
      m_norm_dist(0.0, 1.0), // Mean 0, StdDev 1
      m_accel_bias(Eigen::Vector3d::Zero()),
      m_gyro_bias(Eigen::Vector3d::Zero())
{
    // Seed the random number generator
    std::random_device rd;
    m_rng.seed(rd());

    // Get some realistic MEMS noise params for IMU
    m_accel_noise_density = 0.004; // m/s^2 / sqrt(Hz)
    m_gyro_noise_density = 0.005;  // deg/s / sqrt(Hz)
    m_accel_bias_instability = 0.001; // How fast the bias drifts
    m_gyro_bias_instability = 0.001;
}

// key update function
void IMUSim::update() {
    // incr time
    m_current_time += m_time_step;
    // handle bias change
    update_biases();
    // compute ideal motion now, incorporate noise
    IMUDataSample ideal = get_ideal_motion(m_current_time);
    IMUDataSample noisy = get_noisy_sample(ideal);
    // store latest noisy data
    m_latest_accel = noisy.accel;
    m_latest_gyro = noisy.gyro;
}

void IMUSim::set_curr_activity(const std::string& activity) {
    m_activity_type = activity;
}

// sim ideal motion
IMUDataSample IMUSim::get_ideal_motion(double time_s) {
    IMUDataSample motion;

    if (m_activity_type == "sitting") {
        // when just sitting, only accel is gravity in z dim
        motion.accel = {0.0, 0.0, 9.81};
        motion.gyro = {0.0, 0.0, 0.0};
    }
    else if (m_activity_type == "walking") {
        // walking: simple sinusoidal accel and gyro
        double freq = 2.0; // 2 steps per second
        double accel_amplitude = 0.5; // m/s^2
        double gyro_amplitude = 20.0; // deg/s

        // capture accel change, say sinusoidal changing in speed
        motion.accel[0] = 0.0;
        motion.accel[1] = 0.0;
        motion.accel[2] = 9.81 + (accel_amplitude * std::sin(2 * M_PI * freq * time_s));
        
        // Simulate "leg swinging" i.e. y axis change
        motion.gyro[0] = 0.0;
        motion.gyro[1] = gyro_amplitude * std::sin(2 * M_PI * freq * time_s);
        motion.gyro[2] = 0.0;
    }
    // We can add more options for other actions!
    else {
        // sitting is default
        motion.accel = {0.0, 0.0, 9.81};
        motion.gyro = {0.0, 0.0, 0.0};
    }

    return motion;
}

// incorporate noise
IMUDataSample IMUSim::get_noisy_sample(const IMUDataSample& ideal_sample) {
    // This is the core signal processing math
    // We convert "noise density" (in /sqrt(Hz)) to
    // "standard deviation for our sample rate" (in units)
    double accel_std_dev = m_accel_noise_density / std::sqrt(m_time_step);
    double gyro_std_dev = m_gyro_noise_density / std::sqrt(m_time_step);

    IMUDataSample noisy_sample;
    
    // Add white noise and bias for each axis
    for (int i = 0; i < 3; ++i) {
        noisy_sample.accel[i] = ideal_sample.accel[i] + m_accel_bias[i] + (m_norm_dist(m_rng) * accel_std_dev);
        noisy_sample.gyro[i]  = ideal_sample.gyro[i]  + m_gyro_bias[i]  + (m_norm_dist(m_rng) * gyro_std_dev);
    }
    
    return noisy_sample;
}

// handle changes in bias
void IMUSim::update_biases() {
    // This simulates a "random walk" for the bias
    double accel_drift_std_dev = m_accel_bias_instability * std::sqrt(m_time_step);
    double gyro_drift_std_dev = m_gyro_bias_instability * std::sqrt(m_time_step);

    for (int i = 0; i < 3; ++i) {
        m_accel_bias[i] += m_norm_dist(m_rng) * accel_drift_std_dev;
        m_gyro_bias[i]  += m_norm_dist(m_rng) * gyro_drift_std_dev;
    }
}

// basic test
std::string helloEverybody() {
    return "Hello, Everybody!";
}