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
      m_gyro_bias(Eigen::Vector3d::Zero()),
      m_base_pressure(101325.0), // Standard sea level pressure
      m_pressure_noise_density(4.0), // Pascals (Barometers are noisy!)
      m_latest_pressure(101325.0)
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

    // barometer updates, compute pressure + noise
    double ideal_pressure = get_ideal_pressure(m_current_time);
    double noise = m_norm_dist(m_rng) * m_pressure_noise_density;
    m_latest_pressure = ideal_pressure + noise;
}

void IMUSim::set_curr_activity(const std::string& activity) {
    m_activity_type = activity;
}

// UPGRADED sim of raw ideal motion
IMUDataSample IMUSim::get_ideal_motion(double time_s) {
    IMUDataSample motion;
    // basic gravity
    motion.accel = {0.0, 0.0, 9.81};
    motion.gyro = {0.0, 0.0, 0.0};
    if (m_activity_type == "sitting") {
        // to capture 'breathing' add a tiny 0.3 Hz sine wave
        // some amplitude 0.05 m/s^2 is visible, not just silence
        motion.accel[2] = 9.81 + (0.05 * std::sin(2 * M_PI * 0.3 * time_s));
        // tiny random shifting on gyro ppl naturally move lol
        motion.gyro[0] = 0.5 * std::sin(2 * M_PI * 0.1 * time_s);
    }
    else if (m_activity_type == "walking") {
        double freq = 2.0; 
        double bob_amp = 0.5; 
        double swing_amp = 20.0; 
        motion.accel[2] = 9.81 + (bob_amp * std::sin(2 * M_PI * freq * time_s));
        motion.gyro[1]  = swing_amp * std::sin(2 * M_PI * freq * time_s);
    }
    else if (m_activity_type == "running") {
        double freq = 3.5; 
        double bob_amp = 2.0;  
        double swing_amp = 60.0; 
        motion.accel[2] = 9.81 + (bob_amp * std::sin(2 * M_PI * freq * time_s));
        motion.gyro[1]  = swing_amp * std::sin(2 * M_PI * freq * time_s);
    }
    else if (m_activity_type == "jumping") {
        double cycle_len = 1.5;
        double t_cycle = std::fmod(time_s, cycle_len); 
        if (t_cycle < 0.2) {
            // push off
            motion.accel[2] = 9.81 + 15.0; 
        } 
        else if (t_cycle < 0.5) {
            // in-air
            motion.accel[2] = 0.1; 
        }
        else if (t_cycle < 0.7) {
            // landing impact
            motion.accel[2] = 9.81 + 25.0;
        }
        else {
            // UPGRADE 2: Post-landing stabilization (The "Wobble")
            // Instead of perfect stillness, we simulate a damped oscillation
            // as the person regains balance.
            double t_rec = t_cycle - 0.7; // Time since landing
            // Decay factor: Vibrations stop after ~0.5s
            double decay = std::exp(-t_rec * 5.0); 
            // 5 Hz wobble (shaking legs)
            double wobble = 3.0 * decay * std::sin(2 * M_PI * 5.0 * t_rec);
            motion.accel[2] = 9.81 + wobble;
            // Also wobble the gyro (instability)
            motion.gyro[0] = 10.0 * decay * std::sin(2 * M_PI * 5.0 * t_rec);
        }
        
        // impact noise on gyro
        if (t_cycle < 0.7) {
             motion.gyro[0] += 10.0 * std::sin(time_s * 20.0);
        }
    }

    return motion;
}

// incorporate noise
IMUDataSample IMUSim::get_noisy_sample(const IMUDataSample& ideal_sample) {
    // convert "noise density" (in /sqrt(Hz)) to
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
    // simulates a "random walk" for the bias
    double accel_drift_std_dev = m_accel_bias_instability * std::sqrt(m_time_step);
    double gyro_drift_std_dev = m_gyro_bias_instability * std::sqrt(m_time_step);

    for (int i = 0; i < 3; ++i) {
        m_accel_bias[i] += m_norm_dist(m_rng) * accel_drift_std_dev;
        m_gyro_bias[i]  += m_norm_dist(m_rng) * gyro_drift_std_dev;
    }
}

double IMUSim::get_ideal_pressure(double time_s) {
    double height_meters = 0.0;
    
    if (m_activity_type == "stairs_up") {
        // Climbing: 0.5 m/s linear ascent
        // We add a tiny "step" wobble to match the walking rhythm
        double step_wobble = 0.1 * std::sin(2 * M_PI * 1.5 * time_s);
        height_meters = (0.5 * time_s) + step_wobble;
    }
    else if (m_activity_type == "elevator_up") {
        // Elevator: Fast 2.0 m/s smooth ascent
        // Usually has a "jerk" (acceleration) at start/stop, but linear is fine for now
        height_meters = 2.0 * time_s;
    }
    else {
        // Flat ground: Height is 0 (relative)
        // Add tiny weather drift (slow random walk) if you want to be fancy
        height_meters = 0.0;
    }

    // Physics Formula: P = P0 - (rho * g * h)
    // Approx: Drop 12 Pa per meter
    return m_base_pressure - (12.0 * height_meters);
}

// basic test
std::string helloEverybody() {
    return "Hello, Everybody!";
}