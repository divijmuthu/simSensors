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

// UPGRADED sim of raw ideal motion
IMUDataSample IMUSim::get_ideal_motion(double time_s) {
    IMUDataSample motion;

    // sitting is just plain gravity do nothing more
    motion.accel = {0.0, 0.0, 9.81};
    motion.gyro = {0.0, 0.0, 0.0};

    if (m_activity_type == "sitting") {
    }
    else if (m_activity_type == "walking") {
        // capture w/2 Hz sine wave of bobbing, 2x per sec up/down makes sense
        double freq = 2.0; 
        double bob_amp = 0.5; // m/s^2
        double swing_amp = 20.0; // deg/s

        motion.accel[2] = 9.81 + (bob_amp * std::sin(2 * M_PI * freq * time_s));
        motion.gyro[1]  = swing_amp * std::sin(2 * M_PI * freq * time_s);
    }
    else if (m_activity_type == "running") {
        // faster 3.5 Hz bobbing + bigger amplitude for swinging faster movement of legs makes sense
        double freq = 3.5; 
        double bob_amp = 2.0;  // more vertical bounce
        double swing_amp = 60.0; // aggressive leg swing

        motion.accel[2] = 9.81 + (bob_amp * std::sin(2 * M_PI * freq * time_s));
        motion.gyro[1]  = swing_amp * std::sin(2 * M_PI * freq * time_s);
    }
    else if (m_activity_type == "jumping") {
        // This is complex: IMPULSE physics.
        // Cycle: 1.5 seconds total.
        // 0.0 - 0.2s: Push off (High G)
        // 0.2 - 0.5s: In Air (Zero G)
        // 0.5 - 0.7s: Landing (Massive G)
        // 0.7 - 1.5s: Recovery (Standing still)
        
        double cycle_len = 1.5;
        double t_cycle = std::fmod(time_s, cycle_len); // set up cycle track curr time

        if (t_cycle < 0.2) {
            // push off: +1.5g boost
            motion.accel[2] = 9.81 + 15.0; 
        } 
        else if (t_cycle < 0.5) {
            // in air --> freefall ~0g
            motion.accel[2] = 0.1; 
        }
        else if (t_cycle < 0.7) {
            // landing = hard impact +2.5g
            motion.accel[2] = 9.81 + 25.0;
        }
        else {
            // recovery = just gravity
            motion.accel[2] = 9.81;
        }
        // incorporate noise here for the impact
        if (t_cycle < 0.7) {
             motion.gyro[0] = 10.0 * std::sin(time_s * 20.0);
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

// basic test
std::string helloEverybody() {
    return "Hello, Everybody!";
}