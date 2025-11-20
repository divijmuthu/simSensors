#pragma once
#include <string>
#include <random>
#include <Eigen/Dense>

struct IMUDataSample {
    Eigen::Vector3d accel;
    Eigen::Vector3d gyro;
};

class IMUSim {
    public: 
        // constructor needs fs and current activity
        IMUSim(double sample_rate_hz, const std::string& activityType);
        // prep a update method for state
        void update();
        // getters
        Eigen::Vector3d get_acceleration() const { return m_latest_accel; }
        Eigen::Vector3d get_gyroscope() const    { return m_latest_gyro; }
        double get_current_time() const          { return m_current_time; }
        // be able to update activity
        void set_curr_activity(const std::string& activity);

    private:
        // helper for raw motion info
        IMUDataSample get_ideal_motion(double time_s);

        // Helper function to add realistic noise
        IMUDataSample get_noisy_sample(const IMUDataSample& ideal_sample);
        void update_biases(); // Simulates the slow drift

        double m_current_time;
        double m_sample_rate;
        double m_time_step; // 1.0 / m_sample_rate
        std::string m_activity_type;
        Eigen::Vector3d m_latest_accel;
        Eigen::Vector3d m_latest_gyro;
        std::default_random_engine m_rng;
        std::normal_distribution<double> m_norm_dist;

        // Noise Parameters (grab from real MEMS device specs)
        double m_accel_noise_density; // "White noise"
        double m_gyro_noise_density;  // "White noise"
        double m_accel_bias_instability; // "Bias drift"
        double m_gyro_bias_instability;  // "Bias drift"

        // Noise State drifting over time
        Eigen::Vector3d m_accel_bias;
        Eigen::Vector3d m_gyro_bias;
};

std::string helloEverybody();