// i know its called imu_sim but it captures other sensor data too :D
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
        // getters for IMU
        Eigen::Vector3d get_acceleration() const { return m_latest_accel; }
        Eigen::Vector3d get_gyroscope() const    { return m_latest_gyro; }
        double get_current_time() const          { return m_current_time; }
        // be able to update activity
        void set_curr_activity(const std::string& activity);
        // barometer pressure data
        double get_pressure() const { return m_latest_pressure; }
        // proximity getter to capture PMUT
        double get_proximity() const             { return m_latest_proximity; }
        // setter for the ground truth distance
        void set_obstacle_distance(double distance_meters);

    private:
        // helper for raw motion info
        IMUDataSample get_ideal_motion(double time_s);

        // Helper function to add realistic noise
        IMUDataSample get_noisy_sample(const IMUDataSample& ideal_sample);
        void update_biases(); // Simulates the slow drift

        // helper for proximity sensor 
        double calculate_proximity_reading();

        double m_current_time;
        double m_sample_rate;
        double m_time_step; // 1.0 / m_sample_rate
        std::string m_activity_type;
        Eigen::Vector3d m_latest_accel;
        Eigen::Vector3d m_latest_gyro;

        // RNG noise sources
        std::default_random_engine m_rng;
        std::normal_distribution<double> m_norm_dist;

        // IMU Noise Parameters (can configure from real MEMS device specs)
        double m_accel_noise_density; // "White noise"
        double m_gyro_noise_density;  // "White noise"
        double m_accel_bias_instability; // "Bias drift"
        double m_gyro_bias_instability;  // "Bias drift"

        // Noise State drifting over time
        Eigen::Vector3d m_accel_bias;
        Eigen::Vector3d m_gyro_bias;

        // Barometer state params
        double get_ideal_pressure(double time_s);
        double m_latest_pressure;       // Current reading (Pascals)
        double m_base_pressure;         // Sea level (approx 101325 Pa)
        double m_pressure_noise_density;// Sensor noise

        // Proximity sensor state
        double m_true_distance;       // Ground truth from PyGame
        double m_latest_proximity;    // Noisy sensor output
        double m_prox_max_range;      // e.g., 4.0 meters
        double m_prox_noise_coeff;    // Noise increases with distance
};

std::string helloEverybody();