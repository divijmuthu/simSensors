#pragma once
#include <vector>
#include <Eigen/Dense>
#include <cmath>
#include <numeric> 
#include <complex>
#include <deque>

class FeatureExtractor {
public:
    // we'll maintain a rolling window of fixed size
    FeatureExtractor(int window_size, double computed_sample_rate_hz);
    // Add new samples, pop prev
    void add_sample(const Eigen::Vector3d& accel, const Eigen::Vector3d& gyro, double pressure);
    // compute features over current window
    std::vector<double> compute_features() const;
    std::vector<double> get_z_accel_buffer() const;
    std::vector<double> get_fft_spectrum() const;

private:
    int m_window_size;
    double computed_sample_rate_hz;
    
    // store history for each measure
    std::deque<double> m_acc_x, m_acc_y, m_acc_z;
    std::deque<double> m_gyro_x, m_gyro_y, m_gyro_z;
    std::deque<double> m_pressure;

    // math helpers
    double calc_mean(const std::deque<double>& data) const;
    double calc_variance(const std::deque<double>& data, double mean) const;
    std::pair<double, double> calc_frequency_features(const std::deque<double>& data) const;

    // compute height change over time from pressure data
    double calc_vertical_velocity() const;
};