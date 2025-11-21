#pragma once
#include <vector>
#include <Eigen/Dense>
#include <cmath>
#include <numeric> 
#include <complex>

class FeatureExtractor {
public:
    // we'll maintain a rolling window of fixed size
    FeatureExtractor(int window_size, double computed_sample_rate_hz);

    // Add new samples, pop prev
    void add_sample(const Eigen::Vector3d& accel, const Eigen::Vector3d& gyro);

    // compute features over current window
    std::vector<double> compute_features() const;

private:
    int m_window_size;
    double computed_sample_rate_hz;
    
    // store history for each measure
    std::vector<double> m_acc_x, m_acc_y, m_acc_z;
    std::vector<double> m_gyro_x, m_gyro_y, m_gyro_z;

    // math helpers
    double calc_mean(const std::vector<double>& data) const;
    double calc_variance(const std::vector<double>& data, double mean) const;

    std::pair<double, double> calc_frequency_features(const std::vector<double>& data) const;
};