#include "feature_extractor.h"
#include <iostream>

const double PI = 3.14159265358979323846;

FeatureExtractor::FeatureExtractor(int window_size, double computed_sample_rate_hz) 
    : m_window_size(window_size), computed_sample_rate_hz(computed_sample_rate_hz) {
    // reserve some memory for key vector data
    m_acc_x.reserve(window_size);
    m_acc_y.reserve(window_size);
    m_acc_z.reserve(window_size);
    m_gyro_x.reserve(window_size);
    m_gyro_y.reserve(window_size);
    m_gyro_z.reserve(window_size);
}

void FeatureExtractor::add_sample(const Eigen::Vector3d& accel, const Eigen::Vector3d& gyro) {
    // push new data to buffers
    m_acc_x.push_back(accel.x());
    m_acc_y.push_back(accel.y());
    m_acc_z.push_back(accel.z());
    m_gyro_x.push_back(gyro.x());
    m_gyro_y.push_back(gyro.y());
    m_gyro_z.push_back(gyro.z());

    // pop excess data if we exceed window size
    if (m_acc_x.size() > m_window_size) {
        m_acc_x.erase(m_acc_x.begin());
        m_acc_y.erase(m_acc_y.begin());
        m_acc_z.erase(m_acc_z.begin());
        m_gyro_x.erase(m_gyro_x.begin());
        m_gyro_y.erase(m_gyro_y.begin());
        m_gyro_z.erase(m_gyro_z.begin());
    }
}

std::vector<double> FeatureExtractor::compute_features() const {
    std::vector<double> features;
    // extract some features e.g. mean, variance
    // compute means first, then pass to both mean and variance computation 
    // double mean_ax = calc_mean(m_acc_x);
    // double mean_ay = calc_mean(m_acc_y);
    double mean_az = calc_mean(m_acc_z);    
    // features.push_back(mean_ax);
    // features.push_back(mean_ay);
    features.push_back(mean_az);
    // features.push_back(calc_variance(m_acc_x, mean_ax));
    // features.push_back(calc_variance(m_acc_y, mean_ay));
    features.push_back(calc_variance(m_acc_z, mean_az));

    // let's now compute DFT!
    std::pair<double, double> freq_feats = calc_frequency_features(m_acc_z);
    features.push_back(freq_feats.first);  // Dom freq in Hz
    features.push_back(freq_feats.second); // Spectral energy

    return features;
}

double FeatureExtractor::calc_mean(const std::vector<double>& data) const {
    if (data.empty()) return 0.0;
    // we can sum vector data with accumulate
    double sum = std::accumulate(data.begin(), data.end(), 0.0);
    return sum / data.size();
}

double FeatureExtractor::calc_variance(const std::vector<double>& data, double mean) const {
    if (data.size() < 2) return 0.0;
    // calc variance using accumulte, lambda fxn
    double sum_sq_diff = std::accumulate(data.begin(), data.end(), 0.0,
        [mean](double acc, double val) {
            double diff = val - mean;
            return acc + diff * diff;
        });
    return sum_sq_diff / data.size();
}

std::pair<double, double> FeatureExtractor::calc_frequency_features(const std::vector<double>& data) const {
    // DFT computation --> break into N bins, solve for half
    int N = data.size();
    if (N < 2) return {0.0, 0.0};
    int num_bins = N / 2;
    // grab mag
    double max_magnitude = 0.0;
    int dominant_bin_idx = 0;
    double total_energy = 0.0;
    // demean to capture AC signal
    double mean = calc_mean(data);
    // go thru freq bins 
    for (int k = 1; k < num_bins; ++k) {
        std::complex<double> sum(0.0, 0.0);
        // Check correlation with sin, cos
        for (int n = 0; n < N; ++n) {
            double angle = -2.0 * PI * k * n / N;
            std::complex<double> w(std::cos(angle), std::sin(angle));
            sum += (data[n] - mean) * w;
        }
        double magnitude = std::abs(sum);
        total_energy += magnitude * magnitude;
        if (magnitude > max_magnitude) {
            max_magnitude = magnitude;
            dominant_bin_idx = k;
        }
    }
    // Freq = k * (SampleRate / N) from formula, get freq from bin's index
    double dominant_freq = dominant_bin_idx * (computed_sample_rate_hz / N);
    return {dominant_freq, total_energy};
}