#include "feature_extractor.h"
#include <iostream>

const double PI = 3.14159265358979323846;

FeatureExtractor::FeatureExtractor(int window_size, double computed_sample_rate_hz) 
    : m_window_size(window_size), computed_sample_rate_hz(computed_sample_rate_hz) {
    // reserve some memory for key vector data
}

void FeatureExtractor::add_sample(const Eigen::Vector3d& accel, const Eigen::Vector3d& gyro, const Eigen::Vector3d& mag, double pressure) {
    // new data joins from the end of the deque
    m_acc_x.push_back(accel.x()); m_acc_y.push_back(accel.y()); m_acc_z.push_back(accel.z());
    m_gyro_x.push_back(gyro.x()); m_gyro_y.push_back(gyro.y()); m_gyro_z.push_back(gyro.z());
    m_mag_x.push_back(mag.x()); m_mag_y.push_back(mag.y()); m_mag_z.push_back(mag.z());
    m_pressure.push_back(pressure);

    // remove old data from the front of the deque
    if (m_acc_x.size() > m_window_size) {
        m_acc_x.pop_front(); m_acc_y.pop_front(); m_acc_z.pop_front();
        m_gyro_x.pop_front(); m_gyro_y.pop_front(); m_gyro_z.pop_front();
        m_mag_x.pop_front(); m_mag_y.pop_front(); m_mag_z.pop_front();
        m_pressure.pop_front();
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

    // add our vertical velocity feature via pressure data
    features.push_back(calc_vertical_velocity());

    // Magnetometer Mean X & Y (Heading info)
    features.push_back(calc_mean(m_mag_x));
    features.push_back(calc_mean(m_mag_y));

    return features;
}

double FeatureExtractor::calc_mean(const std::deque<double>& data) const {
    if (data.empty()) return 0.0;
    // we can sum vector data with accumulate
    double sum = std::accumulate(data.begin(), data.end(), 0.0);
    return sum / data.size();
}

double FeatureExtractor::calc_variance(const std::deque<double>& data, double mean) const {
    if (data.size() < 2) return 0.0;
    // calc variance using accumulte, lambda fxn
    double sum_sq_diff = std::accumulate(data.begin(), data.end(), 0.0,
        [mean](double acc, double val) {
            double diff = val - mean;
            return acc + diff * diff;
        });
    return sum_sq_diff / data.size();
}

std::pair<double, double> FeatureExtractor::calc_frequency_features(const std::deque<double>& data) const {
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

std::vector<double> FeatureExtractor::get_z_accel_buffer() const {
    // cast deque to vector to make pybind happy
    return std::vector<double>(m_acc_z.begin(), m_acc_z.end());
}

std::vector<double> FeatureExtractor::get_fft_spectrum() const {
    // does the same FFT logic, gives full info to display later on
    int N = m_acc_z.size();
    if (N < 2) return {};

    int num_bins = N / 2;
    std::vector<double> spectrum;
    spectrum.reserve(num_bins);

    double mean = calc_mean(m_acc_z);

    // calculate magnitude for every bin
    for (int k = 0; k < num_bins; ++k) {
        std::complex<double> sum(0.0, 0.0);
        for (int n = 0; n < N; ++n) {
            double angle = -2.0 * PI * k * n / N;
            std::complex<double> w(std::cos(angle), std::sin(angle));
            sum += (m_acc_z[n] - mean) * w;
        }
        spectrum.push_back(std::abs(sum));
    }
    return spectrum;
}

double FeatureExtractor::calc_vertical_velocity() const {
    // wait 1 sec for pressure difference
    int one_sec_samples = (int)computed_sample_rate_hz;
    if (m_pressure.size() <= one_sec_samples) return 0.0;
    // grab curr pressure
    double p_now = m_pressure.back();
    // check 1 sec ago for prev pressure
    int idx_old = m_pressure.size() - 1 - one_sec_samples;
    if (idx_old < 0) idx_old = 0; // Safety clamp
    double p_old = m_pressure[idx_old];
    // calc diff in pa 
    double diff_pa = p_now - p_old;
    // Convert to Meters (Approx 12 Pa = 1 Meter)
    // Note: Pressure GOES DOWN as height GOES UP. 
    // So negative pressure diff = positive height gain.
    double height_change = -(diff_pa / 12.0);
    // Velocity = Distance / Time (Time is 1.0s)
    return height_change; 
}