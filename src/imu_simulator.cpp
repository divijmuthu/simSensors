#include "imu_simulator.h"

std::string helloEverybody() {
    return "Hello, Everybody!";
}

IMUSim::IMUSim(const std::string& activityType)
    : currActivityType(activityType), currTime(0.0), acceleration(Eigen::Vector3d::Zero()),
      noise_dist(0.0, 1.0) {
    // Initialize random number generator with a seed
    std::random_device rd;
    generator.seed(rd());
}

void IMUSim::update() {
    // Placeholder implementation for update method
}