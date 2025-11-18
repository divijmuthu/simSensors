#pragma once
#include <string>
#include <random>
#include <Eigen/Dense>

class IMUSim {
    public: 
        IMUSim(const std::string& activityType);
        void update();
        double get_current_time() const {
            return currTime;
        }
        void set_curr_activity(std::string activity) {
            currActivityType = activity;
        }
        Eigen::Vector3d get_acceleration() const {
            return acceleration;
        }
    private:
        std::string currActivityType;
        double currTime;
        Eigen::Vector3d acceleration;
        std::default_random_engine generator;
        std::normal_distribution<double> noise_dist;
};

std::string helloEverybody();