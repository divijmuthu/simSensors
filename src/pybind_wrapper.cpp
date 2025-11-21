#include <pybind11/pybind11.h>
#include <pybind11/stl.h> 
#include <pybind11/eigen.h> 
#include "imu_simulator.h"
#include "feature_extractor.h"

namespace py = pybind11;

PYBIND11_MODULE(sensor_sim, m) {
    m.doc() = "A C++ simulator for IMU data!";

    // bind IMUSim
    py::class_<IMUSim>(m, "IMUSim")
        // add constructor
        .def(py::init<double, const std::string&>(), 
             py::arg("sample_rate_hz"), 
             py::arg("activityType") = "sitting") // Default value

        // expose methods
        .def("update", &IMUSim::update)
        .def("get_current_time", &IMUSim::get_current_time)
        .def("set_curr_activity", &IMUSim::set_curr_activity, py::arg("activity"))
        
        // expose getters, eigen will convert to numpy!
        .def("get_acceleration", &IMUSim::get_acceleration)
        .def("get_gyroscope", &IMUSim::get_gyroscope);

    // bind FeatureExtractor to a python class
    py::class_<FeatureExtractor>(m, "FeatureExtractor")
        .def(py::init<int>(), py::arg("window_size"))
        .def("add_sample", &FeatureExtractor::add_sample)
        // smartly converts vector to np array
        .def("compute_features", &FeatureExtractor::compute_features);
        
    // bind test as well lol
    m.def("helloEverybody", &helloEverybody, "A friendly greeting from C++");
}