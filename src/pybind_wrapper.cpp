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
        .def("get_gyroscope", &IMUSim::get_gyroscope)
        // add barometer getter
        .def("get_pressure", &IMUSim::get_pressure);

    // bind FeatureExtractor to a python class
    py::class_<FeatureExtractor>(m, "FeatureExtractor")
        .def(py::init<int, double>(), py::arg("window_size"), py::arg("sample_rate_hz")) 
        .def("add_sample", &FeatureExtractor::add_sample, py::arg("accel"), py::arg("gyro"), py::arg("pressure"))
        .def("compute_features", &FeatureExtractor::compute_features)
        .def("get_z_accel_buffer", &FeatureExtractor::get_z_accel_buffer)
        .def("get_fft_spectrum", &FeatureExtractor::get_fft_spectrum);
        
    // bind test as well lol
    m.def("helloEverybody", &helloEverybody, "A friendly greeting from C++");
}