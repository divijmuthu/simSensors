#include <pybind11/pybind11.h>
#include "imu_simulator.h"

namespace py = pybind11;

PYBIND11_MODULE(sensor_sim, m) {
    m.doc() = "A C++ simulator for IMU data!";

    py::class_<IMUSim>(m, "IMUSim")
        .def(py::init<const std::string&>(), py::arg("activityType"))
        .def("update", &IMUSim::update)
        .def("get_current_time", &IMUSim::get_current_time)
        .def("set_curr_activity", &IMUSim::set_curr_activity, py::arg("activity"))
        .def("get_acceleration", &IMUSim::get_acceleration);

    m.def("helloEverybody", &helloEverybody, "A friendly greeting from C++");
}