# CMakeLists.txt

cmake_minimum_required(VERSION 3.15)
project(sensor_sim) # Give our project a name

# Find required packages
find_package(pybind11 REQUIRED)
find_package(Eigen3 REQUIRED)

# cook up python module from our files
pybind11_add_module(sensor_sim SHARED
    src/imu_simulator.cpp
    src/pybind_wrapper.cpp
)

# Link Eigen3
target_link_libraries(sensor_sim PRIVATE Eigen3::Eigen)

# set output directory for the python module
set_target_properties(sensor_sim PROPERTIES
    LIBRARY_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR}/python)