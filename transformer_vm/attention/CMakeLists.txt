cmake_minimum_required(VERSION 3.18)
project(hull_ext CXX)

find_package(pybind11 REQUIRED)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

pybind11_add_module(hull_ext hull_ext.cpp)
target_include_directories(hull_ext PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
target_compile_options(hull_ext PRIVATE -O3)
