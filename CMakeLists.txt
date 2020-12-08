cmake_minimum_required(VERSION 3.1)
project(Autoencoders)

find_package(xtl REQUIRED)
find_package(xtensor REQUIRED)

add_executable(Autoencoders src/)

if(MSVC)
    set(CMAKE_EXE_LINKER_FLAGS /MANIFEST:NO)
endif()

target_link_libraries(Autoencoders xtensor xtensor::optimize xtensor::use_xsimd)