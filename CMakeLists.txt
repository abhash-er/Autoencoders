cmake_minimum_required(VERSION 3.1)
project(Autoencoders)

find_package(xtl REQUIRED)
find_package(xtensor REQUIRED)

add_library(
    src-deep_learningSigmoid.cpp
    src/deep_learning/Sigmoid.h
)

add_executable(Autoencoders src/main.cpp  src/deep_learning/Sigmoid.cpp)

if(MSVC)
    set(CMAKE_EXE_LINKER_FLAGS /MANIFEST:NO)
endif()

target_link_libraries(Autoencoders xtensor xtensor::optimize)