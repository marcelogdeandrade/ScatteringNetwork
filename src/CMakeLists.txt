cmake_minimum_required(VERSION 3.0)

PROJECT(mnist_example)

# .. -> hint, that the mnist package is one directory level above.
# When using just "find_package(MNIST REQUIRED)", "MNIST_DIR"
#    cmake variable has to be set correctly.
find_package(MNIST PATHS ..)
if(NOT MNIST_FOUND)
    message(FATAL_ERROR "MNIST loader could not be found. It is available under https://github.com/wichtounet/mnist")
endif(NOT MNIST_FOUND)


enable_language(CUDA)

find_package( OpenCV REQUIRED )
include_directories(${MNIST_INCLUDE_DIR})
include_directories( ${OpenCV_INCLUDE_DIRS} )
add_executable(main main.cpp)
add_executable(main_gpu main_gpu.cpp})
set_property(TARGET main_gpu PROPERTY CUDA_STANDARD 11)
target_compile_features(main PRIVATE cxx_range_for)
target_link_libraries( main ${OpenCV_LIBS} )

# Pass MNIST data directory to main.cpp
target_compile_definitions(main PRIVATE MNIST_DATA_LOCATION="${MNIST_DATA_DIR}")
