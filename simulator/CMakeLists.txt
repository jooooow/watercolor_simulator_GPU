cmake_minimum_required(VERSION 3.6)

project(simulator_v2_opt4_GPU_simulator LANGUAGES CXX CUDA)

set(CMAKE_BUILD_TYPE Release)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")

find_package(CUDA REQUIRED)

set(CUDA_SEPARABLE_COMPILATION ON)

set(SOURCES 
    src/main.cu
    #src/watercolor.cpp
    #src/canvas.cpp
    #src/stroke.cpp
)
add_executable(${PROJECT_NAME} ${SOURCES})

find_package(OpenCV REQUIRED)
if(OpenCV_FOUND)
    message("found opencv")
    message("opencv version : ${OpenCV_VERSION}")
endif()

find_package(OpenMP REQUIRED)
if(OpenMP_CXX_FOUND)
    message("found openmp")
    message("openmp version : ${OpenMP_CXX_VERSION}")
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
    set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -Xcompiler=${OpenMP_CXX_FLAGS}") 
    set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${OpenMP_CXX_FLAGS}") 
endif()

target_include_directories(
    ${PROJECT_NAME} PRIVATE 
    ${OpenCV_INCLUDE_DIRS}
    ${PROJECT_SOURCE_DIR}/include
    ${PROJECT_SOURCE_DIR}/..
    ${PROJECT_SOURCE_DIR}/../../CPU/include
)
target_link_libraries(${PROJECT_NAME} ${OpenCV_LIBS})

target_compile_options(${PROJECT_NAME} 
PUBLIC $<$<COMPILE_LANGUAGE:CUDA>:--ptxas-options=-v -arch=sm_80>)


message("CMAKE_CXX_FLAGS_RELEASE : ${CMAKE_CXX_FLAGS_RELEASE}")
message("CMAKE_CUDA_FLAGS_RELEASE : ${CMAKE_CUDA_FLAGS_RELEASE}")
message("CMAKE_BUILD_TYPE : ${CMAKE_BUILD_TYPE}")