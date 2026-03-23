cmake_minimum_required(VERSION 3.16)
project(agx_zed_runtime LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

include(CTest)
enable_testing()

find_package(PkgConfig REQUIRED)
pkg_check_modules(OPENCV4 REQUIRED opencv4)
find_package(yaml-cpp REQUIRED)

add_executable(agx_zed_geometry_tests
    tests/test_geometry.cpp
    src/geometry/depth_sampler.cpp
    src/geometry/projection.cpp
    src/geometry/world_transform.cpp
)
target_include_directories(agx_zed_geometry_tests PRIVATE include)
add_test(NAME agx_zed_geometry_tests COMMAND agx_zed_geometry_tests)
set_tests_properties(agx_zed_geometry_tests PROPERTIES WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR})

add_executable(agx_zed_postprocess_tests
    tests/test_postprocess.cpp
    src/detector/postprocess.cpp
)
target_include_directories(agx_zed_postprocess_tests PRIVATE include)
add_test(NAME agx_zed_postprocess_tests COMMAND agx_zed_postprocess_tests)
set_tests_properties(agx_zed_postprocess_tests PROPERTIES WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR})

add_executable(agx_zed_config_loader_tests
    tests/test_config_loader.cpp
    src/common/config_loader.cpp
)
target_include_directories(agx_zed_config_loader_tests PRIVATE include)
target_link_libraries(agx_zed_config_loader_tests PRIVATE yaml-cpp)
add_test(NAME agx_zed_config_loader_tests COMMAND agx_zed_config_loader_tests)
set_tests_properties(agx_zed_config_loader_tests PROPERTIES WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR})

add_executable(agx_zed_pipeline_runtime_tests
    tests/test_pipeline_runtime.cpp
    src/pipeline/ball_pipeline.cpp
    src/camera/zed_camera.cpp
    src/detector/trt_engine.cpp
    src/detector/detector.cpp
    src/detector/postprocess.cpp
    src/geometry/depth_sampler.cpp
    src/geometry/projection.cpp
    src/geometry/world_transform.cpp
)
target_include_directories(agx_zed_pipeline_runtime_tests PRIVATE
    include
    ${OPENCV4_INCLUDE_DIRS}
    /usr/local/cuda/include
    /usr/local/zed/include
)
target_link_directories(agx_zed_pipeline_runtime_tests PRIVATE
    /usr/local/zed/lib
    /usr/local/cuda/targets/aarch64-linux/lib
)
target_link_libraries(agx_zed_pipeline_runtime_tests PRIVATE
    ${OPENCV4_LIBRARIES}
    sl_zed
    nvinfer
    cudart
)
add_test(NAME agx_zed_pipeline_runtime_tests COMMAND agx_zed_pipeline_runtime_tests)
set_tests_properties(agx_zed_pipeline_runtime_tests PROPERTIES WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR})

add_executable(agx_zed_runtime_overlay_tests
    tests/test_runtime_overlay.cpp
    src/common/visualization.cpp
)
target_include_directories(agx_zed_runtime_overlay_tests PRIVATE
    include
    ${OPENCV4_INCLUDE_DIRS}
)
target_link_libraries(agx_zed_runtime_overlay_tests PRIVATE ${OPENCV4_LIBRARIES})
add_test(NAME agx_zed_runtime_overlay_tests COMMAND agx_zed_runtime_overlay_tests)
set_tests_properties(agx_zed_runtime_overlay_tests PROPERTIES WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR})

add_executable(agx_zed_runtime
    src/main.cpp
    src/common/config_loader.cpp
    src/common/visualization.cpp
    src/camera/zed_camera.cpp
    src/detector/trt_engine.cpp
    src/detector/detector.cpp
    src/detector/postprocess.cpp
    src/geometry/depth_sampler.cpp
    src/geometry/projection.cpp
    src/geometry/world_transform.cpp
    src/pipeline/ball_pipeline.cpp
)

target_include_directories(agx_zed_runtime PRIVATE
    include
    ${OPENCV4_INCLUDE_DIRS}
    /usr/local/cuda/include
    /usr/local/zed/include
)

target_link_directories(agx_zed_runtime PRIVATE
    /usr/local/zed/lib
    /usr/local/cuda/targets/aarch64-linux/lib
)

target_link_libraries(agx_zed_runtime PRIVATE
    ${OPENCV4_LIBRARIES}
    yaml-cpp
    sl_zed
    nvinfer
    cudart
)
