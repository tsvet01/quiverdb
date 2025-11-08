cmake_minimum_required(VERSION 3.20)
project(quiverdb VERSION 0.1.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Header-only library
add_library(quiverdb INTERFACE)
target_include_directories(quiverdb INTERFACE 
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/src>
    $<INSTALL_INTERFACE:include>
)

# Benchmarks
find_package(benchmark REQUIRED)

add_executable(bench_distance benchmarks/bench_distance.cpp)
target_link_libraries(bench_distance PRIVATE quiverdb benchmark::benchmark)
target_compile_options(bench_distance PRIVATE
  $<$<CXX_COMPILER_ID:GNU,Clang,AppleClang>:-march=native -Wall -Wextra -Wpedantic>
  $<$<CXX_COMPILER_ID:MSVC>:/W4 /arch:AVX2>
)

# Tests
include(FetchContent)
FetchContent_Declare(
  Catch2
  GIT_REPOSITORY https://github.com/catchorg/Catch2.git
  GIT_TAG v3.11.0
)
FetchContent_MakeAvailable(Catch2)

enable_testing()

add_executable(test_distance tests/test_distance.cpp)
target_link_libraries(test_distance PRIVATE quiverdb Catch2::Catch2WithMain)
target_compile_options(test_distance PRIVATE
  $<$<CXX_COMPILER_ID:GNU,Clang,AppleClang>:-Wall -Wextra -Wpedantic>
  $<$<CXX_COMPILER_ID:MSVC>:/W4>
)

include(CTest)
list(APPEND CMAKE_MODULE_PATH ${catch2_SOURCE_DIR}/extras)
include(Catch)
catch_discover_tests(test_distance)
