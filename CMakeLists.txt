cmake_minimum_required(VERSION 3.20)
project(quiverdb VERSION 0.1.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

# General warning flags for all targets and configurations
add_compile_options(-Wall -Wextra -Wpedantic)

# Configuration-specific flags
set(CMAKE_CXX_FLAGS_RELEASE "-O3 -march=native")
set(CMAKE_CXX_FLAGS_DEBUG "-O0 -g")
set(CMAKE_CXX_FLAGS_RELWITHDEBINFO "-O2 -g -march=native")

# Header-only library
add_library(quiverdb INTERFACE)
target_include_directories(quiverdb INTERFACE
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/src>
    $<INSTALL_INTERFACE:include>
)

# Options
option(QUIVERDB_BUILD_TESTS "Build tests" ON)
option(QUIVERDB_BUILD_BENCHMARKS "Build benchmarks" ON)
option(QUIVERDB_BUILD_EXAMPLES "Build examples" ON)

# Fetch dependencies
include(FetchContent)

# Google Benchmark
FetchContent_Declare(
  benchmark
  GIT_REPOSITORY https://github.com/google/benchmark.git
  GIT_TAG v1.9.4
)
set(BENCHMARK_ENABLE_TESTING OFF CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(benchmark)

# Benchmarks
if(QUIVERDB_BUILD_BENCHMARKS)
add_executable(bench_distance benchmarks/bench_distance.cpp)
target_link_libraries(bench_distance PRIVATE quiverdb benchmark::benchmark)
target_compile_options(bench_distance PRIVATE
  $<$<CXX_COMPILER_ID:MSVC>:/W4 /arch:AVX2>
)

add_executable(bench_vector_store benchmarks/bench_vector_store.cpp)
target_link_libraries(bench_vector_store PRIVATE quiverdb benchmark::benchmark)
target_compile_options(bench_vector_store PRIVATE
  $<$<CXX_COMPILER_ID:MSVC>:/W4 /arch:AVX2>
)

add_executable(bench_hnsw_index benchmarks/bench_hnsw_index.cpp)
target_link_libraries(bench_hnsw_index PRIVATE quiverdb benchmark::benchmark)
target_compile_options(bench_hnsw_index PRIVATE
  $<$<CXX_COMPILER_ID:MSVC>:/W4 /arch:AVX2>
)
endif() # QUIVERDB_BUILD_BENCHMARKS

# Tests - Catch2
if(QUIVERDB_BUILD_TESTS)
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
  $<$<CXX_COMPILER_ID:MSVC>:/W4>
)

add_executable(test_vector_store tests/test_vector_store.cpp)
target_link_libraries(test_vector_store PRIVATE quiverdb Catch2::Catch2WithMain)
target_compile_options(test_vector_store PRIVATE
  $<$<CXX_COMPILER_ID:MSVC>:/W4>
)

add_executable(test_hnsw_index tests/test_hnsw_index.cpp)
target_link_libraries(test_hnsw_index PRIVATE quiverdb Catch2::Catch2WithMain)
target_compile_options(test_hnsw_index PRIVATE
  $<$<CXX_COMPILER_ID:MSVC>:/W4 /arch:AVX2>
)

include(CTest)
list(APPEND CMAKE_MODULE_PATH ${catch2_SOURCE_DIR}/extras)
include(Catch)
catch_discover_tests(test_distance)
catch_discover_tests(test_vector_store)
catch_discover_tests(test_hnsw_index)
endif() # QUIVERDB_BUILD_TESTS

# Examples
if(QUIVERDB_BUILD_EXAMPLES)
add_executable(semantic_search_example examples/semantic_search_example.cpp)
target_link_libraries(semantic_search_example PRIVATE quiverdb)
target_compile_options(semantic_search_example PRIVATE
  $<$<CXX_COMPILER_ID:MSVC>:/W4>
)
endif() # QUIVERDB_BUILD_EXAMPLES

# Install
install(TARGETS quiverdb EXPORT quiverdb-targets)
install(DIRECTORY src/core/ DESTINATION include/quiverdb FILES_MATCHING PATTERN "*.h")
install(EXPORT quiverdb-targets FILE quiverdb-config.cmake DESTINATION lib/cmake/quiverdb)

# Pybind11 for Python bindings
FetchContent_Declare(
  pybind11
  GIT_REPOSITORY https://github.com/pybind/pybind11.git
  GIT_TAG v2.12.0
)
FetchContent_MakeAvailable(pybind11)

# Python bindings
pybind11_add_module(quiverdb_py MODULE python/quiverdb.cpp)
target_link_libraries(quiverdb_py PRIVATE quiverdb pybind11::embed)
target_compile_options(quiverdb_py PRIVATE
  $<$<CXX_COMPILER_ID:MSVC>:/W4 /arch:AVX2>
)

