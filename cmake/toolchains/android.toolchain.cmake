# Android NDK Toolchain for QuiverDB
# Usage: cmake -B build-android -DCMAKE_TOOLCHAIN_FILE=cmake/toolchains/android.toolchain.cmake
#
# Requires: ANDROID_NDK environment variable or -DANDROID_NDK=/path/to/ndk
# The NDK includes its own toolchain file which this wraps with QuiverDB defaults

# Minimum Android API level (Android 7.0 Nougat)
set(ANDROID_PLATFORM "android-24" CACHE STRING "Android API level")

# Target ARM64 (most modern Android devices)
set(ANDROID_ABI "arm64-v8a" CACHE STRING "Android ABI")

# Use Clang (default in modern NDK)
set(ANDROID_TOOLCHAIN "clang" CACHE STRING "Android toolchain")

# Use libc++ (modern C++ standard library)
set(ANDROID_STL "c++_shared" CACHE STRING "Android STL")

# C++20 support
set(ANDROID_CPP_FEATURES "rtti exceptions" CACHE STRING "C++ features")

# Find NDK path
if(NOT ANDROID_NDK)
    if(DEFINED ENV{ANDROID_NDK})
        set(ANDROID_NDK $ENV{ANDROID_NDK})
    elseif(DEFINED ENV{ANDROID_NDK_HOME})
        set(ANDROID_NDK $ENV{ANDROID_NDK_HOME})
    elseif(DEFINED ENV{ANDROID_HOME})
        set(ANDROID_NDK "$ENV{ANDROID_HOME}/ndk-bundle")
    endif()
endif()

if(NOT ANDROID_NDK)
    message(FATAL_ERROR "ANDROID_NDK not set. Please set ANDROID_NDK environment variable or pass -DANDROID_NDK=/path/to/ndk")
endif()

# Include NDK's toolchain file
set(NDK_TOOLCHAIN_FILE "${ANDROID_NDK}/build/cmake/android.toolchain.cmake")
if(EXISTS "${NDK_TOOLCHAIN_FILE}")
    include("${NDK_TOOLCHAIN_FILE}")
else()
    message(FATAL_ERROR "NDK toolchain file not found: ${NDK_TOOLCHAIN_FILE}")
endif()

# QuiverDB-specific settings
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Disable unsupported features for Android builds
set(QUIVERDB_BUILD_TESTS OFF CACHE BOOL "Disable tests for Android")
set(QUIVERDB_BUILD_BENCHMARKS OFF CACHE BOOL "Disable benchmarks for Android")
set(QUIVERDB_BUILD_PYTHON OFF CACHE BOOL "Disable Python for Android")
set(QUIVERDB_BUILD_EXAMPLES OFF CACHE BOOL "Disable examples for Android")

# Enable NEON for arm64-v8a
if(ANDROID_ABI STREQUAL "arm64-v8a")
    add_compile_definitions(QUIVER_ARM_NEON=1)
elseif(ANDROID_ABI STREQUAL "x86_64")
    add_compile_definitions(QUIVER_AVX2=1)
endif()

message(STATUS "QuiverDB Android toolchain loaded")
message(STATUS "  NDK: ${ANDROID_NDK}")
message(STATUS "  ABI: ${ANDROID_ABI}")
message(STATUS "  Platform: ${ANDROID_PLATFORM}")
