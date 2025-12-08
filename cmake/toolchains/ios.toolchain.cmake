# iOS Toolchain for QuiverDB
# Usage: cmake -B build-ios -DCMAKE_TOOLCHAIN_FILE=cmake/toolchains/ios.toolchain.cmake -GXcode

set(CMAKE_SYSTEM_NAME iOS)
set(CMAKE_OSX_DEPLOYMENT_TARGET "13.0" CACHE STRING "Minimum iOS version")

# Build for arm64 (modern iOS devices)
set(CMAKE_OSX_ARCHITECTURES "arm64" CACHE STRING "iOS architectures")

# Set the SDK (device or simulator)
set(CMAKE_OSX_SYSROOT "iphoneos" CACHE STRING "iOS SDK")

# Standard settings
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Disable unsupported features for iOS builds
set(QUIVERDB_BUILD_TESTS OFF CACHE BOOL "Disable tests for iOS")
set(QUIVERDB_BUILD_BENCHMARKS OFF CACHE BOOL "Disable benchmarks for iOS")
set(QUIVERDB_BUILD_PYTHON OFF CACHE BOOL "Disable Python for iOS")
set(QUIVERDB_BUILD_EXAMPLES OFF CACHE BOOL "Disable examples for iOS")

# Enable NEON (always available on arm64 iOS)
add_compile_definitions(QUIVER_ARM_NEON=1)

message(STATUS "QuiverDB iOS toolchain loaded")
message(STATUS "  Target: iOS ${CMAKE_OSX_DEPLOYMENT_TARGET}+")
message(STATUS "  Architectures: ${CMAKE_OSX_ARCHITECTURES}")
