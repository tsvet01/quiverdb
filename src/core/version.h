#pragma once

/**
 * @file version.h
 * @brief QuiverDB version information.
 */

namespace quiverdb {

/// Major version number
constexpr int VERSION_MAJOR = 0;
/// Minor version number
constexpr int VERSION_MINOR = 1;
/// Patch version number
constexpr int VERSION_PATCH = 0;

/// Full version string
constexpr const char* VERSION_STRING = "0.1.0";

/**
 * @brief Returns the version as a single integer for comparison.
 *
 * Format: MAJOR * 10000 + MINOR * 100 + PATCH
 * Example: 0.1.0 = 100
 */
constexpr int version_number() {
    return VERSION_MAJOR * 10000 + VERSION_MINOR * 100 + VERSION_PATCH;
}

} // namespace quiverdb
