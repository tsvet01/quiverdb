#pragma once

/**
 * @file logging.h
 * @brief Lightweight structured logging for QuiverDB.
 *
 * Simple, header-only logging utility that supports:
 * - Log levels (TRACE, DEBUG, INFO, WARN, ERROR)
 * - Structured key-value logging
 * - Zero overhead when logging is disabled
 * - Thread-safe output
 *
 * Usage:
 *   QUIVERDB_LOG_INFO("Operation completed", "vectors", 1000, "latency_ms", 5.2);
 *   QUIVERDB_LOG_ERROR("Index failed", "reason", "dimension_mismatch");
 *
 * To enable logging, define QUIVERDB_ENABLE_LOGGING before including this header.
 * Log level can be controlled via QUIVERDB_LOG_LEVEL (default: INFO).
 */

#include <chrono>
#include <cstdio>
#include <cstring>
#include <iomanip>
#include <mutex>
#include <sstream>
#include <string>

namespace quiverdb {
namespace logging {

/**
 * @brief Log level enumeration.
 */
enum class Level {
    TRACE = 0,
    DEBUG = 1,
    INFO = 2,
    WARN = 3,
    ERROR = 4,
    OFF = 5
};

/**
 * @brief Convert log level to string.
 */
inline const char* level_to_string(Level level) {
    switch (level) {
        case Level::TRACE: return "TRACE";
        case Level::DEBUG: return "DEBUG";
        case Level::INFO:  return "INFO";
        case Level::WARN:  return "WARN";
        case Level::ERROR: return "ERROR";
        default:           return "UNKNOWN";
    }
}

/**
 * @brief Global log level (can be changed at runtime).
 */
inline Level& global_level() {
    static Level level =
#if defined(QUIVERDB_LOG_LEVEL_TRACE)
        Level::TRACE;
#elif defined(QUIVERDB_LOG_LEVEL_DEBUG)
        Level::DEBUG;
#elif defined(QUIVERDB_LOG_LEVEL_WARN)
        Level::WARN;
#elif defined(QUIVERDB_LOG_LEVEL_ERROR)
        Level::ERROR;
#else
        Level::INFO;
#endif
    return level;
}

/**
 * @brief Set the global log level.
 */
inline void set_level(Level level) {
    global_level() = level;
}

/**
 * @brief Get the global log level.
 */
inline Level get_level() {
    return global_level();
}

/**
 * @brief Mutex for thread-safe logging.
 */
inline std::mutex& log_mutex() {
    static std::mutex mtx;
    return mtx;
}

/**
 * @brief Format a single key-value pair.
 */
template <typename T>
inline void format_kv(std::ostringstream& ss, const char* key, const T& value) {
    ss << " " << key << "=" << value;
}

/**
 * @brief Format key-value pairs (base case).
 */
inline void format_kvs(std::ostringstream&) {}

/**
 * @brief Format key-value pairs (recursive case).
 */
template <typename V, typename... Args>
inline void format_kvs(std::ostringstream& ss, const char* key, const V& value, Args&&... args) {
    format_kv(ss, key, value);
    format_kvs(ss, std::forward<Args>(args)...);
}

/**
 * @brief Core logging function.
 */
template <typename... Args>
inline void log(Level level, const char* file, int line, const char* message, Args&&... args) {
#ifdef QUIVERDB_ENABLE_LOGGING
    if (level < global_level()) {
        return;
    }

    // Get timestamp
    auto now = std::chrono::system_clock::now();
    auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(
        now.time_since_epoch()) % 1000;
    auto time = std::chrono::system_clock::to_time_t(now);
    std::tm tm_buf;
#ifdef _WIN32
    localtime_s(&tm_buf, &time);
#else
    localtime_r(&time, &tm_buf);
#endif

    // Extract filename from path
    const char* filename = std::strrchr(file, '/');
    if (!filename) filename = std::strrchr(file, '\\');
    filename = filename ? filename + 1 : file;

    // Build structured log message
    std::ostringstream ss;
    ss << std::put_time(&tm_buf, "%Y-%m-%d %H:%M:%S") << "."
       << std::setfill('0') << std::setw(3) << ms.count()
       << " [" << level_to_string(level) << "] "
       << filename << ":" << line << " "
       << message;
    format_kvs(ss, std::forward<Args>(args)...);
    ss << "\n";

    // Thread-safe output
    {
        std::lock_guard<std::mutex> lock(log_mutex());
        std::fputs(ss.str().c_str(), stderr);
    }
#else
    (void)level;
    (void)file;
    (void)line;
    (void)message;
    // Suppress unused warnings for args in disabled logging
    (void)std::initializer_list<int>{(static_cast<void>(args), 0)...};
#endif
}

} // namespace logging
} // namespace quiverdb

// Convenience macros
#define QUIVERDB_LOG(level, msg, ...) \
    ::quiverdb::logging::log(level, __FILE__, __LINE__, msg, ##__VA_ARGS__)

#define QUIVERDB_LOG_TRACE(msg, ...) \
    QUIVERDB_LOG(::quiverdb::logging::Level::TRACE, msg, ##__VA_ARGS__)

#define QUIVERDB_LOG_DEBUG(msg, ...) \
    QUIVERDB_LOG(::quiverdb::logging::Level::DEBUG, msg, ##__VA_ARGS__)

#define QUIVERDB_LOG_INFO(msg, ...) \
    QUIVERDB_LOG(::quiverdb::logging::Level::INFO, msg, ##__VA_ARGS__)

#define QUIVERDB_LOG_WARN(msg, ...) \
    QUIVERDB_LOG(::quiverdb::logging::Level::WARN, msg, ##__VA_ARGS__)

#define QUIVERDB_LOG_ERROR(msg, ...) \
    QUIVERDB_LOG(::quiverdb::logging::Level::ERROR, msg, ##__VA_ARGS__)
