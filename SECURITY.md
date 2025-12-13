# Security Policy

## Supported Versions

| Version | Supported          |
| ------- | ------------------ |
| 0.1.x   | :white_check_mark: |

## Reporting a Vulnerability

If you discover a security vulnerability in QuiverDB, please report it responsibly:

1. **Do NOT open a public GitHub issue** for security vulnerabilities
2. **Email the maintainer** with details of the vulnerability
3. Include:
   - Description of the vulnerability
   - Steps to reproduce
   - Potential impact
   - Any suggested fixes (optional)

## Response Timeline

- **Acknowledgment**: Within 48 hours
- **Initial Assessment**: Within 7 days
- **Fix Timeline**: Depends on severity
  - Critical: 24-72 hours
  - High: 1-2 weeks
  - Medium: 2-4 weeks
  - Low: Next release cycle

## Scope

### In Scope

- Memory safety issues (buffer overflows, use-after-free)
- File format vulnerabilities (malicious index files)
- Integer overflows leading to security issues
- Thread safety bugs causing data corruption
- Input validation bypasses

### Out of Scope

- Denial of service via large allocations (documented limitation)
- Performance issues
- Bugs that require physical access to the system
- Issues in dependencies (report to upstream)

## Security Measures

QuiverDB implements several security measures:

- **Input validation**: All public APIs validate inputs
- **Bounds checking**: Array accesses are bounds-checked
- **Overflow protection**: Integer overflow checks in file parsing
- **File format validation**: Magic numbers, version checks, size limits
- **Atomic file operations**: Crash-safe save with temp file + rename
- **Sanitizer testing**: AddressSanitizer and UBSan in CI

## Acknowledgments

We appreciate responsible disclosure and will acknowledge security researchers who report valid vulnerabilities (with their permission).
