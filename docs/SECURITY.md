# Security Policy

## Security Best Practices

### Container Security
- Images run as non-root user (`appuser`)
- Minimal base image (`python:3.11-slim`)
- Multi-stage builds where applicable
- Regular vulnerability scanning with Trivy

### Dependency Management
- Pin major versions in `requirements.txt`
- Automated vulnerability scanning via `pip-audit` in CI
- GitHub Security events integrated with SARIF reporting

### Secrets & Configuration
- Use environment variables for sensitive config (see `.env.example`)
- Secrets are passed via environment, not hardcoded
- API keys and credentials should use GitHub Secrets in CI/CD

### Input Validation
- FastAPI endpoints use Pydantic models for validation
- Request data is validated before inference

### Data Protection
- Model artifacts stored in version-controlled `models/` (can be excluded from public repos)
- Training data should not be stored in the repository

## Reporting Security Issues

If you discover a security vulnerability, please email security@project.local with:
- Description of the issue
- Affected components
- Severity (critical, high, medium, low)
- Suggested fix (if applicable)

Do not open a public GitHub issue for security matters.

## Security Scanning

The CI pipeline includes:
1. **Dependency scanning:** `pip-audit` checks for known vulnerabilities in Python packages
2. **Container scanning:** Trivy scans the built image for OS-level vulnerabilities
3. **Code scanning:** GitHub CodeQL analysis (can be enabled)
4. **Linting:** `flake8` for code quality

## Regular Updates

- Python base image updated regularly (currently Python 3.11)
- Dependencies updated via Dependabot or manual review
- Security patches applied promptly
