# Contributing to MAGE

Thank you for your interest in contributing to MAGE (Mixed Audio Generation Engine)!

## Getting Started

1. Fork the repository
2. Clone your fork:
   ```bash
   git clone https://github.com/yourusername/mage.git
   cd mage
   ```
3. Create a virtual environment:
   ```bash
   python -m venv venv
   venv\Scripts\activate  # Windows
   source venv/bin/activate  # Linux/Mac
   ```
4. Install development dependencies:
   ```bash
   pip install -r requirements.txt
   pip install -e .
   ```

## Development Workflow

### 1. Create a Branch

```bash
git checkout -b feature/your-feature-name
```

### 2. Make Changes

- Write clean, well-documented code
- Follow PEP 8 style guidelines
- Add type hints where appropriate
- Include comprehensive error handling
- Add logging statements for debugging

### 3. Write Tests

- Add tests for new functionality in `tests/`
- Ensure tests pass:
  ```bash
  pytest tests/
  ```
- Aim for high code coverage:
  ```bash
  pytest --cov=mage tests/
  ```

### 4. Format Code

```bash
black mage/ tests/
flake8 mage/ tests/
mypy mage/
```

### 5. Commit Changes

```bash
git add .
git commit -m "feat: add new feature description"
```

Use conventional commit messages:
- `feat:` for new features
- `fix:` for bug fixes
- `docs:` for documentation
- `test:` for tests
- `refactor:` for refactoring

### 6. Push and Create PR

```bash
git push origin feature/your-feature-name
```

Then create a Pull Request on GitHub.

## Code Style

- Follow PEP 8
- Use type hints
- Write docstrings for all public functions/classes
- Keep functions focused and small
- Use descriptive variable names

## Error Handling

- Use custom exceptions from `mage.exceptions`
- Always include error context in exception details
- Log errors appropriately
- Provide helpful error messages

## Logging

- Use the `MAGELogger` for all logging
- Include appropriate log levels:
  - DEBUG: Detailed diagnostic information
  - INFO: General informational messages
  - WARNING: Warning messages
  - ERROR: Error messages
  - CRITICAL: Critical errors
- Use structured logging with extra context

## Testing

- Write unit tests for all new code
- Use pytest fixtures for common setup
- Test edge cases and error conditions
- Mock external dependencies

## Documentation

- Update README.md if adding features
- Add docstrings to all public APIs
- Include usage examples
- Update configuration documentation

## Questions?

Open an issue for discussion before starting major changes.
