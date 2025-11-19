# Contributing to Image Classification Project

Thank you for your interest in contributing! We welcome contributions of all kinds, whether they're bug reports, feature requests, documentation improvements, or code changes.

## Getting Started

### Prerequisites
- Python 3.8 or higher
- Git

### Setup Your Development Environment

1. **Fork and clone the repository:**
   ```bash
   git clone https://github.com/YOUR_USERNAME/image-classification.git
   cd image-classification
   ```

2. **Create a virtual environment:**
   ```bash
   python -m venv venv
   # On Windows
   venv\Scripts\activate
   # On macOS/Linux
   source venv/bin/activate
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

4. **Set up pre-commit hooks (optional but recommended):**
   ```bash
   pre-commit install
   ```

## Development Workflow

### Creating a Feature Branch

Always create a feature branch for your work:

```bash
# For new features
git checkout -b feature/your-feature-name

# For bug fixes
git checkout -b fix/your-bug-fix-name

# For documentation
git checkout -b docs/your-doc-update-name
```

### Code Style

We follow these standards:

- **Python formatting:** Use `black` (configured in `pyproject.toml` if present)
- **Imports:** Organize with `isort` 
- **Linting:** Run `flake8` to catch style issues
- **Type hints:** Encouraged but not required

Before committing, run:

```bash
black .
flake8 src/ tests/
pytest
```

### Writing Tests

- Add tests in the `tests/` directory for any new functionality
- Use `pytest` as the test runner
- Aim for at least 80% code coverage on new code
- Test file naming: `test_*.py` or `*_test.py`

Example test structure:
```python
import pytest
from src.model import MyModel

def test_model_initialization():
    model = MyModel()
    assert model is not None

def test_model_prediction():
    model = MyModel()
    output = model.predict([[1, 2, 3]])
    assert output.shape == (1,)
```

### Notebook Conventions

- Keep notebooks for **exploration and visualization only**
- Move reusable code into `src/` modules
- Run `nbstripout` before committing large outputs:
  ```bash
  nbstripout notebook.ipynb
  ```
- Document assumptions and results clearly in markdown cells

### Commit Messages

Write clear, descriptive commit messages:

```
# Good
git commit -m "Add CNN model for digit classification"
git commit -m "Fix data loading bug in preprocessing"
git commit -m "Update README with quickstart guide"

# Avoid
git commit -m "fix stuff"
git commit -m "WIP"
```

## Submitting a Pull Request

### Before Submitting

1. **Ensure all tests pass:**
   ```bash
   pytest
   ```

2. **Check code formatting:**
   ```bash
   black --check .
   flake8 src/ tests/
   ```

3. **Update documentation** if you changed functionality:
   - Update README.md if adding new features or changing usage
   - Add docstrings to new functions and classes
   - Update data/ or other relevant docs

4. **Keep commits clean:** 
   - Squash or rebase if you have many small commits
   - Ensure commit messages are clear

### Creating the PR

1. Push your branch to your fork:
   ```bash
   git push origin feature/your-feature-name
   ```

2. Open a pull request on GitHub with:
   - **Title:** Clear description of changes
   - **Description:** Reference any related issues (#123), explain what changed and why
   - Include before/after results if applicable

### PR Checklist

Please ensure your PR includes:

- [ ] Descriptive title and clear description
- [ ] Related issue referenced (if applicable)
- [ ] All tests passing (`pytest`)
- [ ] Code formatted with `black`
- [ ] No new linting errors (`flake8`)
- [ ] Updated README if changing usage
- [ ] Added/updated tests for new functionality
- [ ] Notebooks cleaned with `nbstripout` (if modified)
- [ ] No large files committed (use git-lfs for >10MB)

## Code Review Process

- At least one maintainer will review your PR
- We may request changes or have questions
- Once approved, your PR will be merged
- Your contribution will be recognized in the README

## Reporting Issues

### Bug Reports

Include:
- Python version and OS
- Steps to reproduce
- Expected vs actual behavior
- Error messages/tracebacks
- Minimal code example if possible

### Feature Requests

Include:
- Clear description of what you want to add
- Why it would be useful
- Potential implementation approach (if you have ideas)

## Project Structure

```
image-classification/
├── src/                    # Reusable modules
│   ├── data.py            # Data loading and preprocessing
│   ├── model.py           # Model definitions
│   ├── train.py           # Training loops
│   └── utils.py           # Helper functions
├── notebooks/             # Jupyter notebooks for exploration
├── tests/                 # Unit tests
├── data/                  # Data directory (in .gitignore)
├── models/                # Saved model weights (in .gitignore)
├── requirements.txt       # Python dependencies
├── README.md             # Project overview
└── CONTRIBUTING.md       # This file
```

## Questions?

- Open an issue for questions or discussions
- Check existing issues before creating new ones
- Be respectful and constructive in all interactions

## License

By contributing, you agree that your contributions will be licensed under the MIT License.

---

**Happy contributing!** 🎉
