# Contributing to CuFlash-Attn

First off, thank you for considering contributing to CuFlash-Attn! It's people like you that make CuFlash-Attn such a great tool.

## Table of Contents

- [Code of Conduct](#code-of-conduct)
- [Getting Started](#getting-started)
- [Development Setup](#development-setup)
- [Spec-Driven Development Workflow](#spec-driven-development-workflow)
- [How to Contribute](#how-to-contribute)
- [Coding Standards](#coding-standards)
- [Commit Guidelines](#commit-guidelines)
- [Pull Request Process](#pull-request-process)

## Code of Conduct

This project and everyone participating in it is governed by the [Code of Conduct](CODE_OF_CONDUCT.md). By participating, you are expected to uphold this code.

## Getting Started

- Make sure you have a [GitHub account](https://github.com)
- Fork the repository on GitHub
- Check out the [existing issues](https://github.com/LessUp/cuflash-attn/issues) for things to work on
- **Read the spec documents in `/specs/`** to understand the project requirements and design

## Development Setup

### Prerequisites

- **CUDA Toolkit** 11.0+ (tested with 12.4.1)
- **CMake** 3.18+
- **C++17** compatible compiler
  - GCC 9+
  - Clang 10+
  - MSVC 2019+ (experimental)
- **GPU**: NVIDIA GPU with compute capability 7.0+ (V100, A100, RTX 20/30/40 series, H100)

### Building from Source

```bash
# Clone your fork
git clone https://github.com/YOUR_USERNAME/cuflash-attn.git
cd cuflash-attn

# Configure and build
cmake --preset release
cmake --build --preset release

# Run tests
ctest --preset release
```

### Python Test Dependencies (Optional)

If you want to run the PyTorch comparison tests:

```bash
pip install -r requirements-dev.txt
```

## Spec-Driven Development Workflow

This project follows **Spec-Driven Development (SDD)** methodology. All implementation details, requirements, and design decisions are documented in the `/specs/` directory.

### Specification Structure

```
specs/
├── product/            # Product requirements and acceptance criteria
├── rfc/                # Technical design documents (RFCs)
├── api/                # API specifications
├── db/                 # Database/schema specifications (if applicable)
└── testing/            # Testing specifications and BDD test cases
```

### Workflow Guidelines

1. **Read Specs First**: Before implementing any feature or fixing any bug, read the relevant spec documents in `/specs/`.

2. **Spec-First Updates**: If your change requires new functionality or interface changes, **propose updating the spec document first**. Wait for approval before writing code.

3. **Implementation**: Write code that **exactly matches** the spec definitions. Do not add features not defined in specs.

4. **Test Against Spec**: Write tests that validate acceptance criteria defined in specs. Reference spec document IDs in test comments.

For detailed AI agent workflow instructions, see [AGENTS.md](AGENTS.md).

## How to Contribute

### Reporting Bugs

Before creating bug reports, please check the issue list as you might find out that you don't need to create one. When you are creating a bug report, please include as many details as possible:

- **Use a clear and descriptive title**
- **Describe the exact steps to reproduce the problem**
- **Provide specific examples to demonstrate the steps**
- **Describe the behavior you observed and expected**
- **Include your environment details** (OS, CUDA version, GPU model, etc.)

### Suggesting Enhancements

Enhancement suggestions are tracked as GitHub issues. When creating an enhancement suggestion, include:

- **Use a clear and descriptive title**
- **Provide a step-by-step description of the suggested enhancement**
- **Explain why this enhancement would be useful**
- **List some other projects or papers where this enhancement exists**

### Pull Requests

- Fill in the required template
- Do not include issue numbers in the title
- Include screenshots and animated GIFs in your pull request whenever possible
- Follow the coding standards and commit guidelines

## Coding Standards

### C++/CUDA Code Style

We use **clang-format** with LLVM style for code formatting. Run the formatter before committing:

```bash
# Format all files
find . -name "*.cu" -o -name "*.cuh" -o -name "*.cpp" -o -name "*.h" | xargs clang-format -i
```

### General Guidelines

- Use meaningful variable and function names
- Add comments for complex algorithms
- Keep functions focused and modular
- Write tests for new features
- Update documentation for API changes

### File Organization

```
specs/             # Specification documents (SDD)
include/           # Public API headers
src/               # Implementation files
tests/             # Test files
examples/          # Example code
docs/              # Documentation
```

## Commit Guidelines

We follow conventional commit messages:

```
<type>(<scope>): <description>

[optional body]

[optional footer]
```

### Types

- `feat`: A new feature
- `fix`: A bug fix
- `docs`: Documentation only changes
- `style`: Changes that do not affect the meaning of the code
- `refactor`: A code change that neither fixes a bug nor adds a feature
- `test`: Adding missing tests or correcting existing tests
- `chore`: Changes to the build process or auxiliary tools

### Examples

```
feat(api): add FP16 backward support
fix(kernel): correct causal mask boundary condition
docs(guide): update installation instructions
test(backward): add gradient check for edge cases
```

## Pull Request Process

1. **Fork** the repo and create your branch from `main`
2. **Make your changes** following our coding standards
3. **Add tests** for any new functionality
4. **Update documentation** as needed
5. **Run the test suite** to ensure all tests pass
6. **Submit a pull request** with a clear description

### PR Checklist

- [ ] Code compiles without errors
- [ ] All tests pass locally
- [ ] Code follows the project's coding standards
- [ ] New code has appropriate test coverage
- [ ] Documentation is updated if needed
- [ ] Spec documents are updated if interfaces/features changed
- [ ] Commit messages follow our guidelines

## Questions?

Feel free to open an issue with the "question" label or reach out to the maintainers.

Thank you for contributing! 🎉
