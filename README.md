# bs-translator-backend

[![Release](https://img.shields.io/github/v/release/DCC-BS/bs-translator-backend)](https://img.shields.io/github/v/release/DCC-BS/bs-translator-backend)
[![Build status](https://img.shields.io/github/actions/workflow/status/DCC-BS/bs-translator-backend/main.yml?branch=main)](https://github.com/DCC-BS/bs-translator-backend/actions/workflows/main.yml?query=branch%3Amain)
[![codecov](https://codecov.io/gh/DCC-BS/bs-translator-backend/branch/main/graph/badge.svg)](https://codecov.io/gh/DCC-BS/bs-translator-backend)
[![Commit activity](https://img.shields.io/github/commit-activity/m/DCC-BS/bs-translator-backend)](https://img.shields.io/github/commit-activity/m/DCC-BS/bs-translator-backend)
[![License](https://img.shields.io/github/license/DCC-BS/bs-translator-backend)](https://img.shields.io/github/license/DCC-BS/bs-translator-backend)

This is a template repository for Python projects that use uv for their dependency management.

- **Github repository**: <https://github.com/DCC-BS/bs-translator-backend/>
- **Documentation** <https://DCC-BS.github.io/bs-translator-backend/>

## Getting started with your project

#### Pre-requisites

Windows
- Install [Git for Windows](https://git-scm.com/downloads/win)
- Install [Scoop](https://scoop.sh/)
- Install make: `scoop install make`

General
- Install [VSCode](https://code.visualstudio.com/)
- Install [uv](https://docs.astral.sh/uv/getting-started/installation/)


### 1. Create a New Repository

First, create a repository on GitHub with the same name as this project, and then run the following commands:

```bash
git init -b main
git add .
git commit -m "init commit"
git remote add origin git@github.com:DCC-BS/bs-translator-backend.git
git push -u origin main
```

### 2. Set Up Your Development Environment

Then, install the environment and the pre-commit hooks with

```bash
make install
```

This will also generate your `uv.lock` file

### 3. Run the pre-commit hooks

Initially, the CI/CD pipeline might be failing due to formatting issues. To resolve those run:

```bash
uv run pre-commit run -a
```

### 4. Commit the changes

Lastly, commit the changes made by the two steps above to your repository.

```bash
git add .
git commit -m 'Fix formatting issues'
git push origin main
```

You are now ready to start development on your project!
The CI/CD pipeline will be triggered when you open a pull request, merge to main, or when you create a new release.

To finalize the set-up for publishing to PyPI, see [here](https://DCC-BS.github.io/cookiecutter-uv/features/publishing/#set-up-for-pypi).
For activating the automatic documentation with MkDocs, see [here](https://DCC-BS.github.io/cookiecutter-uv/features/mkdocs/#enabling-the-documentation-on-github).
To enable the code coverage reports, see [here](https://DCC-BS.github.io/cookiecutter-uv/features/codecov/).

## Releasing a new version



---

Repository initiated with [DCC-BS/cookiecutter-uv](https://github.com/DCC-BS/cookiecutter-uv).
