# DPGEN2 Development Instructions

DPGEN2 is a Python package that implements concurrent learning workflows for generating machine learning potential energy models. It uses the dflow workflow platform (a wrapper for Argo Workflows) to orchestrate training, exploration, selection, and labeling steps in an iterative concurrent learning algorithm.

Always reference these instructions first and fallback to search or bash commands only when you encounter unexpected information that does not match the info here.

## Working Effectively

### Bootstrap and Installation
Run these commands in sequence to set up the development environment:
```bash
pip install --upgrade pip
pip install -e .
```
- Installation takes approximately 5 minutes. NEVER CANCEL.
- This installs DPGEN2 in development mode with all core dependencies
- Main dependencies: numpy, dpdata, pydflow, dargs, scipy, lbg, packaging, fpop, dpgui, cp2kdata

### Install Development Tools
```bash
pip install mock coverage pytest fakegaussian ruff isort
```
- Takes under 1 minute
- Required for testing and linting

### Test the Installation
```bash
dpgen2 --help
```
- Should show all CLI subcommands: submit, resubmit, showkey, status, download, watch, gui, etc.
- If this fails, the installation is incomplete

### Running Tests
Execute tests from the tests directory:
```bash
cd tests
SKIP_UT_WITH_DFLOW=0 DFLOW_DEBUG=1 python -m unittest <test_module> -v
```
- Individual test files take 1-30 seconds. NEVER CANCEL.
- Debug mode (DFLOW_DEBUG=1) runs workflows locally with mocked operators
- Test artifacts are created in tests/ directory but ignored by git

Full test suite command (as used in CI):
```bash
cd /path/to/repo
SKIP_UT_WITH_DFLOW=0 DFLOW_DEBUG=1 coverage run --source=./dpgen2 -m unittest -v -f && coverage report
```
- Full test suite takes 5-15 minutes. NEVER CANCEL. Set timeout to 30+ minutes.

### Linting and Formatting
Always run these before committing:
```bash
ruff format dpgen2/
isort dpgen2/
```
- Both take under 1 second
- ruff handles code formatting (replaces black)
- isort organizes imports
- CI will fail if code is not properly formatted

## CLI Usage and Validation

### Main CLI Commands
The `dpgen2` command provides these key subcommands:

**Submit a workflow:**
```bash
dpgen2 submit input.json
```
- Requires a JSON configuration file (see examples/almg/input.json)
- Returns a workflow ID (WFID) for tracking

**Check workflow status:**
```bash
dpgen2 status input.json WFID
```
- Shows convergence status and iteration progress

**Watch workflow progress:**
```bash
dpgen2 watch input.json WFID
```
- Live monitoring of workflow execution

**Show step keys:**
```bash
dpgen2 showkey input.json WFID
```
- Lists all workflow steps with unique identifiers

**Download artifacts:**
```bash
dpgen2 download input.json WFID
```
- Downloads results from completed workflow steps

### Validation Scenarios
After making changes, always test these core workflows:

1. **CLI Functionality Test:**
   ```bash
   dpgen2 --help
   dpgen2 submit --help
   dpgen2 status --help
   ```

2. **Run Unit Tests:**
   ```bash
   cd tests
   SKIP_UT_WITH_DFLOW=0 DFLOW_DEBUG=1 python -m unittest entrypoint.test_argparse -v
   ```
   - Takes ~1 second, validates argument parsing

3. **Test Core Workflow Components:**
   ```bash
   cd tests  
   SKIP_UT_WITH_DFLOW=0 DFLOW_DEBUG=1 python -m unittest test_block_cl -v
   ```
   - Takes ~30 seconds, tests core concurrent learning workflow

## Repository Structure and Key Locations

### Main Directories
- `dpgen2/` - Main Python package
  - `dpgen2/entrypoint/` - CLI commands and main entry points
  - `dpgen2/flow/` - Core workflow implementations
  - `dpgen2/op/` - Individual operators (training, exploration, etc.)
  - `dpgen2/exploration/` - Exploration strategies and task groups  
  - `dpgen2/fp/` - First-principles calculation interfaces
- `tests/` - Unit tests (use DFLOW_DEBUG=1 for local execution)
- `examples/` - Example configuration files
- `docs/` - Documentation (requires additional sphinx dependencies to build)

### Important Files
- `dpgen2/entrypoint/main.py` - Main CLI entry point
- `dpgen2/flow/dpgen_loop.py` - Core DPGEN workflow implementation  
- `examples/almg/input.json` - Example workflow configuration
- `pyproject.toml` - Project configuration and dependencies
- `.pre-commit-config.yaml` - Pre-commit hooks configuration

### Core Workflow Understanding
DPGEN2 implements a concurrent learning algorithm with these main components:

**Main Loop:**
- `block` operator - One complete iteration (training → exploration → selection → labeling)
- `exploration_strategy` - Determines convergence and generates next iteration tasks

**Key Steps in Each Iteration:**
- `prep_run_dp_train` - Prepare and run model training
- `prep_run_lmp` - Prepare and run LAMMPS exploration 
- `select_confs` - Select configurations for labeling
- `prep_run_fp` - Prepare and run first-principles calculations
- `collect_data` - Collect results and update dataset

## Common Development Tasks

### Testing Changes
1. **Quick validation** (lightweight tests): ~1-5 seconds
   ```bash
   cd tests
   SKIP_UT_WITH_DFLOW=0 DFLOW_DEBUG=1 python -m unittest entrypoint.test_argparse -v
   ```

2. **Core functionality** (workflow tests): ~30 seconds  
   ```bash
   cd tests
   SKIP_UT_WITH_DFLOW=0 DFLOW_DEBUG=1 python -m unittest test_block_cl -v
   ```

3. **Comprehensive testing** (full suite): 5-15 minutes, set 30+ minute timeout
   ```bash
   SKIP_UT_WITH_DFLOW=0 DFLOW_DEBUG=1 coverage run --source=./dpgen2 -m unittest -v -f
   ```

### Before Committing
Always run these commands and ensure they pass:
```bash
ruff format dpgen2/
isort dpgen2/
cd tests && SKIP_UT_WITH_DFLOW=0 DFLOW_DEBUG=1 python -m unittest entrypoint.test_argparse -v
```

### Working with Examples
Example configurations are in `examples/` directory:
- `examples/almg/input.json` - Al-Mg alloy workflow configuration
- `examples/water/` - Water system examples
- `examples/ch4/` - Methane system examples

### Debugging Workflow Issues
- Use `DFLOW_DEBUG=1` environment variable to run workflows locally
- Check `dpgen2 showkey` output to identify failed steps
- Use `dpgen2 download` to examine step outputs and logs
- Test artifacts are created in tests/ but excluded from git

## Build and Documentation

### No Complex Build Process
DPGEN2 is a pure Python package - no compilation required:
- Install with `pip install -e .`
- No Makefile, configure scripts, or build system
- Dependencies are managed through pyproject.toml

### Documentation
Located in `docs/` directory:
- Uses Sphinx with MyST parser for Markdown support
- Requires additional dependencies: `pip install sphinx sphinx-book-theme myst-parser`
- Build with `cd docs && make html` (may require deepmodeling-sphinx)

### CI/CD Pipeline
GitHub Actions workflow (`.github/workflows/test.yml`):
- Tests Python 3.8, 3.9, 3.10
- Runs full test suite with coverage
- Validates formatting and import organization

## Timing Expectations and "NEVER CANCEL" Guidelines

**Installation and Setup:**
- `pip install -e .`: ~5 minutes - NEVER CANCEL
- Development tools install: <1 minute

**Testing:**
- Individual unit tests: 1-30 seconds  
- Full test suite: 5-15 minutes - NEVER CANCEL, set 30+ minute timeout
- Linting (ruff/isort): <1 second

**No Long-Running Builds:**
- Pure Python package, no compilation required
- Fastest operations are linting and simple unit tests
- Most time-consuming operation is full test suite execution

When running any test command, always wait for completion. The test framework uses dflow in debug mode which can have variable timing depending on system load.