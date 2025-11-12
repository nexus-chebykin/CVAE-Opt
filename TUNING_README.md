# Hyperparameter Tuning Framework

## Overview

This directory contains a comprehensive hyperparameter tuning framework for all 9 optimizers in the CVAE-Opt project, implemented using the [Optuna](https://optuna.org/) optimization framework.

## Available Optimizers

| Optimizer | Key | Tuning Script | Tunable Parameters |
|-----------|-----|---------------|-------------------|
| Standard DE | `de` | `optuna_de_tune.py` | `mutate`, `recombination` |
| EvoX ODE | `ode` | `optuna_ode_tune.py` | `base_vector`, `num_difference_vectors`, `differential_weight`, `cross_probability` |
| EvoX CoDE | `code` | `optuna_code_tune.py` | `diff_padding_num`, `replace` |
| EvoX SHADE | `shade` | `optuna_shade_tune.py` | `diff_padding_num` |
| EvoX SaDE | `sade` | `optuna_sade_tune.py` | `diff_padding_num`, `LP` |
| Standard CMA-ES | `cmaes` | `optuna_cmaes_tune.py` | `sigma0`, `CMA_rankmu`, `CMA_rankone` |
| IPOP-CMA-ES | `ipop` | `optuna_ipop_tune.py` | `sigma0`, `CMA_rankmu`, `CMA_rankone` |
| BIPOP-CMA-ES | `bipop` | `optuna_bipop_tune.py` | `sigma0`, `CMA_rankmu`, `CMA_rankone` |
| SciPy DE | `scipy_de` | `optuna_scipy_de_tune.py` | `strategy`, `mutation`, `recombination_scipy`, `updating` |

## Quick Start

### Tune All Optimizers

```bash
# Using Python master script (recommended)
uv run python run_all_tuning.py \
  --model_path models/tsp_100_model.pt \
  --instances_path instances/tsp/test/tsp100_10inst_w_optimal.pkl \
  --n_trials 50 \
  --tune_n_instances 3

# Using Bash master script (simpler alternative)
./run_all_tuning.sh \
  models/tsp_100_model.pt \
  instances/tsp/test/tsp100_10inst_w_optimal.pkl \
  50 \
  3
```

### Tune Individual Optimizer

```bash
# Example: Tune Standard DE
uv run python optuna_de_tune.py \
  --model_path models/tsp_100_model.pt \
  --instances_path instances/tsp/test/tsp100_10inst_w_optimal.pkl \
  --n_trials 50 \
  --tune_n_instances 3 \
  --batch_size 600 \
  --time_limit 75 \
  --seed 1234
```

## Master Scripts

### Python Master Script: `run_all_tuning.py`

The Python master script provides advanced features:

**Basic usage:**
```bash
uv run python run_all_tuning.py \
  --model_path <model_path> \
  --instances_path <instances_path>
```

**Select specific optimizers:**
```bash
uv run python run_all_tuning.py \
  --model_path models/tsp_100_model.pt \
  --instances_path instances/tsp/test/tsp100_10inst_w_optimal.pkl \
  --optimizers de,cmaes,ipop
```

**Skip certain optimizers:**
```bash
uv run python run_all_tuning.py \
  --model_path models/tsp_100_model.pt \
  --instances_path instances/tsp/test/tsp100_10inst_w_optimal.pkl \
  --skip shade,sade
```

**Resume existing studies:**
```bash
uv run python run_all_tuning.py \
  --model_path models/tsp_100_model.pt \
  --instances_path instances/tsp/test/tsp100_10inst_w_optimal.pkl \
  --n_trials 100 \
  --resume
```

**Full options:**
- `--model_path`: Path to trained CVAE model (.pt file) - **Required**
- `--instances_path`: Path to problem instances (.pkl file) - **Required**
- `--n_trials`: Number of Optuna trials per optimizer (default: 50)
- `--tune_n_instances`: Number of instances for tuning (default: 3)
- `--batch_size`: Population size (default: 600)
- `--time_limit`: Time limit per instance in seconds (default: 75)
- `--seed`: Random seed for reproducibility (default: 1234)
- `--optimizers`: Comma-separated list of optimizers to tune
- `--skip`: Comma-separated list of optimizers to skip
- `--resume`: Resume existing Optuna studies

### Bash Master Script: `run_all_tuning.sh`

Simpler bash alternative for running all optimizers:

```bash
./run_all_tuning.sh \
  <model_path> \
  <instances_path> \
  [n_trials] \
  [tune_n_instances] \
  [batch_size] \
  [time_limit] \
  [seed]
```

**Example:**
```bash
./run_all_tuning.sh \
  models/tsp_100_model.pt \
  instances/tsp/test/tsp100_10inst_w_optimal.pkl \
  50 \
  3 \
  600 \
  75 \
  1234
```

## Individual Tuning Scripts

Each optimizer has its own dedicated tuning script with a consistent interface:

### Common Arguments

All tuning scripts accept the following arguments:

- `--model_path`: Path to trained model (.pt file) - **Required**
- `--instances_path`: Path to problem instances (.pkl file) - **Required**
- `--n_trials`: Number of Optuna trials (default: 50)
- `--tune_n_instances`: Number of instances for tuning (default: 3)
- `--batch_size`: Population size (default: 600)
- `--time_limit`: Time limit per instance in seconds (default: 75)
- `--seed`: Random seed (default: 1234)
- `--storage`: Optuna database path (default: sqlite:///optuna_<optimizer>_tuning.db)
- `--load_if_exists`: Resume existing study if it exists

### Optimizer-Specific Details

#### 1. Standard DE (`optuna_de_tune.py`)

**Tunable parameters:**
- `mutate` (F): Mutation scaling factor [0.4, 1.0], log-scale
- `recombination` (CR): Crossover probability [0.7, 0.95], uniform

**Example:**
```bash
uv run python optuna_de_tune.py \
  --model_path models/tsp_100_model.pt \
  --instances_path instances/tsp/test/tsp100_10inst_w_optimal.pkl
```

#### 2. EvoX ODE (`optuna_ode_tune.py`)

**Tunable parameters:**
- `base_vector`: Mutation base strategy (categorical: 'best', 'rand')
- `num_difference_vectors`: Number of difference vectors (categorical: 1, 2)
- `differential_weight` (F): Mutation scaling factor [0.4, 1.0], log-scale
- `cross_probability` (CR): Crossover probability [0.7, 0.95], uniform

**Example:**
```bash
uv run python optuna_ode_tune.py \
  --model_path models/tsp_100_model.pt \
  --instances_path instances/tsp/test/tsp100_10inst_w_optimal.pkl
```

#### 3. EvoX CoDE (`optuna_code_tune.py`)

**Tunable parameters:**
- `diff_padding_num`: Differential padding number [3, 10], uniform integer
- `replace`: Population replacement strategy (categorical: True, False)

**Example:**
```bash
uv run python optuna_code_tune.py \
  --model_path models/tsp_100_model.pt \
  --instances_path instances/tsp/test/tsp100_10inst_w_optimal.pkl
```

#### 4. EvoX SHADE (`optuna_shade_tune.py`)

**Tunable parameters:**
- `diff_padding_num`: Differential padding number [5, 15], uniform integer

**Note:** SHADE has limited tunable parameters because F and CR are adapted automatically.

**Example:**
```bash
uv run python optuna_shade_tune.py \
  --model_path models/tsp_100_model.pt \
  --instances_path instances/tsp/test/tsp100_10inst_w_optimal.pkl
```

#### 5. EvoX SaDE (`optuna_sade_tune.py`)

**Tunable parameters:**
- `diff_padding_num`: Differential padding number [5, 15], uniform integer
- `LP`: Learning period (memory size) [30, 100], uniform integer

**Example:**
```bash
uv run python optuna_sade_tune.py \
  --model_path models/tsp_100_model.pt \
  --instances_path instances/tsp/test/tsp100_10inst_w_optimal.pkl
```

#### 6. Standard CMA-ES (`optuna_cmaes_tune.py`)

**Tunable parameters:**
- `sigma0`: Initial step size [0.15, 1.0], uniform
- `CMA_rankmu`: Rank-mu update learning rate multiplier [0.5, 2.0], uniform
- `CMA_rankone`: Rank-one update learning rate multiplier [0.5, 2.0], uniform

**Fixed parameters:**
- `use_lhs`: True (Latin Hypercube Sampling)

**Example:**
```bash
uv run python optuna_cmaes_tune.py \
  --model_path models/tsp_100_model.pt \
  --instances_path instances/tsp/test/tsp100_10inst_w_optimal.pkl
```

#### 7. IPOP-CMA-ES (`optuna_ipop_tune.py`)

**Tunable parameters:**
- `sigma0`: Initial step size [0.15, 1.0], uniform
- `CMA_rankmu`: Rank-mu update learning rate multiplier [0.5, 2.0], uniform
- `CMA_rankone`: Rank-one update learning rate multiplier [0.5, 2.0], uniform

**Fixed parameters:**
- `restarts`: 5
- `incpopsize`: 2.0 (population doubling)
- `use_lhs`: True

**Example:**
```bash
uv run python optuna_ipop_tune.py \
  --model_path models/tsp_100_model.pt \
  --instances_path instances/tsp/test/tsp100_10inst_w_optimal.pkl
```

#### 8. BIPOP-CMA-ES (`optuna_bipop_tune.py`)

**Tunable parameters:**
- `sigma0`: Initial step size [0.15, 1.0], uniform
- `CMA_rankmu`: Rank-mu update learning rate multiplier [0.5, 2.0], uniform
- `CMA_rankone`: Rank-one update learning rate multiplier [0.5, 2.0], uniform

**Fixed parameters:**
- `restarts`: 5
- `incpopsize`: 2.0 (population doubling for large restarts)
- `use_lhs`: True

**Example:**
```bash
uv run python optuna_bipop_tune.py \
  --model_path models/tsp_100_model.pt \
  --instances_path instances/tsp/test/tsp100_10inst_w_optimal.pkl
```

#### 9. SciPy DE (`optuna_scipy_de_tune.py`)

**Tunable parameters:**
- `strategy`: DE mutation strategy (categorical: 'best1bin', 'rand1bin', 'randtobest1bin', 'currenttobest1bin')
- `mutation`: Mutation factor (fixed float [0.5, 1.0] OR adaptive dithering [0.3, 0.7] to [0.8, 1.5])
- `recombination_scipy`: Crossover probability [0.7, 0.95], uniform
- `updating`: Population update strategy (categorical: 'deferred', 'immediate')

**Fixed parameters:**
- `init`: 'latinhypercube'

**Example:**
```bash
uv run python optuna_scipy_de_tune.py \
  --model_path models/tsp_100_model.pt \
  --instances_path instances/tsp/test/tsp100_10inst_w_optimal.pkl
```

## Output Files

### Per-Optimizer Outputs

Each tuning script generates the following outputs:

1. **Best parameters JSON:** `best_<optimizer>_params.json`
   - Contains best hyperparameters found
   - Includes best mean gap, trial count, and metadata

2. **Optuna database:** `optuna_<optimizer>_tuning.db`
   - SQLite database with all trial results
   - Can be resumed with `--load_if_exists` flag

3. **Visualization plots** (if kaleido is installed):
   - `<optimizer>_optimization_history.png`: Convergence over trials
   - `<optimizer>_param_importances.png`: Parameter importance analysis
   - `<optimizer>_parallel_coordinate.png`: Parameter interaction visualization

4. **Log file:** `logs/optuna_<optimizer>_tuning_<timestamp>.log`
   - Detailed logging of all trials

### Master Script Outputs

The master scripts generate:

1. **Summary JSON:** `tuning_summary.json`
   - Overall statistics across all optimizers
   - Success/failure status for each optimizer
   - Total time and individual durations

2. **Master log:** `logs/master_tuning_<timestamp>.log`
   - Combined log of entire tuning process

## Tuning Configuration

### Default Configuration

Based on the hyperparameter tuning plan:

- **Number of trials per optimizer:** 50
- **Tuning instances:** 3 (first 3 from dataset)
- **Time limit per instance:** 75 seconds
- **Batch size (population):** 600
- **Random seed:** 1234
- **Sampler:** TPE (Tree-structured Parzen Estimator)

### Estimated Time Budget

With default settings (50 trials, 3 instances, 75s per instance):

- **Time per trial:** ~3.75 minutes (3 instances × 75s + overhead)
- **Time per optimizer:** ~3.1 hours (50 trials × 3.75 min)
- **Total time (9 optimizers):** ~28 hours

### Customizing Configuration

**Faster tuning (fewer trials):**
```bash
uv run python run_all_tuning.py \
  --model_path models/tsp_100_model.pt \
  --instances_path instances/tsp/test/tsp100_10inst_w_optimal.pkl \
  --n_trials 20 \
  --tune_n_instances 2
```

**More thorough tuning:**
```bash
uv run python run_all_tuning.py \
  --model_path models/tsp_100_model.pt \
  --instances_path instances/tsp/test/tsp100_10inst_w_optimal.pkl \
  --n_trials 100 \
  --tune_n_instances 5
```

## Resuming Interrupted Tuning

All tuning scripts support resuming via Optuna's persistent storage:

**Resume individual optimizer:**
```bash
uv run python optuna_de_tune.py \
  --model_path models/tsp_100_model.pt \
  --instances_path instances/tsp/test/tsp100_10inst_w_optimal.pkl \
  --load_if_exists
```

**Resume master tuning:**
```bash
uv run python run_all_tuning.py \
  --model_path models/tsp_100_model.pt \
  --instances_path instances/tsp/test/tsp100_10inst_w_optimal.pkl \
  --resume
```

## Visualization

To enable visualization export, install kaleido:

```bash
uv add kaleido
```

This will generate PNG plots for:
- Optimization history (convergence over trials)
- Parameter importances (which parameters matter most)
- Parallel coordinate plots (parameter interactions)

Without kaleido, you can still view results programmatically using Optuna's API.

## Advanced Usage

### Parallel Tuning

Run multiple optimizers in parallel using different terminals:

```bash
# Terminal 1
uv run python optuna_de_tune.py --model_path ... --instances_path ...

# Terminal 2 (simultaneously)
uv run python optuna_cmaes_tune.py --model_path ... --instances_path ...

# Terminal 3 (simultaneously)
uv run python optuna_ipop_tune.py --model_path ... --instances_path ...
```

### Custom Optuna Analysis

Load and analyze results programmatically:

```python
import optuna

# Load study
study = optuna.load_study(
    study_name='de_tuning',
    storage='sqlite:///optuna_de_tuning.db'
)

# Get best parameters
print(f"Best params: {study.best_params}")
print(f"Best value: {study.best_value}")

# Analyze trials
for trial in study.trials:
    print(f"Trial {trial.number}: {trial.value} - {trial.params}")

# Generate custom plots
from optuna.visualization import plot_optimization_history
fig = plot_optimization_history(study)
fig.show()
```

### Incremental Tuning

Start with fewer trials, then add more:

```bash
# Initial tuning with 25 trials
uv run python optuna_de_tune.py \
  --model_path models/tsp_100_model.pt \
  --instances_path instances/tsp/test/tsp100_10inst_w_optimal.pkl \
  --n_trials 25

# Check results, then add 25 more trials
uv run python optuna_de_tune.py \
  --model_path models/tsp_100_model.pt \
  --instances_path instances/tsp/test/tsp100_10inst_w_optimal.pkl \
  --n_trials 25 \
  --load_if_exists
```

## Troubleshooting

### Common Issues

**1. Out of memory:**
- Reduce `--batch_size` (e.g., from 600 to 400)
- Ensure GPU has sufficient memory
- Use CPU if necessary: `export CUDA_VISIBLE_DEVICES=""`

**2. Tuning taking too long:**
- Reduce `--n_trials` (e.g., 25 instead of 50)
- Reduce `--tune_n_instances` (e.g., 2 instead of 3)
- Reduce `--time_limit` (e.g., 60 instead of 75)

**3. Study already exists error:**
- Use `--load_if_exists` to resume
- Or delete the database: `rm optuna_<optimizer>_tuning.db`

**4. Visualization not working:**
- Install kaleido: `uv add kaleido`
- Or skip visualizations (still generates JSON results)

**5. Model/instances not found:**
- Check file paths are correct
- Use absolute paths if relative paths fail

### Debugging

Enable verbose logging in tuning scripts:

```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

Check Optuna study status:

```bash
uv run python -c "
import optuna
study = optuna.load_study(
    study_name='de_tuning',
    storage='sqlite:///optuna_de_tuning.db'
)
print(f'Trials completed: {len(study.trials)}')
print(f'Best value: {study.best_value}')
"
```

## References

- **Optuna Documentation:** https://optuna.org/
- **Hyperparameter Tuning Plan:** `hyperparameter_tuning_plan.txt`
- **EvoX Documentation:** https://evox.readthedocs.io/
- **CMA-ES Documentation:** https://cma-es.github.io/
- **SciPy DE Documentation:** https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.differential_evolution.html

## File Structure

```
CVAE-Opt/
├── hyperparameter_tuning_plan.txt          # Comprehensive tuning plan
├── TUNING_README.md                        # This file
├── run_all_tuning.py                       # Python master script
├── run_all_tuning.sh                       # Bash master script
├── optuna_de_tune.py                       # Standard DE tuning
├── optuna_ode_tune.py                      # EvoX ODE tuning
├── optuna_code_tune.py                     # EvoX CoDE tuning
├── optuna_shade_tune.py                    # EvoX SHADE tuning
├── optuna_sade_tune.py                     # EvoX SaDE tuning
├── optuna_cmaes_tune.py                    # Standard CMA-ES tuning
├── optuna_ipop_tune.py                     # IPOP-CMA-ES tuning
├── optuna_bipop_tune.py                    # BIPOP-CMA-ES tuning
├── optuna_scipy_de_tune.py                 # SciPy DE tuning
├── de.py                                   # Standard DE implementation
├── evox_ode.py                             # EvoX ODE implementation
├── evox_code.py                            # EvoX CoDE implementation
├── evox_shade.py                           # EvoX SHADE implementation
├── evox_sade.py                            # EvoX SaDE implementation
├── cmaes.py                                # Standard CMA-ES implementation
├── ipop_cmaes.py                           # IPOP-CMA-ES implementation
├── bipop_cmaes.py                          # BIPOP-CMA-ES implementation
└── scipy_de.py                             # SciPy DE implementation
```

## License

See project LICENSE file.
