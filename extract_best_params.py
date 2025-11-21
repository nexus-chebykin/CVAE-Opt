#!/usr/bin/env python3
"""Extract best parameters from all Optuna studies."""

import optuna
import json
from pathlib import Path

# List of all optimizer databases
db_files = [
    "optuna_bipop.db",
    "optuna_cmaes.db",
    "optuna_code.db",
    "optuna_de.db",
    "optuna_ipop.db",
    "optuna_jade.db",
    "optuna_ode.db",
    "optuna_sade.db",
    "optuna_scipy_de.db",
    "optuna_shade.db",
]

results = {}

for db_file in db_files:
    db_path = Path(db_file)
    if not db_path.exists():
        print(f"❌ {db_file} not found")
        continue

    # Extract optimizer name from filename
    optimizer_name = db_file.replace("optuna_", "").replace(".db", "").upper()

    try:
        # Load the study
        storage = f"sqlite:///{db_file}"
        study = optuna.load_study(study_name=f"{optimizer_name.lower()}_tuning", storage=storage)

        # Get best trial
        best_trial = study.best_trial

        results[optimizer_name] = {
            "best_value": best_trial.value,
            "best_params": best_trial.params,
            "trial_number": best_trial.number,
            "n_trials": len(study.trials),
        }

        print(f"\n{'='*60}")
        print(f"🔧 {optimizer_name}")
        print(f"{'='*60}")
        print(f"Best Mean Gap: {best_trial.value:.4f}%")
        print(f"Trial Number: {best_trial.number + 1}/{len(study.trials)}")
        print(f"Best Parameters:")
        for param, value in best_trial.params.items():
            print(f"  - {param}: {value}")

    except Exception as e:
        print(f"❌ Error loading {db_file}: {e}")
        continue

# Save all results to JSON
output_file = "best_parameters_all_optimizers.json"
with open(output_file, "w") as f:
    json.dump(results, f, indent=2)

print(f"\n{'='*60}")
print(f"✅ All results saved to: {output_file}")
print(f"{'='*60}")
