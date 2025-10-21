# IDS-ML: Machine Learning Intrusion Detection System

A machine learning-based network intrusion detection system using XGBoost for detecting various types of network attacks.

## Project Structure

```
capstone-simple/
├── src/
│   └── ids_ml/              # Main package
│       ├── cli/             # Command-line interfaces
│       │   ├── ids.py       # Main IDS CLI
│       │   └── sample_generator.py  # Test sample generator
│       ├── data/            # Data processing modules
│       │   ├── preprocess.py
│       │   └── split.py
│       ├── models/          # Model training & inference
│       │   ├── train.py
│       │   ├── inference.py
│       │   └── utils.py
│       └── visualization/   # Plotting utilities
│
├── data/                    # Data files (gitignored)
│   ├── raw/                 # Raw dataset files
│   ├── samples/             # Test samples
│   └── processed/           # Preprocessed data
│
├── models/                  # Trained models (gitignored)
│   ├── model.json
│   ├── best_hyperparameters.json
│   └── label_mapping.json
│
├── outputs/                 # Generated outputs (gitignored)
│   ├── plots/               # Visualizations
│   └── results/             # Analysis results
│
├── scripts/                 # Utility scripts
├── tests/                   # Unit tests
├── pyproject.toml           # Project configuration
└── README.md
```

## Installation

Install the package in development mode:

```bash
uv pip install -e .
```

This will install the package and make the CLI commands available.

## CLI Commands

### 1. IDS - Intrusion Detection System

Analyze network traffic flows for harmful activity:

```bash
# Analyze a CSV file with traffic flows
uv run ids --file data/samples/test_small.csv

# Analyze with detailed output for each flow
uv run ids --file data/samples/test_small.csv --verbose

# Analyze and save results to a file
uv run ids --file data/samples/test_small.csv --output results.csv

# List all detectable attack types
uv run ids --list-classes

# Use custom model directory
uv run ids --file traffic.csv --model-dir custom_models/
```

### 2. Sample Generator

Create test samples with different traffic distributions:

```bash
# List available scenarios
uv run ids-sample-gen --list-scenarios

# Create a predefined scenario
uv run ids-sample-gen --scenario mostly_benign --output data/samples/benign_test.csv

# Show dataset distribution
uv run ids-sample-gen --show-distribution

# Create custom sample from JSON config
uv run ids-sample-gen --config my_config.json --output data/samples/custom.csv
```

**Available scenarios:**
- `mostly_benign` - 95% benign traffic with scattered attacks
- `under_attack` - Heavy attack scenario (50% attacks)
- `mixed_attacks` - 90% benign with diverse attack types
- `all_attack_types` - Small sample with ALL attack types
- `web_attacks` - Focus on web-based attacks
- `dos_focused` - Various DoS attack types
- `small_test` - Quick 100-flow sample

**Custom config example** (`my_config.json`):
```json
{
  "BENIGN": 1000,
  "DDoS": 100,
  "PortScan": 50,
  "Bot": 50
}
```

Or use percentages:
```json
{
  "BENIGN": 0.95,
  "DDoS": 0.03,
  "PortScan": 0.02
}
```

### 3. Model Training

Train a new model with hyperparameter optimization:

```bash
# Quick training (20% data, 50 trials) - good for testing
uv run ids-train --quick

# Full training (all data, 100 trials) - production model
uv run ids-train --full

# Custom training configuration
uv run ids-train --trials 75 --sample-fraction 0.5

# Training without SMOTE
uv run ids-train --quick --no-smote

# Custom paths
uv run ids-train --dataset data/raw/custom.csv --output-dir trained_models/
```

### 4. Model Evaluation

Evaluate a trained model on the test dataset:

```bash
# Evaluate with default paths
uv run ids-evaluate

# Evaluate with custom paths
uv run ids-evaluate --dataset data/raw/custom.csv --model-dir trained_models/

# Save metrics to custom location
uv run ids-evaluate --output results/metrics.json
```

## Usage Workflow

### 1. Prepare Data

Place your dataset in `data/raw/dataset.csv`

### 2. Train Model (Optional)

```bash
uv run ids-train
```

Model artifacts will be saved to `models/`

### 3. Create Test Samples

```bash
uv run ids-sample-gen --scenario small_test --output data/samples/test.csv
```

### 4. Run IDS Analysis

```bash
uv run ids --file data/samples/test.csv --verbose
```

### 5. Review Results

Check the console output or use `--output` to save results to CSV.

## Development

### Package Structure

The project follows modern Python packaging standards:
- **src-layout**: Clean separation of source code
- **Console scripts**: CLI commands defined in `pyproject.toml`
- **Modular design**: Clear separation of concerns

### Import Examples

```python
# In your code, you can import modules like this:
from ids_ml.models.inference import load_model, predict
from ids_ml.data.split import split_dataset
from ids_ml.models.utils import get_base_params
```

### Running Tests

```bash
# Install test dependencies
uv pip install pytest

# Run tests (when created)
pytest tests/
```

## Detectable Attack Types

The system can detect the following attack types:
- **BENIGN** - Normal traffic
- **Bot** - Botnet traffic
- **DDoS** - Distributed Denial of Service
- **DoS GoldenEye** - DoS attack variant
- **DoS Hulk** - DoS attack variant
- **DoS Slowhttptest** - Slow HTTP DoS attack
- **DoS slowloris** - Slowloris DoS attack
- **FTP-Patator** - FTP brute force
- **Heartbleed** - Heartbleed vulnerability exploit
- **Infiltration** - Network infiltration
- **PortScan** - Port scanning activity
- **SSH-Patator** - SSH brute force
- **Web Attack - Brute Force** - Web brute force attack
- **Web Attack - SQL Injection** - SQL injection attack
- **Web Attack - XSS** - Cross-site scripting attack

## Model Information

- **Algorithm**: XGBoost (Gradient Boosting)
- **Features**: 78 network flow features
- **Classes**: 15 (1 benign + 14 attack types)
- **Optimization**: Optuna hyperparameter tuning
- **Class Balance**: SMOTE oversampling

## References

Iman Sharafaldin, Arash Habibi Lashkari, and Ali A. Ghorbani, “Toward Generating a New Intrusion Detection Dataset and Intrusion Traffic Characterization”, 4th International Conference on Information Systems Security and Privacy (ICISSP), Portugal, January 2018.

## License

[Add your license here]

## Contributing

[Add contribution guidelines here]
