# Murder Model

A machine learning classification model to predict murder likelihood based on location and demographics.

## Project Structure

```
archive/
- data/
  - raw/         # Original dataset files
  - processed/   # Cleaned and preprocessed data
  - output/      # Model outputs and results
- docs/          # Documentation files
- notebooks/     # Jupyter notebooks for analysis
- src/
  - preprocessing/  # Data preparation scripts
  - models/         # Model implementation
  - evaluation/     # Testing and validation
  - utils/          # Helper functions
- tests/         # Test suites
```

## Setup

1. Create and activate the virtual environment (`.venv`):

```bash
# Windows
python -m venv .venv
.\.venv\Scripts\activate

# Unix/MacOS
python -m venv .venv
source .venv/bin/activate
```

2. Install dependencies:

```bash
pip install -r requirements.txt
```

> `.venv` is the only supported environment; any previous `venv/` folder has been removed.

## Usage

```bash
# Activate environment
.\.venv\Scripts\activate        # Windows
source .venv/bin/activate      # Unix/MacOS

# Train models
python src/models/train.py --model logistic --nrows 10000
python src/models/train.py --model random_forest --nrows 10000

# Compare models
python src/models/compare.py --nrows 10000

# Run tests
python -m unittest discover -s tests -p "test_*.py" -v
```

## Documentation

- Project proposal: `docs/projectproposal.md`
- Development phases: `docs/development-phases.md`
- Project roadmap: `docs/project-roadmap.md`
- Version control guidelines: `docs/version-control.md`

## Contributing

1. Create a new branch for your feature (see `docs/version-control.md` for naming/commit conventions)
2. Make your changes
3. Submit a pull request

## License

[To be determined]
