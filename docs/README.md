# DDoS Detection & Mitigation (ML)

This repository implements a machine-learning-powered DDoS detection and mitigation prototype.

Quick start:

1. Create a Python 3.10+ environment and install dependencies:

```bash
python -m pip install -r requirements.txt
```

2. Prepare data and run the pipeline:

```bash
python preprocessing.py
python model_training.py
python threshold_selection.py
python evaluation_reproducibility.py
```

3. Run the API:

```bash
python -m uvicorn api:app --host 127.0.0.1 --port 8000
```

4. Run mitigation simulation:

```bash
python mitigation_engine.py --profile demo
```

Results and artifacts are under `results/`, models under `models/`.
