# HQ-RAPTOR Implementation Guide

Step-by-step instructions to set up and run **HQ-RAPTOR** locally.
## Prerequisites & Environment Setup

1. **Clone the base RAPTOR repository and navigate into the folder:**
   ```bash
   git clone 
   cd hq_raptor
   ```

2. **Create and activate a Python virtual environment:**
   ```bash
   python -m venv .venv

   # Windows (PowerShell):
   .venv\Scripts\Activate.ps1

   # Linux / macOS:
   source .venv/bin/activate
   ```

3. **Install required dependencies:**
   ```bash
   pip install -r requirements.txt
   pip install sentence-transformers transformers torch numpy
   ```


## Run

Execute the evaluation script directly from your activated virtual environment:

```bash
python evaluate_hq_raptor.py
```

On the first run, the script downloads the local Hugging Face models and saves the constructed tree to `demo/cinderella_local`. Subsequent runs load `demo/cinderella_local` directly from disk.
