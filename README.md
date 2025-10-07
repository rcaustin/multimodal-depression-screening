# Multimodal Depression Screening

This repository contains code for a multimodal facial emotion recognition model for early depression screening, using text, audio, and visual features from the DAIC and EDAIC datasets.

---

## Getting Started: Python Environment Setup

Follow these steps to set up your development environment.

### 1. Clone the repository

```bash
git clone <repository-url>
cd <repository-directory>
```

### 2. Add project datasets
```bash
mkdir -p data/raw data/processed/sessions
```

Add compressed, archived data to data/raw/
Add uncompressed, usable data to data/processed/sessions

The current script expects data/processed/sessions to contain a directory for each session named with the corresponding session ID (e.g., 300, 301, 302, etc.) Text features must be present in the session directory (e.g., 300); video and audio features must be present in a features/ sub-directory (e.g., 300/features).

### 3. Create a virtual environment

```bash
mkdir .venv
virtualenv .venv
```

### 4. Activate the virtual environment

Linux/macOS
```bash
source .venv/bin/activate
```

Windows
```cmd
.venv\Scripts\activate
```

### 5. Install project dependencies

```bash
pip install -r requirements.txt
```

---

## Dataset Notice

This project depends on the **DAIC and EDAIC datasets**, which are **not included**.  

- You must obtain the datasets separately under their respective End-User License Agreements:
  - [DAIC Dataset EULA](https://dcapswoz.ict.usc.edu/daic-woz-database-download/)
  - [EDAIC Dataset EULA](https://dcapswoz.ict.usc.edu/extended-daic-database-download/)
- Redistribution of these datasets or any portion of them is **strictly prohibited**.
- Use of this project for **commercial purposes is not allowed**.

Please cite the datasets in any publication or presentation using this project.
