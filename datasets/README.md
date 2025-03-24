# 📂 datasets/

This directory stores input data used for training, testing, and evaluating the XSecIoT system.

## 📊 Structure

Each dataset should be placed in its own subdirectory. For example:

```
datasets/
├── DFAIR/
│   └── combined_data_with_okpVacc_modified.csv
├── CICIDS/
│   └── flow_data.csv
├── UNSW/
    └── nf_unsw.csv
```

## ✅ Guidelines for Adding New Datasets

1. Create a new folder under `datasets/` with a descriptive name (e.g., `CICIDS`, `UNSW`, etc.)
2. Place the relevant `.csv` file(s) inside that folder
3. Ensure all datasets conform to the **flow-based CSV format** expected by the classifiers

## 📃 Current Datasets

* **DFAIR/**: Dataset collected and maintained by the DFAIR Lab at Augusta University, including IoT device traffic under attack scenarios.

---

🔐 All datasets are intended for **research use only**. Do not use in production or share outside approved channels.
