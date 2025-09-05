# Motivation
This repository contains scripts for generating **image chunks** related to **price information** from receipts.

## Main Scripts
- `yolo_totalLabelSimple_data_collection_chunk_only_number_u50.py`
- `yolo_totalLabelSimple_data_collection_chunk_only_number.py`

These scripts are designed to detect and crop number-related regions (such as totals and prices) from receipt images.

## Why Do I Need These Chunks?
The generated chunks can be used to:
- Train and build **CNN** or **CRNN** models for digit/price recognition
- Prepare datasets with corresponding **annotation labels** for supervised training


### Credits for image datasets
- This project uses the ExpressExpense Sample Receipt Dataset (SRD),
available at ExpressExpense.com under the MIT License.


- This project uses the "mahb receipt" dataset/model created by MAHB TEST,
available on Roboflow Universe under CC BY 4.0.
