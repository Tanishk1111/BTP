import csv
import os
from typing import Tuple

REQUIRED_COLUMNS = ["barcode","id","x_pixel","y_pixel"]

def normalize_prediction_csv(original_path: str, target_wsi_id: str, output_dir: str = "uploads") -> Tuple[str,bool,bool]:
    """Normalize a user-provided coordinate CSV into required format.

    Acceptable input columns (case-insensitive variants allowed):
      - x_pixel / x / X
      - y_pixel / y / Y
      - barcode (optional)
      - id / wsi_id (optional)

    If barcode missing, generate sequential SPOT0001,...
    If id missing, fill with target_wsi_id.

    Returns: (normalized_csv_path, generated_barcodes, added_id)
    """
    if not os.path.exists(original_path):
        raise FileNotFoundError(f"Prediction CSV not found: {original_path}")

    # Read all rows
    with open(original_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    if not rows:
        raise ValueError("Prediction CSV is empty")

    # Normalize headers mapping
    def find_col(candidates):
        for c in candidates:
            for key in reader.fieldnames or []:
                if key.lower() == c.lower():
                    return key
        return None

    x_col = find_col(["x_pixel","x","X"])
    y_col = find_col(["y_pixel","y","Y"])
    barcode_col = find_col(["barcode"])
    id_col = find_col(["id","wsi_id"])  # optional

    if not x_col or not y_col:
        raise ValueError("CSV must contain x/y columns (x_pixel or x / y_pixel or y)")

    generated_barcodes = False
    added_id = False

    out_rows = []
    for i,row in enumerate(rows, start=1):
        try:
            x_val = float(row[x_col])
            y_val = float(row[y_col])
        except (TypeError,ValueError):
            continue
        barcode = row[barcode_col].strip() if barcode_col and row.get(barcode_col) else f"SPOT{i:04d}"
        if not barcode_col and barcode.startswith("SPOT"):
            generated_barcodes = True
        wsi_id = row[id_col].strip() if id_col and row.get(id_col) else target_wsi_id
        if not id_col:
            added_id = True
        out_rows.append({
            'barcode': barcode,
            'id': wsi_id,
            'x_pixel': x_val,
            'y_pixel': y_val
        })

    if not out_rows:
        raise ValueError("No valid coordinate rows after parsing")

    base_name = os.path.splitext(os.path.basename(original_path))[0]
    norm_name = f"normalized_{base_name}.csv"
    norm_path = os.path.join(output_dir, norm_name)
    os.makedirs(output_dir, exist_ok=True)
    with open(norm_path, 'w', newline='', encoding='utf-8') as out_f:
        writer = csv.DictWriter(out_f, fieldnames=REQUIRED_COLUMNS)
        writer.writeheader()
        writer.writerows(out_rows)

    return norm_path, generated_barcodes, added_id
