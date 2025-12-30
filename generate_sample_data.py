#!/usr/bin/env python3
"""
Generate sample spatial transcriptomics data for testing Pratyaksha viewer.
Creates:
1. tissue_positions_list.csv - Barcode positions matching your image
2. sample_expression.h5 - 10x Genomics format expression data
"""

import pandas as pd
import numpy as np
import os

# Image dimensions
IMG_WIDTH = 1400
IMG_HEIGHT = 920

# Spot grid parameters (Visium-like hexagonal grid)
SPOT_DIAMETER = 55  # pixels
SPOT_SPACING = 100  # center-to-center distance
NUM_ROWS = int(IMG_HEIGHT / (SPOT_SPACING * 0.866))  # hex grid row spacing
NUM_COLS = int(IMG_WIDTH / SPOT_SPACING)

# Margin from edges
MARGIN_X = 80
MARGIN_Y = 60

print(f"Generating spots for {IMG_WIDTH}x{IMG_HEIGHT} image...")
print(f"Grid: ~{NUM_ROWS} rows x ~{NUM_COLS} cols")

# Generate barcode positions
barcodes = []
barcode_id = 1

for row in range(NUM_ROWS):
    # Hex grid offset for alternating rows
    x_offset = MARGIN_X + (SPOT_SPACING / 2 if row % 2 else 0)
    y_pos = MARGIN_Y + row * SPOT_SPACING * 0.866  # sqrt(3)/2 for hex
    
    if y_pos > IMG_HEIGHT - MARGIN_Y:
        break
        
    for col in range(NUM_COLS):
        x_pos = x_offset + col * SPOT_SPACING
        
        if x_pos > IMG_WIDTH - MARGIN_X:
            break
        
        # Generate barcode name (like Visium format)
        barcode = f"AAACAAGTATCTCCCA-{barcode_id}"
        
        # Simulate "in_tissue" - most spots are in tissue, some at edges are not
        in_tissue = 1
        if x_pos < MARGIN_X + 50 or x_pos > IMG_WIDTH - MARGIN_X - 50:
            in_tissue = np.random.choice([0, 1], p=[0.3, 0.7])
        if y_pos < MARGIN_Y + 50 or y_pos > IMG_HEIGHT - MARGIN_Y - 50:
            in_tissue = np.random.choice([0, 1], p=[0.4, 0.6])
        
        barcodes.append({
            'barcode': barcode,
            'in_tissue': in_tissue,
            'array_row': row,
            'array_col': col,
            'pxl_row_in_fullres': int(y_pos),
            'pxl_col_in_fullres': int(x_pos)
        })
        
        barcode_id += 1

# Create DataFrame
df = pd.DataFrame(barcodes)
print(f"Generated {len(df)} total spots, {df['in_tissue'].sum()} in tissue")

# Save CSV (no header, as per 10x format)
csv_path = "sample_tissue_positions_list.csv"
df.to_csv(csv_path, index=False, header=False)
print(f"✅ Saved: {csv_path}")

# Generate H5 expression data
print("\nGenerating expression data...")

try:
    import h5py
    import scipy.sparse as sp
    
    # Filter to only in-tissue barcodes
    in_tissue_df = df[df['in_tissue'] == 1]
    n_barcodes = len(in_tissue_df)
    
    # Sample genes (common breast cancer markers + housekeeping)
    genes = [
        # Breast cancer markers (from SpatX)
        "ESR1", "ERBB2", "GATA3", "KRT5", "CD3E", "FASN", "ANKRD30A",
        "TOP2A", "MKI67", "BRCA1", "BRCA2", "TP53", "EGFR", "PGR",
        # Immune markers
        "CD4", "CD8A", "FOXP3", "PDCD1", "CD274", "CTLA4",
        # Housekeeping
        "ACTB", "GAPDH", "B2M", "RPL13A", "HPRT1",
        # Additional markers
        "VIM", "CDH1", "CDH2", "EPCAM", "KRT19", "KRT8",
        "MMP2", "MMP9", "VEGFA", "HIF1A", "COL1A1", "FN1",
        "TGFB1", "IL6", "TNF", "CXCL8", "CCL2", "CXCR4",
        "WNT1", "NOTCH1", "SHH", "BMI1", "NANOG", "SOX2"
    ]
    n_genes = len(genes)
    
    print(f"Creating matrix: {n_barcodes} barcodes x {n_genes} genes")
    
    # Generate sparse expression matrix with realistic distribution
    # Most values are 0, some low counts, few high counts
    density = 0.3  # 30% of gene-barcode pairs have expression
    
    # Create random sparse matrix
    data = []
    row_indices = []
    col_indices = []
    
    for i in range(n_barcodes):
        # Each spot expresses ~30% of genes
        n_expressed = int(n_genes * density * np.random.uniform(0.5, 1.5))
        expressed_genes = np.random.choice(n_genes, min(n_expressed, n_genes), replace=False)
        
        for g in expressed_genes:
            # Expression values follow negative binomial-like distribution
            count = int(np.random.exponential(5) + 1)
            if np.random.random() < 0.1:  # 10% chance of higher expression
                count = int(np.random.exponential(50) + 10)
            
            data.append(count)
            row_indices.append(g)  # genes are rows in 10x format
            col_indices.append(i)  # barcodes are columns
    
    # Create sparse matrix (genes x barcodes)
    matrix = sp.csc_matrix(
        (data, (row_indices, col_indices)), 
        shape=(n_genes, n_barcodes),
        dtype=np.int32
    )
    
    print(f"Matrix density: {matrix.nnz / (n_genes * n_barcodes):.2%}")
    
    # Save as 10x Genomics H5 format
    h5_path = "sample_expression.h5"
    
    with h5py.File(h5_path, 'w') as f:
        # Create matrix group
        grp = f.create_group('matrix')
        
        # Store sparse matrix components
        grp.create_dataset('data', data=matrix.data)
        grp.create_dataset('indices', data=matrix.indices)
        grp.create_dataset('indptr', data=matrix.indptr)
        grp.create_dataset('shape', data=matrix.shape)
        
        # Store barcodes
        barcodes_encoded = [b.encode() for b in in_tissue_df['barcode'].values]
        grp.create_dataset('barcodes', data=barcodes_encoded)
        
        # Store features (genes)
        features = grp.create_group('features')
        gene_ids = [f"ENSG{i:011d}".encode() for i in range(n_genes)]
        gene_names = [g.encode() for g in genes]
        feature_types = [b'Gene Expression'] * n_genes
        
        features.create_dataset('id', data=gene_ids)
        features.create_dataset('name', data=gene_names)
        features.create_dataset('feature_type', data=feature_types)
    
    print(f"✅ Saved: {h5_path}")
    
    # Print some stats
    print(f"\n📊 Expression Statistics:")
    print(f"   Total UMI counts: {matrix.sum():,}")
    print(f"   Mean counts/spot: {matrix.sum() / n_barcodes:.1f}")
    print(f"   Genes detected: {n_genes}")
    
except ImportError as e:
    print(f"\n⚠️ Could not create H5 file: {e}")
    print("   Install h5py and scipy: pip install h5py scipy")
    
    # Create a simpler CSV version instead
    print("\n   Creating CSV expression file as fallback...")
    in_tissue_df = df[df['in_tissue'] == 1]
    
    genes = ["ESR1", "ERBB2", "GATA3", "KRT5", "CD3E", "FASN", "ACTB", "GAPDH"]
    expr_data = {'barcode': in_tissue_df['barcode'].values}
    
    for gene in genes:
        # Random expression values
        expr_data[gene] = np.random.exponential(10, len(in_tissue_df)).astype(int)
    
    expr_df = pd.DataFrame(expr_data)
    expr_df.to_csv("sample_expression.csv", index=False)
    print(f"✅ Saved: sample_expression.csv (CSV fallback)")

print("\n" + "="*50)
print("Sample data generated! Files created:")
print(f"  1. {csv_path}")
print(f"  2. sample_expression.h5 (or .csv)")
print("\nUse these files in the Pratyaksha upload page.")
print("="*50)





