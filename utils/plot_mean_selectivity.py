import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import ast

# ======================================================================
# CONFIGURATION
# ======================================================================
# Path to the folder containing your benchmark CSV files
# Use '.' if the script is in the same folder as the CSVs
DATA_FOLDER = r'C:\Users\eddyt\Desktop\GitHub\Regenerative_Peripheral_Nerve_Interface_Simulation\data\benchmark_adaptive_2'

# Output filenames
SUMMARY_CSV = "benchmark_summary_stats.csv"
SUMMARY_PLOT = "benchmark_summary_plot.png"

def parse_target_point(filename):
    """
    Extracts target coordinates (x, y, z) from the filename if they are present.
    Filename format expected: "..._x0.0_y0.0_z0.0.csv"
    """
    try:
        # Remove extension
        base = os.path.splitext(filename)[0]
        parts = base.split('_')
        
        x, y, z = None, None, None
        for part in parts:
            if part.startswith('x'): x = float(part[1:])
            elif part.startswith('y'): y = float(part[1:])
            elif part.startswith('z'): z = float(part[1:])
            
        if x is not None and y is not None and z is not None:
            # Format to match the string representation in the original script's print statements
            return f"({x:.4f}, {y:.4f}, {z:.4f})" # e.g. (0.0020, 0.0035, 0.0100)
        return "Unknown Target"
    except:
        return "Unknown Target"

def load_data(folder):
    all_data = []
    
    if not os.path.exists(folder):
        print(f"Error: Folder '{folder}' not found.")
        return pd.DataFrame()

    files = [f for f in os.listdir(folder) if f.endswith('.csv')]
    print(f"Found {len(files)} CSV files in {folder}...")

    for f in files:
        file_path = os.path.join(folder, f)
        try:
            # Use low_memory=False or specify dtypes to prevent mixed-type warnings
            df = pd.read_csv(file_path, low_memory=False)
            
            # Check required columns
            if 'best_so_far' not in df.columns:
                continue
            
            # --- CRITICAL FIX FOR TYPE ERRORS ---
            # Force 'best_so_far' to numeric. Coerce errors (strings/headers) to NaN.
            df['best_so_far'] = pd.to_numeric(df['best_so_far'], errors='coerce')
            
            # Robustness: Get the absolute maximum 'best_so_far' from the entire file
            # Since we coerced errors, NaNs are ignored by max()
            best_score = df['best_so_far'].max()
            
            # If best_score is NaN (e.g. file contained only headers or garbage), skip
            if pd.isna(best_score):
                continue
            
            # Get optimizer name (assuming it's constant in the file)
            optimizer = df['optimizer'].iloc[0] if 'optimizer' in df.columns else "Unknown"
            
            # Extract Target
            # Priority 1: From columns if they exist (more accurate)
            if {'target_x', 'target_y', 'target_z'}.issubset(df.columns):
                # Ensure target columns are numeric too, just in case
                tx = pd.to_numeric(df['target_x'], errors='coerce').iloc[0]
                ty = pd.to_numeric(df['target_y'], errors='coerce').iloc[0]
                tz = pd.to_numeric(df['target_z'], errors='coerce').iloc[0]
                if pd.isna(tx) or pd.isna(ty) or pd.isna(tz):
                     target_str = parse_target_point(f)
                else:
                    target_str = f"({tx}, {ty}, {tz})" # Matches original tuple format roughly
            else:
                # Priority 2: From filename
                target_str = parse_target_point(f)

            all_data.append({
                'optimizer': optimizer,
                'target': target_str,
                'selectivity': best_score,
                'file': f
            })
            
        except Exception as e:
            print(f"Skipping {f}: {e}")

    df_res = pd.DataFrame(all_data)
    
    # Ensure the final 'selectivity' column is strictly float before we try to calculate mean
    if not df_res.empty and 'selectivity' in df_res.columns:
        df_res['selectivity'] = pd.to_numeric(df_res['selectivity'], errors='coerce')
        
    return df_res

def plot_results(df_stats, output_path):
    if df_stats.empty:
        print("No data to plot.")
        return

    targets = df_stats['target'].unique()
    optimizers = df_stats['optimizer'].unique()
    
    # Pivot for easier plotting: Index=Target, Columns=Optimizer, Values=Mean
    df_mean = df_stats.pivot(index='target', columns='optimizer', values='mean')
    df_std = df_stats.pivot(index='target', columns='optimizer', values='std')

    # Plot settings
    n_targets = len(targets)
    n_opts = len(optimizers)
    
    # Dynamic figure size
    fig, ax = plt.subplots(figsize=(max(12, n_targets * 2.5), 7))
    
    # Bar width setup
    width = 0.8 / n_opts
    x = np.arange(n_targets)
    
    # Generate bars
    for i, opt in enumerate(optimizers):
        means = df_mean[opt].values
        stds = df_std[opt].fillna(0).values
        
        # Offset bars
        x_pos = x + (i - n_opts/2 + 0.5) * width
        
        ax.bar(x_pos, means, width, label=opt, yerr=stds, capsize=3, alpha=0.85, edgecolor='black', linewidth=0.5)

    ax.set_xlabel('Target Point (x, y, z)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Mean Selectivity', fontsize=12, fontweight='bold')
    ax.set_title('Benchmark Results: Mean Selectivity per Model (Final Value)', fontsize=14, pad=20)
    
    ax.set_xticks(x)
    ax.set_xticklabels(targets, rotation=45, ha='right', fontsize=10)
    
    # Move legend outside
    ax.legend(title="Optimizer", bbox_to_anchor=(1.02, 1), loc='upper left', borderaxespad=0.)
    
    ax.grid(axis='y', linestyle='--', alpha=0.3)
    ax.set_axisbelow(True) # Put grid behind bars
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=200, bbox_inches='tight')
    print(f"Plot saved to {output_path}")
    plt.close()

if __name__ == "__main__":
    print("--- Analyzing Benchmark Results (Identical Calculation to Original) ---")
    
    # 1. Load Data
    df = load_data(DATA_FOLDER)
    
    if not df.empty:
        # 2. Group and Calculate Stats
        # This calculates the Mean Selectivity across all repeats for each (Optimizer, Target) pair.
        print(f"\nComputing statistics for {len(df)} runs...")
        
        # Ensure no NaNs in selectivity before grouping (though groupby usually handles them, agg might not)
        df = df.dropna(subset=['selectivity'])
        
        stats = df.groupby(['optimizer', 'target'])['selectivity'].agg(['mean', 'std', 'count', 'max', 'min']).reset_index()
        
        # Sort by target string to keep plot consistent
        stats = stats.sort_values(by='target')
        
        # 3. Save CSV Summary
        out_csv = os.path.join(DATA_FOLDER, SUMMARY_CSV)
        stats.to_csv(out_csv, index=False)
        print(f"Summary stats saved to {out_csv}")
        
        # 4. Generate Plot
        out_plot = os.path.join(DATA_FOLDER, SUMMARY_PLOT)
        plot_results(stats, out_plot)
        
        print("\n" + "="*50)
        print("WINNER PER TARGET (Highest Mean Selectivity)")
        print("="*50)
        for t in stats['target'].unique():
            subset = stats[stats['target'] == t]
            best_row = subset.loc[subset['mean'].idxmax()]
            print(f"Target {t}")
            print(f"  WINNER: {best_row['optimizer']}")
            print(f"  Mean:   {best_row['mean']:.4f}")
            print(f"  Std:    {best_row['std']:.4f}")
            print("-" * 30)
            
    else:
        print(f"No valid data found in {DATA_FOLDER}. Check your folder path.")