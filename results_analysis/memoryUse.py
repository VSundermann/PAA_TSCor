import os
import glob
import pandas as pd
import numpy as np

def analyze_memory(results_dir="experimental_results", output_csv="memory_results.csv"):
    algorithms = ['DTW', 'CDTW', 'LCSS', 'SDTW', 'ADTW']
    all_files = glob.glob(os.path.join(results_dir, "*.csv"))
    
    data = []
    
    for file in all_files:
        filename = os.path.basename(file)
        parts = filename.split('_')
        if len(parts) >= 2:
            dataset = parts[0]
            algo = parts[1]
            if algo in algorithms:
                try:
                    df = pd.read_csv(file)
                    if 'Space Usage (MB)' in df.columns:
                        memory = df['Space Usage (MB)'].dropna()
                        if len(memory) > 0:
                            mean_mem = memory.mean()
                            std_mem = memory.std(ddof=1) if len(memory) > 1 else 0.0
                            data.append({
                                'Dataset': dataset,
                                'Classifier': algo,
                                'Mean': mean_mem,
                                'Std': std_mem,
                                'AllMemory': memory.tolist()
                            })
                except Exception as e:
                    print(f"Error reading {file}: {e}")

    if not data:
        print("No valid data found.")
        return

    # Create dataset-level strings
    formatted_data = []
    for d in data:
        formatted_data.append({
            'Dataset': d['Dataset'],
            'Classifier': d['Classifier'],
            'MemoryFormat': f"{d['Mean']:.6f} (+-{d['Std']:.6f})"
        })
        
    df_formatted = pd.DataFrame(formatted_data)
    
    # Pivot the table so Datasets are rows and Classifiers are columns
    pivot_df = df_formatted.pivot_table(index='Dataset', columns='Classifier', values='MemoryFormat', aggfunc='first')
    
    # Add a Mean row for overall statistics
    overall_stats = {}
    for algo in algorithms:
        algo_mem = []
        for d in data:
            if d['Classifier'] == algo:
                algo_mem.extend(d['AllMemory'])
        
        if algo_mem:
            overall_mean = np.mean(algo_mem)
            overall_std = np.std(algo_mem, ddof=1)
            overall_stats[algo] = f"{overall_mean:.6f} (+-{overall_std:.6f})"
        else:
            overall_stats[algo] = pd.NA
            
    # Add Mean row to pivot_df
    pivot_df.loc['Mean'] = pd.Series(overall_stats)
    
    # Ensure correct column order
    ordered_columns = [col for col in algorithms if col in pivot_df.columns]
    pivot_df = pivot_df[ordered_columns]
    
    # Save to CSV
    pivot_df.to_csv(output_csv)
    print(f"Results saved to {output_csv}")
    print("\nPreview of the table:")
    print(pivot_df.head(10))
    print("...")
    print(pivot_df.tail(2))

if __name__ == "__main__":
    analyze_memory()
