import os
import glob
import pandas as pd
import numpy as np

def analyze_time(results_dir="experimental_results", output_csv="time_results.csv"):
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
                    if 'Computing Time (s)' in df.columns:
                        # Convert seconds to milliseconds
                        times = df['Computing Time (s)'].dropna() * 1000
                        if len(times) > 0:
                            mean_time = times.mean()
                            std_time = times.std(ddof=1) if len(times) > 1 else 0.0
                            data.append({
                                'Dataset': dataset,
                                'Classifier': algo,
                                'Mean': mean_time,
                                'Std': std_time,
                                'AllTimes': times.tolist()
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
            'TimeFormat': f"{d['Mean']:.1f} (+-{d['Std']:.1f})"
        })
        
    df_formatted = pd.DataFrame(formatted_data)
    
    # Pivot the table so Datasets are rows and Classifiers are columns
    pivot_df = df_formatted.pivot_table(index='Dataset', columns='Classifier', values='TimeFormat', aggfunc='first')
    
    # Add a Mean row for overall statistics
    overall_stats = {}
    for algo in algorithms:
        algo_times = []
        for d in data:
            if d['Classifier'] == algo:
                algo_times.extend(d['AllTimes'])
        
        if algo_times:
            overall_mean = np.mean(algo_times)
            overall_std = np.std(algo_times, ddof=1)
            overall_stats[algo] = f"{overall_mean:.1f} (+-{overall_std:.1f})"
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
    analyze_time()
