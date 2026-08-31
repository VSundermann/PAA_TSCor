import os
import glob
import pandas as pd
import numpy as np

def analyze_total_time(results_dir="experimental_results", output_csv="total_time_results.csv"):
    algorithms = ['DTW', 'CDTW', 'LCSS', 'SDTW', 'ADTW', '1NN', 'LinearSVM', 'Naive Bayes']
    
    # Mapping to match filename/content classifiers to the target names
    name_mapping = {
        '1-NN': '1NN',
        'Linear SVM': 'LinearSVM',
        'Linear_SVM': 'LinearSVM',
        'Naive Bayes': 'Naive Bayes',
        'Naive_Bayes': 'Naive Bayes'
    }
    
    all_files = glob.glob(os.path.join(results_dir, "*.csv"))
    
    data = []
    
    for file in all_files:
        filename = os.path.basename(file)
        parts = filename.replace('.csv', '').split('_')
        if len(parts) >= 2:
            dataset = parts[0]
            
            if 'results' in parts:
                algo_parts = parts[1:parts.index('results')]
            else:
                algo_parts = parts[1:]
                
            # Filter out time strings like '0.98s'
            algo_parts = [p for p in algo_parts if not (p.endswith('s') and p[0].isdigit())]
            algo_raw = "_".join(algo_parts)
            
            # Map name
            algo = name_mapping.get(algo_raw, algo_raw)
            algo = algo.replace('_', ' ')
            algo = name_mapping.get(algo, algo)
            
            if algo not in algorithms:
                try:
                    df = pd.read_csv(file)
                    if 'Classifier' in df.columns:
                        val = df['Classifier'].iloc[0]
                        algo = name_mapping.get(val, val)
                except:
                    pass
            
            if algo in algorithms:
                try:
                    df = pd.read_csv(file)
                    if 'Computing Time (s)' in df.columns:
                        total_time = df['Computing Time (s)'].sum()
                        data.append({
                            'Dataset': dataset,
                            'Classifier': algo,
                            'TotalTime': total_time
                        })
                except Exception as e:
                    print(f"Error reading {file}: {e}")

    if not data:
        print("No valid data found.")
        return
        
    df_all = pd.DataFrame(data)
    
    # Group by dataset and classifier just in case of duplicate files
    df_all = df_all.groupby(['Dataset', 'Classifier'])['TotalTime'].sum().reset_index()
    
    # Pivot the table so Datasets are rows and Classifiers are columns
    pivot_df = df_all.pivot_table(index='Dataset', columns='Classifier', values='TotalTime')
    
    # Add a Total row
    pivot_df.loc['Total'] = pivot_df.sum()
    
    # Ensure correct column order
    ordered_columns = [col for col in algorithms if col in pivot_df.columns]
    
    # Add any missing columns
    for col in ordered_columns:
        if col not in pivot_df.columns:
            pivot_df[col] = pd.NA
            
    pivot_df = pivot_df[ordered_columns]
    
    # Save to CSV
    pivot_df.to_csv(output_csv, float_format='%.4f')
    print(f"Results saved to {output_csv}")
    print("\nPreview of the table:")
    print(pivot_df.head(10))
    print("...")
    print(pivot_df.tail(2))

if __name__ == "__main__":
    analyze_total_time()
