import os
import pandas as pd
import glob

def analyze_results(results_dir="experimental_results", output_csv="summary_results.csv"):
    all_files = glob.glob(os.path.join(results_dir, "*.csv"))
    
    data = []
    for file in all_files:
        try:
            df = pd.read_csv(file)
            if 'Dataset' in df.columns and 'Classifier' in df.columns and 'Accuracy' in df.columns:
                for _, row in df.iterrows():
                    data.append({
                        'Dataset': row['Dataset'],
                        'Classifier': row['Classifier'],
                        'Accuracy': row['Accuracy']
                    })
            elif 'Same Class' in df.columns:
                filename = os.path.basename(file)
                parts = filename.split('_')
                dataset = parts[0]
                classifier = parts[1]
                
                # Calculate accuracy as the proportion of True values
                if df['Same Class'].dtype == object:
                    accuracy = (df['Same Class'].astype(str).str.lower() == 'true').mean()
                else:
                    accuracy = df['Same Class'].mean()
                    
                data.append({
                    'Dataset': dataset,
                    'Classifier': classifier,
                    'Accuracy': accuracy
                })
        except Exception as e:
            print(f"Error reading {file}: {e}")
            
    if not data:
        print("No valid data found.")
        return

    df_all = pd.DataFrame(data)
    
    # Map classifier names to match user requested order names
    name_mapping = {
        '1-NN': '1NN',
        'Linear SVM': 'LinearSVM',
        'Naive Bayes': 'Naive Bayes'
    }
    df_all['Classifier'] = df_all['Classifier'].replace(name_mapping)
    
    # Pivot the table so Datasets are rows and Classifiers are columns
    pivot_df = df_all.pivot_table(index='Dataset', columns='Classifier', values='Accuracy')
    
    # Add a Mean row
    pivot_df.loc['Mean'] = pivot_df.mean()
    
    # Ensure correct column order
    ordered_columns = ['DTW', 'CDTW', 'LCSS', 'SDTW', 'ADTW', '1NN', 'LinearSVM', 'Naive Bayes']
    
    # Add any missing columns with NaN to prevent KeyError
    for col in ordered_columns:
        if col not in pivot_df.columns:
            pivot_df[col] = pd.NA
            
    # Reorder columns
    pivot_df = pivot_df[ordered_columns]
    
    # Convert to percentages
    pivot_df = pivot_df * 100
    
    # Save to CSV with 1 decimal space
    pivot_df.to_csv(output_csv, float_format='%.1f')
    print(f"Results saved to {output_csv}")
    print("\nPreview of the table:")
    print(pivot_df.head(10))
    print("...")
    print(pivot_df.tail(2))

if __name__ == "__main__":
    analyze_results()
