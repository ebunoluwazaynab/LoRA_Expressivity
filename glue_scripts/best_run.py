import os
import pandas as pd
import shutil
import glob

def find_best_run(base_dir, final_save_dir):
    """
    Scans a directory of experiment runs, finds the one with the lowest
    evaluation loss, and copies all its files to a new directory.
    """
    best_loss = float('inf')
    best_run_path = None
    
    # Use glob to find all eval_per_epoch.csv files across all subdirectories
    eval_files = glob.glob(os.path.join(base_dir, '**', 'eval_per_epoch.csv'), recursive=True)
    
    if not eval_files:
        print(f"No 'eval_per_epoch.csv' files found in {base_dir}. Exiting.")
        return

    print(f"Found {len(eval_files)} experiment log files. Analyzing...")
    
    # Iterate through all evaluation files to find the best run
    for file_path in eval_files:
        try:
            df = pd.read_csv(file_path)
            # Ensure 'loss' column exists and is not empty
            if 'loss' in df.columns and not df.empty:
                min_loss_in_file = df['loss'].min()
                
                # Check if this run is the new best
                if min_loss_in_file < best_loss:
                    best_loss = min_loss_in_file
                    # Store the path to the best run's directory
                    best_run_path = os.path.dirname(file_path)
                    
        except Exception as e:
            print(f"Could not process {file_path}. Skipping. Error: {e}")

    if best_run_path is None:
        print("No valid runs found with a recorded loss. Exiting.")
        return

    # Create the final directory to store the best run's files
    # The new directory name is based on the best run's name
    best_run_name = os.path.basename(best_run_path)
    final_dest_dir = os.path.join(final_save_dir, best_run_name)

    print("\n" + "="*50)
    print(f"Best run found: '{best_run_name}'")
    print(f"Lowest evaluation loss: {best_loss:.6f}")
    print("="*50 + "\n")
    
    # Copy the entire directory of the best run
    try:
        if os.path.exists(final_dest_dir):
            print(f"Warning: Directory '{final_dest_dir}' already exists. Overwriting content.")
        
        # This is a much cleaner way to copy all files and subdirectories
        shutil.copytree(best_run_path, final_dest_dir, dirs_exist_ok=True)
        print(f"Successfully copied all files to '{final_dest_dir}'")
        
    except Exception as e:
        print(f"An error occurred while copying files: {e}")

if __name__ == "__main__":
    # Define your base experiments folder
    BASE_EXPERIMENTS_DIR = "experiments"
    # Define the directory where the best run will be saved
    FINAL_SAVE_DIR = "experiments_best_run_analysis"

    find_best_run(BASE_EXPERIMENTS_DIR, FINAL_SAVE_DIR)