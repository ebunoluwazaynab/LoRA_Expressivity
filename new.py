import pandas as pd
import os

def clean_sr_layerwise():
    """
    Removes duplicated epochs by dropping rows up to a user-specified index
    and replaces the original sr_layerwise.csv with the cleaned version.
    """
    try:
        # Prompt the user for the input file name (must be sr_layerwise.csv ideally)
        input_file = input("Enter the path to sr_layerwise.csv: ").strip()
        
        # Check if the file exists
        if not os.path.exists(input_file):
            print(f"Error: The file '{input_file}' was not found.")
            return

        row_number_str = input("Enter the number of the last duplicate row to remove: ")

        try:
            row_to_start_from = int(row_number_str)
            if row_to_start_from <= 0:
                print("Error: The row number must be a positive integer.")
                return
            start_index = row_to_start_from  # pandas is 0-indexed
        except ValueError:
            print("Error: The row number must be a valid integer.")
            return

        # Read the CSV into a DataFrame
        df = pd.read_csv(input_file)

        # Bounds check
        if start_index >= len(df):
            print(
                f"Error: The row number you provided ({row_to_start_from}) "
                f"is >= total number of rows in the file ({len(df)})."
            )
            return

        # Slice from start_index onwards
        new_df = df.iloc[start_index:]

        # Delete the original file
        os.remove(input_file)

        # Save back with the same name
        new_df.to_csv(input_file, index=False)

        print(f"Successfully cleaned and replaced '{input_file}'.")

    except Exception as e:
        print(f"Unexpected error: {e}")


if __name__ == "__main__":
    clean_sr_layerwise()
