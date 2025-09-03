import pandas as pd
import os

def remove_duplicate_epochs():
    """
    Removes the first set of duplicated epochs from a CSV file
    by dropping all rows up to a specified row number.
    """
    try:
        # Prompt the user for the input file name and row number
        input_file = input("Please enter the name of your input CSV file (e.g., data.csv): ")
        
        # Check if the file exists
        if not os.path.exists(input_file):
            print(f"Error: The file '{input_file}' was not found.")
            return

        row_number_str = input("Please enter the number of the last row you want to remove: ")

        try:
            # Convert the row number to an integer. We subtract 1 to make it 0-indexed for pandas.
            row_to_start_from = int(row_number_str)
            if row_to_start_from <= 0:
                print("Error: The row number must be a positive integer.")
                return
            # Pandas is 0-indexed, so we use the given row number directly
            start_index = row_to_start_from
        except ValueError:
            print("Error: The row number must be a valid integer.")
            return
        
        # Read the CSV file into a pandas DataFrame
        df = pd.read_csv(input_file)
        
        # Check if the specified row number is within the DataFrame's bounds
        if start_index >= len(df):
            print(f"Error: The row number you provided ({row_to_start_from}) is greater than or equal to the total number of rows in the file ({len(df)}).")
            return
        
        # Slice the DataFrame to keep only the rows from the specified index onwards
        # The .iloc function is used for integer-based indexing
        # Slicing from `start_index` to the end of the DataFrame
        new_df = df.iloc[start_index:]
        
        # Create a new file name for the output
        file_name, file_ext = os.path.splitext(input_file)
        output_file = f"{file_name}_clean{file_ext}"
        
        # Save the new DataFrame to the output CSV file without the index column
        new_df.to_csv(output_file, index=False)
        
        print(f"Successfully processed the file. The cleaned data has been saved to '{output_file}'.")
        
    except Exception as e:
        print(f"An unexpected error occurred: {e}")

if __name__ == "__main__":
    remove_duplicate_epochs()
