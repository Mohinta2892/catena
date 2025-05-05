import pandas as pd
import os


def convert_synapse_coords_to_nm(input_csv_path, output_csv_path=None, scale_factor=8):
    """
    Convert synapse coordinates from voxel space to nanometers by multiplying by a scale factor.
    
    Args:
        input_csv_path (str): Path to the input CSV file
        output_csv_path (str, optional): Path to save the output CSV file. If None, will use input filename with '_nm' suffix
        scale_factor (float, optional): Scale factor to multiply coordinates by. Default is 8.
        
    Returns:
        pd.DataFrame: DataFrame with converted coordinates
    """
    # Read the input CSV file
    df = pd.read_csv(input_csv_path)

    # Make a copy to avoid modifying the original
    df_nm = df.copy()

    # Convert axis columns to nanometers
    axis_columns = ['axis-0', 'axis-1', 'axis-2']
    for col in axis_columns:
        if col in df_nm.columns:
            df_nm[col] = df_nm[col] * scale_factor

    # Generate output path if not provided
    if output_csv_path is None:
        base_name = os.path.splitext(input_csv_path)[0]
        output_csv_path = f"{base_name}_nm.csv"

    # Save to new CSV file
    df_nm.to_csv(output_csv_path, index=False)
    print(f"Converted coordinates saved to {output_csv_path}")

    return df_nm


# Example usage
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='Convert synapse coordinates from voxel space to nanometers')
    parser.add_argument('input_csv', help='Path to the input CSV file')
    parser.add_argument('--output_csv', help='Path to save the output CSV file (optional)')
    parser.add_argument('--scale_factor', type=float, default=8,
                        help='Scale factor to multiply coordinates by (default: 8)')

    args = parser.parse_args()

    convert_synapse_coords_to_nm(args.input_csv, args.output_csv, args.scale_factor)
