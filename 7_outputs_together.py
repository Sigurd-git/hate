import pandas as pd
from pathlib import Path


def merge_excels_in_folder(folder_path: str) -> pd.DataFrame:
    """
    Merge all .xlsx files in a folder into a single DataFrame.

    Parameters
    ----------
    folder_path : str
        Path to the folder that contains Excel files.

    Returns
    -------
    pd.DataFrame
        Concatenated DataFrame with an extra 'source_file' column.
    """
    folder = Path(folder_path)
    if not folder.is_dir():
        raise NotADirectoryError(f"{folder} is not a valid directory")

    # Find all Excel files in the folder (non-recursive)
    excel_files = sorted(folder.glob("*.xlsx"))

    if not excel_files:
        raise FileNotFoundError(f"No .xlsx files found in {folder}")

    dataframes = []
    for file in excel_files:
        # Read each Excel file
        df = pd.read_excel(file)

        # Add a column to record which file each row came from
        df["source_file"] = file.name

        dataframes.append(df)

    # Concatenate all DataFrames
    merged_df = pd.concat(dataframes, ignore_index=True)
    return merged_df


def save_merged_excel(folder_path: str, output_filename: str) -> None:
    """
    Merge all Excel files in a folder and save the result as a new Excel file.

    Parameters
    ----------
    folder_path : str
        Path to the folder that contains Excel files.
    output_filename : str
        Name of the merged Excel file to be created (without folder path).
    """
    folder = Path(folder_path)
    merged_df = merge_excels_in_folder(folder_path)

    # Save the merged result in the same parent directory as the folder
    output_path = folder.parent / output_filename

    # Use index=False to avoid writing the DataFrame index as a column
    merged_df.to_excel(output_path, index=False)
    print(f"Merged Excel saved to: {output_path}")


if __name__ == "__main__":
    # TODO: Change these two paths to your own folders

    # Folder 1: chain-of-thought condition
    cot_folder = "/Users/baoxuan/Desktop/研究生研究/llm毕业论文/hate/outputs/natural/5_new_englishhatedata_2400_balanced/en/chain_of_thought"

    # Folder 2: zero-shot condition
    zs_folder = "/Users/baoxuan/Desktop/研究生研究/llm毕业论文/hate/outputs/natural/5_new_englishhatedata_2400_balanced/en/zero_shot"

    # Merge each condition separately
    save_merged_excel(cot_folder, "merged_chain_of_thought.xlsx")
    save_merged_excel(zs_folder, "merged_zero_shot.xlsx")

    # If you also want a single file that combines both conditions, uncomment below:
    # cot_df = merge_excels_in_folder(cot_folder)
    # zs_df = merge_excels_in_folder(zs_folder)
    # both_df = pd.concat([cot_df, zs_df], ignore_index=True)
    # both_output = Path(cot_folder).parent / "merged_all_conditions.xlsx"
    # both_df.to_excel(both_output, index=False)
    # print(f"All conditions merged Excel saved to: {both_output}")