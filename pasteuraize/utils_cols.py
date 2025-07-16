import pandas as pd
import os
import dotenv
import glob

dotenv.load_dotenv()


folder_path = os.getenv("FOLDER_PATH_TEST")


def clean_name(name):
    return (
        name.replace(" ", "").replace("_", "").replace("-", "").replace(".", "").lower()
    )


def get_name_from_path(path: str):
    """Get the name of the dataframe from the path"""
    parts = path.split("/")
    name = parts[-1].split(".")[0]
    name = clean_name(name)
    return name


def get_files_name(path: str, type_file: str):
    """give a list of the files names of type 'type_file' in a folder at the path 'path'"""
    path = path + "/*." + type_file
    return glob.glob(path)


files_path = get_files_name(folder_path, "parquet")
list_df = {get_name_from_path(file): pd.read_parquet(file) for file in files_path}


def rename_cols_FK(df: pd.DataFrame, cols_name: list):
    """Rename the specified column in the DataFrame to set it as a foreign key."""
    for name in cols_name:
        if name in df.columns:
            new_col_name = name + "_FK"
            df.rename(columns={name: new_col_name}, inplace=True)
        else:
            raise ValueError(f"Column '{name}' does not exist in the DataFrame.")

    return df


def rename_cols(df: pd.DataFrame, cols_name: list):
    """Rename the specified column in the DataFrame"""
    """argument cols_name is a list of tuples with column names to rename and new names"""

    for previous, new in cols_name:
        if previous in df.columns:
            df.rename(columns={previous: new}, inplace=True)
        else:
            raise ValueError(f"Column '{previous}' does not exist in the DataFrame.")

    return df


def rename_cols_FK_in_all_dfs(list_df: dict, cols_names: dict):
    """
    Rename specified columns in all DataFrames in the list_df dictionary.

    :param list_df: Dictionary of DataFrames.
    :param cols_names: Dictionary where keys are DataFrame names and values are lists of column names to rename.
    :return: Updated dictionary of DataFrames with renamed columns.
    """
    for name, df in list_df.items():
        # Check if the DataFrame name exists in cols_names

        if name in cols_names:
            to_rename = [col for col in cols_names[name] if col in df.columns]
            # for col_name in cols_names[name]:
            #     # print(f"Renaming column '{col_name}' in DataFrame '{name}'")
            if len(to_rename) > 1:
                df = rename_cols_FK(df, to_rename)
                list_df[name] = df
    return list_df


def set_first_col(df: pd.DataFrame, col_name: str):
    """Set the first column of the DataFrame to the specified column name."""
    if col_name in df.columns:
        df = df[[col_name] + [col for col in df.columns if col != col_name]]
    else:
        raise ValueError(f"Column '{col_name}' does not exist in the DataFrame.")
    return df


# print(set_first_col(list_df["metadatawithcoords"], "id_mgp"))


# print(list_df.keys())
# for name, df in list_df.items():
#     if name == "abondancetablesub20mrandreduced":
#         df = rename_cols(df, [("sample", "id_mgp"), ("bacteria", "msp_name")])
#         list_df[name] = df
#         print(df.head())


# renamed_cols = rename_cols_FK_in_all_dfs(
#     list_df=list_df,
#     cols_names={"abondancetablesub20mrandreduced": ["id_mgp", "msp_name"]},
# )
# for name, df in renamed_cols.items():
#     print(f"DataFrame '{name}' after renaming columns:")
#     print(df.head())
#     print("\n")
