import psycopg2
import pandas as pd
from vanna.chromadb.chromadb_vector import ChromaDB_VectorStore
from vanna.openai.openai_chat import OpenAI_Chat
import os
import dotenv
from openai import OpenAI
import yaml
import glob
import itertools
from utils_cols import (
    rename_cols,
    rename_cols_FK_in_all_dfs,
    set_first_col,
)
from manage_schema import create_schema, use_schema
from create_yaml import create_yaml

dotenv.load_dotenv()
# Load environment variables from .env file

folder_path = os.getenv("FOLDER_PATH")
api_key = os.getenv("API_KEY")
base_url = os.getenv("BASE_URL")
llm_model = os.getenv("DATA_MODEL")
db = os.getenv("DB_NAME")
# Create a database
schema_name = str(input("Enter the name of the project to create: "))
schema_name = schema_name.lower()
if not schema_name:
    project_name = "Project/test"
    create_schema(schema_name)
else:
    project_name = "Project/" + schema_name
    create_schema(schema_name)

# Set the schema to use (WATCH OUT IT IS NOT ACTUALLY USING A SEARCH_PATH BASED ON USER BUT ON DB)
use_schema(schema_name)

# Connect to the PostgreSQL database

conn = psycopg2.connect(
    host="localhost",
    port="5432",
    dbname=db,  # Change this to your database name
    user="postgres",
    password=os.getenv("POSTGRES_PASSWORD"),
)

conn.autocommit = True


def create_count_table(df, sample, species):
    # Create a DataFrame with all combinations of msp and fg
    print(f"Creating Count Table")
    species_value = df[species]
    sample_values = [name for name in df.columns if name != species]
    # print(f"Sample values: {sample_values}")
    all_combinations = list(itertools.product(sample_values, species_value))
    # Initialize the reduced DataFrame
    df_reduced = pd.DataFrame(all_combinations, columns=[sample, species])
    # Add the values from the original DataFrame
    values = {}
    for smpl in sample_values:
        values[smpl] = df[smpl].values

    # Create a mapping of bacteria to their indices
    species_to_idx = {species: idx for idx, species in enumerate(df[species])}

    # Add the values to df_reduced
    df_reduced["abundance"] = df_reduced.apply(
        lambda row: values[row[sample]][species_to_idx[row[species]]], axis=1
    )

    # Rename columns to add _FK suffix for foreign keys
    df_reduced.rename(
        columns={sample: f"{sample}_FK", species: f"{species}_FK"}, inplace=True
    )
    return df_reduced


def table_foreign_keys(list_df):
    """
    Create a dict of the foreign key link bewteen the tables from the list of dataframes.

    it Return a dict with the table name as key and a tuple of the table name linked and foreign key as value.
    """
    pk = []
    fk = []
    for name, df in list_df.items():
        # Get the keys

        for col in df.columns:
            if "_FK" in col:
                col_name = col.replace("_FK", "")
                df.rename(columns={col: col_name}, inplace=True)  # Rename the column
                fk.append((name, col_name))
        pk.append((name, df.columns[0]))  # Assuming the first column is the primary key
    link_table = {}
    for i in range(len(fk)):
        table_keys = []
        for name, df in list_df.items():
            if fk[i][1] in df.columns and fk[i][0] != name:
                # If the foreign key is in the dataframe, add it to the link table
                table_keys.append((name, fk[i][1]))
        if fk[i][0] not in link_table:
            link_table[fk[i][0]] = table_keys
        else:
            link_table[fk[i][0]].extend(table_keys)
    return link_table


def define_col_types(df):
    col_types = {}
    for col in df.columns:
        if pd.api.types.is_integer_dtype(df[col]):
            col_types[col] = "INTEGER"
        elif pd.api.types.is_float_dtype(df[col]):
            col_types[col] = "REAL"
        elif pd.api.types.is_datetime64_any_dtype(df[col]):
            col_types[col] = "TIMESTAMP"
        elif pd.api.types.is_string_dtype(df[col]):
            col_types[col] = "TEXT"
        else:
            col_types[col] = "TEXT"  # Default to TEXT for other types
    return col_types


def create_table_from_df(df, table_name, conn, linked_col=[], join={}):
    # Create a cursor object
    cursor = conn.cursor()

    # Create a copy of the dataframe to avoid modifying the original
    df_copy = df.copy()

    # Get column types
    defined_col_types = define_col_types(df_copy)

    # Set the id column with name and type
    id_col = list(defined_col_types.items())[0]

    # Create the table with the same schema as the DataFrame
    create_table_query = f"CREATE TABLE IF NOT EXISTS {table_name} ("
    # If there is more than one join it will be a composed Primary Key
    print(f"Len linked_col: {len(linked_col)}")
    if len(linked_col) > 1:
        for column in df_copy.columns:
            print(f'Column "{column}" type: {defined_col_types[column]}')
            create_table_query += f'"{column}" {defined_col_types[column]},'
        create_table_query += f"PRIMARY KEY("
        for link in linked_col:
            print(f"Link: {link}")
            create_table_query += f'"{link[1]}", '
        # Setting the end of the primary key line
        create_table_query = create_table_query[:-2] + "),"
    else:
        create_table_query += f'"{id_col[0]}" {id_col[1]} PRIMARY KEY,'
        # Add all columns except 'id' which we already defined
        for column in df_copy.columns:
            if column != id_col[0]:  # Skip the id column as we already defined it
                create_table_query += f'"{column}" {defined_col_types[column]},'
    print(f"Create table query: {create_table_query}")
    if table_name in join:
        create_table_query += join[table_name]

    # Remove last comma and close parenthesis
    create_table_query = create_table_query[:-1] + ");"
    print(f"Final create table query: {create_table_query}")
    cursor.execute(create_table_query)

    # Use pandas to_sql with a more direct approach
    try:
        # Insert row by row with explicit column mapping
        for idx, row in df_copy.iterrows():
            # Build values list ensuring values match their column positions
            values = []
            # Add all other column values in order
            for col in df_copy.columns:
                val = row[col]
                # Convert NaT to None
                if pd.isna(val):
                    val = None
                values.append(val)

            # Create columns list with proper quoting
            columns = ", ".join([f'"{col}"' for col in df_copy.columns])
            placeholders = ", ".join(["%s"] * len(df_copy.columns))

            # Execute insert with proper values
            insert_query = (
                f'INSERT INTO "{table_name}" ({columns}) VALUES ({placeholders})'
            )
            cursor.execute(insert_query, values)

    except Exception as e:
        print(f"Error during insertion: {e}")
        conn.rollback()
        raise

    # Commit the changes
    conn.commit()
    cursor.close()
    return (create_table_query, insert_query)


def create_join(link_table):
    """create the SQL JOIN queries"""
    query_list = {}
    query = ""
    for table, keys in link_table.items():
        for link_table, fk in keys:
            query += f'FOREIGN KEY ("{fk}") REFERENCES {link_table}("{fk}"),'
        query_list[table] = query
    return query_list


def define_order(list_df, link_table):
    """Define the order of creation of each table"""
    count = {table: 0 for table in list_df.keys()}

    for table, link in link_table.items():
        if table in count:
            count[table] = len(link)

    print(f"Count of links for each table: {count}")
    ordered = {
        key: list_df[key]
        for key, _ in sorted(count.items(), key=lambda occurence: occurence[1])
    }

    return ordered


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


def chunk_list(lst, n):
    chunked_list = []
    for i in range(0, len(lst), n):
        chunked_list.append(lst[i : i + n])
    return chunked_list


def get_df_from_path(folder_path: str, filetype: list):
    """Get the DataFrame from the path"""
    all_df = {}
    if "parquet" in filetype:
        parquet_path = get_files_name(folder_path, "parquet")
        all_parquet = {
            get_name_from_path(file): pd.read_parquet(file) for file in parquet_path
        }
        all_df.update(all_parquet)
    if "tsv" in filetype:
        tsv_path = get_files_name(folder_path, "tsv")
        all_tsv = {
            get_name_from_path(file): pd.read_csv(file, sep="\t") for file in tsv_path
        }
        all_df.update(all_tsv)
    if "csv" in filetype:
        csv_path = get_files_name(folder_path, "csv")
        all_csv = {get_name_from_path(file): pd.read_csv(file) for file in csv_path}
        all_df.update(all_csv)
    else:
        raise ValueError(f"Unsupported file type: {filetype}")
    return all_df


def create_missing_yaml(list_df: dict, yaml_path: list):
    existing_yaml = [get_name_from_path(path) for path in yaml_path]

    for name, df in list_df.items():
        schema_name = name + "_schema"
        if schema_name not in existing_yaml:
            # If the YAML file does not exist, create it
            if name in get_files_name(folder_path, "yaml"):
                filetype = "yaml"
            elif name in get_files_name(folder_path, "parquet"):
                filetype = "parquet"
            else:
                filetype = "csv"
            create_yaml(
                df_name=name,
                df=df,
                project_name=project_name,
                filetype=filetype,
                folder_path=folder_path,
            )
            print(f"YAML file created for {name}")
        else:
            print(f"YAML file already exists for {name}, skipping creation.")
    updated_folder_path = get_files_name(folder_path, "yaml")
    return updated_folder_path


def main():
    """Main function to run the auto table creation process."""
    list_df = get_df_from_path(folder_path, ["parquet", "tsv", "csv"])
    # print(f"List of DataFrames found: {list_df}")
    # Create the links as a dict with the foreign keys
    for name, df in list_df.items():
        if "counttable" in name:
            # If the table is a count table, we need to create it with the count function
            count_table = name
        elif "taxonomy" in name:
            print(f"Creating table {name}")
            species = df.columns[0]
        elif "target" in name:
            print(f"Creating table {name}")
            sample = df.columns[0]
    try:
        list_df[count_table] = create_count_table(list_df[count_table], sample, species)

    except Exception as e:
        print(f"No counting Table or error with taxonomy or target: {e}")
    link_table = table_foreign_keys(list_df)
    # Create the join queries for the foreign keys
    join = create_join(link_table)
    # Define the order of creation of the tables based on the foreign keys
    list_df = define_order(list_df=list_df, link_table=link_table)
    for name in list_df[count_table].columns:
        if "_FK" in name:
            list_df[count_table].rename(
                columns={name: name.replace("_FK", "")}, inplace=True
            )
    # print(f"List of DataFrames to create: {list_df}")
    # Connect to OpenAI client
    client = OpenAI(api_key=api_key, base_url=base_url)

    class MyVanna(ChromaDB_VectorStore, OpenAI_Chat):
        def __init__(self, config=None):
            ChromaDB_VectorStore.__init__(self, config=config)
            OpenAI_Chat.__init__(self, client=client, config=config)

    vn = MyVanna(config={"model": llm_model, "path": project_name})

    list_queries = []
    # Create tables in the order defined by the foreign keys and a list of SQL queries
    for name, df in list_df.items():
        if name in link_table:
            list_queries.append(
                create_table_from_df(
                    df, name, conn, linked_col=link_table[name], join=join
                )
            )
        else:
            list_queries.append(
                create_table_from_df(
                    df,
                    name,
                    conn,
                )
            )

    # Train the Vanna model with the queries generated from the DataFrames insertion in SQL
    for queries in list_queries:
        vn.train(ddl=queries[0])

    # Create YAML files for the DataFrames
    yaml_path = get_files_name(folder_path, "yaml")
    updated_yaml_path = create_missing_yaml(list_df=list_df, yaml_path=yaml_path)

    list_yaml = []
    for path in updated_yaml_path:
        with open(path) as file:
            list_yaml.append(yaml.safe_load(file))

    df_metadata = {}
    for data in list_yaml:
        for categories in data.items():
            if type(categories[1]) == str:  # LE NOM DU DATAFRAME
                df_metadata[categories[0]] = categories[1]
            if (
                type(categories[1]) == dict
            ):  # Type : parquet, csv, etc. , path: data.parquet
                df_metadata[categories[0]] = str(categories[1])
            # Donc le nom et la source peuvent être envoyé ensemble
            if type(categories[1]) == list:  # liste de dict: nom, type, description
                chunk_lists = chunk_list(categories[1], 1)
                i = 0
                for chunk in chunk_lists:
                    df_metadata[f"chunk_{i}"] = str(chunk)
                    i += 1

    for elt in df_metadata.items():
        vn.train(documentation=str(elt))


# # EXEMPLE OF UTILIZATION
# list_df = get_df_from_path(folder_path, ["parquet", "tsv", "csv"])

# # Rename specific columns in the DataFrames that are supposed to be foreign keys
# for name, df in list_df.items():
#     if name == "abondancetablesub20mrandreduced":
#         df = rename_cols(df, [("sample", "id_mgp"), ("bacteria", "msp_name")])
#         list_df[name] = df
# list_df = rename_cols_FK_in_all_dfs(
#     list_df=list_df,
#     cols_names={"abondancetablesub20mrandreduced": ["id_mgp", "msp_name"]},
# )
# # Set the first column as id_mgp so that it can be used as a primary key (also foreign key in other tables)
# list_df["metadatawithcoords"] = set_first_col(list_df["metadatawithcoords"], "id_mgp")

# # Create the links as a dict with the foreign keys
# link_table = table_foreign_keys(list_df)
# # Create the join queries for the foreign keys
# join = create_join(link_table)
# # Define the order of creation of the tables based on the foreign keys
# list_df = define_order(list_df=list_df, link_table=link_table)

# # Connect to OpenAI client
# client = OpenAI(api_key=api_key, base_url=base_url)


# class MyVanna(ChromaDB_VectorStore, OpenAI_Chat):
#     def __init__(self, config=None):
#         ChromaDB_VectorStore.__init__(self, config=config)
#         OpenAI_Chat.__init__(self, client=client, config=config)


# vn = MyVanna(config={"model": llm_model, "path": project_name})


# list_queries = []
# # Create tables in the order defined by the foreign keys and a list of SQL queries

# for name, df in list_df.items():
#     print(f"Creating table {name}")
#     if "count_table" in name:
#         # If the table is a count table, we need to create it with the count function
#         count_table = df
#     elif "taxonomy" in name:
#         species = df.columns[0]
#     elif "target" in name:
#         sample = df.columns[0]
#         print("No count table found, creating table from DataFrame")

# for name, df in list_df.items():
#     if name in link_table:
#         if "count_table" in name:
#             # If the table is a count table, we need to create it with the count function
#             df = create_count_table(df, sample, species)
#         list_queries.append(
#             create_table_from_df(df, name, conn, linked_col=link_table[name], join=join)
#         )
#     else:
#         list_queries.append(
#             create_table_from_df(
#                 df,
#                 name,
#                 conn,
#             )
#         )

# # Train the Vanna model with the queries generated from the DataFrames insertion in SQL
# for queries in list_queries:
#     vn.train(ddl=queries[0])

# # Create YAML files for the DataFrames
# yaml_path = get_files_name(folder_path, "yaml")
# updated_yaml_path = create_missing_yaml(list_df=list_df, yaml_path=yaml_path)

# list_yaml = []
# for path in updated_yaml_path:
#     with open(path) as file:
#         list_yaml.append(yaml.safe_load(file))


# df_metadata = {}
# for data in list_yaml:
#     for categories in data.items():
#         if type(categories[1]) == str:  # LE NOM DU DATAFRAME
#             df_metadata[categories[0]] = categories[1]
#         if (
#             type(categories[1]) == dict
#         ):  # Type : parquet, csv, etc. , path: data.parquet
#             df_metadata[categories[0]] = str(categories[1])
#         # Donc le nom et la source peuvent être envoyé ensemble
#         if type(categories[1]) == list:  # liste de dict: nom, type, description
#             chunk_lists = chunk_list(categories[1], 1)
#             i = 0
#             for chunk in chunk_lists:
#                 df_metadata[f"chunk_{i}"] = str(chunk)
#                 i += 1

# for elt in df_metadata.items():
#     vn.train(documentation=str(elt))


def add_french_gut():
    list_df = get_df_from_path(folder_path, ["parquet", "tsv", "csv"])
    # Rename specific columns in the DataFrames that are supposed to be foreign keys
    for name, df in list_df.items():
        if name == "counttable":
            df = rename_cols(df, [("msp", "msp_name")])
            df = set_first_col(df, "msp_name")
            list_df[name] = df

        if name == "target":
            df = rename_cols(df, [("id_mgp", "sample")])
            df = set_first_col(df, "sample")
            list_df[name] = df

    for name, df in list_df.items():
        if "counttable" in name:
            # If the table is a count table, we need to create it with the count function
            count_table = name
        elif "taxonomy" in name:
            print(f"Creating table {name}")
            species = df.columns[0]
        elif "target" in name:
            print(f"Creating table {name}")
            sample = df.columns[0]
    try:
        list_df[count_table] = create_count_table(list_df[count_table], sample, species)

    except Exception as e:
        print(f"No counting Table or error with taxonomy or target: {e}")
    link_table = table_foreign_keys(list_df)
    # Create the join queries for the foreign keys
    join = create_join(link_table)
    # Define the order of creation of the tables based on the foreign keys
    list_df = define_order(list_df=list_df, link_table=link_table)
    for name in list_df[count_table].columns:
        if "_FK" in name:
            list_df[count_table].rename(
                columns={name: name.replace("_FK", "")}, inplace=True
            )
    # print(f"List of DataFrames to create: {list_df}")
    # Connect to OpenAI client
    client = OpenAI(api_key=api_key, base_url=base_url)

    class MyVanna(ChromaDB_VectorStore, OpenAI_Chat):
        def __init__(self, config=None):
            ChromaDB_VectorStore.__init__(self, config=config)
            OpenAI_Chat.__init__(self, client=client, config=config)

    vn = MyVanna(config={"model": llm_model, "path": project_name})

    list_queries = []
    # Create tables in the order defined by the foreign keys and a list of SQL queries
    for name, df in list_df.items():
        if name in link_table:
            list_queries.append(
                create_table_from_df(
                    df, name, conn, linked_col=link_table[name], join=join
                )
            )
        else:
            list_queries.append(
                create_table_from_df(
                    df,
                    name,
                    conn,
                )
            )

    # Train the Vanna model with the queries generated from the DataFrames insertion in SQL
    for queries in list_queries:
        vn.train(ddl=queries[0])

    # Create YAML files for the DataFrames
    yaml_path = get_files_name(folder_path, "yaml")
    updated_yaml_path = create_missing_yaml(list_df=list_df, yaml_path=yaml_path)

    list_yaml = []
    for path in updated_yaml_path:
        with open(path) as file:
            list_yaml.append(yaml.safe_load(file))

    df_metadata = {}
    for data in list_yaml:
        for categories in data.items():
            if type(categories[1]) == str:  # LE NOM DU DATAFRAME
                df_metadata[categories[0]] = categories[1]
            if (
                type(categories[1]) == dict
            ):  # Type : parquet, csv, etc. , path: data.parquet
                df_metadata[categories[0]] = str(categories[1])
            # Donc le nom et la source peuvent être envoyé ensemble
            if type(categories[1]) == list:  # liste de dict: nom, type, description
                chunk_lists = chunk_list(categories[1], 1)
                i = 0
                for chunk in chunk_lists:
                    df_metadata[f"chunk_{i}"] = str(chunk)
                    i += 1

    for elt in df_metadata.items():
        vn.train(documentation=str(elt))


if __name__ == "__main__":
    # main()
    add_french_gut()
    print("Auto table creation completed successfully.")
    print("You can now use the Vanna model to interact with your data.")
    print(
        "Remember to check the created tables and YAML files in the specified folder."
    )
