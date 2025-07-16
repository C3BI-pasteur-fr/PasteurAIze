import pandas as pd
import asyncio
import json

import os
from pydantic_ai import Agent
from pydantic_ai.models.openai import OpenAIModel
from pydantic_ai.providers.openai import OpenAIProvider
import dotenv
import re
from ruamel.yaml import YAML
import glob

dotenv.load_dotenv()


llm_model = "pixtral-large-2411-local"
llm_url = os.getenv("BASE_URL")
llm_api_key = os.getenv("API_KEY_LARGE")


def get_local_model():
    """Return a pydantic agent set with the LLM wanted"""
    model = OpenAIModel(
        llm_model,
        provider=OpenAIProvider(
            base_url=llm_url,
            api_key=llm_api_key,
        ),
    )
    return Agent(model, instrument=True)


llm = get_local_model()


def map_dtype_to_category(dtype):
    dtype_str = str(dtype)
    if "object" in dtype_str or "string" in dtype_str:
        return "string"
    elif "int" in dtype_str:
        return "integer"
    elif "float" in dtype_str:
        return "float"
    elif "datetime" in dtype_str or "date" in dtype_str or "time" in dtype_str:
        return "datetime"
    elif "bool" in dtype_str:
        return "boolean"
    else:
        return "string"


async def get_description(subset):
    description = await llm.run(
        user_prompt="Given these information about a dataframe create a json dictionnary following these rules : \n"
        "** -First ** : Find a global description/theme of the subject treated by this dataframe\n "
        "** -Second ** : Create the dictionnary with the columns name as key and a descritpion of the column as value. The description must be detailed and related to the theme found earlier and to the values of the column. If the number of differents values is inferior or equal to 5, give them in the description\n"
        "** -Third  ** : Give only the dictionnary as answer\n\n"
        " ** HERE ARE THE COLUMNS NAME OF THE DATAFRAME AND A SNIPPET OF DIFFERENTS VALUES POSSIBLE INSIDE: ** \n"
        + json.dumps(subset, default=str)
    )
    return description.output


def create_subset_list(df: pd.DataFrame, chunk_size: int, dict_type: dict):
    """Create a list of subsets of the dataframe, each containing a specified number of columns and some values example."""
    value_subsets = []
    total_cols = len(df.columns)

    for i in range(0, total_cols, chunk_size):
        end = min(i + chunk_size, total_cols)
        subset_cols = df.columns[i:end]
        subset_values = {}

        for column in subset_cols:
            typ = dict_type[column]
            if column == "inscription_date":
                print("Type: ", typ)
            if typ == "string":
                subset_values[column] = df[column].unique().tolist()[:5]

            else:
                subset_values[column] = "Numbers or dates or boolean values"

        value_subsets.append(subset_values)
    return value_subsets


def create_schema(df: pd.DataFrame):

    # Use the dict_type from the LargeDataframe instance

    dict_type = {
        df.columns[i]: map_dtype_to_category(df.dtypes.iloc[i])
        for i in range(len(df.columns))
    }
    value_subsets = create_subset_list(df, chunk_size=20, dict_type=dict_type)

    desc_columns = []
    for subset in value_subsets:
        max_retries = 3
        retry_count = 0
        # print("Subset: ", subset)
        while retry_count < max_retries:
            try:
                description = asyncio.run(get_description(subset))
                # print(
                #     f"Description (attempt {retry_count + 1}/{max_retries}): ",
                #     description,
                # )

                try:
                    description_dict = json.loads(description)
                except json.decoder.JSONDecodeError:
                    # Try to extract JSON-like content
                    pattern = r"(\{.*\})"
                    find = re.search(pattern, description, re.DOTALL)
                    if find:
                        try:
                            description_dict = json.loads(find.group(1))
                        except json.decoder.JSONDecodeError:
                            retry_count += 1
                            if retry_count < max_retries:
                                print(
                                    f"Failed to parse JSON. Retrying... ({retry_count}/{max_retries})"
                                )
                                continue
                            else:
                                raise
                    else:
                        retry_count += 1
                        if retry_count < max_retries:
                            print(
                                f"No JSON structure found. Retrying... ({retry_count}/{max_retries})"
                            )
                            continue
                        else:
                            raise ValueError(
                                "Could not find JSON structure in LLM output"
                            )

                # If we get here, parsing was successful
                for column in subset.keys():
                    if column in description_dict:
                        desc_columns.append(
                            {
                                "name": f"{column}",
                                "type": f"{dict_type.get(column, 'string')}",
                                "description": f"{description_dict[column]}",
                            }
                        )
                    else:
                        print(
                            f"Warning: Column '{column}' not found in LLM description output"
                        )

                # Successfully processed this subset
                break

            except Exception as e:
                print(f"Error during processing: {str(e)}")
                retry_count += 1
                if retry_count >= max_retries:
                    # Add fallback descriptions if all attempts fail
                    for column in subset.keys():
                        desc_columns.append(
                            {
                                "name": f"{column}",
                                "type": f"{dict_type.get(column, 'string')}",
                                "description": f"Column containing {subset[column] if isinstance(subset[column], list) else 'data'}",
                            }
                        )
                    print(
                        f"Using fallback descriptions after {max_retries} failed attempts"
                    )
                    break

        print("Desc_columns updated")

    return desc_columns


def create_yaml(
    df_name, df: pd.DataFrame, project_name: str, folder_path: str, filetype: str
):
    """
    Create a YAML file with the schema of the DataFrame.

    Args:
        df (pd.DataFrame): The DataFrame to create the schema from.
        project_name (str): The name of the project.
    """
    schema = create_schema(df)
    # current_dir = os.getcwd()
    # output_dir = os.path.join(current_dir, project_name)
    output_dir = folder_path
    # os.makedirs(output_dir, exist_ok=True)

    yaml_path = os.path.join(output_dir, f"{df_name}_schema.yaml")
    yaml_content = {
        "name": df_name,
        "source": {
            "type": filetype,
            "path": yaml_path,
        },
        "columns": schema,
    }
    # print("YAML content to be written:")
    # print("-------------------")
    # print(yaml_content)

    # print(f"Current working directory: {current_dir}")

    # Create directory if it doesn't exist
    yaml = YAML()
    with open(yaml_path, "w") as file:
        yaml.dump(yaml_content, file)

    # print(f"YAML file created at {yaml_path}")


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


# Example usage:
# folder_path = (
#     "/xxxx/yaml_test"
# )

# files_path = get_files_name(folder_path, "tsv")
# # print("Files found: ", files_path)
# list_df = {get_name_from_path(file): pd.read_csv(file, sep="\t") for file in files_path}

# create_yaml(
#     "target", list_df["target"], "yaml_test", folder_path=folder_path, filetype="tsv"
# )
