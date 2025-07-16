import os
import dotenv
import psycopg2

dotenv.load_dotenv()
db = os.getenv("DB_NAME")


def create_schema(schema_name):
    # Create a connection to the PostgreSQL database
    conn = psycopg2.connect(
        host="localhost",
        port="5432",
        dbname=db,
        user="postgres",
        password=os.getenv("POSTGRES_PASSWORD"),
    )
    cursor = conn.cursor()

    # Create the schema if it doesn't exist
    try:
        cursor.execute(f"CREATE SCHEMA {schema_name};")
        conn.commit()
        print(f"Schema '{schema_name}' created successfully.")
    except psycopg2.errors.DuplicateSchema:
        print(f"Schema '{schema_name}' already exists.")

    cursor.close()
    conn.close()


def use_schema(schema_name):
    # todo MODIFY TO THE SQL TO SET THE SEARCH PATH TO USERS WHEN ADDING USER CONCEPT
    # Create a connection to the PostgreSQL database
    conn = psycopg2.connect(
        host="localhost",
        port="5432",
        dbname=db,
        user="postgres",
        password=os.getenv("POSTGRES_PASSWORD"),
    )
    cursor = conn.cursor()

    # Set the search path to the specified schema
    cursor.execute(f'ALTER DATABASE "{db}" SET SEARCH_PATH TO {schema_name};')
    conn.commit()
    print(f"Search path set to schema '{schema_name}'.")

    cursor.close()
    conn.close()


def show_current_schema():
    # Create a connection to the PostgreSQL database
    conn = psycopg2.connect(
        host="localhost",
        port="5432",
        dbname=db,
        user="postgres",
        password=os.getenv("POSTGRES_PASSWORD"),
    )
    cursor = conn.cursor()

    # Query to get the current schema
    cursor.execute("SELECT current_schema();")
    current_schema = cursor.fetchone()[0]

    cursor.close()
    conn.close()
    return current_schema


def show_all_schemas():
    # Create a connection to the PostgreSQL database
    # todo MODIFY TO THE SQL TO SET THE SCHEMA SHOWN TO SPECIFIC USERS WHEN ADDING USER CONCEPT
    conn = psycopg2.connect(
        host="localhost",
        port="5432",
        dbname=db,
        user="postgres",
        password=os.getenv("POSTGRES_PASSWORD"),
    )
    cursor = conn.cursor()

    # Query to list all schemas
    cursor.execute("SELECT schema_name FROM information_schema.schemata;")
    schemas = cursor.fetchall()
    avaible_schemas = [
        schema[0]
        for schema in schemas
        if schema[0] not in ("information_schema", "pg_catalog", "public", "pg_toast")
    ]

    cursor.close()
    conn.close()
    return avaible_schemas


def create_table(schema_name, table_name):
    use_schema(schema_name)
    conn = psycopg2.connect(
        host="localhost",
        port="5432",
        dbname=db,
        user="postgres",
        password=os.getenv("POSTGRES_PASSWORD"),
    )
    cursor = conn.cursor()
    # Create a table in the specified schema

    conn.commit()

    cursor.execute(
        f"""
        CREATE TABLE IF NOT EXISTS {table_name} (
            id SERIAL PRIMARY KEY,
            name VARCHAR(100),
            value INT
        );
    """
    )
    conn.commit()
    print(f"Table '{table_name}' created in schema '{schema_name}'.")

    cursor.close()
    conn.close()
