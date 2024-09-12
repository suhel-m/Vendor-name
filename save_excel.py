import sqlite3

# Create a connection to the SQLite database file
# This will create 'your_database.db' file in the current directory if it doesn't exist


# Dynamically generate a CREATE TABLE statement based on df.columns
def create_table_from_df(df, table_name):
    
    
    conn = sqlite3.connect('database.db')
    # Optional: Create a cursor object using the connection to execute SQL commands
    cursor = conn.cursor()
    
    # Optional: Print a success message
    print("Database created and connected successfully.")

    columns = df.columns
    column_definitions = []

    # Map Python types to SQLite types
    type_mapping = {
        'int64': 'INTEGER',
        'float64': 'REAL',
        'object': 'TEXT',
        'bool': 'INTEGER',  # SQLite uses 0 and 1 for boolean values
        'datetime64[ns]': 'TEXT'  # Store datetime as TEXT
    } 
    print(columns.tolist())
    for column in columns.tolist():
        # Get the dtype of the column and map it to SQLite type
        
        dtype = str(df[column].dtype)
        sqlite_type = type_mapping.get(dtype, 'TEXT')  # Default to TEXT if unknown
        column_definitions.append(f"{column} {sqlite_type}")
   
    # Combine column definitions into a CREATE TABLE statement
    create_table_statement = f"CREATE TABLE IF NOT EXISTS {table_name} ({', '.join(column_definitions)});"
    
    cursor.execute(create_table_statement)

    df.to_sql(table_name, conn, if_exists='replace', index=False)
    print('value added')
# Define the table name

    cursor.close()
    # Close the connection
    conn.close()
