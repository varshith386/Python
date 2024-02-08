import csv
import mysql.connector

# MySQL database configuration
db_config = {
    'user': 'root',
    'password': 'Gokss^^',
    'host': 'localhost',
    'database': 'dbms'
}

# CSV file path
csv_file_path = r'C:\Users\admin\Downloads\archive\StudentsPerformance.csv'

# Connect to MySQL database
try:
    conn = mysql.connector.connect(**db_config)
    cursor = conn.cursor()

    # Open and read the CSV file
    with open(csv_file_path, 'r') as csv_file:
        csv_reader = csv.reader(csv_file)

        # Skip header row
        header = next(csv_reader)

        # Create table (if not exists)
        table_name = 'students'
        create_table_query = f"CREATE TABLE IF NOT EXISTS {table_name} ({', '.join([f'{col} VARCHAR(255)' for col in header])})"
        cursor.execute(create_table_query)

        # Insert data into MySQL table
        insert_query = f"INSERT INTO {table_name} ({', '.join(header)}) VALUES ({', '.join(['%s' for _ in header])})"
        for row in csv_reader:
            cursor.execute(insert_query, row)

    # Commit changes
    conn.commit()
    print("Data imported successfully!")


except mysql.connector.Error as err:
    print(f"Error: {err}")

finally:
    # Close connections
    if 'conn' in locals() and conn.is_connected():
        cursor.close()
        conn.close()
        print("MySQL connection closed.")
