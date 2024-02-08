import openai
import mysql.connector

def generate_mysql_query(api_key, user_input):
    """
    Function to convert user input into a MySQL query using GPT-3.5-turbo.

    Parameters:
    api_key (str): Your OpenAI API key.
    user_input (str): The user input to be converted into a MySQL query.

    Returns:
    str: A MySQL query generated from the user input.
    """

    openai.api_key = api_key

    try:
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are an assistant capable of generating MySQL queries."},
                {"role": "user", "content": f"Generate a MySQL query for: '{user_input}'"}
            ]
        )
        full_response = response.choices[0].message['content']

        # Keywords to look for in the response
        keywords = ["select", "create", "alter", "delete", "drop", "rename", 
                    "update", "insert", "grant", "revoke", "rollback", "commit"]

        # Extract only the MySQL query from the response
        query_start = min([full_response.lower().find(keyword) for keyword in keywords if full_response.lower().find(keyword) != -1], default=-1)
        query_end = full_response.find(";")

        if query_start != -1 and query_end != -1:
            mysql_query = full_response[query_start:query_end+1]
        else:
            mysql_query = "No valid query could be generated."

        return mysql_query
    except Exception as e:
        print("Error:", e)
        return None


def execute_query(query, host, user, password, database):
    """
    Function to execute a given MySQL query on a specified database.

    Parameters:
    query (str): The MySQL query to be executed.
    host (str): Hostname for the database server.
    user (str): Username for the database.
    password (str): Password for the database.
    database (str): Name of the database to use.

    Returns:
    str: Results of the query or an error message.
    """
    try:
        connection = mysql.connector.connect(host=host, user=user, password=password, database=database)
        cursor = connection.cursor()
        cursor.execute(query)
        if query.lower().startswith("select"):
            results = cursor.fetchall()
            return results
        else:
            connection.commit()
            return "Query executed successfully."
    except Error as e:
        return str(e)
    finally:
        if connection.is_connected():
            cursor.close()
            connection.close()

# Example usage
api_key = "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"  # Replace with your actual API key
user_input = input("Please describe the data retrieval or action for MySQL: ")
mysql_query = generate_mysql_query(api_key, user_input)
print("MySQL Query:", mysql_query)


# Database credentials and details
db_host = "localhost"
db_user = "root"
db_password = "Gokss^^"
db_database = "dbms"

# Execute the query
result = execute_query(mysql_query, db_host, db_user, db_password, db_database)
print("Query Execution Result:", result)


