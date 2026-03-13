import os

import pandas as pd
import psycopg2
from psycopg2 import extras




def import_csv_data_to_db(
    db_config,
    user_data_path=None,
    send_logs_path=None,
    response_logs_path=None,
):
    user_data_path = user_data_path or os.path.join("data", "user_data.csv")
    send_logs_path = send_logs_path or os.path.join("data", "logs", "send_logs.csv")
    response_logs_path = response_logs_path or os.path.join("data", "logs", "response_logs.csv")

    try:
        conn = psycopg2.connect(**db_config)
        cur = conn.cursor()
        print("Succesfully connect to DB!")

        # clean state
        cur.execute(
            """
            DROP TABLE IF EXISTS raw_users;
            CREATE TABLE raw_users (
                name TEXT PRIMARY KEY,
                user_type TEXT,
                base_engagement FLOAT,
                weekend_modifier FLOAT,
                hourly_weights JSONB
            );

            DROP TABLE IF EXISTS raw_sends;
            CREATE TABLE raw_sends (
                event_id TEXT PRIMARY KEY,
                user_id TEXT,
                day INT,
                hour INT,
                day_of_week INT,
                is_weekend INT,
                send_timestamp TIMESTAMP 
            );

            DROP TABLE IF EXISTS raw_responses;
            CREATE TABLE raw_responses (
                event_id TEXT PRIMARY KEY,
                opened INT,
                response_delay_minutes INT,
                open_timestamp TIMESTAMP NULL
            );
            """
        )

        # Load CSVs as DataFrames
        df_user_data = pd.read_csv(user_data_path)
        df_response_logs = pd.read_csv(response_logs_path)
        df_response_logs["open_timestamp"] = df_response_logs["open_timestamp"].replace({pd.NA: None, float("nan"): None})
        df_send_logs = pd.read_csv(send_logs_path)


        user_tuples = [tuple(x) for x in df_user_data.to_numpy()]
        response_tuples = [tuple(x) for x in df_response_logs.to_numpy()]
        send_tuples = [tuple(x) for x in df_send_logs.to_numpy()]

        insert_user_query = "INSERT INTO raw_users (name, user_type, base_engagement, weekend_modifier, hourly_weights) VALUES %s"
        extras.execute_values(cur, insert_user_query, user_tuples)  # use for bulk load

        insert_send_query = "INSERT INTO raw_sends (event_id, user_id, day, hour, day_of_week, is_weekend, send_timestamp) VALUES %s"
        extras.execute_values(cur, insert_send_query, send_tuples)  # bulk load for sends

        insert_response_query = "INSERT INTO raw_responses (event_id, opened, response_delay_minutes, open_timestamp) VALUES %s"
        extras.execute_values(cur, insert_response_query, response_tuples)  # bulk load for responses

        #commit changes
        conn.commit()
        print("ENttered logs and user data")
    
    except Exception as e:
        print("Error : ", e)
        if conn:
            conn.rollback()
    finally:
        if cur: cur.close()
        if conn: conn.close()


if __name__ == "__main__":
    
    DB_CONFIG = {
    "host": "notification_db",
    "database": "notification_db",
    "user": "admin",
    "password": "pypass",
    "port": "5432"
    }
    import_csv_data_to_db(DB_CONFIG)

