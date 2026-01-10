import pandas as pd
import numpy as np
import random
import os
import time


class ControlPolicy(): 
    """
    Controls the policy for the users
    """
    
    @staticmethod
    def fixed_policy():
        """ give out notification at 9am always """
        return 9

class LogGenerator:
    """
    Generates send and response logs for the users
    """

    def __init__(self, user_data_path, logs_path):
        self.logs_path = logs_path
        self.user_data = pd.read_csv(user_data_path)
        
        send_logs = []
        response_logs = []

        for day in range(30):  # 30 days
            for index, row in self.user_data.iterrows():
                hour = ControlPolicy.fixed_policy()
                user_id = row["name"]
                peak_hour = row["peak_hour"]
                peak_probability = row["peak_probability"]
                user_type = row["user_type"]
                event_id = f"{user_id}_{day}"


                send_logs.append(
                    {
                        "event_id": event_id,
                        "user_id": user_id,
                        "day": day,
                        "hour": hour
                    })

                p = peak_probability if hour == peak_hour else peak_probability * 0.4
                opened = random.random() < p

                response_logs.append(
                    {
                        "event_id": event_id,
                        "opened": int(opened)
                    })
        
        self.send_logs = pd.DataFrame(send_logs)
        self.response_logs = pd.DataFrame(response_logs)

        self.save_logs()

    def save_logs(self):
        """ save the logs to a csv file """
        self.send_logs.to_csv(os.path.join(self.logs_path, "send_logs.csv"), index=False)
        self.response_logs.to_csv(os.path.join(self.logs_path, "response_logs.csv"), index=False)
    

if __name__ == "__main__":
    user_data_path = "data/user_data.csv"
    logs_path = "data/logs"
    os.makedirs(logs_path, exist_ok=True)
    log_generator = LogGenerator(user_data_path, logs_path)