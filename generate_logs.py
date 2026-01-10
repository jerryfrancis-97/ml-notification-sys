import pandas as pd
import numpy as np
import random
import os
import time


def ControlPolicy(user_data):
    """
    Controls the policy for the users
    """
    
    def fixed_policy(self):
        """ give out notification at 9am always """
        return 9

class LogGenerator:
    """
    Generates send and response logs for the users
    """

    def __init__(self, user_data):

        self.user_data = pd.read_csv(user_data)
        
        send_logs = []
        response_logs = []

        for day in range(30):  # 30 days
            for index, row in self.user_data.iterrows():
                hour = ControlPolicy.fixed_policy(user_id)
                user_id = row["user_id"]
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
            return send_logs, response_logs