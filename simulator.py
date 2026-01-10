"""
Creates a system to simulate users interacting with a web application
"""

import random
import time
import numpy as np
import os
import pandas as pd


class User:

    
    def __init__(self, count_id):
        """
        Defines the user's attributes
        """
        self.id = count_id
        self.name = f"U_{self.id}"
        self.user_type = random.choice(["early_bird", "night_owl", "regular", "sporadic"])
        self.peak_hours = self.get_peak_hours()

    def define_user(self):
        """
        Defines the user's attributes
        """
        self.name = f"U_{self.id}"
        self.user_type = random.choice(["early_bird", "night_owl", "regular", "sporadic"])
        self.peak_hours = self.get_peak_hours()
        self.peak_probability = np.random.uniform(0.5, 1) #samples betwene 0.5 and 1
        return self

    def get_peak_hours(self):
        """
        Gets the user's peak hours
        """
        if self.user_type == "early_bird":
            return random.randint(7, 10)
        elif self.user_type == "night_owl":
            return random.randint(20, 23)
        elif self.user_type == "regular":
            return random.randint(10, 18)
        elif self.user_type == "sporadic":
            return random.randint(8, 22)



class Simulation:
    """
    Simulates the users
    """
    def __init__(self, num_users):
        self.users = []
        for i in range(num_users):
            user = User(i+1).define_user()
            self.users.append(user)
            # print(user.name, user.peak_hours, user.user_type)
            # raise Exception("Stop")
        self.save_user_data()

    def save_user_data(self):
        """
        Saves the user's data to a CSV file
        """

        user_df = pd.DataFrame([{
            "name": user.name,
            "peak_hours": user.peak_hours,
            "user_type": user.user_type,
            "peak_hours": user.peak_hours
        } for user in self.users])

        if os.path.exists("user_data.csv"):
            all_users = pd.read_csv("user_data.csv")
            all_users = pd.concat([all_users, user_df], ignore_index=True)
        else:
            all_users = user_df

        all_users.to_csv("user_data.csv", index=False)

    def load_user_data(self):
        """
        Loads the user's data from a CSV file
        """

        return pd.read_csv("user_data.csv")



if __name__ == "__main__":
    simulation = Simulation(100)