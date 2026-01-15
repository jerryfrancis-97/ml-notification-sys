"""
Creates a system to simulate users interacting with a web application
"""

import random
import numpy as np
import os
import pandas as pd
import json


class User:
    """
    Represents a user with realistic engagement patterns
    """
    
    def __init__(self, count_id):
        """
        Defines the user's attributes
        """
        self.id = count_id
        self.name = f"U_{self.id}"
        self.user_type = random.choice(["early_bird", "night_owl", "regular", "sporadic"])
        self.base_engagement = np.random.uniform(0.3, 0.9)  # Overall engagement level
        self.weekend_modifier = self._get_weekend_modifier()
        self.hourly_weights = self._generate_hourly_curve()

    def _get_weekend_modifier(self):
        """
        Get weekend behavior modifier based on user type
        """
        if self.user_type == "early_bird":
            return np.random.uniform(0.7, 0.9)  # Less active on weekends
        elif self.user_type == "night_owl":
            return np.random.uniform(1.1, 1.4)  # More active on weekends
        elif self.user_type == "regular":
            return np.random.uniform(0.8, 1.0)  # Slightly less active
        else:  # sporadic
            return np.random.uniform(0.9, 1.3)  # Variable

    def _generate_hourly_curve(self):
        """
        Generate a 24-hour probability curve based on user type
        Returns array of 24 weights (one per hour)
        """
        weights = np.zeros(24)
        
        if self.user_type == "early_bird":
            # Peak between 6-10 AM
            peak_center = random.randint(7, 9)
            for hour in range(24):
                # Gaussian-like distribution centered at peak
                distance = min(abs(hour - peak_center), 24 - abs(hour - peak_center))
                weights[hour] = np.exp(-0.5 * (distance / 2.5) ** 2)
            # Add small secondary peak around lunch
            for hour in range(11, 14):
                weights[hour] += 0.3 * np.exp(-0.5 * ((hour - 12) / 1.5) ** 2)
                
        elif self.user_type == "night_owl":
            # Peak between 8-11 PM
            peak_center = random.randint(20, 22)
            for hour in range(24):
                distance = min(abs(hour - peak_center), 24 - abs(hour - peak_center))
                weights[hour] = np.exp(-0.5 * (distance / 3) ** 2)
            # Add afternoon activity
            for hour in range(14, 18):
                weights[hour] += 0.25 * np.exp(-0.5 * ((hour - 16) / 2) ** 2)
                
        elif self.user_type == "regular":
            # Multiple peaks: morning commute, lunch, evening
            peaks = [9, 12, 18]
            for peak in peaks:
                for hour in range(24):
                    distance = min(abs(hour - peak), 24 - abs(hour - peak))
                    weights[hour] += 0.6 * np.exp(-0.5 * (distance / 2) ** 2)
            # General daytime activity
            for hour in range(8, 20):
                weights[hour] += 0.2
                
        else:  # sporadic
            # Random peaks throughout the day
            num_peaks = random.randint(2, 4)
            peak_hours = random.sample(range(8, 23), num_peaks)
            for peak in peak_hours:
                for hour in range(24):
                    distance = min(abs(hour - peak), 24 - abs(hour - peak))
                    weights[hour] += np.exp(-0.5 * (distance / 2.5) ** 2)
            # Add random noise
            weights += np.random.uniform(0, 0.2, 24)
        
        # Ensure very low activity during sleep hours (1-5 AM)
        for hour in range(1, 6):
            weights[hour] *= 0.1
        
        # Normalize weights to have max = base_engagement
        if weights.max() > 0:
            weights = (weights / weights.max()) * self.base_engagement
        
        return weights.tolist()

    def define_user(self):
        """
        Returns self for chaining (backward compatibility)
        """
        return self


class Simulation:
    """
    Simulates the users
    """
    def __init__(self, num_users):
        self.users = []
        for i in range(num_users):
            user = User(i+1).define_user()
            self.users.append(user)
        self.save_user_data()

    def save_user_data(self):
        """
        Saves the user's data to a CSV file
        """
        user_df = pd.DataFrame([{
            "name": user.name,
            "user_type": user.user_type,
            "base_engagement": user.base_engagement,
            "weekend_modifier": user.weekend_modifier,
            "hourly_weights": json.dumps(user.hourly_weights)  # Store as JSON string
        } for user in self.users])

        os.makedirs("data", exist_ok=True)
        user_df.to_csv("data/user_data.csv", index=False)

    @staticmethod
    def load_user_data():
        """
        Loads the user's data from a CSV file
        """
        df = pd.read_csv("data/user_data.csv")
        # Parse hourly_weights back from JSON
        df['hourly_weights'] = df['hourly_weights'].apply(json.loads)
        return df


if __name__ == "__main__":
    simulation = Simulation(100)
    print("Generated 100 users with realistic hourly engagement curves")
