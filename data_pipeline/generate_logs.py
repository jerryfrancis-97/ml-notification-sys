import pandas as pd
import numpy as np
import random
import os
import json
from collections import defaultdict
from datetime import timedelta


class ControlPolicy:
    """
    Controls the policy for sending notifications to users
    """
    
    @staticmethod
    def fixed_policy():
        """Give out notification at 9am always"""
        return 9
    
    @staticmethod
    def random_policy():
        """Random hour between 7am and 10pm for training data coverage"""
        return random.randint(7, 22)
    
    @staticmethod
    def weighted_random_policy():
        """
        Weighted random - more likely during typical active hours
        Ensures good coverage while being somewhat realistic
        """
        # Weight distribution favoring daytime hours
        hours = list(range(24))
        weights = [
            0.5, 0.2, 0.1, 0.1, 0.1, 0.3,  # 0-5 AM (low)
            0.8, 1.5, 2.0, 2.5, 2.5, 2.0,  # 6-11 AM (rising)
            2.5, 2.5, 2.0, 2.0, 2.0, 2.5,  # 12-5 PM (high)
            3.0, 3.0, 2.5, 2.0, 1.5, 1.0   # 6-11 PM (evening peak, then declining)
        ]
        return random.choices(hours, weights=weights, k=1)[0]


class UserHistory:
    """
    Tracks notification and engagement history per user
    """
    
    def __init__(self):
        # Track all notifications: {user_id: [(day, hour, opened), ...]}
        self.notifications = defaultdict(list)
    
    def record_notification(self, user_id, day, hour, opened):
        """Record a notification event"""
        self.notifications[user_id].append({
            'day': day,
            'hour': hour,
            'opened': opened
        })
    
    def get_notifications_last_24h(self, user_id, current_day, current_hour):
        """Count notifications sent in the last 24 hours"""
        count = 0
        for notif in self.notifications[user_id]:
            hours_ago = (current_day - notif['day']) * 24 + (current_hour - notif['hour'])
            if 0 < hours_ago <= 24:
                count += 1
        return count
    
    def get_hours_since_last_notification(self, user_id, current_day, current_hour):
        """Get hours since last notification (None if no previous)"""
        if not self.notifications[user_id]:
            return None
        
        last = self.notifications[user_id][-1]
        hours = (current_day - last['day']) * 24 + (current_hour - last['hour'])
        return max(0, hours)
    
    def get_opens_last_7_days(self, user_id, current_day):
        """Count opens in the last 7 days"""
        count = 0
        for notif in self.notifications[user_id]:
            if current_day - notif['day'] <= 7 and notif['opened']:
                count += 1
        return count
    
    def get_total_notifications(self, user_id):
        """Get total notifications received by user"""
        return len(self.notifications[user_id])


class LogGenerator:
    """
    Generates realistic send and response logs for notification ML training
    """

    def __init__(self, user_data_path, logs_path, num_days=30, policy='weighted_random',
                 start_date="2025-01-01", open_cutoff_minutes=480):
        self.logs_path = logs_path
        self.user_data = pd.read_csv(user_data_path)
        # Parse hourly_weights from JSON string
        self.user_data['hourly_weights'] = self.user_data['hourly_weights'].apply(json.loads)

        self.num_days = num_days
        self.policy = policy
        self.start_date = pd.Timestamp(start_date)
        self.open_cutoff_minutes = open_cutoff_minutes
        self.history = UserHistory()

        self._generate_logs()
        self.save_logs()

    def _get_send_hour(self):
        """Get the hour to send notification based on policy"""
        if self.policy == 'fixed':
            return ControlPolicy.fixed_policy()
        elif self.policy == 'random':
            return ControlPolicy.random_policy()
        else:  # weighted_random
            return ControlPolicy.weighted_random_policy()

    def _calculate_open_probability(self, user_row, hour, day, is_weekend):
        """
        Calculate probability of user opening notification
        Factors: hourly curve, weekend modifier, fatigue, recency
        """
        user_id = user_row['name']
        hourly_weights = user_row['hourly_weights']
        weekend_modifier = user_row['weekend_modifier']
        
        # Base probability from hourly curve
        base_prob = hourly_weights[hour]
        
        # Apply weekend modifier
        if is_weekend:
            base_prob *= weekend_modifier
        
        # Apply fatigue penalty (0.9^notifications_last_24h)
        notifs_24h = self.history.get_notifications_last_24h(user_id, day, hour)
        fatigue_penalty = 0.9 ** notifs_24h
        base_prob *= fatigue_penalty
        
        # Apply recency boost (users not notified recently are more likely to engage)
        hours_since_last = self.history.get_hours_since_last_notification(user_id, day, hour)
        if hours_since_last is not None:
            if hours_since_last > 48:  # More than 2 days - boost engagement
                recency_boost = min(1.3, 1 + (hours_since_last - 48) / 100)
            elif hours_since_last < 6:  # Very recent - reduce engagement
                recency_boost = 0.7
            else:
                recency_boost = 1.0
            base_prob *= recency_boost
        
        # Clamp probability to [0, 1]
        return max(0, min(1, base_prob))

    def _generate_logs(self):
        """Generate all send and response logs"""
        send_logs = []
        response_logs = []
        
        # Start simulation from a Monday (day 0 = Monday)
        for day in range(self.num_days):
            day_of_week = day % 7  # 0=Mon, 1=Tue, ..., 6=Sun
            is_weekend = day_of_week >= 5  # Saturday=5, Sunday=6
            
            for _, user_row in self.user_data.iterrows():
                user_id = user_row['name']
                hour = self._get_send_hour()
                event_id = f"{user_id}_day{day}_h{hour}"

                # Derive send_timestamp with sub-hour jitter
                random_minute = random.randint(0, 59)
                send_timestamp = self.start_date + timedelta(
                    days=day, hours=hour, minutes=random_minute
                )

                # Send log entry
                send_logs.append({
                    "event_id": event_id,
                    "user_id": user_id,
                    "day": day,
                    "hour": hour,
                    "day_of_week": day_of_week,
                    "is_weekend": int(is_weekend),
                    "send_timestamp": send_timestamp,
                })

                # Calculate open probability and determine outcome
                open_prob = self._calculate_open_probability(user_row, hour, day, is_weekend)
                engaged = random.random() < open_prob

                if engaged:
                    # Response delay follows log-normal distribution
                    response_delay = int(np.random.lognormal(mean=2, sigma=1.5))
                    response_delay = min(response_delay, 1440)  # Cap at 24 hours
                    opened = 1 if response_delay <= self.open_cutoff_minutes else 0
                    open_timestamp = (
                        send_timestamp + timedelta(minutes=response_delay)
                        if opened else pd.NaT
                    )
                else:
                    response_delay = -1
                    opened = 0
                    open_timestamp = pd.NaT

                # Response log entry
                response_logs.append({
                    "event_id": event_id,
                    "opened": int(opened),
                    "response_delay_minutes": response_delay,
                    "open_timestamp": open_timestamp,
                })

                # Update history AFTER generating the log
                self.history.record_notification(user_id, day, hour, opened)
        
        self.send_logs = pd.DataFrame(send_logs)
        self.response_logs = pd.DataFrame(response_logs)

    def save_logs(self):
        """Save the logs to CSV files"""
        os.makedirs(self.logs_path, exist_ok=True)
        self.send_logs.to_csv(os.path.join(self.logs_path, "send_logs.csv"), index=False)
        self.response_logs.to_csv(os.path.join(self.logs_path, "response_logs.csv"), index=False)
        
        # Also save a merged version for convenience
        merged = pd.merge(self.send_logs, self.response_logs, on='event_id')
        merged.to_csv(os.path.join(self.logs_path, "training_data.csv"), index=False)
        
        print(f"Generated {len(self.send_logs)} notification events")
        print(f"Open rate: {self.response_logs['opened'].mean():.2%}")
        print(f"Logs saved to {self.logs_path}/")


if __name__ == "__main__":
    user_data_path = "data/user_data.csv"
    logs_path = "data/logs"

    log_generator = LogGenerator(
        user_data_path, logs_path,
        num_days=30, policy='weighted_random',
        start_date="2025-01-01", open_cutoff_minutes=480
    )
