"""Unit tests for data_pipeline.generate_logs."""

import json

import pandas as pd
import pytest

from data_pipeline.generate_logs import ControlPolicy, LogGenerator, UserHistory


class TestControlPolicy:
    """Tests for ControlPolicy."""

    def test_control_policy_fixed(self):
        """ControlPolicy.fixed_policy() always returns 9."""
        for _ in range(10):
            assert ControlPolicy.fixed_policy() == 9

    def test_control_policy_random(self):
        """ControlPolicy.random_policy() returns value in [7, 22]."""
        for _ in range(20):
            h = ControlPolicy.random_policy()
            assert 7 <= h <= 22

    def test_control_policy_weighted(self):
        """ControlPolicy.weighted_random_policy() returns value in [0, 23]."""
        for _ in range(20):
            h = ControlPolicy.weighted_random_policy()
            assert 0 <= h <= 23


class TestUserHistory:
    """Tests for UserHistory."""

    def test_user_history_record_and_query(self):
        """Record notifications, verify query methods."""
        hist = UserHistory()
        hist.record_notification("U_1", day=0, hour=10, opened=True)
        hist.record_notification("U_1", day=0, hour=11, opened=False)
        hist.record_notification("U_1", day=1, hour=9, opened=True)
        hist.record_notification("U_1", day=8, hour=10, opened=True)

        # At (0, 12): 2 notifications in last 24h (at 10 and 11)
        assert hist.get_notifications_last_24h("U_1", 0, 12) == 2
        # At (9, 10): last notification was (8, 10), so 24 hours ago
        assert hist.get_hours_since_last_notification("U_1", 9, 10) == 24
        # Opens in last 7 days when at day 2: (0,10), (1,9), (8,10) match day diff <= 7
        assert hist.get_opens_last_7_days("U_1", 2) == 3
        assert hist.get_total_notifications("U_1") == 4


class TestLogGenerator:
    """Tests for LogGenerator with timestamps and cutoff."""

    @pytest.fixture
    def minimal_user_data(self, tmp_path):
        """Create minimal user_data.csv for testing."""
        data = [
            {
                "name": "U_1",
                "user_type": "regular",
                "base_engagement": 0.5,
                "weekend_modifier": 1.0,
                "hourly_weights": json.dumps([0.1] * 24),
            },
            {
                "name": "U_2",
                "user_type": "night_owl",
                "base_engagement": 0.6,
                "weekend_modifier": 1.2,
                "hourly_weights": json.dumps([0.2] * 24),
            },
        ]
        df = pd.DataFrame(data)
        path = tmp_path / "user_data.csv"
        df.to_csv(path, index=False)
        return path

    def test_log_generator_output_columns(self, minimal_user_data, tmp_path):
        """Run LogGenerator and verify output column schema."""
        logs_path = tmp_path / "logs"
        gen = LogGenerator(
            str(minimal_user_data),
            str(logs_path),
            num_days=2,
            policy="fixed",
            start_date="2025-01-01",
            open_cutoff_minutes=480,
        )
        send_cols = set(gen.send_logs.columns)
        resp_cols = set(gen.response_logs.columns)
        assert send_cols >= {
            "event_id", "user_id", "day", "hour", "day_of_week", "is_weekend", "send_timestamp"
        }
        assert resp_cols >= {"event_id", "opened", "response_delay_minutes", "open_timestamp"}

    def test_send_timestamp_derived_correctly(self, minimal_user_data, tmp_path):
        """Verify send_timestamp dates fall within [start_date, start_date + num_days)."""
        logs_path = tmp_path / "logs"
        gen = LogGenerator(
            str(minimal_user_data),
            str(logs_path),
            num_days=2,
            policy="fixed",
            start_date="2025-01-01",
            open_cutoff_minutes=480,
        )
        start = pd.Timestamp("2025-01-01")
        end = pd.Timestamp("2025-01-03")
        for ts in gen.send_logs["send_timestamp"]:
            ts = pd.Timestamp(ts)
            assert start <= ts < end

    def test_open_timestamp_only_when_opened(self, minimal_user_data, tmp_path):
        """opened=0 -> NaT for open_timestamp; opened=1 -> valid timestamp."""
        logs_path = tmp_path / "logs"
        gen = LogGenerator(
            str(minimal_user_data),
            str(logs_path),
            num_days=2,
            policy="fixed",
            start_date="2025-01-01",
            open_cutoff_minutes=480,
        )
        df = gen.response_logs
        for _, row in df.iterrows():
            if row["opened"] == 0:
                assert pd.isna(row["open_timestamp"])
            else:
                assert not pd.isna(row["open_timestamp"])

    def test_cutoff_window_applied(self, minimal_user_data, tmp_path):
        """No row with opened=1 has response_delay_minutes > open_cutoff_minutes."""
        cutoff = 480
        logs_path = tmp_path / "logs"
        gen = LogGenerator(
            str(minimal_user_data),
            str(logs_path),
            num_days=2,
            policy="fixed",
            start_date="2025-01-01",
            open_cutoff_minutes=cutoff,
        )
        opened_rows = gen.response_logs[gen.response_logs["opened"] == 1]
        for _, row in opened_rows.iterrows():
            assert row["response_delay_minutes"] <= cutoff

    def test_response_delay_range(self, minimal_user_data, tmp_path):
        """All response_delay_minutes are -1 or in [0, 1440]."""
        logs_path = tmp_path / "logs"
        gen = LogGenerator(
            str(minimal_user_data),
            str(logs_path),
            num_days=2,
            policy="fixed",
            start_date="2025-01-01",
            open_cutoff_minutes=480,
        )
        for val in gen.response_logs["response_delay_minutes"]:
            assert val == -1 or (0 <= val <= 1440)

    def test_merged_training_data(self, minimal_user_data, tmp_path):
        """Verify merged CSV has all expected columns from both logs."""
        logs_path = tmp_path / "logs"
        gen = LogGenerator(
            str(minimal_user_data),
            str(logs_path),
            num_days=2,
            policy="fixed",
            start_date="2025-01-01",
            open_cutoff_minutes=480,
        )
        merged_path = logs_path / "training_data.csv"
        assert merged_path.exists()
        merged = pd.read_csv(merged_path)
        expected = {
            "event_id", "user_id", "day", "hour", "day_of_week", "is_weekend",
            "send_timestamp", "opened", "response_delay_minutes", "open_timestamp",
        }
        assert expected <= set(merged.columns)
