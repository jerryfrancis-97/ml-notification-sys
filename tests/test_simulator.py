"""Unit tests for data_pipeline.simulator."""

import json
import os

import pytest

from data_pipeline.simulator import Simulation, User


class TestUser:
    """Tests for the User class."""

    def test_user_creation(self):
        """Verify User attributes are valid."""
        user = User(1)
        assert user.id == 1
        assert user.name == "U_1"
        assert user.user_type in ["early_bird", "night_owl", "regular", "sporadic"]
        assert 0.3 <= user.base_engagement <= 0.9
        assert len(user.hourly_weights) == 24

    def test_user_types(self):
        """Confirm user_type is one of the valid types."""
        valid_types = {"early_bird", "night_owl", "regular", "sporadic"}
        for _ in range(20):
            user = User(1)
            assert user.user_type in valid_types

    def test_hourly_weights_shape(self):
        """All 24 weights are non-negative and max <= base_engagement."""
        user = User(1)
        weights = user.hourly_weights
        assert len(weights) == 24
        assert all(w >= 0 for w in weights)
        if max(weights) > 0:
            assert max(weights) <= user.base_engagement + 1e-6  # small tolerance

    def test_weekend_modifier_ranges(self):
        """Each user type produces a modifier in the expected range."""
        ranges = {
            "early_bird": (0.7, 0.9),
            "night_owl": (1.1, 1.4),
            "regular": (0.8, 1.0),
            "sporadic": (0.9, 1.3),
        }
        for _ in range(50):
            user = User(1)
            low, high = ranges[user.user_type]
            assert low <= user.weekend_modifier <= high


class TestSimulation:
    """Tests for the Simulation class."""

    def test_simulation_creates_users(self, tmp_path):
        """Simulation(n) creates n users."""
        sim = Simulation(10, output_dir=str(tmp_path))
        assert len(sim.users) == 10

    def test_simulation_saves_csv(self, tmp_path):
        """Verify user_data.csv is created with correct columns."""
        Simulation(5, output_dir=str(tmp_path))
        csv_path = tmp_path / "user_data.csv"
        assert csv_path.exists()
        import pandas as pd
        df = pd.read_csv(csv_path)
        expected_cols = {"name", "user_type", "base_engagement", "weekend_modifier", "hourly_weights"}
        assert set(df.columns) == expected_cols
        assert len(df) == 5

    def test_load_user_data(self, tmp_path):
        """Load saved CSV and verify hourly_weights is parsed back into lists."""
        Simulation(3, output_dir=str(tmp_path))
        path = tmp_path / "user_data.csv"
        df = Simulation.load_user_data(path=str(path))
        assert "hourly_weights" in df.columns
        for val in df["hourly_weights"]:
            assert isinstance(val, list)
            assert len(val) == 24
