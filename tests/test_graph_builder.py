import pandas as pd
from src.graph_builder import completed_pass_mask, calculate_pass_angle


def test_completed_pass_requires_null_outcome():
    events = pd.DataFrame({
        "team": ["A", "A"], "type": ["Pass", "Pass"], "pass_recipient": ["P2", "P3"],
        "pass_outcome": [None, "Incomplete"], "pass_end_location": [[20, 20], [30, 30]],
    })
    assert completed_pass_mask(events, "A").tolist() == [True, False]


def test_horizontal_pass_angle():
    row = pd.Series({"location": [10.0, 20.0], "pass_end_location": [30.0, 20.0], "pass_angle": None})
    assert abs(calculate_pass_angle(row)) < 1e-9
