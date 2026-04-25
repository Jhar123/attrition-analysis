import pandas as pd
from src.metrics import (
    attrition_rate,
    attrition_by_department,
    attrition_by_overtime,
    average_income_by_attrition,
    satisfaction_summary,
)


# --- shared fixtures ---

def make_df():
    """Small controlled dataset used across multiple tests."""
    return pd.DataFrame({
        "employee_id": [1, 2, 3, 4, 5, 6],
        "department":  ["Sales", "Sales", "Sales", "HR", "HR", "HR"],
        "overtime":    ["Yes", "Yes", "No", "Yes", "No", "No"],
        "monthly_income": [4000, 5000, 6000, 3000, 7000, 8000],
        "job_satisfaction": [1, 2, 2, 1, 3, 3],
        "attrition": ["Yes", "Yes", "No", "Yes", "No", "No"],
    })


# --- attrition_rate ---

def test_attrition_rate_returns_expected_percent():
    df = pd.DataFrame({
        "employee_id": [1, 2, 3, 4],
        "department": ["Sales", "Sales", "HR", "HR"],
        "attrition": ["Yes", "No", "No", "Yes"],
    })
    assert attrition_rate(df) == 50.0


def test_attrition_rate_all_leavers():
    df = pd.DataFrame({
        "employee_id": [1, 2],
        "attrition": ["Yes", "Yes"],
    })
    assert attrition_rate(df) == 100.0


def test_attrition_rate_no_leavers():
    df = pd.DataFrame({
        "employee_id": [1, 2],
        "attrition": ["No", "No"],
    })
    assert attrition_rate(df) == 0.0


# --- attrition_by_department ---

def test_attrition_by_department_returns_expected_columns():
    df = pd.DataFrame({
        "employee_id": [1, 2, 3, 4],
        "department": ["Sales", "Sales", "HR", "HR"],
        "attrition": ["Yes", "No", "No", "Yes"],
    })
    result = attrition_by_department(df)
    assert list(result.columns) == ["department", "employees", "leavers", "attrition_rate"]


def test_attrition_by_department_calculates_correct_rates():
    df = make_df()
    result = attrition_by_department(df)
    sales_row = result[result["department"] == "Sales"].iloc[0]
    hr_row = result[result["department"] == "HR"].iloc[0]
    # Sales: 2 leavers out of 3 = 66.67%
    assert sales_row["employees"] == 3
    assert sales_row["leavers"] == 2
    assert sales_row["attrition_rate"] == 66.67
    # HR: 1 leaver out of 3 = 33.33%
    assert hr_row["employees"] == 3
    assert hr_row["leavers"] == 1
    assert hr_row["attrition_rate"] == 33.33


def test_attrition_by_department_sorted_descending():
    df = make_df()
    result = attrition_by_department(df)
    rates = list(result["attrition_rate"])
    assert rates == sorted(rates, reverse=True)


# --- attrition_by_overtime ---

def test_attrition_by_overtime_returns_expected_columns():
    result = attrition_by_overtime(make_df())
    assert list(result.columns) == ["overtime", "employees", "leavers", "attrition_rate"]


def test_attrition_by_overtime_calculates_correct_rates():
    df = make_df()
    result = attrition_by_overtime(df)
    yes_row = result[result["overtime"] == "Yes"].iloc[0]
    no_row = result[result["overtime"] == "No"].iloc[0]
    # Overtime Yes: employees 1(Yes),2(Yes),4(Yes) — 3 employees, 3 leavers = 100%
    assert yes_row["employees"] == 3
    assert yes_row["leavers"] == 3
    assert yes_row["attrition_rate"] == 100.0
    # Overtime No: employees 3(No),5(No),6(No) — 3 employees, 0 leavers = 0%
    assert no_row["employees"] == 3
    assert no_row["leavers"] == 0
    assert no_row["attrition_rate"] == 0.0


# --- average_income_by_attrition ---

def test_average_income_by_attrition_returns_expected_columns():
    result = average_income_by_attrition(make_df())
    assert list(result.columns) == ["attrition", "avg_monthly_income"]


def test_average_income_by_attrition_calculates_correct_means():
    df = make_df()
    result = average_income_by_attrition(df)
    yes_row = result[result["attrition"] == "Yes"].iloc[0]
    no_row = result[result["attrition"] == "No"].iloc[0]
    # Leavers: incomes 4000, 5000, 3000 → mean = 4000.0
    assert yes_row["avg_monthly_income"] == 4000.0
    # Stayers: incomes 6000, 7000, 8000 → mean = 7000.0
    assert no_row["avg_monthly_income"] == 7000.0


# --- satisfaction_summary ---

def test_satisfaction_summary_returns_expected_columns():
    result = satisfaction_summary(make_df())
    assert list(result.columns) == ["job_satisfaction", "total_employees", "leavers", "attrition_rate"]


def test_satisfaction_summary_uses_group_size_as_denominator():
    """
    Validates the corrected formula: leavers / total_employees per group.

    With the old (wrong) formula — leavers / total_company_leavers — sat=1
    would show 66.67% (2 of 3 leavers), not the true rate of 100% (2 of 2
    employees at that level). This test catches that regression.
    """
    df = make_df()
    result = satisfaction_summary(df)
    sat1 = result[result["job_satisfaction"] == 1].iloc[0]
    sat2 = result[result["job_satisfaction"] == 2].iloc[0]
    sat3 = result[result["job_satisfaction"] == 3].iloc[0]
    # sat=1: employees 1(Yes),4(Yes) — 2 of 2 left = 100%
    assert sat1["total_employees"] == 2
    assert sat1["leavers"] == 2
    assert sat1["attrition_rate"] == 100.0
    # sat=2: employees 2(Yes),3(No) — 1 of 2 left = 50%
    assert sat2["total_employees"] == 2
    assert sat2["leavers"] == 1
    assert sat2["attrition_rate"] == 50.0
    # sat=3: employees 5(No),6(No) — 0 of 2 left = 0%
    assert sat3["total_employees"] == 2
    assert sat3["leavers"] == 0
    assert sat3["attrition_rate"] == 0.0


def test_satisfaction_summary_sorted_by_satisfaction_ascending():
    result = satisfaction_summary(make_df())
    levels = list(result["job_satisfaction"])
    assert levels == sorted(levels)
