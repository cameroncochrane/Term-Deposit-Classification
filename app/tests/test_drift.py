from monitoring.drift_report import MIN_SAMPLE_SIZE, load_reference, run_drift_check

REFERENCE = load_reference()


def test_no_drift_on_a_sample_of_the_reference_itself():
    current = REFERENCE.sample(200, random_state=1).reset_index(drop=True)

    result = run_drift_check(REFERENCE, current)

    assert result.dataset_drift is False
    assert result.drifted_columns == []


def test_simulated_shift_triggers_drift():
    """
    The "simulate feature distribution shift to prove alerts fire correctly" check from the
    deployment plan - deliberately skews duration and job away from the training distribution
    and confirms the drift check actually catches it, not just that it runs without error.
    """
    current = REFERENCE.sample(200, random_state=1).reset_index(drop=True).copy()
    current["duration"] = current["duration"] + 5000
    current["job"] = "student"

    result = run_drift_check(REFERENCE, current)

    assert result.dataset_drift is True
    assert "duration" in result.drifted_columns
    assert "job" in result.drifted_columns
    assert result.report_html is not None


def test_small_sample_is_skipped_not_flagged():
    current = REFERENCE.sample(MIN_SAMPLE_SIZE - 1, random_state=1).reset_index(drop=True)

    result = run_drift_check(REFERENCE, current)

    assert result.dataset_drift is False
    assert result.report_html is None
