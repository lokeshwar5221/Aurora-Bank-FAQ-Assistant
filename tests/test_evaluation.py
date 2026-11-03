from scripts.evaluate_live import CASES, _offline_demonstrations


def test_offline_evaluation_demonstrates_updates_conflicts_and_api_interruption():
    demonstrations = _offline_demonstrations()

    assert demonstrations["policy_update"]["pass"]
    assert demonstrations["policy_update"]["active_versions_today"] == ["1"]
    assert demonstrations["policy_update"]["active_versions_after_effective_date"] == ["2"]
    assert demonstrations["policy_conflict"]["pass"]
    assert demonstrations["policy_conflict"]["active_document_count"] == 0
    assert demonstrations["interrupted_api_response"]["pass"]
    assert demonstrations["interrupted_api_response"]["attempts"] == 1


def test_evaluation_case_questions_are_unique_and_credit_card_case_is_current():
    cases = {case["id"]: case for case in CASES}

    assert len(cases) == len(CASES)
    assert cases["credit-card-late-fee"]["question"] == (
        "What is the late-payment fee for a total amount due between ₹5,001 and ₹20,000?"
    )
    assert {"expected_refusal"} <= cases["unsupported-gold-loan-question"].keys()
    assert {"expected_refusal"} <= cases["personal-account-balance"].keys()
