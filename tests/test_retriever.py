from rag.retriever import contextualize_query


def test_explicit_follow_up_uses_previous_user_topic():
    history = [{"role": "user", "content": "What is the home loan interest rate?"}]

    assert contextualize_query("And for senior citizens?", history) == (
        "What is the home loan interest rate? And for senior citizens?"
    )
    assert contextualize_query("What about women?", history).startswith(
        "What is the home loan interest rate?"
    )


def test_short_independent_topic_is_not_combined_with_history():
    history = [{"role": "user", "content": "What is the home loan interest rate?"}]

    assert contextualize_query("What are ATM charges?", history) == "What are ATM charges?"
    assert contextualize_query("ATM fees?", history) == "ATM fees?"
    assert contextualize_query("What about ATM charges?", history) == "What about ATM charges?"


def test_question_without_referential_language_stays_standalone():
    history = [{"role": "user", "content": "What is the home loan interest rate?"}]

    assert contextualize_query("What is the minimum savings balance?", history) == (
        "What is the minimum savings balance?"
    )


def test_consecutive_followups_keep_the_resolved_topic():
    history = [
        {"role": "user", "content": "What is the home loan rate?"},
        {"role": "assistant", "content": "The rate is policy based."},
        {"role": "user", "content": "And for women?"},
        {"role": "assistant", "content": "The policy has no separate rate."},
    ]

    query = contextualize_query("And for senior citizens?", history)

    assert "home loan" in query
    assert "women?" not in query
    assert "senior citizens" in query


def test_followup_with_a_new_topic_is_independent():
    history = [{"role": "user", "content": "What is the home loan rate?"}]

    assert contextualize_query("And what are ATM charges?", history) == (
        "And what are ATM charges?"
    )
