from rag.verification import unverified_numbers, verification_issues, verification_warnings
from rag.ingest import Chunk
from rag.retriever import Hit


def _hit(text: str, rank: int = 1) -> Hit:
    chunk = Chunk(
        chunk_id=f"chunk-{rank}", doc_id="policy", doc_title="Policy",
        source="policy.md", section="Rates", text=text,
    )
    return Hit(chunk=chunk, score=0.9, rank=rank)


def test_number_must_match_the_product_in_its_cited_passage():
    source = "Savings rate: 3%; personal loan rate: 12%."

    assert unverified_numbers("Savings rate is 12% [1].", [_hit(source)]) == ["12"]
    assert unverified_numbers("Savings rate is 3% [1].", [_hit(source)]) == []


def test_single_digit_numbers_are_checked():
    assert unverified_numbers("Savings rate is 4% [1].", [_hit("Savings rate is 3%.")]) == ["4"]


def test_number_units_must_be_consistent():
    assert unverified_numbers(
        "Savings rate is 3 years [1].", [_hit("Savings rate is 3%.")]
    ) == ["3"]
    assert unverified_numbers(
        "The rate is 2% per year [1].", [_hit("The rate is 2% per month.")]
    ) == ["2"]
    warnings = verification_warnings(
        "The amount is ₹200 per year [1].",
        [_hit("The amount is ₹200 per month.")],
    )
    assert any("₹200 is not verified" in warning for warning in warnings)


def test_missing_and_invalid_citations_are_reported():
    missing = verification_warnings("Savings rate is 3%.", [_hit("Savings rate is 3%.")])
    invalid = verification_warnings("Savings rate is 3% [4].", [_hit("Savings rate is 3%.")])

    assert any("missing a source citation" in warning for warning in missing)
    assert any("Citation [4]" in warning for warning in invalid)
    assert any("3%" in warning for warning in invalid)


def test_wide_citation_brackets_are_recognized_and_validated():
    source = [_hit("Savings rate is 3%.")]

    assert unverified_numbers("Savings rate is 3% 【1】.", source) == []
    warnings = verification_warnings("Savings rate is 3% 【2】.", source)
    assert any("Citation [2]" in warning for warning in warnings)


def test_numbers_in_an_uncited_passage_do_not_verify_a_claim():
    hits = [_hit("Savings rate is 3%.", rank=1), _hit("Loan rate is 12%.", rank=2)]

    assert unverified_numbers("Savings rate is 12% [1].", hits) == ["12"]
    assert unverified_numbers("Loan rate is 12% [2].", hits) == []


def test_conditions_must_match_the_cited_figure():
    source = "Women borrowers receive a 4% rate."

    assert unverified_numbers("Senior citizens receive a 4% rate [1].", [_hit(source)]) == ["4"]
    assert unverified_numbers("Women borrowers receive a 4% rate [1].", [_hit(source)]) == []


def test_range_endpoints_support_equivalent_minimum_and_maximum_claims():
    source = [_hit("Age eligibility is 21 years to 60 years at application.")]

    assert unverified_numbers(
        "Applicants must be at least 21 years old [1].", source
    ) == []
    assert unverified_numbers(
        "Applicants may be no older than 60 years [1].", source
    ) == []


def test_tenure_range_lower_and_upper_endpoints_keep_their_own_conditions():
    source = [_hit("Minimum fixed deposit amount is ₹10,000; tenure ranges from 7 days to 10 years.")]

    assert unverified_numbers(
        "The minimum is ₹10,000 and tenure is from 7 days up to 10 years [1].",
        source,
    ) == []


def test_table_headers_keep_product_and_condition_associated_with_figures():
    source = (
        "Savings account minimum balance by branch:\n"
        "| Product | Metro / Urban | Semi-urban | Rural |\n"
        "|---|---|---|---|\n"
        "| Regular Savings | ₹5,000 | ₹2,500 | ₹1,000 |"
    )

    assert unverified_numbers("Regular Savings needs ₹5,000 in rural branches [1].", [_hit(source)]) == [
        "5,000"
    ]
    assert unverified_numbers("Regular Savings needs ₹1,000 in rural branches [1].", [_hit(source)]) == []


def test_table_rate_frequency_is_inherited_from_header():
    source = (
        "| Product | Interest (p.a.) |\n"
        "|---|---|\n"
        "| Savings | 3% |"
    )

    assert unverified_numbers(
        "Savings interest is 3% per year [1].", [_hit(source)]
    ) == []


def test_customer_provided_amount_is_not_mistaken_for_a_policy_figure():
    source = [_hit("Late fee is ₹750 for amounts due between ₹5,001 and ₹20,000.")]

    warnings = verification_warnings(
        "A ₹10,000 amount due has a ₹750 late fee [1].",
        source,
        question="What fee applies to my ₹10,000 amount due?",
    )

    assert warnings == []


def test_number_in_question_does_not_verify_a_policy_rate_claim():
    warnings = verification_warnings(
        "The savings rate is 12% [1].",
        [_hit("Savings rate is 3%.")],
        question="Is the savings rate 12%?",
    )

    assert any("12%" in warning and "not verified" in warning for warning in warnings)


def test_customer_amount_due_is_input_but_interest_rate_still_requires_support():
    source = [_hit("Credit-card late fee is ₹750 for amounts due from ₹5,001 to ₹20,000.")]

    assert verification_warnings(
        "For an amount due of ₹10,000, the fee is ₹750 [1].",
        source,
        question="What fee applies to my amount due of ₹10,000?",
    ) == []
    assert unverified_numbers(
        "The credit-card interest rate is 12% [1].",
        [_hit("Credit-card interest rate is 3.6% per month.")],
    ) == ["12"]


def test_verification_normalizes_markdown_nonbreaking_spaces_and_percent_spacing():
    warnings = verification_warnings(
        "The **senior-citizen rate** is **7.50 %** per annum [1].",
        [_hit("Senior-citizen rate is 7.50% per annum.")],
    )

    assert warnings == []


def test_currency_magnitudes_are_equivalent():
    assert unverified_numbers(
        "The deposit limit is ₹30,000,000 [1].",
        [_hit("The deposit limit is ₹3 crore.")],
    ) == []


def test_below_and_up_to_are_not_treated_as_equivalent():
    warnings = verification_warnings(
        "Fixed deposits are available up to ₹3 crore [1].",
        [_hit("Fixed deposits are available below ₹3 crore.")],
    )

    assert any("₹3" in warning and "not verified" in warning for warning in warnings)


def test_missing_citations_and_unsupported_numbers_are_distinct():
    issues = verification_issues(
        "The savings rate is 12%.",
        [_hit("Savings rate is 3%.")],
    )

    assert issues["missing_citations"]
    assert issues["unsupported_numbers"]
    assert issues["invalid_citations"] == []


def test_evaluation_formatting_and_table_claims_do_not_raise_false_warnings():
    fd_source = (
        "Interest rates on domestic fixed deposits below ₹3 crore:\n"
        "| Tenor | General public (p.a.) | Senior citizens (p.a.) |\n"
        "|---|---|---|\n"
        "| 1 year to less than 2 years | 7.00% | 7.50% |\n"
        "Senior citizens receive an additional 0.50% per annum."
    )
    card_source = (
        "| Total amount due | Late payment fee |\n|---|---|\n"
        "| ₹5,001 to ₹20,000 | ₹750 |"
    )
    upi_source = "Maximum UPI payment limit: ₹1,00,000 per day per customer."

    assert verification_warnings(
        "The senior-citizen FD rate is **7.50 %** per annum [1]. "
        "Deposits are below ₹30,000,000 [1].",
        [_hit(fd_source)],
    ) == []
    assert verification_warnings(
        "A ₹10,000 amount due has a ₹750 late-payment fee 【1】.",
        [_hit(card_source)],
        question="What fee applies to my amount due of ₹10,000?",
    ) == []
    assert verification_warnings(
        "The regular UPI limit is ₹1,00,000 per day [1].",
        [_hit(upi_source)],
    ) == []
    assert verification_warnings(
        "The regular daily UPI limit is ₹1,00,000 [1].",
        [_hit(upi_source)],
    ) == []


def test_period_abbreviations_do_not_detach_citations_from_claims():
    source = (
        "Interest rates on domestic fixed deposits below ₹3 crore:\n"
        "| Tenor | General public (p.a.) | Senior citizens (p.a.) |\n"
        "|---|---|---|\n"
        "| 1 year to less than 2 years | 7.00% | 7.50% |\n"
        "Senior citizens (60 years and above) receive an additional 0.50% per annum "
        "over the general rate."
    )
    answer = (
        "The fixed-deposit rate for the 1-year-to-less-than-2-years tenor is "
        "**7.00 % per annum for the general public and 7.50 % per annum for senior citizens**【1】.\n\n"
        "- General public: 7.00 % p.a. for 1 year to less than 2 years【1】\n"
        "- Senior citizens (60 years and above): 7.50 % p.a. for the same tenor, "
        "reflecting the additional 0.50 % benefit【1】"
    )

    assert verification_warnings(answer, [_hit(source)]) == []
