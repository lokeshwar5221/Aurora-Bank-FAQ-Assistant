from rag.generator import unverified_numbers, verification_warnings
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
