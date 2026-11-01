"""Run explicitly opted-in live semantic retrieval and LLM evaluations."""
from __future__ import annotations

import argparse
import json
import re
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from rag.config import POLICY_DIR, get_llm_settings  # noqa: E402
from rag.generator import LLMClient, LLMError, NOT_FOUND_MESSAGE, stream_answer, verification_warnings  # noqa: E402
from rag.ingest import Chunk  # noqa: E402
from rag.pipeline import KnowledgeBase  # noqa: E402
from rag.retriever import Hit  # noqa: E402

CASES = [
    {
        "id": "savings-minimum-balance",
        "question": "What is the minimum balance for a Regular Savings account?",
        "expected_answer": "₹5,000 in metro/urban areas, ₹2,500 in semi-urban areas and ₹1,000 in rural areas.",
        "expected_source": "Savings Account Policy",
        "required_terms": ["5,000", "2,500", "1,000"],
    },
    {
        "id": "senior-citizen-fd-rate",
        "question": "What is the senior-citizen fixed-deposit rate for 1 year to less than 2 years?",
        "expected_answer": "The senior-citizen rate for that tenor is 7.50% per annum.",
        "expected_source": "Fixed Deposit and Recurring Deposit Policy",
        "required_terms": ["7.50%"],
    },
    {
        "id": "personal-loan-age",
        "question": "What is the age eligibility for a salaried personal loan?",
        "expected_answer": "Applicants must be 21 to 60 years old at application and no older than 65 at maturity.",
        "expected_source": "Personal Loan Policy",
        "required_terms": ["21", "60", "65"],
    },
    {
        "id": "home-loan-processing-fee",
        "question": "What is the processing fee for a home loan?",
        "expected_answer": "0.50% of the loan amount, with a ₹5,000 minimum and ₹15,000 maximum, plus GST.",
        "expected_source": "Home Loan Policy",
        "required_terms": ["0.50%", "5,000", "15,000"],
    },
    {
        "id": "credit-card-late-fee",
        "question": "What is the late-payment fee for a total amount due between ₹5,001 and ₹20,000?",
        "expected_answer": "The late fee is ₹750 for a total amount due from ₹5,001 to ₹20,000.",
        "expected_source": "Credit Card Policy",
        "required_terms": ["750", "5,001", "20,000"],
    },
    {
        "id": "account-opening-documents",
        "question": "Which documents are required to open a savings account?",
        "expected_answer": "Identity/address proof, PAN or Form 60, a recent photograph, and an Aadhaar-linked mobile number.",
        "expected_source": "KYC and Account Opening Policy",
        "required_terms": ["Aadhaar", "PAN", "photograph"],
    },
    {
        "id": "other-bank-atm-limit",
        "question": "How many free transactions do I get at other banks' ATMs?",
        "expected_answer": "Five monthly transactions are free, or three in the listed metro cities.",
        "expected_source": "Schedule of Fees",
        "required_terms": ["5", "3", "per month"],
    },
    {
        "id": "unauthorized-transaction-reporting",
        "question": "How do I report an unauthorized transaction?",
        "expected_answer": "Report it through the app, net banking, or the 24x7 helpline 1800-120-4567.",
        "expected_source": "Digital Banking",
        "required_terms": ["app", "net banking", "1800-120-4567"],
    },
    {
        "id": "upi-daily-limit",
        "question": "What is the regular daily UPI payment limit?",
        "expected_answer": "₹1,00,000 and 20 transactions per customer per day.",
        "expected_source": "Digital Banking",
        "required_terms": ["1,00,000", "20 UPI transactions"],
    },
    {
        "id": "home-loan-tenure",
        "question": "What is the maximum home-loan tenure for salaried and self-employed applicants?",
        "expected_answer": "30 years for salaried applicants and 25 years for self-employed applicants.",
        "expected_source": "Home Loan Policy",
        "required_terms": ["30 years", "25 years"],
    },
]
JUDGMENT_VERSION = "citation-scope-v3"


def _normalise(text: str) -> str:
    text = re.sub(r"(?<=\d),(?=\d)", "", text.casefold())
    return re.sub(r"[^a-z0-9]+", " ", text).strip()


def _record_sources(hits: list[Hit], answerable: bool) -> list[dict]:
    sources = hits if answerable else hits[:3]
    return [
        {
            "citation": hit.rank,
            "source_id": hit.chunk.doc_id,
            "title": hit.chunk.doc_title,
            "source": hit.chunk.source,
            "section": hit.chunk.section,
            "version": hit.chunk.version,
            "effective_date": hit.chunk.effective_date,
            "score": round(hit.score, 4),
            "passage": hit.chunk.text,
        }
        for hit in sources
    ]


def _judge(record: dict, case: dict, hits: list[Hit]) -> dict:
    answer = record["observed_output"]
    warnings = (
        verification_warnings(answer, hits, record["question"])
        if answer and record.get("answerable", True) else []
    )
    normalized_answer = _normalise(answer)
    missing_terms = [
        term for term in case["required_terms"]
        if _normalise(term) not in normalized_answer
    ]
    source_found = record.get("answerable", True) and any(
        case["expected_source"].casefold() in source["title"].casefold()
        for source in record["retrieved_sources"]
    )
    answer_pass = record.get("answerable", True) and not missing_terms and not record.get("error")
    record["required_terms"] = case["required_terms"]
    record["verification_warnings"] = warnings
    record["judgment"] = {
        "expected_terms_present": not missing_terms,
        "missing_terms": missing_terms,
        "expected_source_retrieved": source_found,
        "citation_and_number_checks_passed": not warnings,
        "pass": answer_pass and source_found and not warnings,
    }
    return record["judgment"]


def rejudge(output_path: Path) -> dict:
    report = json.loads(output_path.read_text(encoding="utf-8"))
    cases = {case["id"]: case for case in CASES}
    for record in report["evaluations"]:
        case = cases[record["id"]]
        hits = [
            Hit(
                chunk=Chunk(
                    chunk_id=f"eval-{source['citation']}",
                    doc_id=source["source_id"],
                    doc_title=source["title"],
                    source=source["source"],
                    section=source["section"],
                    text=source["passage"],
                    version=source.get("version"),
                    effective_date=source.get("effective_date"),
                ),
                score=source["score"],
                rank=source["citation"],
            )
            for source in record["retrieved_sources"]
        ]
        _judge(record, case, hits)
    report["passed_cases"] = sum(record["judgment"]["pass"] for record in report["evaluations"])
    report["failed_cases"] = sum(not record["judgment"]["pass"] for record in report["evaluations"])
    report["judged_at"] = datetime.now(timezone.utc).isoformat()
    report["judgment_version"] = JUDGMENT_VERSION
    output_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return report


def run(output_path: Path) -> dict:
    if output_path.exists():
        raise FileExistsError(
            f"Evaluation report already exists: {output_path}. Choose a new --output path."
        )
    api_key, model_name = get_llm_settings()
    if not api_key:
        raise RuntimeError("GROQ_API_KEY is required for live evaluations.")

    knowledge_base = KnowledgeBase([POLICY_DIR], embedder_preference="fastembed").build(persist=False)
    if knowledge_base.embedder is None:
        raise RuntimeError("The semantic embedder was not initialized.")
    llm = LLMClient(api_key, model_name)
    records: list[dict] = []

    for case in CASES:
        result = knowledge_base.retriever.retrieve(case["question"])
        hits = result.hits if result.answerable else result.closest[:3]
        error = None
        if result.answerable:
            try:
                answer = "".join(stream_answer(llm, case["question"], result.hits, []))
            except LLMError as exc:
                answer, error = "", str(exc)
        else:
            answer = NOT_FOUND_MESSAGE

        record = {
            "id": case["id"],
            "question": case["question"],
            "expected_answer": case["expected_answer"],
            "expected_source": case["expected_source"],
            "retrieved_sources": _record_sources(hits, result.answerable),
            "observed_output": answer,
            "answerable": result.answerable,
            "error": error,
        }
        _judge(record, case, result.hits)
        records.append(record)
        if error:
            break

    report = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "provider": "Groq OpenAI-compatible API",
        "model": model_name,
        "embedder": knowledge_base.embedder.name,
        "judged_at": datetime.now(timezone.utc).isoformat(),
        "judgment_version": JUDGMENT_VERSION,
        "total_cases": len(CASES),
        "completed_cases": len(records),
        "passed_cases": sum(record["judgment"]["pass"] for record in records),
        "failed_cases": sum(not record["judgment"]["pass"] for record in records),
        "evaluations": records,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--confirm-live",
        action="store_true",
        help="Confirm that questions and retrieved fictional policy passages may be sent to Groq.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "tests" / "evaluations" / "live_results.json",
        help="Path for the recorded evaluation report.",
    )
    parser.add_argument(
        "--rejudge",
        type=Path,
        help="Recompute judgments from a saved report without making API calls.",
    )
    args = parser.parse_args()
    if args.rejudge:
        try:
            report = rejudge(args.rejudge)
        except (OSError, ValueError, KeyError, TypeError) as exc:
            print(f"Could not rejudge report: {exc}", file=sys.stderr)
            return 1
        print(
            f"Rejudged {report['completed_cases']}/{report['total_cases']} evaluations; "
            f"{report['passed_cases']} passed and {report['failed_cases']} failed. "
            f"Report: {args.rejudge}"
        )
        return 0
    if not args.confirm_live:
        parser.error("Pass --confirm-live to authorize API calls and quota usage.")
    try:
        report = run(args.output)
    except (RuntimeError, ImportError, OSError, ValueError) as exc:
        print(f"Live evaluation failed: {exc}", file=sys.stderr)
        return 1
    print(
        f"Recorded {report['completed_cases']}/{report['total_cases']} evaluations; "
        f"{report['passed_cases']} passed and {report['failed_cases']} failed. "
        f"Report: {args.output}"
    )
    return 0 if report["completed_cases"] == report["total_cases"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
