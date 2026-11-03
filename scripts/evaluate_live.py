"""Run explicitly opted-in live semantic retrieval and LLM evaluations."""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import subprocess
import sys
from datetime import date, datetime, timezone
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from rag.config import POLICY_DIR, get_llm_settings  # noqa: E402
from rag.generator import LLMClient, LLMError, NOT_FOUND_MESSAGE, stream_answer  # noqa: E402
from rag.ingest import Chunk, Document  # noqa: E402
from rag.pipeline import KnowledgeBase, _active_policy_documents  # noqa: E402
from rag.retriever import Hit, contextualize_query  # noqa: E402
from rag.verification import verification_issues  # noqa: E402

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
        "required_terms": ["1,00,000", "20 transactions per day"],
    },
    {
        "id": "home-loan-tenure",
        "question": "What is the maximum home-loan tenure for salaried and self-employed applicants?",
        "expected_answer": "30 years for salaried applicants and 25 years for self-employed applicants.",
        "expected_source": "Home Loan Policy",
        "required_terms": ["30 years", "25 years"],
    },
    {
        "id": "unsupported-gold-loan-question",
        "question": "What is the gold-loan interest rate?",
        "expected_answer": "The answer must be declined because this policy is not in the documents.",
        "expected_source": None,
        "required_terms": [],
        "expected_refusal": True,
    },
    {
        "id": "personal-account-balance",
        "question": "What is my current savings-account balance?",
        "expected_answer": "The assistant must not claim access to a customer's personal balance.",
        "expected_source": None,
        "required_terms": [],
        "expected_refusal": True,
    },
    {
        "id": "fd-rate-follow-up",
        "history": [
            {"role": "user", "content": "What is the fixed-deposit rate for 1 year to less than 2 years?"}
        ],
        "question": "And for senior citizens?",
        "expected_answer": "The senior-citizen rate for that tenor is 7.50% per annum.",
        "expected_source": "Fixed Deposit and Recurring Deposit Policy",
        "required_terms": ["7.50%"],
    },
]
CASE_VERSION = "aurora-live-v4"
JUDGMENT_VERSION = "categorized-citation-scope-v5"


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


def _case_set_sha256() -> str:
    return hashlib.sha256(
        json.dumps(CASES, ensure_ascii=False, sort_keys=True).encode("utf-8")
    ).hexdigest()


def _code_provenance() -> dict:
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, check=True,
        capture_output=True, text=True,
    ).stdout.strip()
    diff = subprocess.run(
        ["git", "diff", "--binary", "HEAD"], cwd=ROOT, check=True,
        capture_output=True,
    ).stdout
    untracked = subprocess.run(
        ["git", "ls-files", "--others", "--exclude-standard", "-z"],
        cwd=ROOT, check=True, capture_output=True,
    ).stdout.split(b"\0")
    digest = hashlib.sha256(diff)
    dirty = bool(diff)
    for raw_path in untracked:
        if not raw_path:
            continue
        relative = Path(raw_path.decode("utf-8"))
        if relative.parts[:2] == ("tests", "evaluations"):
            continue
        dirty = True
        digest.update(raw_path)
        digest.update((ROOT / relative).read_bytes())
    return {
        "code_commit": commit,
        "working_tree_dirty": dirty,
        "working_tree_diff_sha256": digest.hexdigest() if dirty else None,
    }


def _offline_demonstrations() -> dict:
    current = Document(
        "current", "Demo Policy", "current.md", "Current rules.",
        policy_id="demo-policy", version="1", effective_date="2026-01-01",
    )
    future = Document(
        "future", "Demo Policy", "future.md", "Replacement rules.",
        policy_id="demo-policy", version="2", effective_date="2099-01-01",
        supersedes=["1"],
    )
    active_today, today_conflicts = _active_policy_documents(
        [current, future], as_of_date=date.today()
    )
    active_after_update, future_conflicts = _active_policy_documents(
        [current, future], as_of_date=date(2099, 1, 1)
    )

    conflict_a = Document(
        "conflict-a", "Conflicting Policy", "a.md", "Rule A.",
        policy_id="conflict-demo",
    )
    conflict_b = Document(
        "conflict-b", "Conflicting Policy", "b.md", "Rule B.",
        policy_id="conflict-demo",
    )
    active_conflicts, conflict_details = _active_policy_documents(
        [conflict_a, conflict_b], as_of_date=date.today()
    )

    def interrupted_response():
        yield SimpleNamespace(
            choices=[SimpleNamespace(delta=SimpleNamespace(content="Partial answer."))]
        )
        raise RuntimeError("simulated connection interruption")

    llm = object.__new__(LLMClient)
    llm.model = "offline-demo"
    completions = SimpleNamespace(calls=0)

    def create_response(**kwargs):
        completions.calls += 1
        return interrupted_response()

    completions.create = create_response
    llm.client = SimpleNamespace(
        chat=SimpleNamespace(completions=completions)
    )
    partial_text = ""
    try:
        for delta in llm.stream([], attempts=1):
            partial_text += delta
        interruption = None
    except LLMError as exc:
        interruption = str(exc)

    return {
        "policy_update": {
            "as_of_today": date.today().isoformat(),
            "active_versions_today": [document.version for document in active_today],
            "today_conflicts": today_conflicts,
            "active_versions_after_effective_date": [
                document.version for document in active_after_update
            ],
            "future_date_conflicts": future_conflicts,
            "pass": (
                [document.version for document in active_today] == ["1"]
                and [document.version for document in active_after_update] == ["2"]
                and not today_conflicts and not future_conflicts
            ),
        },
        "policy_conflict": {
            "active_document_count": len(active_conflicts),
            "conflicts": conflict_details,
            "pass": not active_conflicts and len(conflict_details) == 1,
        },
        "interrupted_api_response": {
            "partial_text_received": bool(partial_text),
            "error": interruption,
            "attempts": completions.calls,
            "pass": bool(partial_text) and bool(interruption) and completions.calls == 1,
        },
    }


def _judge(record: dict, case: dict, hits: list[Hit]) -> dict:
    answer = record["observed_output"]
    issues = (
        verification_issues(answer, hits, record["question"])
        if answer and record.get("answerable", True)
        else {"missing_citations": [], "invalid_citations": [], "unsupported_numbers": []}
    )
    warnings = [warning for category in issues.values() for warning in category]
    normalized_answer = _normalise(answer)
    missing_terms = [
        term for term in case["required_terms"]
        if _normalise(term) not in normalized_answer
    ]
    expected_refusal = case.get("expected_refusal", False)
    if expected_refusal:
        answer_pass = (
            not answer.strip()
            or answer.strip() == NOT_FOUND_MESSAGE
            or "[[NOT_IN_POLICY]]" in answer
        ) and not record.get("error")
        source_found = None
    else:
        source_found = record.get("answerable", True) and any(
            case["expected_source"].casefold() in source["title"].casefold()
            for source in record["retrieved_sources"]
        )
        answer_pass = (
            record.get("answerable", True) and not missing_terms and not record.get("error")
        )
    record["required_terms"] = case["required_terms"]
    record["verification"] = issues
    record["verification_warnings"] = warnings
    record["judgment"] = {
        "expected_terms_present": None if expected_refusal else not missing_terms,
        "missing_terms": missing_terms,
        "expected_source_retrieved": source_found,
        "citation_and_number_checks_passed": not warnings,
        "expected_refusal": expected_refusal,
        "pass": answer_pass and (expected_refusal or source_found) and not warnings,
    }
    return record["judgment"]


def rejudge(output_path: Path) -> dict:
    report = json.loads(output_path.read_text(encoding="utf-8"))
    if report.get("case_version") != CASE_VERSION:
        raise ValueError(
            f"Report case version {report.get('case_version')!r} does not match "
            f"current version {CASE_VERSION!r}; retain the original report unchanged."
        )
    if report.get("case_set_sha256") != _case_set_sha256():
        raise ValueError(
            "Report case-set digest does not match the current evaluation cases."
        )
    cases = {case["id"]: case for case in CASES}
    for record in report["evaluations"]:
        case = cases[record["id"]]
        if record["question"] != case["question"]:
            raise ValueError(
                f"Question mismatch for {record['id']!r}; report and evaluation cases differ."
            )
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
    report["offline_demonstrations"] = _offline_demonstrations()
    report.update(_code_provenance())
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
        history = case.get("history", [])
        query = contextualize_query(case["question"], history)
        result = knowledge_base.retriever.retrieve(query)
        hits = result.hits if result.answerable else result.closest[:3]
        error = None
        if result.answerable and not case.get("expected_refusal", False):
            try:
                answer = "".join(stream_answer(llm, case["question"], result.hits, history))
            except LLMError as exc:
                answer, error = "", str(exc)
        elif case.get("expected_refusal", False) and result.answerable:
            try:
                answer = "".join(stream_answer(llm, case["question"], result.hits, history))
            except LLMError as exc:
                answer, error = "", str(exc)
        else:
            answer = NOT_FOUND_MESSAGE

        record = {
            "id": case["id"],
            "case_version": CASE_VERSION,
            "question": case["question"],
            "history": history,
            "retrieval_query": query,
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
        "as_of_date": knowledge_base.as_of_date.isoformat(),
        **_code_provenance(),
        "case_version": CASE_VERSION,
        "case_set_sha256": _case_set_sha256(),
        "judged_at": datetime.now(timezone.utc).isoformat(),
        "judgment_version": JUDGMENT_VERSION,
        "offline_demonstrations": _offline_demonstrations(),
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
        default=ROOT / "tests" / "evaluations" / "live_results_v5.json",
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
