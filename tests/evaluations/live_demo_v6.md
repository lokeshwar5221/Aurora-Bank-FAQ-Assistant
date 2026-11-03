# Aurora FAQ Assistant — Live Demonstration

**Recorded:** 2026-11-03  
**Implementation commit:** `b700480716b0546c25d1055e37b351f584feae97`  
**Evaluation cases:** `aurora-live-v4`  
**Live evaluation:** 13/13 automated cases passed  
**Full evidence:** [live_results_v6.json](./live_results_v6.json)

> **Academic sample only:** Aurora National Bank is fictional. All rates, policies, fees and contact details below are sample data and are not real financial advice or bank services.

The answer and refusal examples below were recorded from the live Groq evaluation against the committed implementation. The policy-update, conflict and interrupted-stream scenarios are deterministic offline demonstrations included in the report; they do not represent actual bank policy changes or an outage of the live API.

## 1. Sourced FAQ answer

**Customer:** What is the minimum balance for a Regular Savings account?

**Aria:**

> The minimum balance for a Regular Savings account varies by branch location [2].
>
> - **Metro / Urban:** ₹5,000 [2]
> - **Semi-urban:** ₹2,500 [2]
> - **Rural:** ₹1,000 [2]

**Retrieved policy source [2]:** Savings Account Policy, “Minimum balance requirements,” version 4.2, effective 2026-04-01. The report retains the exact passage and other retrieved sources. No citation or numeric warning was raised.

## 2. Unsupported question

**Customer:** What is the gold-loan interest rate?

**Observed behavior:** The model declined the question; the UI converts the refusal token to:

> I couldn't find this in the bank's policy documents, so I can't answer it reliably.
>
> You can try rephrasing your question or contact the bank through its official customer-care channels.

## 3. Multi-turn follow-up

**Customer:** What is the fixed-deposit rate for 1 year to less than 2 years?  
**Customer (follow-up):** And for senior citizens?

The follow-up retrieval query retained the original fixed-deposit tenor:
`What is the fixed-deposit rate for 1 year to less than 2 years? And for senior citizens?`

**Aria:**

> The fixed-deposit rate for a tenure of 1 year to less than 2 years is 7.00% p.a. for the general public and 7.50% p.a. for senior citizens [2].
>
> - General public: 7.00% per annum [2]
> - Senior citizens (60 years and above): 7.50% per annum [2]

**Retrieved policy source [2]:** Fixed Deposit and Recurring Deposit Policy, “Fixed deposit interest rates,” version 5.0, effective 2026-04-01.

## 4. Effective-dated policy update

An offline fixture has version 1 effective 2026-01-01 and version 2 effective 2099-01-01, with version 2 superseding version 1.

- As of 2026-11-03, version 1 remains active; the future version does not supersede it early.
- As of 2099-01-01, version 2 becomes active.
- Both checks passed without a policy conflict.

## 5. Interrupted API response

The offline stream test delivered a partial answer, then raised a simulated connection interruption. The stream client surfaced an interruption error after **one attempt** and did not restart or duplicate the partial answer; the chat interface displays caught stream errors.

## 6. Ambiguous policy conflict

Two conflicting, undated fixture revisions were excluded from the active knowledge base. The conflict was recorded rather than selecting one arbitrarily.

## Reproducibility and limits

The JSON report records the exact code commit, clean worktree state, case-set version and digest, retrieved passages, live model outputs, judgments and offline demonstration results. Automated citation and numeric checks are safeguards, not proof that every sentence is factually correct. Human review of answers against the cited passages remains necessary.
