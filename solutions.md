## 1. Design
ticket ─► hints (module alias, error codes in text, injection flag)
        ─► LLM ⇄ tools loop  (LLM picks the next tool, reads the result, decides to continue or stop)
        ─► finish(...) ─► guardrails ──fail──► problems sent back to LLM (max 2 rounds)
                                   └─ok──► output JSON + reasoning_trace
```
- **Tools** (`tam_agent/tools.py`, deterministic Python): `resolve_account` (domain match, warns about look-alike names), `get_account` (tier, days to renewal, licensed modules, SSO change date, escalations), `get_usage_trend` (3-month active-user change **and** whether the KB-2914 risk override applies, computed exactly), `search_tickets`, `check_duplicates` (KB-5930 rule), `search_kb` (BM25 + error-code/module boosts, flags deprecated and active-incident articles), `get_kb_article` (full text, `related`, `superseded_by`, so the agent can follow links), `get_policy`, and `finish`.
- **Split of work:** code computes facts (dates, percentages, matching); the LLM makes judgements (is this "FYI" really a data-exposure report? is a "bug" a by-design limit?).
- **Agentic loop** (`agent.py`): native tool calling for Anthropic, OpenAI and Ollama behind one provider-neutral message format (`llm.py`). Paths differ per ticket. A how-to may need two lookups; an SSO-related bid failure needs account → KB-4795 → follow KB-7889.
- **Stopping:** the model stops by calling `finish`. Hard limits are 10 tool calls (then only `finish` is offered) and 2 correction rounds. If the model still fails, a safe fallback is returned (needs human review, confidence 0.1).
- **Guardrails on `finish`:** schema valid; cited KB ids exist, were actually retrieved and are not deprecated; `duplicate_of` is a real earlier ticket of the same account; security ⇒ P1/security/escalate; P1 ⇒ escalate; no other customer's name in the reply. Ticket text is wrapped as untrusted content, and injection phrases are flagged as a hint.

## 2. Evaluation
- Build a **labelled gold set** (~200 tickets, double-labelled by support engineers, disagreements adjudicated). Measure per-field accuracy: category, priority, route, escalate, notify_tam, duplicate_of.
- Weight errors by cost: **under-prioritising** (P1 called P3) and **missed security** are tracked separately with a near-zero target; over-prioritising is cheaper.
- **Retrieval:** recall@5 and precision of `kb_references` against gold articles. Separate checks that no deprecated article is cited and active incidents are linked.
- **Adversarial suite:** prompt injections, look-alike company names, blank account/module, disguised security reports, by-design limits. Every one must pass before release.
- **Agent behaviour:** tool calls per ticket, guardrail rejection rate, fallback rate, and trace review (did it look before deciding?).
- **Reply quality:** rubric review (correct, specific, no leakage, no false promises) on a sample, with LLM-as-judge only as a pre-filter.
- **Shadow mode** for 2–4 weeks: the agent runs alongside humans. Compare decisions and track the human override rate, then roll out by category, starting with the lowest risk.
- Calibration: is confidence 0.9 right ~90% of the time? Use it to auto-route low-confidence tickets to humans.

## 3. Scale (10x: 30k tickets, 4k articles)
- **Cost/latency:** per-ticket cost does not grow with data size. Prompts only contain tool results, which are capped (top-5 summaries, 10 tickets, 3.5k chars). The typical run is 4–8 LLM calls; the main cost lever is model choice. Caching the system prompt and tool specs (prompt caching) cuts input cost.
- **Lookups:** account, ticket and usage lookups are in-memory dict/groupby indexes, built in <1 s now and a few seconds at 10x. Beyond that, move to Postgres with indexes on `account_id` and `created_at`.
- **Retrieval quality** is the real risk: with 4k articles, BM25 alone returns more near-misses. I would switch to **hybrid search** (BM25 + embeddings in a vector store such as pgvector) with metadata filters (module, status≠deprecated, type) and a small re-ranker. I would also index `related` links as a graph and run a KB hygiene job to auto-flag stale or duplicate articles.
- **Duplicates** at volume: embedding similarity within the 7-day window, not only module + error code.

## 4. Limitations
- Not yet measured on a labelled set. The 10 review tickets were checked by hand against the policies.
- Quality depends on the model's tool use. Small local models (8B) sometimes skip lookups or misread policy. The guardrails catch hard violations, not every judgement error.
- Symptom-only duplicates (no error code) rely on the LLM comparing the candidate descriptions.
- The usage trend compares the latest month (October, partial) with three months earlier. A partial month can exaggerate a drop.
- The system prompt carries general triage guidance. The exact rules come from the policy articles, but the risk override is also re-implemented in code (`get_usage_trend`), so a policy change must be made in both places.
- No retry/backoff for provider rate limits yet.

## 5. How to run
```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env                 # set LLM_PROVIDER, LLM_MODEL and the API key
python scripts/validate_data.py
python tests/test_agent.py           # offline tests, no LLM needed
python run.py --set review           # writes outputs/*.json and outputs/run_log.jsonl
python scripts/check_outputs.py      # expect 10/10 valid
```
````

---

## STEP 10 – Check that every file is right

Paste in the terminal:

```bash
ls tam_agent tests SOLUTION.md .env; grep -c chat_tools tam_agent/llm.py tam_agent/agent.py; grep -c TOOL_SPECS tam_agent/tools.py
```

You should see `retrieval.py` and `tools.py` inside `tam_agent`, the `tests` folder, `SOLUTION.md`, `.env`, and **every number above 0**. If a number is 0, that file still has old code – redo its step.

