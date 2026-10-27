"""Tools the triage agent can call.

Design rules:
  - Tools are deterministic Python. They look things up and compute facts
    (e.g. "users down 49% in 3 months"); the LLM makes the judgement calls.
  - Every tool returns a small JSON-able dict so the prompt stays short even on
    the full dataset. Nothing returns a whole table or the whole KB.
  - One `TicketSession` per ticket remembers what the agent has looked at, so the
    final answer can be checked against it (no citing articles it never saw).
"""
import re
from datetime import date

import pandas as pd

from .data import AS_OF_DATE

# ---------------------------------------------------------------------------
# Module labels in tickets are free text. Map aliases to canonical names
# (the names used in accounts.modules_licensed and the KB).
# ---------------------------------------------------------------------------
CANONICAL_MODULES = ["Contracts", "Integrations", "Platform", "Sourcing", "Spend Analytics",
                     "Supplier Management", "eInvoice", "eProcurement"]
MODULE_ALIASES = {
    "platform": "Platform", "sso": "Platform", "login": "Platform",
    "contracts": "Contracts", "contract": "Contracts", "clm": "Contracts", "contract mgmt": "Contracts",
    "sourcing": "Sourcing", "rfx": "Sourcing", "esourcing": "Sourcing", "sourcing events": "Sourcing",
    "spend analytics": "Spend Analytics", "spend": "Spend Analytics", "analytics": "Spend Analytics",
    "reports": "Spend Analytics",
    "supplier management": "Supplier Management", "sim": "Supplier Management",
    "supplier portal": "Supplier Management", "supplier mgmt": "Supplier Management",
    "einvoice": "eInvoice", "e-invoice": "eInvoice", "einv": "eInvoice", "invoicing": "eInvoice", "ap": "eInvoice",
    "eprocurement": "eProcurement", "e-procurement": "eProcurement", "eproc": "eProcurement",
    "procurement": "eProcurement",
    "integrations": "Integrations", "erp integration": "Integrations", "api": "Integrations",
    "connector": "Integrations",
}
ERROR_CODE = re.compile(r"\bE-\d{4}\b")
# Phrases that suggest text addressed to an automated system (prompt injection).
INJECTION = re.compile(r"(ignore (all|previous|prior)|system (note|prompt|message)|\bAI\b|assistant|"
                       r"classify this ticket|set (this|priority) to|list of (all )?customer)", re.I)


def canonical_module(raw: str) -> str | None:
    return MODULE_ALIASES.get((raw or "").strip().lower())


def ticket_hints(ticket: dict) -> dict:
    """Deterministic pre-parsing of the ticket. Facts only, no decisions."""
    text = f"{ticket.get('subject', '')} {ticket.get('description', '')}"
    codes = sorted(set(ERROR_CODE.findall(text)) | ({ticket["error_code"]} if ticket.get("error_code") else set()))
    return {
        "module_raw": ticket.get("module") or "(blank)",
        "module_canonical": canonical_module(ticket.get("module", "")) or "unknown",
        "error_codes_found": codes,
        "account_id_given": ticket.get("account_id") or "(blank - resolve it)",
        "possible_prompt_injection": bool(INJECTION.search(text)),
    }


def _days_between(a: str, b: str) -> int:
    return (date.fromisoformat(b[:10]) - date.fromisoformat(a[:10])).days


class TicketSession:
    """All tools for one ticket. `ctx` holds the shared data and indexes."""

    def __init__(self, ctx, ticket: dict):
        self.ctx, self.ticket = ctx, ticket
        self.seen_kb: set[str] = set()      # every KB id shown to the agent
        self.opened_kb: set[str] = set()    # KB ids read in full
        self.resolved_account: str | None = None
        self.checked: set[str] = set()      # names of tools already called

    # ---------------- account ----------------
    def resolve_account(self, requester_domain: str = "", company_name_raw: str = "") -> dict:
        domain = (requester_domain or self.ticket.get("requester_domain", "")).strip().lower()
        acc = self.ctx.accounts_by_domain.get(domain)
        if acc is None:
            return {"found": False, "message": f"No account owns domain '{domain}'. Do not guess from the company name."}
        self.resolved_account = acc["account_id"]
        out = {"found": True, "account_id": acc["account_id"], "company_name": acc["company_name"],
               "matched_on": "requester_domain (exact)"}
        given = self.ticket.get("account_id")
        if given and given != acc["account_id"]:
            out["warning"] = f"Ticket account_id {given} differs from domain owner {acc['account_id']}."
        # Warn about look-alike names so the agent doesn't mix up e.g. 'Ironleaf Foods' and 'Ironleaf Foods Europe'.
        first = acc["company_name"].split()[0]
        similar = [n for n in self.ctx.accounts["company_name"] if n.split()[0] == first and n != acc["company_name"]]
        if similar:
            out["note"] = (f"{len(similar)} other accounts have similar names (e.g. {', '.join(similar[:3])}). "
                           f"Domain match is authoritative.")
        return out

    def get_account(self, account_id: str) -> dict:
        acc = self.ctx.accounts_by_id.get(account_id)
        if acc is None:
            return {"error": f"Unknown account_id {account_id}"}
        days_to_renewal = _days_between(AS_OF_DATE, acc["renewal_date"])
        sso = acc["sso_enabled_date"]
        return {
            "account_id": account_id, "company_name": acc["company_name"], "tier": acc["tier"],
            "arr_band": acc["arr_band"], "region": acc["region"],
            "renewal_date": acc["renewal_date"], "days_to_renewal": days_to_renewal,
            "modules_licensed": acc["modules_licensed"].split(";"),
            "sso_enabled_or_reconfigured": sso or "never",
            "days_since_sso_change": _days_between(sso, AS_OF_DATE) if sso else None,
            "open_escalations": int(acc["open_escalations"] or 0),
            "nps_last": acc["nps_last"] or None, "last_qbr_date": acc["last_qbr_date"] or None,
        }

    def get_usage_trend(self, account_id: str) -> dict:
        u = self.ctx.usage_by_account.get(account_id)
        if u is None or u.empty:
            return {"error": f"No usage data for {account_id}"}
        u = u.sort_values("month")
        last = u.iloc[-1]
        three_back = u.iloc[-4] if len(u) >= 4 else u.iloc[0]
        change = round(100 * (last.active_users - three_back.active_users) / max(three_back.active_users, 1), 1)
        acc = self.get_account(account_id)
        # Account risk override from policy KB-2914, computed exactly so the LLM doesn't do date maths.
        within_90 = 0 <= acc["days_to_renewal"] <= 90
        reasons = []
        if change < -25:
            reasons.append(f"active users down {abs(change)}% over last 3 months")
        if acc["open_escalations"] >= 2:
            reasons.append(f"{acc['open_escalations']} open escalations")
        applies = within_90 and bool(reasons)
        return {
            "account_id": account_id,
            "recent_months": u.tail(4)[["month", "active_users", "login_rate_pct", "tickets_opened"]].to_dict("records"),
            "active_users_change_3m_pct": change,
            "risk_override_check (KB-2914)": {
                "applies": applies,
                "days_to_renewal": acc["days_to_renewal"],
                "within_90_days_of_renewal": within_90,
                "risk_factors": reasons or ["none"],
                "meaning": ("Raise priority one level (max P2), notify TAM, escalate=true" if applies
                            else "Override does not apply"),
            },
        }

    # ---------------- tickets ----------------
    def search_tickets(self, account_id: str, query: str = "", module: str = "", error_code: str = "",
                       days: int = 30) -> dict:
        """Recent tickets from the same account, created before this ticket."""
        t = self.ctx.tickets_by_account.get(account_id)
        if t is None:
            return {"tickets": [], "note": "no history"}
        created = self.ticket["created_at"]
        t = t[(t["created_at"] < created) & (t["ticket_id"] != self.ticket["ticket_id"])]
        cutoff = (pd.Timestamp(created) - pd.Timedelta(days=int(days))).isoformat()
        t = t[t["created_at"] >= cutoff[:16]]
        if module:
            want = canonical_module(module) or module
            t = t[t["module"].map(lambda m: (canonical_module(m) or m) == want)]
        if error_code:
            t = t[t["error_code"].eq(error_code) | t["description"].str.contains(error_code, regex=False)
                  | t["subject"].str.contains(error_code, regex=False)]
        if query:
            words = [w for w in re.findall(r"[a-z0-9-]{4,}", query.lower())]
            if words:
                text = (t["subject"] + " " + t["description"]).str.lower()
                t = t[text.map(lambda s: any(w in s for w in words))]
        t = t.sort_values("created_at", ascending=False).head(10)
        return {"count": len(t), "window_days": days, "tickets": [
            {"ticket_id": r.ticket_id, "created_at": r.created_at, "module": r.module,
             "module_canonical": canonical_module(r.module), "subject": r.subject,
             "error_code": r.error_code, "status": r.status, "customer_priority": r.customer_priority}
            for r in t.itertuples()]}

    def check_duplicates(self) -> dict:
        """Apply the duplicate rule from KB-5930 to this ticket."""
        acc = self.resolved_account or self.ticket.get("account_id")
        if not acc:
            return {"error": "Resolve the account first (resolve_account)."}
        hints = ticket_hints(self.ticket)
        module = hints["module_canonical"]
        codes = set(hints["error_codes_found"])
        recent = self.search_tickets(acc, days=7)["tickets"]
        cands = []
        for r in recent:
            if r["status"] not in ("open", "pending_customer"):
                continue
            same_module = module != "unknown" and r["module_canonical"] == module
            row = self.ctx.tickets_by_id[r["ticket_id"]]
            r_codes = set(ERROR_CODE.findall(row["subject"] + " " + row["description"])) | (
                {row["error_code"]} if row["error_code"] else set())
            same_code = bool(codes & r_codes)
            if same_module and same_code:
                cands.append(r | {"description": row["description"][:200], "match": "same module + same error code"})
            elif same_module:
                cands.append(r | {"description": row["description"][:200],
                                  "match": "same module only - compare symptoms yourself"})
        return {"rule": "KB-5930: same account, open/pending, same module, same error code or symptom, previous 7 days",
                "candidates": cands, "candidate_count": len(cands)}

    # ---------------- knowledge base ----------------
    def search_kb(self, query: str, module: str = "", error_code: str = "", doc_type: str = "") -> dict:
        mod = canonical_module(module) or (module if module in CANONICAL_MODULES else "")
        hits = self.ctx.kb.search(query, module=mod, error_code=error_code, doc_type=doc_type, k=5)
        self.seen_kb.update(h["id"] for h in hits)
        return {"results": hits}

    def get_kb_article(self, kb_id: str) -> dict:
        art = self.ctx.kb.by_id.get(kb_id.strip().upper())
        if art is None:
            return {"error": f"No article {kb_id}"}
        self.seen_kb.add(art["id"])
        self.opened_kb.add(art["id"])
        meta = art["meta"]
        out = {"id": art["id"], "title": meta.get("title"), "type": meta.get("type"), "status": meta.get("status"),
               "last_updated": meta.get("last_updated"), "related": meta.get("related") or [],
               "body": art["body"][:3000]}
        if meta.get("status") == "deprecated":
            out["warning"] = f"DEPRECATED. Do not cite. Read {meta.get('superseded_by')} instead."
            out["superseded_by"] = meta.get("superseded_by")
        return out

    def get_policy(self, topic: str) -> dict:
        ids = {"priority_routing_escalation": "KB-2914", "duplicates": "KB-5930", "data_exposure": "KB-6401"}
        kb_id = ids.get(topic)
        if not kb_id:
            return {"error": f"topic must be one of {list(ids)}"}
        return self.get_kb_article(kb_id)

    def call(self, name: str, args: dict) -> dict:
        fn = {"resolve_account": self.resolve_account, "get_account": self.get_account,
              "get_usage_trend": self.get_usage_trend, "search_tickets": self.search_tickets,
              "check_duplicates": self.check_duplicates, "search_kb": self.search_kb,
              "get_kb_article": self.get_kb_article, "get_policy": self.get_policy}.get(name)
        if fn is None:
            return {"error": f"Unknown tool {name}"}
        self.checked.add(name)
        try:
            return fn(**(args or {}))
        except TypeError as e:  # bad arguments from the model -> tell it, don't crash
            return {"error": f"Bad arguments for {name}: {e}"}


# ---------------------------------------------------------------------------
# Tool definitions shown to the LLM (JSON schema). Descriptions matter: they are
# how the model decides which tool to use.
# ---------------------------------------------------------------------------
def _p(**props):
    return {k: {"type": v[0], "description": v[1]} for k, v in props.items()}


TOOL_SPECS = [
    {"name": "resolve_account",
     "description": "Find the customer account from the requester's email domain (authoritative). Use when "
                    "account_id is blank or to confirm it.",
     "parameters": {"type": "object", "properties": _p(requester_domain=("string", "Domain from the ticket"),
                                                       company_name_raw=("string", "Company name as typed")),
                    "required": ["requester_domain"]}},
    {"name": "get_account",
     "description": "Account facts: tier, ARR band, days to renewal, licensed modules, SSO change date, open "
                    "escalations, NPS.",
     "parameters": {"type": "object", "properties": _p(account_id=("string", "ACC-####")), "required": ["account_id"]}},
    {"name": "get_usage_trend",
     "description": "Last 4 months of usage, 3-month active-user change, and whether the KB-2914 account risk "
                    "override applies.",
     "parameters": {"type": "object", "properties": _p(account_id=("string", "ACC-####")), "required": ["account_id"]}},
    {"name": "search_tickets",
     "description": "Search this account's earlier tickets (history, repeat problems). Filters are optional.",
     "parameters": {"type": "object", "properties": _p(account_id=("string", "ACC-####"),
                                                       query=("string", "keywords"),
                                                       module=("string", "module name"),
                                                       error_code=("string", "e.g. E-1234"),
                                                       days=("integer", "look-back window, default 30")),
                    "required": ["account_id"]}},
    {"name": "check_duplicates",
     "description": "Apply the duplicate-ticket rule (KB-5930) to the current ticket and list candidate originals.",
     "parameters": {"type": "object", "properties": {}, "required": []}},
    {"name": "search_kb",
     "description": "Search the knowledge base. Returns top 5 article summaries with status (current, deprecated, "
                    "active_incident). Use error_code when you have one.",
     "parameters": {"type": "object", "properties": _p(query=("string", "symptom keywords"),
                                                       module=("string", "module name"),
                                                       error_code=("string", "e.g. E-1234"),
                                                       doc_type=("string", "optional: troubleshooting, how_to, limit, "
                                                                           "known_issue, policy, faq, reference")),
                    "required": ["query"]}},
    {"name": "get_kb_article",
     "description": "Read one KB article in full, including related articles and superseded_by. Use to follow "
                    "links between articles.",
     "parameters": {"type": "object", "properties": _p(kb_id=("string", "KB-####")), "required": ["kb_id"]}},
    {"name": "get_policy",
     "description": "Read a support policy. Topics: priority_routing_escalation (KB-2914), duplicates (KB-5930), "
                    "data_exposure (KB-6401).",
     "parameters": {"type": "object", "properties": _p(topic=("string", "policy topic")), "required": ["topic"]}},
    {"name": "finish",
     "description": "Submit the final triage decision. Call only when every field is justified by what you found.",
     "parameters": {"type": "object", "properties": {
         "category": {"type": "string", "enum": ["incident", "bug", "how_to", "feature_request", "security",
                                                 "duplicate", "needs_info"]},
         "priority": {"type": "string", "enum": ["P1", "P2", "P3", "P4"]},
         "route_to": {"type": "string", "enum": ["L1_support", "L2_engineering", "security", "product"]},
         "escalate": {"type": "boolean"},
         "notify_tam": {"type": "boolean"},
         "duplicate_of": {"type": ["string", "null"], "description": "original ticket_id, or null"},
         "needs_clarification": {"type": "boolean"},
         "kb_references": {"type": "array", "items": {"type": "string"},
                           "description": "KB ids that support the decision (current articles, policies)"},
         "account_signals": {"type": "array", "items": {"type": "string"},
                             "description": "facts from account/usage/history that influenced the decision"},
         "customer_reply": {"type": "string"},
         "internal_note": {"type": "string"},
         "confidence": {"type": "number", "description": "0.0 - 1.0"},
     }, "required": ["category", "priority", "route_to", "escalate", "notify_tam", "duplicate_of",
                     "needs_clarification", "kb_references", "account_signals", "customer_reply",
                     "internal_note", "confidence"]}},
]