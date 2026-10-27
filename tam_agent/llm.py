"""Thin LLM client with native tool calling. Pick a provider with env vars (see .env.example).

  LLM_PROVIDER=ollama     free, local (https://ollama.com) - no API key needed
  LLM_PROVIDER=anthropic  needs ANTHROPIC_API_KEY and `pip install anthropic`
  LLM_PROVIDER=openai     needs OPENAI_API_KEY and `pip install openai`

The agent keeps its conversation in ONE provider-neutral format:
  {"role": "user", "content": str}
  {"role": "assistant", "content": str, "tool_calls": [{"id", "name", "args"}]}
  {"role": "tool", "tool_call_id": str, "name": str, "content": str}
`chat_tools` converts that to each provider's wire format and back, so agent.py
never contains provider-specific code.
"""
import json
import os
import urllib.request
import uuid
from pathlib import Path

# minimal .env loader (no extra dependency)
_env = Path(__file__).resolve().parent.parent / ".env"
if _env.exists():
    for _line in _env.read_text().splitlines():
        _line = _line.split("#", 1)[0].strip()
        if "=" in _line:
            _k, _v = _line.split("=", 1)
            os.environ.setdefault(_k.strip(), _v.strip())


class LLMClient:
    def __init__(self, provider: str | None = None, model: str | None = None):
        self.provider = (provider or os.getenv("LLM_PROVIDER", "ollama")).lower()
        self.model = model or os.getenv("LLM_MODEL", {
            "ollama": "llama3.1:8b", "anthropic": "claude-sonnet-4-5", "openai": "gpt-4o-mini"}[self.provider])
        self.calls = 0            # number of LLM requests (run.py reports this)
        self.input_tokens = 0     # running totals for cost reporting
        self.output_tokens = 0
        self._client = None       # SDK client, created once

    # ------------------------------------------------------------------ plain chat (kept from starter)
    def chat(self, messages: list[dict], system: str | None = None, max_tokens: int = 1024,
             temperature: float = 0.0) -> str:
        return self.chat_tools(messages, tools=[], system=system, max_tokens=max_tokens,
                               temperature=temperature)["content"]

    # ------------------------------------------------------------------ tool calling
    def chat_tools(self, messages: list[dict], tools: list[dict], system: str | None = None,
                   max_tokens: int = 2048, temperature: float = 0.0, force_tool: bool = True) -> dict:
        """Returns {"content": str, "tool_calls": [{"id", "name", "args"}]}.

        tools: [{"name", "description", "parameters": <JSON schema>}]
        force_tool: ask the provider to always answer with a tool call (where supported).
        """
        self.calls += 1
        fn = {"anthropic": self._anthropic, "openai": self._openai, "ollama": self._ollama}.get(self.provider)
        if fn is None:
            raise ValueError(f"Unknown LLM_PROVIDER: {self.provider}")
        return fn(messages, tools, system, max_tokens, temperature, force_tool)

    # ---- Anthropic
    def _anthropic(self, messages, tools, system, max_tokens, temperature, force_tool):
        import anthropic
        self._client = self._client or anthropic.Anthropic()
        wire = []
        for m in messages:
            if m["role"] == "user":
                if wire and wire[-1]["role"] == "user" and isinstance(wire[-1]["content"], list):
                    wire[-1]["content"].append({"type": "text", "text": m["content"]})  # after tool results
                else:
                    wire.append({"role": "user", "content": m["content"]})
            elif m["role"] == "assistant":
                blocks = [{"type": "text", "text": m["content"]}] if m.get("content") else []
                blocks += [{"type": "tool_use", "id": c["id"], "name": c["name"], "input": c["args"]}
                           for c in m.get("tool_calls", [])]
                wire.append({"role": "assistant", "content": blocks})
            elif m["role"] == "tool":
                block = {"type": "tool_result", "tool_use_id": m["tool_call_id"], "content": m["content"]}
                # consecutive tool results must share one user message
                if wire and wire[-1]["role"] == "user" and isinstance(wire[-1]["content"], list):
                    wire[-1]["content"].append(block)
                else:
                    wire.append({"role": "user", "content": [block]})
        kwargs = dict(model=self.model, max_tokens=max_tokens, system=system or "", messages=wire)
        # Newer SDK versions dropped `temperature`; only send it where it is still accepted.
        import inspect
        if "temperature" in inspect.signature(self._client.messages.create).parameters:
            kwargs["temperature"] = temperature
        if tools:
            kwargs["tools"] = [{"name": t["name"], "description": t["description"],
                                "input_schema": t["parameters"]} for t in tools]
            if force_tool:
                kwargs["tool_choice"] = {"type": "any"}
        resp = self._client.messages.create(**kwargs)
        self.input_tokens += resp.usage.input_tokens
        self.output_tokens += resp.usage.output_tokens
        text = "".join(b.text for b in resp.content if b.type == "text")
        calls = [{"id": b.id, "name": b.name, "args": b.input or {}} for b in resp.content if b.type == "tool_use"]
        return {"content": text, "tool_calls": calls}

    # ---- OpenAI
    def _openai(self, messages, tools, system, max_tokens, temperature, force_tool):
        from openai import OpenAI
        self._client = self._client or OpenAI()
        wire = [{"role": "system", "content": system}] if system else []
        for m in messages:
            if m["role"] == "assistant":
                msg = {"role": "assistant", "content": m.get("content") or None}
                if m.get("tool_calls"):
                    msg["tool_calls"] = [{"id": c["id"], "type": "function",
                                          "function": {"name": c["name"], "arguments": json.dumps(c["args"])}}
                                         for c in m["tool_calls"]]
                wire.append(msg)
            elif m["role"] == "tool":
                wire.append({"role": "tool", "tool_call_id": m["tool_call_id"], "content": m["content"]})
            else:
                wire.append({"role": "user", "content": m["content"]})
        # MAX_TOKENS / TOOL_CHOICE / REASONING_EFFORT let OpenAI-compatible providers (e.g. Gemini) work too.
        kwargs = dict(model=self.model, messages=wire, temperature=temperature,
                      max_tokens=int(os.getenv("MAX_TOKENS", max_tokens)))
        if os.getenv("REASONING_EFFORT"):
            kwargs["reasoning_effort"] = os.getenv("REASONING_EFFORT")
        if tools:
            kwargs["tools"] = [{"type": "function", "function": t} for t in tools]
            kwargs["tool_choice"] = os.getenv("TOOL_CHOICE", "required" if force_tool else "auto")
        resp = None
        for attempt in range(8):  # free tiers have per-minute limits: wait and retry on 429
            try:
                resp = self._client.chat.completions.create(**kwargs)
                break
            except Exception as e:
                if "429" not in str(e) or "credit" in str(e).lower() or attempt == 7:
                    raise
                import time
                time.sleep(min(10 * (attempt + 1), 60))
        if resp.usage:
            self.input_tokens += resp.usage.prompt_tokens
            self.output_tokens += resp.usage.completion_tokens
        msg = resp.choices[0].message
        calls = []
        for c in msg.tool_calls or []:
            try:
                args = json.loads(c.function.arguments or "{}")
            except json.JSONDecodeError:
                args = {"_unparseable": c.function.arguments}
            calls.append({"id": c.id, "name": c.function.name, "args": args})
        return {"content": msg.content or "", "tool_calls": calls}

    # ---- Ollama (local)
    def _ollama(self, messages, tools, system, max_tokens, temperature, force_tool):
        host = os.getenv("OLLAMA_HOST", "http://localhost:11434")
        wire = [{"role": "system", "content": system}] if system else []
        for m in messages:
            if m["role"] == "assistant":
                msg = {"role": "assistant", "content": m.get("content") or ""}
                if m.get("tool_calls"):
                    msg["tool_calls"] = [{"function": {"name": c["name"], "arguments": c["args"]}}
                                         for c in m["tool_calls"]]
                wire.append(msg)
            elif m["role"] == "tool":
                wire.append({"role": "tool", "content": m["content"], "tool_name": m["name"]})
            else:
                wire.append({"role": "user", "content": m["content"]})
        body = {"model": self.model, "messages": wire, "stream": False,
                "options": {"temperature": temperature, "num_predict": max_tokens, "num_ctx": 16384}}
        if tools:
            body["tools"] = [{"type": "function", "function": t} for t in tools]
        req = urllib.request.Request(f"{host}/api/chat", data=json.dumps(body).encode(),
                                     headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=600) as r:
            data = json.loads(r.read())
        self.input_tokens += data.get("prompt_eval_count", 0)
        self.output_tokens += data.get("eval_count", 0)
        msg = data["message"]
        calls = []
        for c in msg.get("tool_calls") or []:
            args = c["function"].get("arguments") or {}
            if isinstance(args, str):
                try:
                    args = json.loads(args)
                except json.JSONDecodeError:
                    args = {"_unparseable": args}
            calls.append({"id": uuid.uuid4().hex[:12], "name": c["function"]["name"], "args": args})
        return {"content": msg.get("content", ""), "tool_calls": calls}