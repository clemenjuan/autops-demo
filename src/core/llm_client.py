"""
LLM Client — Dual-provider interface (Ollama + OpenAI).

Provides a synchronous LLM call interface with:
- TUM Ollama as primary provider (via OLLAMA_HOST env var)
- OpenAI API as fallback (via OPENAI_API_KEY env var)
- File-based response cache for reproducibility and cost savings
- Mock mode for CI/testing (no live LLM calls)
- Automatic failover with backoff

Configuration is driven entirely by experiment YAML (representation_config).
"""
from __future__ import annotations

import hashlib
import json
import logging
import os
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

# Cache identity deliberately follows decision content, not campaign seed or
# endpoint, so paired cells can share deterministic responses. The schema
# label makes future identity changes explicit and auditable.
CACHE_KEY_SCHEMA = "llm-decision-v2"


class _ProviderCompletion(str):
    """String-compatible provider response carrying non-decision metadata."""

    provenance: Dict[str, Any]

    def __new__(
        cls, text: str, provenance: Dict[str, Any] | None = None
    ) -> "_ProviderCompletion":
        instance = str.__new__(cls, text)
        instance.provenance = dict(provenance or {})
        return instance


class LLMClient:
    """Synchronous LLM client with dual-provider failover and caching.

    Args:
        config: Representation config dict from experiment YAML. Relevant keys:
            - ``llm_provider``: ``"ollama"`` (default), ``"openai"``, or ``"auto"``
            - ``llm_model``: Model name (provider-specific)
            - ``llm_temperature``: Sampling temperature (default 0.0)
            - ``llm_mock``: If ``True``, return mock responses (for CI)
            - ``llm_cache_dir``: Cache directory (default ``data/llm_cache``)
            - ``ollama_host``: Ollama server URL (default from ``OLLAMA_HOST`` env)
    """

    def __init__(self, config: Dict[str, Any] | None = None) -> None:
        self.config = config or {}
        self.mock_mode: bool = self.config.get("llm_mock", False)
        self.temperature: float = self.config.get("llm_temperature", 0.0)
        # Default to the small/fast model; experiments pin their model explicitly
        # (qwen3.6:35b). Avoids an unset-config path silently using the 122b giant.
        self.model: str = self.config.get("llm_model", "qwen3.5:4b")
        self.provider: str = self.config.get("llm_provider", "auto")

        # Cache
        cache_dir = self.config.get("llm_cache_dir", "data/llm_cache")
        self.cache_dir = Path(cache_dir) / self.model.replace("/", "_").replace(":", "_")
        if not self.mock_mode:
            self.cache_dir.mkdir(parents=True, exist_ok=True)

        # Provider state
        self.ollama_host: str = self.config.get(
            "ollama_host", os.getenv("OLLAMA_HOST", "https://ollama.sps.ed.tum.de")
        )
        self._ollama_available: Optional[bool] = None
        self._openai_available: Optional[bool] = None
        self._ollama_backoff_until: float = 0.0

        # Metrics
        self._total_calls: int = 0
        self._cache_hits: int = 0
        self._total_prompt_tokens: int = 0
        self._total_completion_tokens: int = 0
        self._total_latency_s: float = 0.0
        self._last_latency_s: float = 0.0
        self._last_provider: str = "none"
        self._last_call_provenance: Dict[str, Any] = {}
        self._call_provenance_counts: Dict[str, Dict[str, Any]] = {}

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def generate(
        self,
        system_prompt: str,
        user_prompt: str,
        *,
        temperature: float | None = None,
        json_mode: bool = False,
    ) -> str:
        """Generate a completion from the LLM.

        Args:
            system_prompt: System-level instructions.
            user_prompt: User-level prompt with the actual query.
            temperature: Override default temperature for this call.
            json_mode: Request JSON output format (provider-specific).

        Returns:
            The LLM's text response.

        Raises:
            RuntimeError: If no LLM provider is available and not in mock mode.
        """
        if self.mock_mode:
            return self._mock_response(user_prompt)

        temp = temperature if temperature is not None else self.temperature
        response = self._call_with_failover(system_prompt, user_prompt, temp, json_mode)

        logger.debug(
            "LLM [%s] prompt: %.200s...", self._last_provider, user_prompt[:200]
        )
        logger.debug(
            "LLM [%s] response: %.500s", self._last_provider, response[:500]
        )

        return response

    def get_metrics(self) -> Dict[str, float]:
        """Return LLM client metrics."""
        live_calls = self._total_calls - self._cache_hits
        return {
            "llm_api_calls": float(self._total_calls),
            "llm_cache_hits": float(self._cache_hits),
            "llm_cache_hit_rate": (
                self._cache_hits / self._total_calls if self._total_calls > 0 else 0.0
            ),
            "llm_total_latency_s": self._total_latency_s,
            "llm_last_latency_s": self._last_latency_s,
            # Mean wall-clock per *live* LLM inference (cache hits are ~0 s) — the
            # M-07 "cost of cognition" signal for LLM cells.
            "llm_mean_call_latency_s": (
                self._total_latency_s / live_calls if live_calls > 0 else 0.0
            ),
            "llm_tokens_prompt": float(self._total_prompt_tokens),
            "llm_tokens_completion": float(self._total_completion_tokens),
        }

    def get_provenance(self) -> Dict[str, Any]:
        """Return non-decision LLM provenance for result artifacts.

        Endpoint and provider-returned revision belong in provenance, not the
        cache key: the cache intentionally shares identical decision content
        across paired seeds and equivalent serving endpoints.
        """

        calls = sorted(
            (dict(record) for record in self._call_provenance_counts.values()),
            key=lambda record: (
                str(record.get("provider", "")),
                str(record.get("actual_model", "")),
                str(record.get("endpoint", "")),
                str(record.get("invocation", "")),
                str(record.get("cache_origin_invocation", "")),
                bool(record.get("cache_hit", False)),
            ),
        )
        return {
            "llm_cache_key_schema": CACHE_KEY_SCHEMA,
            "llm_mock": bool(self.mock_mode),
            "llm_configured_provider": self.provider,
            "llm_configured_model": self.model,
            "llm_last_call": dict(self._last_call_provenance),
            "llm_call_provenance": calls,
        }

    def _record_call_provenance(self, record: Dict[str, Any]) -> None:
        """Aggregate a provider/cache outcome without storing prompt content."""

        normalized = {
            "provider": str(record.get("provider", "unknown")),
            "configured_model": str(record.get("configured_model", "")),
            "actual_model": str(record.get("actual_model", "")),
            "model_revision": str(record.get("model_revision", "")),
            "response_id": str(record.get("response_id", "")),
            "endpoint": str(record.get("endpoint", "")),
            "invocation": str(record.get("invocation", "direct")),
            "cache_origin_invocation": str(
                record.get("cache_origin_invocation", "")
            ),
            "cache_hit": bool(record.get("cache_hit", False)),
        }
        key = json.dumps(normalized, sort_keys=True, separators=(",", ":"))
        existing = self._call_provenance_counts.get(key)
        if existing is None:
            existing = {**normalized, "count": 0}
            self._call_provenance_counts[key] = existing
        existing["count"] = int(existing["count"]) + 1
        self._last_call_provenance = dict(normalized)

    def reset_metrics(self) -> None:
        """Zero the per-episode metric counters (the on-disk cache is untouched, so
        cross-episode cache hits still register within the next episode)."""
        self._total_calls = 0
        self._cache_hits = 0
        self._total_prompt_tokens = 0
        self._total_completion_tokens = 0
        self._total_latency_s = 0.0
        self._last_latency_s = 0.0

    # ------------------------------------------------------------------
    # Provider calls
    # ------------------------------------------------------------------

    def _call_with_failover(
        self, system_prompt: str, user_prompt: str, temperature: float, json_mode: bool
    ) -> str:
        """Try Ollama first, fall back to OpenAI."""
        self._total_calls += 1
        now = time.time()

        providers = self._resolve_provider_order()
        last_error: Exception | None = None

        # Patient retry so a transient Ollama outage (GPU OOM dropping the model,
        # cold reload ~44 s, colleague-VM restart) doesn't abort a multi-hour run.
        # waits with cap=300 over 10 retries: 15,30,60,120,240,300,300,300,300,300
        # ≈ 33 min per provider pass; ×3 scheduler retries ≈ survives a ~1.5 h outage.
        max_retries = self.config.get("llm_retries", 10)
        backoff_cap_s = self.config.get("llm_backoff_cap_s", 300)
        ollama_cooldown_s = self.config.get("llm_ollama_cooldown_s", 300)
        for provider_index, provider in enumerate(providers):
            provider_model = self._provider_model(provider)
            invocation = (
                "fallback"
                if provider_index > 0 or (self.provider == "auto" and provider != "ollama")
                else "direct"
            )
            cache_key = self._cache_key(
                system_prompt,
                user_prompt,
                temperature,
                json_mode=json_mode,
                provider=provider,
                model=provider_model,
            )
            cached_entry = self._cache_get_entry(cache_key)
            if cached_entry is not None:
                self._cache_hits += 1
                self._last_provider = f"cache:{provider}"
                self._last_latency_s = 0.0
                cached_provenance = {
                    "provider": cached_entry.get("provider", provider),
                    "configured_model": cached_entry.get("model", provider_model),
                    "actual_model": cached_entry.get(
                        "actual_model", cached_entry.get("model", provider_model)
                    ),
                    "model_revision": cached_entry.get("model_revision", ""),
                    "response_id": cached_entry.get("response_id", ""),
                    "endpoint": cached_entry.get("endpoint", self._provider_endpoint(provider)),
                    # ``invocation`` describes how this run reached the answer;
                    # the cached entry's historical route is separate provenance.
                    "invocation": invocation,
                    "cache_origin_invocation": cached_entry.get("invocation", ""),
                    "cache_hit": True,
                }
                self._record_call_provenance(cached_provenance)
                return str(cached_entry.get("response", ""))

            for attempt in range(max_retries + 1):
                try:
                    t0 = time.perf_counter()
                    if provider == "ollama":
                        response = self._call_ollama(system_prompt, user_prompt, temperature, json_mode)
                    else:
                        response = self._call_openai(system_prompt, user_prompt, temperature, json_mode)
                    elapsed = time.perf_counter() - t0

                    self._last_latency_s = elapsed
                    self._total_latency_s += elapsed
                    self._last_provider = provider
                    call_provenance = {
                        "provider": provider,
                        "configured_model": provider_model,
                        "actual_model": provider_model,
                        "model_revision": "",
                        "response_id": "",
                        "endpoint": self._provider_endpoint(provider),
                        "invocation": invocation,
                        "cache_hit": False,
                        **dict(getattr(response, "provenance", {}) or {}),
                    }
                    self._record_call_provenance(call_provenance)
                    # Store under the provider that actually answered so an
                    # OpenAI fallback can never masquerade as Ollama/Qwen (or
                    # vice versa). Empty responses are never cached.
                    if response and response.strip():
                        self._cache_put(
                            cache_key,
                            response,
                            system_prompt,
                            user_prompt,
                            provider=provider,
                            model=provider_model,
                            temperature=temperature,
                            json_mode=json_mode,
                            provenance=call_provenance,
                        )
                    return response

                except Exception as e:
                    last_error = e
                    if attempt < max_retries:
                        wait = min(15 * (2 ** attempt), backoff_cap_s)
                        logger.warning(
                            "LLM provider '%s' attempt %d/%d failed: %s — retrying in %ds",
                            provider, attempt + 1, max_retries + 1, e, wait,
                        )
                        time.sleep(wait)
                    else:
                        logger.warning("LLM provider '%s' failed after %d attempts: %s", provider, max_retries + 1, e)
                        if provider == "ollama":
                            self._ollama_backoff_until = now + ollama_cooldown_s

        raise RuntimeError(
            f"All LLM providers failed. Last error: {last_error}"
        )

    def _resolve_provider_order(self) -> List[str]:
        """Determine provider order based on config and availability."""
        now = time.time()

        if self.provider == "ollama":
            return ["ollama"]
        elif self.provider == "openai":
            return ["openai"]

        # auto: prefer Ollama, fall back to OpenAI
        order = []
        if now >= self._ollama_backoff_until:
            order.append("ollama")
        if os.getenv("OPENAI_API_KEY"):
            order.append("openai")
        if not order and self._ollama_backoff_until > now:
            # Backoff expired check
            order.append("ollama")
        if not order:
            order.append("ollama")  # Will fail with clear error
        return order

    def _provider_model(self, provider: str) -> str:
        """Return the configured model alias used for cache identity."""
        if provider == "openai":
            return str(self.config.get("openai_model", "gpt-4o-mini"))
        if provider == "ollama":
            return self.model
        if provider == "mock":
            return "mock"
        raise ValueError(f"Unsupported LLM provider '{provider}'")

    def _provider_endpoint(self, provider: str) -> str:
        """Return the configured serving endpoint for provenance only."""

        if provider == "ollama":
            return self.ollama_host
        if provider == "openai":
            return str(
                self.config.get(
                    "openai_base_url",
                    os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1"),
                )
            )
        if provider == "mock":
            return "mock://local"
        return ""

    def _call_ollama(
        self, system_prompt: str, user_prompt: str, temperature: float, json_mode: bool
    ) -> str:
        """Call Ollama API with a hard wall-clock timeout enforced from the
        outside.

        The TUM Ollama gateway can keep an HTTPS streaming connection
        open indefinitely while emitting no chunks. requests/urllib3's
        ``timeout=`` does not enforce per-recv reads under
        ``stream=True``, and shutting down the underlying socket from
        another thread does not unblock a blocked SSL read. The only
        reliable escape is to run the HTTP call in a daemon worker
        thread and give up from the calling thread via a bounded
        ``queue.Queue.get(timeout=...)``.

        On timeout a RuntimeError is raised so the retry loop in
        ``_call_with_failover`` fires. The worker thread cannot be
        interrupted mid-read, so it is abandoned: it is a daemon (dies
        at process exit) and writes only to a local queue, so it touches
        no shared state after we move on. The number of simultaneously
        abandoned workers per call site is bounded by ``llm_retries``
        (default 8); they unwind as the dead gateway connections finally
        error out or the process exits.
        """
        import queue
        import threading

        # Worker-thread wall-clock cap (the real wedge escape — see docstring).
        # Default 300s, not 120s: streaming keeps the nginx gateway alive while
        # *tokens flow*, but reasoning models (qwen3.6 emits a long `thinking`
        # trace) over the agentic prompts can run well past 2 min on the
        # occasionally-loaded TUM GPU. A healthy call is ~12s, so the headroom is
        # free in the happy path; it only matters during a slow spell, where it
        # lets a slow-but-progressing call finish instead of being killed and
        # retried (which wastes the elapsed time and can starve a decision).
        # Override per-config with ``llm_hard_timeout_s``.
        hard_timeout_s = self.config.get("llm_hard_timeout_s", 300)

        result_q: "queue.Queue[tuple[str, Any]]" = queue.Queue()

        def _worker() -> None:
            try:
                out = self._call_ollama_inner(
                    system_prompt, user_prompt, temperature, json_mode
                )
                result_q.put(("ok", out))
            except Exception as e:  # noqa: BLE001
                result_q.put(("err", e))

        t = threading.Thread(target=_worker, daemon=True, name="ollama-call")
        t.start()
        try:
            status, payload = result_q.get(timeout=hard_timeout_s)
        except queue.Empty:
            raise RuntimeError(
                f"Ollama call exceeded hard timeout of {hard_timeout_s}s "
                f"(worker thread abandoned)"
            )
        if status == "ok":
            return payload
        raise payload

    def _call_ollama_inner(
        self, system_prompt: str, user_prompt: str, temperature: float, json_mode: bool
    ) -> str:
        """Actual HTTP call (streaming by default). Runs in a worker thread.

        Set ``llm_stream: false`` to fall back to the legacy
        non-streaming path.
        """
        import requests

        url = f"{self.ollama_host}/api/chat"
        stream = bool(self.config.get("llm_stream", True))
        payload: Dict[str, Any] = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "stream": stream,
            "options": {"temperature": temperature},
        }
        if json_mode:
            payload["format"] = "json"

        connect_timeout = self.config.get("llm_connect_timeout", 15)
        # Best-effort socket-level read timeout. Not relied on for hang
        # detection — the outer worker-thread wrapper in _call_ollama is
        # the real escape hatch.
        read_timeout = self.config.get("llm_timeout", 90)

        if not stream:
            resp = requests.post(
                url, json=payload, timeout=(connect_timeout, read_timeout)
            )
            resp.raise_for_status()
            data = resp.json()
            if "eval_count" in data:
                self._total_completion_tokens += data["eval_count"]
            if "prompt_eval_count" in data:
                self._total_prompt_tokens += data["prompt_eval_count"]
            content = data["message"]["content"]
            if not content:
                raise RuntimeError("Ollama non-streaming response empty")
            return _ProviderCompletion(
                content,
                {
                    "actual_model": data.get("model") or self.model,
                    "model_revision": data.get("model_digest")
                    or data.get("digest")
                    or "",
                    "response_id": data.get("id") or "",
                    "endpoint": self.ollama_host,
                },
            )

        # Streaming path
        content_parts: List[str] = []
        eval_count = 0
        prompt_eval_count = 0
        response_model = self.model
        model_revision = ""
        response_id = ""
        with requests.post(
            url, json=payload, timeout=(connect_timeout, read_timeout), stream=True
        ) as resp:
            resp.raise_for_status()
            for raw in resp.iter_lines(decode_unicode=True):
                if not raw:
                    continue
                try:
                    chunk = json.loads(raw)
                except json.JSONDecodeError:
                    continue
                message = chunk.get("message") or {}
                content_parts.append(message.get("content", ""))
                response_model = chunk.get("model") or response_model
                model_revision = (
                    chunk.get("model_digest") or chunk.get("digest") or model_revision
                )
                response_id = chunk.get("id") or response_id
                if chunk.get("done"):
                    eval_count = chunk.get("eval_count", 0)
                    prompt_eval_count = chunk.get("prompt_eval_count", 0)
                    break

        if eval_count:
            self._total_completion_tokens += eval_count
        if prompt_eval_count:
            self._total_prompt_tokens += prompt_eval_count

        content = "".join(content_parts)
        # Guard against a silently-aborted stream (zero chunks / no tokens).
        # Returning "" here would let _call_with_failover treat it as a
        # success and poison the response cache permanently. Raise so the
        # retry/backoff loop fires instead.
        if not content and eval_count == 0 and prompt_eval_count == 0:
            raise RuntimeError(
                "Ollama streaming response empty (no chunks from gateway)"
            )
        return _ProviderCompletion(
            content,
            {
                "actual_model": response_model,
                "model_revision": model_revision,
                "response_id": response_id,
                "endpoint": self.ollama_host,
            },
        )

    def _call_openai(
        self, system_prompt: str, user_prompt: str, temperature: float, json_mode: bool
    ) -> str:
        """Call OpenAI API."""
        from openai import OpenAI

        endpoint = self._provider_endpoint("openai")
        client_kwargs: Dict[str, Any] = {}
        if self.config.get("openai_base_url"):
            client_kwargs["base_url"] = endpoint
        client = OpenAI(**client_kwargs)  # Uses OPENAI_API_KEY env var
        model = self.config.get("openai_model", "gpt-4o-mini")

        kwargs: Dict[str, Any] = {
            "model": model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "temperature": temperature,
        }
        if json_mode:
            kwargs["response_format"] = {"type": "json_object"}

        response = client.chat.completions.create(**kwargs)
        choice = response.choices[0]

        # Track tokens
        if response.usage:
            self._total_prompt_tokens += response.usage.prompt_tokens
            self._total_completion_tokens += response.usage.completion_tokens

        return _ProviderCompletion(
            choice.message.content or "",
            {
                "actual_model": getattr(response, "model", None) or model,
                "model_revision": getattr(response, "system_fingerprint", None) or "",
                "response_id": getattr(response, "id", None) or "",
                "endpoint": endpoint,
            },
        )

    # ------------------------------------------------------------------
    # Mock mode
    # ------------------------------------------------------------------

    def _mock_response(self, user_prompt: str) -> str:
        """Return a deterministic mock response for testing."""
        self._total_calls += 1
        self._last_provider = "mock"
        self._last_latency_s = 0.001
        self._record_call_provenance(
            {
                "provider": "mock",
                "configured_model": "mock",
                "actual_model": "mock",
                "endpoint": "mock://local",
                "invocation": "direct",
                "cache_hit": False,
            }
        )
        # Schedule planner prompt → return a small valid schedule (the planner
        # clamps/pads it to the gap). Otherwise a single-mode selection.
        if "schedule" in user_prompt.lower():
            return json.dumps({
                "mode": "communication",
                "schedule": [["payload_observe", 3], ["payload_compress", 4], ["charging", 10]],
                "rationale": "Mock LLM schedule: communicate during contact, then observe, compress, and charge.",
            })
        return json.dumps({
            "mode": "charging",
            "rationale": "Mock LLM response: defaulting to charging for safety.",
        })

    # ------------------------------------------------------------------
    # Cache
    # ------------------------------------------------------------------

    def _cache_key(
        self,
        system_prompt: str,
        user_prompt: str,
        temperature: float | None,
        *,
        json_mode: bool = False,
        provider: str | None = None,
        model: str | None = None,
    ) -> str:
        """Generate a deterministic, decision-complete cache key.

        ``provider`` and ``model`` are explicit on live paths because ``auto`` is
        a failover policy, not an answering provider. Defaults keep this helper
        convenient for unit tests while still distinguishing configured models.
        """
        effective_temperature = (
            self.temperature if temperature is None else float(temperature)
        )
        effective_provider = provider or self.provider
        effective_model = model or (
            self._provider_model(effective_provider)
            if effective_provider in {"ollama", "openai", "mock"}
            else self.model
        )
        content = json.dumps(
            {
                "schema": CACHE_KEY_SCHEMA,
                "provider": effective_provider,
                "model": effective_model,
                "temperature": effective_temperature,
                "json_mode": bool(json_mode),
                "system_prompt": system_prompt,
                "user_prompt": user_prompt,
            },
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
        )
        return hashlib.sha256(content.encode("utf-8")).hexdigest()

    def _cache_get_entry(self, key: str) -> Optional[Dict[str, Any]]:
        """Read a complete cache entry for response and provenance replay."""

        path = self.cache_dir / f"{key}.json"
        if path.exists():
            try:
                data = json.loads(path.read_text(encoding="utf-8"))
                if not isinstance(data, dict) or data.get("response") is None:
                    return None
                return data
            except (json.JSONDecodeError, KeyError, OSError):
                return None
        return None

    def _cache_get(self, key: str) -> Optional[str]:
        """Backward-compatible response-only cache reader."""

        entry = self._cache_get_entry(key)
        return str(entry["response"]) if entry is not None else None

    def _cache_put(
        self,
        key: str,
        response: str,
        system_prompt: str = "",
        user_prompt: str = "",
        *,
        provider: str | None = None,
        model: str | None = None,
        temperature: float | None = None,
        json_mode: bool = False,
        provenance: Dict[str, Any] | None = None,
    ) -> None:
        """Write a response to cache (with prompts for debugging)."""
        path = self.cache_dir / f"{key}.json"
        effective_temperature = (
            self.temperature if temperature is None else float(temperature)
        )
        call_provenance = dict(provenance or {})
        data = {
            "cache_key_schema": CACHE_KEY_SCHEMA,
            "provider": provider or self.provider,
            "model": model or self.model,
            "actual_model": call_provenance.get(
                "actual_model", model or self.model
            ),
            "model_revision": call_provenance.get("model_revision", ""),
            "response_id": call_provenance.get("response_id", ""),
            "endpoint": call_provenance.get("endpoint", ""),
            "invocation": call_provenance.get("invocation", "direct"),
            "temperature": effective_temperature,
            "json_mode": bool(json_mode),
            "system_prompt": system_prompt,
            "user_prompt": user_prompt,
            "response": response,
            "timestamp": time.time(),
        }
        path.write_text(json.dumps(data, indent=2), encoding="utf-8")
