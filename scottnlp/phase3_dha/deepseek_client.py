"""Cached, rate-limited DeepSeek API client using OpenAI-compatible interface."""

import hashlib
import json
import os
import re
import tempfile
import threading
import time
from pathlib import Path

from dotenv import load_dotenv
from openai import (
    APIConnectionError,
    APITimeoutError,
    BadRequestError,
    OpenAI,
    RateLimitError,
)

from scottnlp.config import (
    DEEPSEEK_BASE_URL,
    DEEPSEEK_MAX_RETRIES,
    DEEPSEEK_MAX_TOKENS,
    DEEPSEEK_MODEL,
    DEEPSEEK_RATE_LIMIT_RPS,
    DEEPSEEK_RETRY_BASE_DELAY,
    DEEPSEEK_TEMPERATURE,
    OUTPUT_DIR,
)

SYSTEM_PROMPT = (
    "You are an expert in Critical Discourse Analysis, specifically the "
    "Discourse-Historical Approach (DHA) developed by Reisigl & Wodak (2001, 2009). "
    "You are analyzing Scottish legal and policy documents about language policy "
    "(English, Gaelic, and Scots languages). "
    "Always respond with valid JSON matching the requested schema exactly. "
    "Base all analysis strictly on the provided text. Do not hallucinate evidence."
)


class DeepSeekClient:
    """Cached, rate-limited wrapper around the DeepSeek chat API."""

    def __init__(
        self,
        api_key: str | None = None,
        model: str = DEEPSEEK_MODEL,
        base_url: str = DEEPSEEK_BASE_URL,
        cache_path: Path | None = None,
    ):
        # Load .env file from project root
        load_dotenv(Path(__file__).resolve().parent.parent.parent / ".env")
        self._api_key = api_key or os.environ.get("DEEPSEEK_API_KEY")
        if not self._api_key:
            raise EnvironmentError(
                "DEEPSEEK_API_KEY not set. Export it as an environment variable:\n"
                "  export DEEPSEEK_API_KEY='your-key-here'"
            )

        self._model = model
        self._client = OpenAI(base_url=base_url, api_key=self._api_key)
        self._cache_path = cache_path or (OUTPUT_DIR / "phase3" / "api_cache.json")
        self._cache: dict[str, dict] = self._load_cache()

        # Rate limiting
        self._min_interval = 1.0 / DEEPSEEK_RATE_LIMIT_RPS
        self._last_request_time = 0.0

        # Stats
        self._cache_hits = 0
        self._api_calls = 0
        self._api_errors = 0

        # Thread safety
        self._lock = threading.Lock()
        self._save_counter = 0

    # ── Cache ─────────────────────────────────────────────────────────

    def _load_cache(self) -> dict[str, dict]:
        if self._cache_path.exists():
            with open(self._cache_path, "r", encoding="utf-8") as f:
                return json.load(f)
        return {}

    def _save_cache(self) -> None:
        self._cache_path.parent.mkdir(parents=True, exist_ok=True)
        # Atomic write via temp file + rename
        fd, tmp = tempfile.mkstemp(
            dir=self._cache_path.parent, suffix=".tmp"
        )
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as f:
                json.dump(self._cache, f, ensure_ascii=False)
            os.replace(tmp, self._cache_path)
        except BaseException:
            os.unlink(tmp)
            raise

    @staticmethod
    def _cache_key(prompt: str) -> str:
        return hashlib.sha256(prompt.encode("utf-8")).hexdigest()

    def clear_cache(self) -> None:
        self._cache = {}
        if self._cache_path.exists():
            self._cache_path.unlink()

    # ── Rate limiting ─────────────────────────────────────────────────

    def _rate_limit(self) -> None:
        now = time.monotonic()
        elapsed = now - self._last_request_time
        if elapsed < self._min_interval:
            time.sleep(self._min_interval - elapsed)
        self._last_request_time = time.monotonic()

    # ── API call ──────────────────────────────────────────────────────

    def _call_api(self, prompt: str) -> dict | None:
        """Make a single API call with retry logic. Returns parsed JSON or None."""
        for attempt in range(DEEPSEEK_MAX_RETRIES):
            try:
                response = self._client.chat.completions.create(
                    model=self._model,
                    messages=[
                        {"role": "system", "content": SYSTEM_PROMPT},
                        {"role": "user", "content": prompt},
                    ],
                    temperature=DEEPSEEK_TEMPERATURE,
                    max_tokens=DEEPSEEK_MAX_TOKENS,
                    response_format={"type": "json_object"},
                )
                with self._lock:
                    self._api_calls += 1
                content = response.choices[0].message.content
                if content is None:
                    print("  Empty response from API")
                    with self._lock:
                        self._api_errors += 1
                    return None
                return self._parse_response(content)

            except (RateLimitError, APIConnectionError, APITimeoutError) as e:
                delay = DEEPSEEK_RETRY_BASE_DELAY * (2 ** attempt)
                print(f"  API error (attempt {attempt + 1}/{DEEPSEEK_MAX_RETRIES}): {e}")
                print(f"  Retrying in {delay:.1f}s...")
                time.sleep(delay)

            except BadRequestError as e:
                print(f"  Bad request (not retrying): {e}")
                with self._lock:
                    self._api_errors += 1
                return None

        with self._lock:
            self._api_errors += 1
        print(f"  All {DEEPSEEK_MAX_RETRIES} attempts failed.")
        return None

    @staticmethod
    def _parse_response(content: str) -> dict | None:
        """Parse JSON from the API response, with fallback for markdown blocks."""
        # Direct parse
        try:
            return json.loads(content)
        except json.JSONDecodeError:
            pass

        # Fallback: extract JSON from markdown code blocks
        match = re.search(r"```(?:json)?\s*\n?(.*?)\n?```", content, re.DOTALL)
        if match:
            try:
                return json.loads(match.group(1))
            except json.JSONDecodeError:
                pass

        # Fallback: repair truncated JSON (max_tokens exceeded)
        repaired = DeepSeekClient._repair_truncated_json(content)
        if repaired:
            print(f"  Repaired truncated JSON for strategy={repaired.get('strategy_name')}")
            return repaired

        print(f"  Failed to parse JSON response: {content[:200]}...")
        return None

    @staticmethod
    def _repair_truncated_json(content: str) -> dict | None:
        """Attempt to recover key fields from truncated JSON responses."""
        result = {}

        m = re.search(r'"strategy_name":\s*"([^"]+)"', content)
        if not m:
            return None
        result["strategy_name"] = m.group(1)

        m = re.search(r'"present":\s*(true|false)', content)
        if m:
            result["present"] = m.group(1) == "true"
        else:
            return None

        m = re.search(r'"confidence":\s*([0-9.]+)', content)
        if m:
            result["confidence"] = float(m.group(1))
        else:
            result["confidence"] = 0.0

        # Extract whatever complete quotes we can
        quotes = []
        eq_match = re.search(r'"evidence_quotes":\s*\[', content)
        if eq_match:
            rest = content[eq_match.end():]
            for qm in re.finditer(r'"((?:[^"\\]|\\.)*)"', rest):
                q = qm.group(1)
                if len(q) > 5:
                    quotes.append(q)
        result["evidence_quotes"] = quotes
        result["linguistic_devices"] = []
        result["target_languages"] = []
        result["notes"] = "Repaired from truncated API response (max_tokens exceeded)."
        result["repaired"] = True

        return result

    # ── Public interface ──────────────────────────────────────────────

    def classify(self, prompt: str) -> dict | None:
        """Classify via DeepSeek with caching. Thread-safe."""
        key = self._cache_key(prompt)

        with self._lock:
            if key in self._cache:
                self._cache_hits += 1
                return self._cache[key]

        # API call outside lock (allows concurrency)
        result = self._call_api(prompt)

        if result is not None:
            with self._lock:
                self._cache[key] = result
                self._save_counter += 1
                # Batch save every 10 results to reduce I/O
                if self._save_counter % 10 == 0:
                    self._save_cache()

        return result

    def flush_cache(self) -> None:
        """Force save any unsaved cache entries to disk."""
        with self._lock:
            self._save_cache()

    @property
    def stats(self) -> dict:
        return {
            "cache_hits": self._cache_hits,
            "api_calls": self._api_calls,
            "api_errors": self._api_errors,
            "cache_size": len(self._cache),
        }
