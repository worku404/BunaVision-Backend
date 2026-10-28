"""Gemini advisory writer (sync). Structured output, validated again on our side.

Any failure is an explicit `AdvisoryFailed`: there is no offline or canned advisory, because a
made-up advisory would look like a real one. The SDK's own HTTP timeout is the only per-call
bound here (gunicorn's worker timeout is the backstop). SDK names and units (timeout in
milliseconds) were checked against google-genai 2.32.0 by an earlier session, not re-checked.
"""

import logging
import re
from collections.abc import Sequence
from functools import lru_cache
from typing import Any

import httpx
from django.conf import settings
from google import genai
from google.genai import errors as genai_errors
from google.genai import types
from pydantic import ValidationError

from apps.assessments.clients.prompts import (
    MAX_RECOMMENDATIONS,
    SYSTEM_INSTRUCTION,
    AdvisoryPayload,
    BilingualPayload,
    render_facts,
)
from apps.assessments.taxonomy import DefectClass
from apps.common.errors import AdvisoryFailed

logger = logging.getLogger(__name__)

_ETHIOPIC = re.compile(r"[\u1200-\u137f]")
_LATIN = re.compile(r"[A-Za-z]")
_MAX_TEXT_CHARS = 1200

logger = logging.getLogger(__name__)

_KEY_NAMES = ("API1_KEY", "API2_KEY", "API3_KEY", "API4_KEY")  # priority order


def _api_keys() -> list[str]:
    """Configured Gemini keys in priority order; unset or blank ones are skipped."""
    keys = [str(getattr(settings, name, "")).strip() for name in _KEY_NAMES]
    return [key for key in keys if key]


@lru_cache(maxsize=len(_KEY_NAMES))
def get_client(api_key: str) -> genai.Client:
    """One cached client per key."""
    return genai.Client(api_key=api_key)


def _call(api_key: str, contents: str, config: types.GenerateContentConfig) -> str | None:
    try:
        response = get_client(api_key).models.generate_content(
            model=settings.GEMINI_MODEL, contents=contents, config=config
        )
    except genai_errors.APIError as exc:
        raise AdvisoryFailed(f"Gemini API error {exc.code}: {exc}") from exc
    except httpx.HTTPError as exc:
        raise AdvisoryFailed(f"Gemini transport error: {exc!r}") from exc
    return response.text


def generate(
    summary: dict[str, Any], scanned: Sequence[DefectClass], not_scanned: Sequence[DefectClass]
) -> dict[str, Any]:
    keys = _api_keys()
    if not keys:
        raise AdvisoryFailed("no Gemini API key is configured (API1_KEY to API4_KEY)")

    config = types.GenerateContentConfig(
        system_instruction=SYSTEM_INSTRUCTION,
        response_mime_type="application/json",
        response_schema=AdvisoryPayload,
        temperature=settings.GEMINI_TEMPERATURE,
        max_output_tokens=settings.GEMINI_MAX_OUTPUT_TOKENS,
        http_options=types.HttpOptions(timeout=int(settings.GEMINI_TIMEOUT_S * 1000)),
    )
    contents = render_facts(summary, scanned, not_scanned)

    failures: list[AdvisoryFailed] = []
    for position, api_key in enumerate(keys, start=1):
        try:
            text = _call(api_key, contents, config)
        except AdvisoryFailed as exc:
            failures.append(exc)
            logger.warning("Gemini key %d of %d failed: %s", position, len(keys), exc)  # never log the key
            continue
        return _checked(text)  # a call that succeeded is judged on its content, not retried
    raise failures[-1]  # every key failed: surface the last error Gemini returned


def _checked(text: str | None) -> dict[str, Any]:
    if not text:
        raise AdvisoryFailed("Gemini returned no text (blocked, empty or truncated)")
    try:
        payload = AdvisoryPayload.model_validate_json(text)
    except ValidationError as exc:
        raise AdvisoryFailed(f"Gemini reply did not match the schema: {exc}") from exc
    if not 1 <= len(payload.recommendations) <= MAX_RECOMMENDATIONS:
        raise AdvisoryFailed(f"expected 1-{MAX_RECOMMENDATIONS} recommendations, got {len(payload.recommendations)}")
    return {
        "generator_model": settings.GEMINI_MODEL,
        "summary": _bilingual(payload.summary),
        "recommendations": [_bilingual(item) for item in payload.recommendations],
    }


def _bilingual(text: BilingualPayload) -> dict[str, str]:
    """Reject empty, over-long, or wrong-script text (e.g. Amharic written in Latin letters)."""
    en, am = text.en.strip(), text.am.strip()
    if not en or not am:
        raise AdvisoryFailed("advisory text was empty in one language")
    if len(en) > _MAX_TEXT_CHARS or len(am) > _MAX_TEXT_CHARS:
        raise AdvisoryFailed("advisory text exceeded the length limit")
    if not _LATIN.search(en):
        raise AdvisoryFailed("English text contained no Latin letters")
    if not _ETHIOPIC.search(am):
        raise AdvisoryFailed("Amharic text contained no Ethiopic characters")
    return {"en": en, "am": am}
