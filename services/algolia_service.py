"""
Step 3 — Algolia Typo-Tolerant Inventory Search.

For every line flagged as requires_review, this service queries the Algolia
index using the official Python client (v4 async API).  Algolia's built-in
typo-tolerance handles the garbled OCR strings without any pre-processing.
"""

import asyncio
import logging
from typing import Any

from algoliasearch.search.client import SearchClient
from algoliasearch.search.models.search_params_object import SearchParamsObject

from config import settings
from models import AlgoliaMatch

logger = logging.getLogger(__name__)

# Module-level client — created once and reused across requests.
# The v4 async client is safe to share between coroutines.
_client: SearchClient | None = None


def _get_client() -> SearchClient:
    global _client
    if _client is None:
        _client = SearchClient(
            app_id=settings.ALGOLIA_APP_ID,
            api_key=settings.ALGOLIA_API_KEY,
        )
        logger.info("Algolia SearchClient initialised (app_id=%s).", settings.ALGOLIA_APP_ID)
    return _client


def _parse_hit(hit: Any) -> AlgoliaMatch:
    """
    Extract a clean AlgoliaMatch from a single Algolia Hit object.

    algoliasearch v4 uses Pydantic with extra="allow", so user-defined index
    fields (item_name, standard_quantity, …) are stored in hit.model_extra.
    Reserved fields (objectID → object_id) are first-class attributes.
    """
    # User-defined fields land in model_extra for Pydantic v2 extra="allow" models.
    attrs: dict[str, Any] = getattr(hit, "model_extra", {}) or {}

    object_id: str = getattr(hit, "object_id", "") or ""
    item_name: str = attrs.get("item_name", "")
    standard_quantity: str | None = attrs.get("standard_quantity")

    # Everything else passes through as metadata.
    metadata = {
        k: v for k, v in attrs.items()
        if k not in {"item_name", "standard_quantity"}
    }

    return AlgoliaMatch(
        object_id=object_id,
        matched_item_name=item_name,
        standard_quantity=standard_quantity,
        metadata=metadata,
    )


async def search_inventory(query_text: str) -> AlgoliaMatch | None:
    """
    Query the Algolia inventory index for *query_text*.

    Returns the top AlgoliaMatch, or None if there are no results or the
    request fails.  Algolia's typo-tolerance is enabled by default on all
    indices and requires no extra configuration here.
    """
    client = _get_client()

    params = SearchParamsObject(
        query=query_text,
        hits_per_page=1,        # we only need the best match
        attributes_to_retrieve=["item_name", "standard_quantity"],
        attributes_to_highlight=[],  # skip highlight markup — we don't need it
    )

    try:
        response = await client.search_single_index(
            index_name=settings.ALGOLIA_INDEX_NAME,
            search_params=params,
        )
    except Exception as exc:
        logger.error("Algolia search error for query '%s': %s", query_text, exc)
        return None

    hits = response.hits if response.hits else []
    if not hits:
        logger.warning("Algolia returned no hits for query: '%s'", query_text)
        return None

    match = _parse_hit(hits[0])
    logger.info(
        "Algolia matched '%s' → '%s' (objectID=%s)",
        query_text,
        match.matched_item_name,
        match.object_id,
    )
    return match


async def search_flagged_lines(flagged_texts: list[str]) -> list[AlgoliaMatch | None]:
    """
    Run search_inventory concurrently for every flagged line and return
    results in the same order as the input list.
    """
    tasks = [search_inventory(text) for text in flagged_texts]
    return list(await asyncio.gather(*tasks))
