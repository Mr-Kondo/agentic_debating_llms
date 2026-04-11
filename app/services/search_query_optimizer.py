"""Optional search query optimizer (DSPy-backed when available)."""

from __future__ import annotations


def optimize_search_query(*, query: str, topic: str, mode: str) -> str:
    """Return optimized query when enabled; otherwise return input unchanged.

    This optimizer is best-effort only. Any failure must return original query so
    search availability is never reduced by optimization.
    """
    if mode != "dspy":
        return query

    try:
        import dspy

        class SearchQueryRewriteSignature(dspy.Signature):
            """Rewrite search query to concise web-search keywords."""

            topic = dspy.InputField(desc="debate topic")
            query = dspy.InputField(desc="original search query")
            optimized_query = dspy.OutputField(desc="optimized query in one line")

        module = dspy.Predict(SearchQueryRewriteSignature)
        result = module(topic=topic, query=query)
        optimized = str(getattr(result, "optimized_query", "")).strip()
        if not optimized:
            return query
        if "\n" in optimized or "\r" in optimized or "\x00" in optimized:
            return query
        return optimized
    except Exception:
        return query
