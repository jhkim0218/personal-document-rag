from __future__ import annotations


def related_bundle(query: str, results, limit: int = 5) -> dict[str, object]:
    """Diversify an already-ranked retrieval list without inventing document relations."""
    sources = []
    paths = set()
    for rank, result in enumerate(results, start=1):
        if result.path in paths:
            continue
        paths.add(result.path)
        source = result.as_dict()
        source['retrieval_rank'] = rank
        sources.append(source)
        if len(sources) == limit:
            break
    return {'query': query, 'sources': sources, 'candidate_chunks': len(results),
            'candidate_documents': len({result.path for result in results}), 'documents_selected': len(paths),
            'provenance': 'One highest-ranked chunk per document from the normal retrieval result; no inferred relationships',
            'limitations': 'Diversity cannot recover a document absent from retrieval. Verify whether these documents answer complementary parts of the question.'}
