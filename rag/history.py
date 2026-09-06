from __future__ import annotations

from datetime import date
import re

from .text import split_sentences


_DATE = re.compile(r"(?P<year>20\d{2})[-./년\s]+(?P<month>\d{1,2})[-./월\s]+(?P<day>\d{1,2})일?")
_CHANGE = re.compile(r"\b(changed|updated|revised)\b|변경(?:됨|했다|되었습니다)?|바뀜", re.IGNORECASE)
_ARROW = re.compile(r"(?P<before>[^\n.。;]{1,120}?)\s*(?:→|->)\s*(?P<after>[^\n.。;]{1,120})")
_ENGLISH_FROM_TO = re.compile(r"from\s+(?P<before>[^\n.。;]{1,100}?)\s+to\s+(?P<after>[^\n.。;]{1,100}?)(?=\s+(?:because|due to|reason|changed|updated|revised)|[.。;]|$)", re.IGNORECASE)
_KOREAN_CHANGE_VALUE = r"(?:20\d{2}[-./]\d{1,2}[-./]\d{1,2}|\d+\s*월\s*\d+\s*일)"
_KOREAN_FROM_TO = re.compile(rf"(?P<before>{_KOREAN_CHANGE_VALUE})에서\s+(?P<after>{_KOREAN_CHANGE_VALUE})(?:으로|로)\s*변경", re.IGNORECASE)
_REASON = re.compile(r"(?:because|due to|reason|이유(?:는|은|가)?|원인(?:은|이었|이)?)\s*[:：]?\s*(?P<reason>[^\n.。]+)", re.IGNORECASE)


def _date_value(sentence: str):
    match = _DATE.search(sentence)
    if not match:
        return None, None
    raw = match.group(0)
    try:
        return raw, date(int(match['year']), int(match['month']), int(match['day']))
    except ValueError:
        return raw, None


def _change_values(sentence: str):
    match = _ARROW.search(sentence) or _ENGLISH_FROM_TO.search(sentence) or _KOREAN_FROM_TO.search(sentence)
    if not match:
        return None, None
    before = re.split(r":\s*", match['before'].strip())[-1]
    after = re.split(r"\s+(?:changed|updated|revised|because|due to|reason|변경(?:됨|했다|되었습니다)?|이유|원인)\b", match['after'].strip(), flags=re.IGNORECASE)[0]
    return before, after.strip()


def decision_history(query: str, results) -> dict[str, object]:
    """Return only literal change statements; absent fields are never inferred."""
    events = []
    seen = set()
    for result in results:
        sentences = split_sentences(result.text)
        for number, sentence in enumerate(sentences):
            if not _CHANGE.search(sentence):
                continue
            key = (result.path, result.location, sentence)
            if key in seen:
                continue
            seen.add(key)
            raw_date, sortable_date = _date_value(sentence)
            before, after = _change_values(sentence)
            reason_sentence = sentence if _REASON.search(sentence) else (sentences[number + 1] if number + 1 < len(sentences) and _REASON.search(sentences[number + 1]) else None)
            reason_match = _REASON.search(reason_sentence) if reason_sentence else None
            reason = reason_match['reason'].strip() if reason_match else None
            events.append({'date': raw_date, 'before': before, 'after': after, 'reason': reason, 'reason_statement': reason_sentence,
                           'statement': sentence, 'source': {'chunk_id': result.chunk_id, 'title': result.title,
                           'path': result.path, 'location': result.location}, '_sort_date': sortable_date})
    events.sort(key=lambda event: (event['_sort_date'] is None, event['_sort_date'] or date.max, event['source']['path'], event['source']['location']))
    for event in events:
        event.pop('_sort_date')
    return {'query': query, 'events': events,
            'provenance': 'Literal source sentences only; dates, before/after values, and reasons are null when not explicit',
            'limitations': 'This is not semantic entity resolution. Check each linked source before treating events as one decision history.'}
