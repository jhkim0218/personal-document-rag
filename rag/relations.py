from __future__ import annotations

import json
from pathlib import Path


class Relations:
    """Small explicit project → decision → indexed-document map; no graph database or inference."""

    def __init__(self, path: Path):
        self.path = path
        self.data = json.loads(path.read_text(encoding='utf-8')) if path.exists() else {'projects': []}

    @staticmethod
    def _validate(data, document_source):
        if not isinstance(data, dict) or not isinstance(data.get('projects'), list) or len(data['projects']) > 30:
            raise ValueError('projects must be a list of at most 30 entries')
        projects, names = [], set()
        for project in data['projects']:
            if not isinstance(project, dict) or not isinstance(project.get('name'), str) or not 1 <= len(project['name'].strip()) <= 120 or project['name'].strip() in names or not isinstance(project.get('decisions'), list) or len(project['decisions']) > 100:
                raise ValueError('Each project needs a unique name and at most 100 decisions')
            name = project['name'].strip()
            names.add(name)
            decisions, decision_names = [], set()
            for decision in project['decisions']:
                if not isinstance(decision, dict) or not isinstance(decision.get('name'), str) or not 1 <= len(decision['name'].strip()) <= 200 or decision['name'].strip() in decision_names or not isinstance(decision.get('documents'), list) or not 1 <= len(decision['documents']) <= 20:
                    raise ValueError('Each decision needs a unique name and 1–20 indexed documents')
                decision_name = decision['name'].strip()
                decision_names.add(decision_name)
                documents = []
                for value in decision['documents']:
                    if not isinstance(value, str) or not document_source(value):
                        raise ValueError('Relation documents must be indexed files in active source folders')
                    documents.append(str(Path(value).resolve()))
                decisions.append({'name': decision_name, 'documents': documents})
            projects.append({'name': name, 'decisions': decisions})
        return {'projects': projects}

    def configure(self, data, document_source):
        self.data = self._validate(data, document_source)
        temporary = self.path.with_suffix(self.path.suffix + '.tmp')
        temporary.write_text(json.dumps(self.data, ensure_ascii=False, indent=2), encoding='utf-8')
        temporary.replace(self.path)
        return self.data

    def view(self, query, document_source):
        if not isinstance(query, str):
            raise ValueError('query must be text')
        terms, projects, unavailable = query.casefold().strip(), [], 0
        for project in self.data['projects']:
            decisions = []
            for decision in project['decisions']:
                if terms and terms not in project['name'].casefold() and terms not in decision['name'].casefold():
                    continue
                documents = []
                for path in decision['documents']:
                    source = document_source(path)
                    if source:
                        documents.append({key: source[key] for key in ('chunk_id', 'title', 'path', 'location')})
                    else:
                        unavailable += 1
                decisions.append({'name': decision['name'], 'documents': documents})
            if decisions:
                projects.append({'name': project['name'], 'decisions': decisions})
        return {'query': query, 'projects': projects, 'unavailable_documents': unavailable,
                'configuration': self.data,
                'provenance': 'User-entered explicit project, decision and document relations; no inferred links',
                'limitations': 'Only currently indexed active-source documents are shown. This does not prove document coverage or semantic relevance.'}
