# Hybrid retrieval 실험 노트

Hybrid retrieval은 BM25 키워드 검색과 embedding 기반 vector 검색의 후보 순위를 reciprocal rank fusion(RRF)으로 합친다. 약어, 에러 코드, 제품명처럼 정확한 단어가 중요한 질의는 BM25가 강했다.

의미가 비슷하지만 표현이 달라지는 질문은 vector 검색이 보완했다. 그래서 Nebula의 기본 검색 방식은 hybrid retrieval로 결정했다.
