# Nebula 문서 검색 아키텍처

Nebula의 개인 문서 검색기는 SQLite 인덱스와 로컬 원본 폴더를 분리한다. 원본 파일은 절대 수정하지 않으며, `content_hash`가 달라진 파일만 다시 색인한다.

첫 번째 기준선은 vector-only retrieval이다. 검색 결과는 항상 파일 경로, 헤딩 또는 페이지, 원문 발췌를 함께 반환한다.
