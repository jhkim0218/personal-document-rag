# 0.1 릴리스 노트

0.1 릴리스는 PDF, DOCX, Markdown, TXT 파일을 지원한다. PDF 텍스트 추출은 pypdf 라이브러리를 사용하고, DOCX는 ZIP 내부의 WordprocessingML 문단을 읽는다.

지원하지 않는 확장자나 파싱 실패 파일은 색인 결과에 실패 원인으로 남기되 다른 파일의 색인은 계속한다.
