# 로컬 운영 런북

처음 실행할 때 `pip install -r requirements.txt`로 PDF 파서 의존성을 설치한다. 그 다음 `python app.py --data <문서폴더>`로 로컬 서버를 시작하고 브라우저에서 127.0.0.1:8000을 연다.

문서를 추가하거나 수정한 뒤에는 “변경 파일 색인”을 누른다. 인덱스 SQLite 파일은 원본 폴더가 아닌 `.local` 경로에 둔다.
