#!/usr/bin/env python3
"""
md2pdf.py -- Markdown to PDF 변환 스크립트

한국어 완벽 지원 (NanumGothic / NanumGothicCoding)을 포함한
Markdown -> HTML -> PDF 변환 도구.

의존 패키지:
    pip install markdown weasyprint pygments

사용법:
    # 단일 파일 변환
    python md2pdf.py input.md

    # 출력 파일 지정
    python md2pdf.py input.md -o output.pdf

    # 디렉토리 내 모든 .md 파일 일괄 변환
    python md2pdf.py --dir /path/to/dir

    # 디렉토리 재귀 변환 (하위 디렉토리 포함)
    python md2pdf.py --dir /path/to/dir --recursive

    # 자세한 출력
    python md2pdf.py input.md -v
"""

import argparse
import os
import re
import sys
import logging
from datetime import datetime
from pathlib import Path

import markdown
from markdown.extensions.toc import TocExtension
from pygments.formatters import HtmlFormatter
import weasyprint

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# 내장 CSS 스타일
# ---------------------------------------------------------------------------
CSS_STYLE = """
/* ── 페이지 설정 ────────────────────────────────────── */
@page {
    size: A4;
    margin: 2.5cm;

    @bottom-center {
        content: counter(page);
        font-family: 'NanumGothic', sans-serif;
        font-size: 9pt;
        color: #888;
    }
}

/* ── 기본 본문 ──────────────────────────────────────── */
body {
    font-family: 'NanumGothic', 'Nanum Gothic', sans-serif;
    font-size: 11pt;
    line-height: 1.7;
    color: #222;
    word-break: keep-all;
    overflow-wrap: break-word;
}

/* ── 제목 ───────────────────────────────────────────── */
h1 {
    font-size: 22pt;
    font-weight: bold;
    margin-top: 1.5em;
    margin-bottom: 0.6em;
    padding-bottom: 0.3em;
    border-bottom: 2px solid #333;
    color: #111;
    page-break-after: avoid;
}
h2 {
    font-size: 17pt;
    font-weight: bold;
    margin-top: 1.3em;
    margin-bottom: 0.5em;
    padding-bottom: 0.2em;
    border-bottom: 1px solid #ccc;
    color: #222;
    page-break-after: avoid;
}
h3 {
    font-size: 14pt;
    font-weight: bold;
    margin-top: 1.1em;
    margin-bottom: 0.4em;
    color: #333;
    page-break-after: avoid;
}
h4 {
    font-size: 12pt;
    font-weight: bold;
    margin-top: 1em;
    margin-bottom: 0.3em;
    color: #444;
    page-break-after: avoid;
}
h5, h6 {
    font-size: 11pt;
    font-weight: bold;
    margin-top: 0.8em;
    margin-bottom: 0.3em;
    color: #555;
    page-break-after: avoid;
}

/* ── 목차 (TOC) ─────────────────────────────────────── */
.toc {
    background: #f8f9fa;
    border: 1px solid #dee2e6;
    border-radius: 6px;
    padding: 1em 1.5em;
    margin-bottom: 2em;
    page-break-after: always;
}
.toc > ul {
    margin: 0;
    padding-left: 1.2em;
}
.toc ul {
    list-style-type: none;
    padding-left: 1.2em;
}
.toc li {
    margin: 0.2em 0;
    font-size: 10.5pt;
}
.toc a {
    text-decoration: none;
    color: #2c5282;
}

/* ── 단락, 리스트 ──────────────────────────────────── */
p {
    margin: 0.5em 0;
}
ul, ol {
    margin: 0.5em 0 0.5em 1.5em;
    padding: 0;
}
li {
    margin: 0.2em 0;
}

/* ── 인라인 코드 ───────────────────────────────────── */
code {
    font-family: 'NanumGothicCoding', 'Nanum Gothic Coding', 'Consolas', monospace;
    font-size: 9.5pt;
    background-color: #f0f0f0;
    padding: 0.15em 0.4em;
    border-radius: 3px;
    color: #c7254e;
}

/* ── 코드 블록 ─────────────────────────────────────── */
pre {
    font-family: 'NanumGothicCoding', 'Nanum Gothic Coding', 'Consolas', monospace;
    font-size: 9pt;
    line-height: 1.5;
    background-color: #f5f5f5;
    border: 1px solid #ddd;
    border-radius: 4px;
    padding: 0.8em 1em;
    overflow-x: auto;
    white-space: pre-wrap;
    word-wrap: break-word;
    page-break-inside: avoid;
}
pre code {
    background: none;
    padding: 0;
    border-radius: 0;
    color: inherit;
    font-size: inherit;
}

/* ── Pygments 코드 하이라이트 ──────────────────────── */
.codehilite {
    font-family: 'NanumGothicCoding', 'Nanum Gothic Coding', 'Consolas', monospace;
    font-size: 9pt;
    line-height: 1.5;
    background-color: #f5f5f5;
    border: 1px solid #ddd;
    border-radius: 4px;
    padding: 0.8em 1em;
    overflow-x: auto;
    white-space: pre-wrap;
    word-wrap: break-word;
    page-break-inside: avoid;
}
.codehilite pre {
    background: none;
    border: none;
    padding: 0;
    margin: 0;
}

/* ── 테이블 ────────────────────────────────────────── */
table {
    border-collapse: collapse;
    width: 100%;
    margin: 1em 0;
    font-size: 10pt;
    page-break-inside: avoid;
}
th, td {
    border: 1px solid #bbb;
    padding: 0.5em 0.7em;
    text-align: left;
    vertical-align: top;
}
th {
    background-color: #e8eef4;
    font-weight: bold;
    color: #333;
}
tr:nth-child(even) {
    background-color: #f7f9fb;
}
tr:hover {
    background-color: #edf2f7;
}

/* ── 인용 블록 ─────────────────────────────────────── */
blockquote {
    margin: 0.8em 0;
    padding: 0.5em 1em;
    border-left: 4px solid #4a90d9;
    background-color: #f0f5fa;
    color: #444;
    font-size: 10.5pt;
}
blockquote p {
    margin: 0.3em 0;
}

/* ── 수평선 ────────────────────────────────────────── */
hr {
    border: none;
    border-top: 1px solid #ccc;
    margin: 1.5em 0;
}

/* ── 링크 ──────────────────────────────────────────── */
a {
    color: #2c5282;
    text-decoration: none;
}

/* ── 이미지 ────────────────────────────────────────── */
img {
    max-width: 100%;
    height: auto;
    display: block;
    margin: 1em auto;
}

/* ── 강조 ──────────────────────────────────────────── */
strong {
    font-weight: bold;
    color: #1a1a1a;
}
em {
    font-style: italic;
}

/* ── 정의 리스트 ───────────────────────────────────── */
dt {
    font-weight: bold;
    margin-top: 0.5em;
}
dd {
    margin-left: 1.5em;
    margin-bottom: 0.3em;
}

/* ── 표지 페이지 ──────────────────────────────────────── */
@page cover {
    margin: 2.5cm;
    @bottom-center { content: none; }
}
.cover-page {
    page: cover;
    page-break-after: always;
    display: flex;
    flex-direction: column;
    justify-content: center;
    align-items: center;
    min-height: 80vh;
    text-align: center;
}
.cover-page .cover-title {
    font-size: 28pt;
    font-weight: bold;
    color: #1a1a2e;
    line-height: 1.3;
    margin-bottom: 0.8em;
    padding: 0 1em;
}
.cover-page .cover-divider {
    width: 60%;
    height: 3px;
    background: linear-gradient(90deg, transparent, #2c5282, transparent);
    margin: 1em auto;
}
.cover-page .cover-meta {
    font-size: 13pt;
    color: #555;
    line-height: 2;
    margin-top: 1em;
}
.cover-page .cover-meta .label {
    color: #888;
    font-size: 10pt;
}
.cover-page .cover-project {
    font-size: 11pt;
    color: #777;
    margin-top: 3em;
    padding-top: 1.5em;
    border-top: 1px solid #ddd;
}
"""

# Pygments 스타일 CSS (코드 하이라이트)
PYGMENTS_CSS = HtmlFormatter(style="default").get_style_defs(".codehilite")


def _extract_cover_info(md_text: str) -> dict:
    """Markdown 텍스트에서 표지에 사용할 정보를 추출한다.

    추출 항목:
        - title: 첫 번째 H1 제목
        - date: 본문 내 날짜 패턴 (YYYY-MM-DD) 또는 오늘 날짜
        - author: '작성자:' 패턴에서 추출, 없으면 기본값
    """
    info: dict = {}

    # 제목: 첫 번째 # 헤딩
    m = re.search(r"^#\s+(.+)$", md_text, re.MULTILINE)
    info["title"] = m.group(1).strip() if m else "Untitled"

    # 날짜: '> ... YYYY-MM-DD' 또는 본문 내 날짜
    m = re.search(r"(\d{4}-\d{2}-\d{2})", md_text)
    info["date"] = m.group(1) if m else datetime.now().strftime("%Y-%m-%d")

    info["author"] = "노승우"

    return info


def _build_cover_html(info: dict) -> str:
    """표지 페이지 HTML을 생성한다."""
    author_line = ""
    if info["author"]:
        author_line = f'<div><span class="label">작성자</span><br>{info["author"]}</div>'

    return f"""<div class="cover-page">
    <div class="cover-title">{info["title"]}</div>
    <div class="cover-divider"></div>
    <div class="cover-meta">
        <div><span class="label">날짜</span><br>{info["date"]}</div>
        {author_line}
    </div>
    <div class="cover-project">Alpamayo VRAM Optimization Research</div>
</div>
"""


def md_to_html(md_text: str, base_path: str = ".") -> str:
    """Markdown 텍스트를 HTML 문서로 변환한다.

    Args:
        md_text: Markdown 원문 문자열.
        base_path: 이미지 등 상대 경로의 기준 디렉토리.

    Returns:
        완전한 HTML 문서 문자열.
    """
    extensions = [
        "tables",
        "fenced_code",
        "codehilite",
        "attr_list",
        "def_list",
        "footnotes",
        "md_in_html",
        "sane_lists",
        "smarty",
        TocExtension(
            title="목차 (Table of Contents)",
            toc_depth="2-3",
            permalink=False,
        ),
    ]
    extension_configs = {
        "codehilite": {
            "css_class": "codehilite",
            "guess_lang": True,
            "use_pygments": True,
        },
    }

    md = markdown.Markdown(
        extensions=extensions,
        extension_configs=extension_configs,
    )
    body = md.convert(md_text)
    toc = getattr(md, "toc", "")

    # 표지 생성
    cover_info = _extract_cover_info(md_text)
    cover_html = _build_cover_html(cover_info)

    html = f"""<!DOCTYPE html>
<html lang="ko">
<head>
<meta charset="utf-8">
<style>
{CSS_STYLE}
{PYGMENTS_CSS}
</style>
</head>
<body>
{cover_html}
{toc}
{body}
</body>
</html>"""
    return html


def html_to_pdf(html: str, output_path: str, base_url: str = ".") -> None:
    """HTML 문자열을 PDF 파일로 변환한다.

    Args:
        html: 변환할 HTML 문서 문자열.
        output_path: 출력 PDF 파일 경로.
        base_url: 상대 경로 리소스(이미지 등)의 기준 URL/경로.
    """
    doc = weasyprint.HTML(string=html, base_url=base_url)
    doc.write_pdf(output_path)


def convert_file(input_path: str, output_path: str | None = None) -> str:
    """단일 Markdown 파일을 PDF로 변환한다.

    Args:
        input_path: 입력 .md 파일 경로.
        output_path: 출력 .pdf 파일 경로. None이면 입력 파일과 같은
                     디렉토리에 확장자만 .pdf로 바꿔 생성한다.

    Returns:
        생성된 PDF 파일의 절대 경로.
    """
    input_path = os.path.abspath(input_path)
    if output_path is None:
        output_path = os.path.splitext(input_path)[0] + ".pdf"
    else:
        output_path = os.path.abspath(output_path)

    base_dir = os.path.dirname(input_path)

    logger.info("변환 중: %s -> %s", input_path, output_path)

    with open(input_path, "r", encoding="utf-8") as f:
        md_text = f.read()

    html = md_to_html(md_text, base_path=base_dir)
    # base_url을 file:// URI로 설정하여 상대 경로 이미지를 올바르게 해석
    base_url = Path(base_dir).as_uri() + "/"
    html_to_pdf(html, output_path, base_url=base_url)

    logger.info("완료: %s (%.1f KB)", output_path, os.path.getsize(output_path) / 1024)
    return output_path


def convert_directory(dir_path: str, recursive: bool = False) -> list[str]:
    """디렉토리 내 모든 .md 파일을 PDF로 일괄 변환한다.

    Args:
        dir_path: 대상 디렉토리 경로.
        recursive: True이면 하위 디렉토리까지 재귀 탐색.

    Returns:
        생성된 PDF 파일 경로 리스트.
    """
    dir_path = os.path.abspath(dir_path)
    results: list[str] = []

    if recursive:
        md_files = sorted(Path(dir_path).rglob("*.md"))
    else:
        md_files = sorted(Path(dir_path).glob("*.md"))

    if not md_files:
        logger.warning("변환할 .md 파일이 없습니다: %s", dir_path)
        return results

    logger.info("%d개의 .md 파일 발견 (%s)", len(md_files), dir_path)

    for md_file in md_files:
        try:
            pdf_path = convert_file(str(md_file))
            results.append(pdf_path)
        except Exception as e:
            logger.error("변환 실패: %s -- %s", md_file, e)

    return results


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Markdown -> PDF 변환 (한국어 지원, WeasyPrint 기반)",
        epilog=(
            "예시:\n"
            "  python md2pdf.py README.md\n"
            "  python md2pdf.py README.md -o docs/readme.pdf\n"
            "  python md2pdf.py --dir ./docs\n"
            "  python md2pdf.py --dir ./docs --recursive\n"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "input",
        nargs="?",
        help="변환할 Markdown 파일 경로",
    )
    parser.add_argument(
        "-o", "--output",
        help="출력 PDF 파일 경로 (단일 파일 변환 시)",
    )
    parser.add_argument(
        "--dir",
        help="일괄 변환할 디렉토리 경로",
    )
    parser.add_argument(
        "--recursive", "-r",
        action="store_true",
        help="하위 디렉토리까지 재귀 탐색 (--dir와 함께 사용)",
    )
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="상세 로그 출력",
    )

    args = parser.parse_args()

    # 로깅 설정
    log_level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=log_level,
        format="[%(levelname)s] %(message)s",
    )

    # 인자 검증
    if args.input is None and args.dir is None:
        parser.error("변환할 파일(input) 또는 디렉토리(--dir)를 지정하세요.")

    if args.input and args.dir:
        parser.error("input과 --dir은 동시에 사용할 수 없습니다.")

    if args.output and args.dir:
        parser.error("-o/--output은 단일 파일 변환 시에만 사용할 수 있습니다.")

    if args.recursive and not args.dir:
        parser.error("--recursive는 --dir과 함께 사용해야 합니다.")

    # 변환 실행
    if args.dir:
        results = convert_directory(args.dir, recursive=args.recursive)
        print(f"\n변환 완료: {len(results)}개 파일")
        for pdf in results:
            print(f"  {pdf}")
    else:
        if not os.path.isfile(args.input):
            logger.error("파일을 찾을 수 없습니다: %s", args.input)
            sys.exit(1)
        try:
            pdf_path = convert_file(args.input, args.output)
            print(f"\n변환 완료: {pdf_path}")
        except Exception as e:
            logger.error("변환 실패: %s", e)
            sys.exit(1)


if __name__ == "__main__":
    main()
