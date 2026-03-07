---
name: notion-sync
description: 실험 결과를 Notion Carlamayo 페이지에 동기화. 최신 work-log 또는 지정된 실험 결과를 Notion에 반영.
disable-model-invocation: false
allowed-tools: Read, Bash, Glob, Grep
---

# Notion 동기화

Carlamayo 프로젝트의 Notion 페이지에 실험 결과를 동기화합니다.

## Notion 정보
- **페이지 ID**: 3190badf-7252-80a7-91d8-f31f9bfb80bb
- **주요 섹션**: Feasibility Test

## 절차

1. $ARGUMENTS가 있으면 해당 실험 폴더의 결과를 읽음
2. $ARGUMENTS가 없으면 최신 work-log 파일을 읽음
3. Notion MCP 도구(`mcp__notion__notion-fetch`)로 현재 페이지 내용 확인
4. `mcp__notion__notion-update-page`로 Feasibility Test 섹션에 결과 추가

## 작성 규칙
- 어투: 간결한 명사형 ("실험 진행", "확인", "개선")
- 수치 데이터 필수 포함
- 이전 내용과 중복되지 않도록 확인
