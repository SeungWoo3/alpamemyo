#!/bin/bash
# Alpamayo-R1-10B 파이프라인 분석 — 전체 시각화 일괄 생성
# 사용법: bash analysis/generate_all.sh

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")/scripts" && pwd)"
FIG_DIR="$(cd "$(dirname "$0")/figures" && pwd)"

echo "=== Alpamayo-R1-10B 시각화 생성 시작 ==="
echo "Scripts: $SCRIPT_DIR"
echo "Figures: $FIG_DIR"
echo ""

# 1. 전체 파이프라인 흐름도
echo "[1/6] 파이프라인 흐름도 생성..."
python3 "$SCRIPT_DIR/visualize_pipeline.py"

# 2. 텐서 차원 변화 차트
echo "[2/6] 텐서 차원 변화 차트 생성..."
python3 "$SCRIPT_DIR/visualize_dimensions.py"

# 3. Vision Encoder 내부 구조
echo "[3/6] Vision Encoder 구조도 생성..."
python3 "$SCRIPT_DIR/visualize_vision_encoder.py"

# 4. Diffusion 10-step 과정
echo "[4/6] Diffusion 과정 시각화 생성..."
python3 "$SCRIPT_DIR/visualize_diffusion.py"

# 5. Action→Trajectory 변환
echo "[5/6] Action→Trajectory 변환 시각화 생성..."
python3 "$SCRIPT_DIR/visualize_action_traj.py"

# 6. 전체 모델 아키텍처 블록도
echo "[6/6] 전체 아키텍처 블록도 생성..."
python3 "$SCRIPT_DIR/visualize_architecture.py"

echo ""
echo "=== 완료 ==="
echo "생성된 파일:"
ls -la "$FIG_DIR"/*.png 2>/dev/null || echo "(PNG 파일 없음)"
