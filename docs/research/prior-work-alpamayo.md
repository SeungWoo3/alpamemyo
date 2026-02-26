# Alpamayo 선행연구 정리

> 조사일: 2026-02-25

---

## 1. Alpamayo 원본 연구

### 1.1 Alpamayo-R1 (Alpamayo 1) 논문

- **제목**: Alpamayo-R1: Bridging Reasoning and Action Prediction for Generalizable Autonomous Driving in the Long Tail
- **저자**: Yan Wang, Wenjie Luo, Junjie Bai, Yulong Cao, Tong Che, Ke Chen, Yuxiao Chen, Jenna Diamond, Yifan Ding, Wenhao Ding, Liang Feng, Greg Heinrich, Jack Huang, Peter Karkus, Boyi Li, Pinyi Li, Tsung-Yi Lin, Dongran Liu, Ming-Yu Liu, Langechuan Liu, Zhijian Liu, Jason Lu, Yunxiang Mao, Pavlo Molchanov, Lindsey Pavao, Zhenghao Peng, Mike Ranzinger, Ed Schmerling, Shida Shen, Yunfei Shi, Sarah Tariq, Ran Tian, Tilman Wekel, Xinshuo Weng, Tianjun Xiao, Eric Yang, Xiaodong Yang, Yurong You, Xiaohui Zeng, Wenyuan Zhang, Boris Ivanovic, Marco Pavone (총 42명, NVIDIA + Stanford)
- **발표처**: arXiv preprint (arXiv:2511.00088)
- **날짜**: 2025년 10월 30일 초판 제출, 2026년 1월 7일 개정판
- **링크**: https://arxiv.org/abs/2511.00088

#### 핵심 내용 요약

Alpamayo-R1(AR1)은 해석 가능한 추론(interpretable reasoning)과 궤적 계획(trajectory planning)을 통합한 **Vision-Language-Action (VLA) 모델**이다. 자율주행에서 안전에 중요한 롱테일(long-tail) 시나리오를 다루기 위해 설계되었다.

**3대 핵심 기여:**

1. **Chain of Causation (CoC) 데이터셋**: 하이브리드 자동 라벨링 + 인간 참여(human-in-the-loop) 파이프라인으로 구축된 700,000개의 결정 근거 기반(decision-grounded) 인과 추론 트레이스. 각 추론 트레이스는 명시적 운전 결정과 연관되며, 해당 결정을 유발하는 인과 요인만 포함한다.

2. **모듈형 VLA 아키텍처**: Cosmos-Reason (Physical AI용으로 사전학습된 VLM, 8.2B 파라미터) 백본과 diffusion 기반 궤적 디코더(2.3B 파라미터)를 결합. 총 10.5B 파라미터. 입력은 멀티카메라 이미지(4대: front-wide, front-tele, cross-left, cross-right) + 텍스트 명령 + 자기 이동 이력. 출력은 CoC 추론 트레이스 + 6.4초 미래 궤적(64 웨이포인트, 10Hz).

3. **다단계 학습 전략**: Supervised Fine-Tuning(SFT)으로 추론 능력을 유도하고, Reinforcement Learning(RL)으로 후처리하여 추론 품질 45% 향상, 추론-행동 일관성 37% 향상 달성.

**성능:**
- 도전적 사례에서 계획 정확도 최대 12% 향상 (궤적 전용 베이스라인 대비)
- 폐쇄 루프 시뮬레이션에서 근접 조우율(close encounter rate) 35% 감소
- 실시간 지연시간 99ms
- 모델 크기 0.5B~7B까지 스케일링 검증
- 실차 도로 테스트로 배포 가능성 확인

### 1.2 Alpamayo 에코시스템 구성 요소

| 구성 요소 | 설명 | 링크 |
|-----------|------|------|
| **Alpamayo 1 모델** | 10.5B 파라미터 reasoning VLA 모델 | https://huggingface.co/nvidia/Alpamayo-R1-10B |
| **Physical AI AV Dataset** | 25개국, 2,500+ 도시, 1,727시간 주행 데이터, 310,895 클립 | https://huggingface.co/datasets/nvidia/PhysicalAI-Autonomous-Vehicles |
| **Physical AI AV NuRec Dataset** | 폐쇄 루프 평가용 재구성 장면 데이터셋 (~900 장면) | https://huggingface.co/datasets/nvidia/PhysicalAI-Autonomous-Vehicles-NuRec |
| **AlpaSim** | 마이크로서비스 기반 폐쇄 루프 시뮬레이션 플랫폼 | https://github.com/NVlabs/alpasim |
| **Alpamayo 코드** | 추론 코드 (Apache 2.0 라이선스) | https://github.com/NVlabs/alpamayo |

---

## 2. Alpamayo 활용 / 인용 연구

> Alpamayo-R1은 2025년 10월에 발표되었으며, CES 2026(2026년 1월)에서 공식 공개되었다. 비교적 최근 연구이므로 인용 논문의 수가 아직 제한적이다.

### 2.1 ReasonPlan

- **제목**: ReasonPlan: Unified Scene Prediction and Decision Reasoning for Closed-loop Autonomous Driving
- **저자**: Yupeng Zheng 외
- **날짜**: 2025년 5월 (arXiv:2505.20024)
- **링크**: https://arxiv.org/abs/2505.20024
- **Alpamayo와의 관계**: Alpamayo-R1의 Chain of Causation 추론 접근법과 비교. ReasonPlan은 자기 지도 Next Scene Prediction(NSP) + 지도 Decision Chain-of-Thought(DeCoT) 프레임워크를 제안. Bench2Drive에서 E2E 모방학습 대비 L2 19%, 주행 점수 16.1% 향상.

### 2.2 Drive-R1

- **제목**: Drive-R1: Bridging Reasoning and Planning in VLMs for Autonomous Driving with Reinforcement Learning
- **저자**: Li 외
- **날짜**: 2025년 6월 (arXiv:2506.18234)
- **링크**: https://arxiv.org/abs/2506.18234
- **Alpamayo와의 관계**: Alpamayo-R1과 유사하게 VLM에서의 추론과 궤적 계획을 연결하되, InternVL2 기반의 소규모 도메인 특화 VLM 사용. 300만 샘플 대규모 데이터셋으로 후학습. nuScenes, DriveLM-nuScenes에서 SOTA 달성.

### 2.3 AutoVLA

- **제목**: AutoVLA: A Vision-Language-Action Model for End-to-End Autonomous Driving with Adaptive Reasoning and Reinforcement Fine-Tuning
- **저자**: Zhou 외
- **날짜**: 2025년
- **링크**: https://autovla.github.io/ / https://openreview.net/forum?id=28qUA2bSe5
- **Alpamayo와의 관계**: Alpamayo-R1 논문에서 비교 대상으로 언급. 단일 자기회귀 모델 내에서 추론과 행동 생성을 통합. 이중 사고 모드(빠른 사고 + 느린 사고)를 도입하고, GRPO 기반 강화 미세조정으로 불필요한 추론 감소. Waymo Vision-based E2E Driving Challenge에서 RFS Spotlight 최고 점수.

### 2.4 산업 파트너의 활용

CES 2026에서 발표된 얼리 어답터:
- **Mercedes-Benz**: Alpamayo 기반 NVIDIA DRIVE AV 풀스택이 탑재된 최초의 양산 차량(CLA) — 2026년 Q1 미국 출시
- **Jaguar Land Rover (JLR)**: Level 4 자율주행 로드맵에 Alpamayo 채택
- **Lucid Motors**: Alpamayo 에코시스템 활용
- **Uber**: 자율주행 모빌리티 연구에 Alpamayo 활용
- **Berkeley DeepDrive**: 학술 연구에서 Alpamayo 활용

---

## 3. 기반 기술 연구

### 3.1 VLM 백본: Cosmos-Reason1

- **제목**: Cosmos-Reason1: From Physical Common Sense to Embodied Reasoning
- **저자**: Alisson Azzolini, Junjie Bai, Hannah Brandon 외 (NVIDIA, 50+ 저자)
- **날짜**: 2025년 3월 (arXiv:2503.15558)
- **링크**: https://arxiv.org/abs/2503.15558
- **핵심 내용**: Physical AI를 위한 물리적 상식(physical common sense)과 체화된 추론(embodied reasoning)을 갖춘 VLM. Qwen2.5-VL 기반 아키텍처에 Mamba-MLP-Transformer 하이브리드 백본 사용. 두 가지 크기: 8B, 56B. 4단계 학습: 비전 사전학습 → 일반 SFT → Physical AI SFT → Physical AI RL. 3.7M VQA 샘플로 후학습. 체화 추론 벤치마크에서 베이스라인 VLM 대비 10%+ 향상.
- **Alpamayo에서의 역할**: Alpamayo-R1의 VLM 백본(8.2B)으로 사용됨.

### 3.2 Cosmos-Reason2 (후속 버전)

- **제목**: 확인 불가 (별도 논문 미확인, GitHub 릴리즈 기반)
- **날짜**: 2025년 12월 19일 (코드/모델 공개)
- **링크**: https://github.com/nvidia-cosmos/cosmos-reason2
- **핵심 내용**: Cosmos-Reason1의 후속 버전. 물리적 상식과 체화 추론 성능을 향상. 2B, 8B 모델 제공. 리더보드 최상위 추론 VLM.

### 3.3 Cosmos World Foundation Model

- **제목**: Cosmos World Foundation Model Platform for Physical AI
- **저자**: Niket Agarwal 외 (NVIDIA, 76+ 저자)
- **날짜**: 2025년 1월 (arXiv:2501.03575)
- **링크**: https://arxiv.org/abs/2501.03575
- **핵심 내용**: Physical AI를 위한 세계 기반 모델 플랫폼. 비디오 큐레이션 파이프라인, 사전학습된 세계 기반 모델, 후학습 예시, 비디오 토크나이저를 포함. 오픈소스로 공개.

### 3.4 Cosmos-Drive-Dreams

- **제목**: Cosmos-Drive-Dreams: Scalable Synthetic Driving Data Generation with World Foundation Models
- **날짜**: 2025년 6월 (arXiv:2506.09042)
- **링크**: https://arxiv.org/abs/2506.09042
- **핵심 내용**: Cosmos World Foundation Model을 주행 도메인에 특화하여 제어 가능한 고품질 합성 주행 데이터 생성. 롱테일 분포 문제 완화. 3D 차선 감지에서 최대 +9.4%, 3D 객체 감지에서 최대 +6.0% 성능 향상.

### 3.5 VLM 기반 아키텍처: Qwen2.5-VL

- **제목**: Qwen2.5-VL Technical Report
- **저자**: Qwen Team, Alibaba Group
- **날짜**: 2025년 2월 (arXiv:2502.13923)
- **링크**: https://arxiv.org/abs/2502.13923
- **핵심 내용**: Cosmos-Reason1의 기반 아키텍처. 동적 해상도 Vision Transformer(ViT)를 처음부터 학습하며 Window Attention으로 계산 오버헤드 절감. 문서/다이어그램 이해에서 GPT-4o, Claude 3.5 Sonnet에 필적하는 성능. 3가지 크기로 제공.
- **Alpamayo에서의 역할**: Cosmos-Reason1 → Alpamayo-R1 파이프라인의 VLM 기반 아키텍처.

### 3.6 Qwen3-VL (Qwen2.5-VL 후속)

- **제목**: Qwen3-VL Technical Report
- **저자**: Qwen Team, Alibaba Group
- **날짜**: 2025년 11월 (arXiv:2511.21631)
- **링크**: https://arxiv.org/abs/2511.21631
- **핵심 내용**: Qwen 시리즈의 최신 VLM. 256K 토큰까지의 인터리빙 컨텍스트 지원. Dense(2B/4B/8B/32B) + MoE(30B-A3B/235B-A22B) 변형 제공. 멀티모달 추론에서 SOTA.

### 3.7 Diffusion/Flow Matching 기반 궤적 디코더

Alpamayo-R1의 궤적 디코더는 diffusion 기반 접근법을 사용한다. 관련 핵심 연구:

#### 3.7.1 Flow Matching 이론적 기초
- **Lipman et al. (2023)**: "Flow Matching for Generative Modeling" — 연속 확률 흐름을 직접 모델링하는 생성 모델링 방법론
- **Zhong et al. (2023)**: Flow matching 관련 이론적 연구

#### 3.7.2 π₀ / π₀.₅ (Physical Intelligence)
- **제목**: π₀: A Vision-Language-Action Flow Model for General Robot Control
- **저자**: Danny Driess 외 (Physical Intelligence)
- **날짜**: 2024년 10월 (arXiv:2410.24164)
- **링크**: https://arxiv.org/abs/2410.24164
- **핵심 내용**: 사전학습된 VLM 위에 flow matching 아키텍처를 구축한 로봇 제어용 VLA 모델. Knowledge Insulating(KI) 접근법으로 VLM 지식 유지 + 고주파 연속 행동 출력. Alpamayo-R1 논문에서 "π₀.₅-KI" 형태로 제어 기반 표현과 flow matching을 결합한 사례로 인용.

#### 3.7.3 DiffusionDrive
- **제목**: DiffusionDrive: Truncated Diffusion Model for End-to-End Autonomous Driving
- **발표처**: CVPR 2025
- **핵심 내용**: 절단된 확산 모델로 자율주행 궤적 생성.

#### 3.7.4 GoalFlow
- **제목**: GoalFlow: Goal-Driven Flow Matching for Multimodal Trajectories Generation in End-to-End Autonomous Driving
- **발표처**: CVPR 2025
- **핵심 내용**: 목표점 기반 flow matching으로 멀티모달 궤적 생성.

### 3.8 추론 강화 학습 기법: DeepSeek-R1

- **제목**: DeepSeek-R1: Incentivizing Reasoning Capability in LLMs via Reinforcement Learning
- **저자**: DeepSeek-AI
- **날짜**: 2025년 1월 (arXiv:2501.12948)
- **링크**: https://arxiv.org/abs/2501.12948
- **핵심 내용**: 순수 RL만으로 LLM의 추론 능력을 유도. DeepSeek-V3-Base를 GRPO로 학습. 자기 성찰, 검증, 동적 전략 적응 등의 고급 추론 패턴이 자연 발생. AIME 2024 pass@1: 15.6% → 71.0%.
- **Alpamayo에서의 역할**: Alpamayo-R1의 RL 후학습 전략에 영감을 제공.

### 3.9 GRPO (Group Relative Policy Optimization)

- **제목**: DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models
- **저자**: Zhihong Shao 외
- **날짜**: 2024년 2월 (arXiv:2402.03300)
- **링크**: https://arxiv.org/abs/2402.03300
- **핵심 내용**: PPO의 변형으로, critic 모델 없이 그룹 점수에서 베이스라인을 추정하여 학습 자원을 대폭 절감. 수학적 추론에서 큰 성능 향상.
- **Alpamayo에서의 역할**: Alpamayo-R1의 RL 후학습 및 관련 VLA 연구들(AdaThinkDrive, AutoVLA 등)에서 공통적으로 사용되는 핵심 최적화 알고리즘.

---

## 4. Alpamayo 논문에서 참조한 관련 연구 (Related Work)

### 4.1 End-to-End 자율주행 모델

| 연구 | 저자 | 연도 | 핵심 내용 |
|------|------|------|-----------|
| End-to-End Neural Driving | Bojarski et al. | 2016 | 원시 센서 → 제어로 직접 매핑하는 초기 신경망 기반 주행 |
| UniAD | Hu et al. | 2023 | 트랜스포머 기반 통합 자율주행 프레임워크 |
| EMMA | Hwang et al. (Waymo) | 2024 | Gemini 기반 멀티모달 E2E 모델, 자연어 텍스트로 입출력 통합 |
| Alpamayo-VA | Wu | 2025 | Alpamayo-R1의 비전-행동(vision-action) 베이스라인 |

### 4.2 VLM/VLA 기반 자율주행

| 연구 | 저자 | 연도 | 핵심 내용 |
|------|------|------|-----------|
| Drive-GPT | Mao et al. | 2023 | LLM을 활용한 초기 자율주행 계획 |
| Agent-Driver | Mao et al. | 2024 | LLM을 인지 에이전트로 활용, 도구 라이브러리 + 인지 기억 + 추론 엔진 |
| Cube-LLM | Cho et al. | 2024 | 3D 이해를 위한 언어-이미지 모델, 3D 공간 추론 + CoT 프롬프팅 |
| OpenDriveVLA | Zhou et al. | 2025 | 자기회귀적 궤적 웨이포인트 생성, 계층적 비전-언어 정렬 (AAAI 2026) |
| SimLingo | Renz et al. | 2025 | 비전 전용 폐쇄 루프 자율주행, Action Dreaming 도입 (CVPR 2025 Spotlight) |

### 4.3 추론 강화 VLA

| 연구 | 저자 | 연도 | 핵심 내용 |
|------|------|------|-----------|
| AdaThinkDrive | Luo et al. | 2025 | 빠른/느린 사고 이중 모드 + GRPO, Navsim PDMS 90.3 |
| AutoDrive-R2 | Yuan et al. | 2025 | 물리 기반 보상 + GRPO, 6K CoT 샘플만으로 SOTA |
| Poutine | Rowe et al. | 2025 | VLM 기반 VLT 사전학습 + GRPO 후학습, 2025 Waymo E2E Challenge 1위 (RFS 7.99) |
| AgentThink | Qian et al. | 2025 | 멀티 에이전트 추론 접근법 |
| DSDrive | Liu et al. | 2025 | 다양한 추론 및 멀티 에이전트 접근법 |

### 4.4 Chain-of-Thought 추론 기초

| 연구 | 저자 | 연도 | 핵심 내용 |
|------|------|------|-----------|
| Chain-of-Thought Prompting | Wei et al. | 2022 | CoT 프롬프팅 원본 논문 |
| Tree-of-Thought | Yao et al. | 2023 | 트리 구조 추론 방법론 |
| OpenAI o1 | OpenAI | 2024 | 추론 시간 계산을 활용한 추론 모델 |

### 4.5 자율주행 추론 데이터셋

| 데이터셋 | 저자 | 연도 | 핵심 내용 |
|----------|------|------|-----------|
| nuScenes | Caesar et al. | 2020 | 6카메라 + 5레이더 + 1라이다, 1000 장면, CVPR 2020 |
| BDD-X | Kim et al. | 2018 | 운전 행동에 대한 인간 작성 설명 |
| Drama | Malla et al. | 2023 | 자아 행동에 영향을 미치는 중요 객체 주석 |
| DriveCoT | Wang et al. | 2024 | 1058 시나리오, 36K 라벨 샘플, CoT 형식 운전 이해 라벨 |
| DriveLM | Sima et al. | 2024 | 그래프 구조 QA, 지각→예측→계획의 논리적 연결 |
| Reason2Drive | Nie et al. | 2024 | 600K+ 비디오-텍스트 쌍, 해석 가능한 체인 기반 추론 벤치마크 |
| WOMD-reasoning | Li et al. | 2024 | Waymo Open Motion Dataset에 비전-언어 주석 확장 |

### 4.6 시뮬레이션 및 평가

| 연구 | 핵심 내용 |
|------|-----------|
| CARLA (Dosovitskiy et al., 2017) | 오픈소스 주행 시뮬레이터 |
| Waymo Open Motion Dataset (Ettinger et al., 2021) | 대규모 모션 데이터 |
| Bench2Drive (Jia et al., 2024) | CARLA 기반 폐쇄 루프 주행 벤치마크 |
| Sim2Val (NVIDIA) | 시뮬레이션-실세계 상관관계 활용 검증 프레임워크, 분산 최대 83% 감소 |
| RoaD (NVIDIA) | 개방 루프 학습 ↔ 폐쇄 루프 배포 간 공변량 이동(covariate shift) 완화 |

---

## 5. 관련 기술 블로그 / 발표

### 5.1 NeurIPS 2025 (2025년 12월 1일, 샌디에이고)

- **발표 내용**: Alpamayo-R1 최초 공개. "세계 최초의 산업 규모 오픈 추론 VLA 모델"로 소개.
- **링크**: https://blogs.nvidia.com/blog/neurips-open-source-digital-physical-ai/
- **비고**: NVIDIA는 NeurIPS 2025에서 70+ 논문/발표/워크샵 진행. Physical AI AV 데이터셋도 동시 공개.

### 5.2 CES 2026 (2026년 1월 5일, 라스베이거스)

- **발표자**: Jensen Huang (NVIDIA CEO) 키노트
- **발표 내용**: Alpamayo 에코시스템 공식 출시. Alpamayo-R1을 Alpamayo 1으로 리브랜딩. Mercedes-Benz CLA에 NVIDIA DRIVE AV 탑재 발표.
- **주요 링크**:
  - NVIDIA 뉴스룸: https://nvidianews.nvidia.com/news/alpamayo-autonomous-vehicle-development
  - NVIDIA 블로그: https://blogs.nvidia.com/blog/2026-ces-special-presentation/
  - TechCrunch 기사: https://techcrunch.com/2026/01/05/nvidia-launches-alpamayo-open-ai-models-that-allow-autonomous-vehicles-to-think-like-a-human/

### 5.3 NVIDIA 기술 블로그

- **제목**: Building Autonomous Vehicles That Reason with NVIDIA Alpamayo
- **링크**: https://developer.nvidia.com/blog/building-autonomous-vehicles-that-reason-with-nvidia-alpamayo/
- **핵심 내용**: Alpamayo 1 모델, Physical AI AV 데이터셋, AlpaSim 시뮬레이터의 기술적 세부사항 상세 설명. 파이프라인 병렬화, Sim2Val 프레임워크, RoaD 알고리즘 소개.

### 5.4 HuggingFace 블로그

- **제목**: Building Autonomous Vehicles That Reason with the NVIDIA Alpamayo Open Ecosystem
- **저자**: Marco Pavone (drmapavone)
- **링크**: https://huggingface.co/blog/drmapavone/nvidia-alpamayo
- **핵심 내용**: Alpamayo 에코시스템의 개요, 모델 사용법, 데이터셋 접근 방법 안내.

### 5.5 NVIDIA Alpamayo 공식 페이지

- **링크**: https://www.nvidia.com/en-us/solutions/autonomous-vehicles/alpamayo/
- **핵심 내용**: Level 4 자율주행을 위한 오픈 AI 포트폴리오 소개. 추론 VLA 모델, 시뮬레이션 블루프린트, 데이터셋 포함.

### 5.6 NVIDIA Developer 페이지

- **링크**: https://developer.nvidia.com/drive/alpamayo
- **핵심 내용**: 개발자를 위한 Alpamayo 모델 접근, 시뮬레이션 도구, 데이터셋 활용 가이드.

### 5.7 NVIDIA Research 페이지

- **링크**: https://research.nvidia.com/publication/2025-10_alpamayo-r1
- **링크**: https://research.nvidia.com/labs/avg/publication/wang.luo.etal.arxiv2025/
- **핵심 내용**: Alpamayo-R1 논문의 연구 그룹(Autonomous Vehicle Research Group) 페이지.

---

## 6. VLA for Autonomous Driving 서베이 논문

### 6.1 서베이 1

- **제목**: A Survey on Vision-Language-Action Models for Autonomous Driving
- **저자**: Jiang et al.
- **날짜**: 2025년 6월 (arXiv:2506.24044)
- **링크**: https://arxiv.org/abs/2506.24044
- **발표처**: ICCV 2025 Workshop
- **핵심 내용**: VLA4AD의 첫 종합 서베이. 아키텍처 빌딩 블록 형식화, 초기 explainer에서 추론 중심 VLA로의 진화 추적.

### 6.2 서베이 2

- **제목**: Vision-Language-Action Models for Autonomous Driving: Past, Present, and Future
- **저자**: Hu et al.
- **날짜**: 2025년 12월 (arXiv:2512.16760, 2026년 1월 업데이트)
- **링크**: https://arxiv.org/abs/2512.16760
- **핵심 내용**: E2E VLA vs. Dual-System VLA로 분류. 텍스트 vs. 수치 행동 생성기, 명시적 vs. 암시적 안내 메커니즘 등 세부 분류.

---

## 7. 요약 및 연구 공백 (Research Gap)

### 7.1 기존 연구 요약

Alpamayo-R1은 자율주행 분야에서 **추론과 행동 예측의 통합**이라는 핵심 과제를 다룬다. 기존 연구 흐름을 정리하면:

1. **E2E 자율주행 진화**: 단순 모방학습(Bojarski 2016) → 트랜스포머 기반 통합(UniAD 2023) → 멀티모달 LLM 활용(EMMA 2024) → 추론 VLA(Alpamayo 2025)
2. **추론 능력의 도입**: CoT 프롬프팅(Wei 2022) → 자율주행 특화 CoT(DriveCoT, DriveLM) → 인과 추론 체인(Chain of Causation, Alpamayo)
3. **RL 후학습의 확산**: DeepSeek-R1의 GRPO → 자율주행 VLA에 적용(AdaThinkDrive, AutoDrive-R2, Poutine, Alpamayo)
4. **궤적 생성 방법론**: 자기회귀적 웨이포인트(OpenDriveVLA) → Diffusion 기반(DiffusionDrive) → Flow Matching 기반(GoalFlow, π₀)

### 7.2 연구 공백 (Research Gaps)

| 영역 | 공백 설명 |
|------|-----------|
| **실시간 추론 효율성** | Alpamayo-R1은 99ms 지연시간을 달성했으나, 10.5B 파라미터 모델의 엣지 디바이스 배포 최적화는 미해결. 모델 경량화(pruning, quantization, distillation)에 대한 심층 연구 부족. |
| **추론-행동 인과적 정합성** | CoC 데이터셋이 인과 추론을 제공하지만, 추론 트레이스가 실제로 궤적 생성에 *인과적으로* 영향을 미치는지(단순 상관관계가 아닌) 검증하는 방법론 부족. |
| **다중 센서 융합** | Alpamayo-R1은 카메라 4대 + 자기 이동 이력만 사용. LiDAR, Radar를 VLA 프레임워크에 통합하는 연구는 제한적. |
| **폐쇄 루프 학습** | 대부분의 연구(Alpamayo 포함)가 개방 루프 SFT + 시뮬레이션 기반 RL에 의존. 실세계 폐쇄 루프에서의 온라인 학습/적응 연구 부족. |
| **다국어/다문화 추론** | CoC 추론 트레이스가 영어로만 제공. 다양한 교통 문화, 규제, 도로 표지판에 대한 추론 능력 검증 부족. |
| **안전 보장 및 검증** | 추론 VLA의 형식적 안전 보장(formal safety guarantee) 방법론이 부재. 추론 오류가 궤적에 미치는 영향을 체계적으로 분석한 연구 부족. |
| **장기 시간적 추론** | Alpamayo-R1은 6.4초 미래 궤적 예측. 30초 이상의 전략적 수준 추론(경로 계획, 차선 변경 전략 등)은 다루지 않음. |
| **비정형 도로 환경** | 대부분의 연구가 구조화된 도로(structured road) 중심. 비포장 도로, 공사 구간, 주차장 등 비정형 환경에서의 추론 능력은 미검증. |
| **RL 후학습의 v1.0 미포함** | Alpamayo 1 v1.0 공개 버전에는 RL 후학습이 포함되지 않음(SFT만 적용). RL 후학습 코드/모델의 오픈소스 공개 예정 시기 미정. |

### 7.3 향후 연구 방향 제안

1. **경량 추론 VLA**: 모델 증류(distillation) 또는 MoE 구조를 활용하여 엣지 디바이스에서 실시간 추론 가능한 소형 VLA 개발
2. **인과적 개입 실험**: 추론 트레이스를 조작(intervention)하여 궤적 변화를 관찰하는 인과 검증 프레임워크 설계
3. **멀티모달 센서 통합 VLA**: LiDAR/Radar 정보를 VLA의 추론 체인에 명시적으로 통합
4. **실세계 온라인 적응**: 시뮬레이션에서 학습한 추론 모델을 실세계에서 점진적으로 적응시키는 continual learning 프레임워크
5. **안전 인식 추론**: 추론 오류의 안전 영향을 평가하고, 안전 제약을 추론 과정에 내재화하는 방법론

---

## 부록: 주요 참고문헌 전체 목록

```
[1]  Wang et al. (2025). "Alpamayo-R1: Bridging Reasoning and Action Prediction for
     Generalizable Autonomous Driving in the Long Tail." arXiv:2511.00088.
[2]  NVIDIA et al. (2025). "Cosmos-Reason1: From Physical Common Sense to Embodied
     Reasoning." arXiv:2503.15558.
[3]  NVIDIA et al. (2025). "Cosmos World Foundation Model Platform for Physical AI."
     arXiv:2501.03575.
[4]  NVIDIA et al. (2025). "Cosmos-Drive-Dreams: Scalable Synthetic Driving Data Generation
     with World Foundation Models." arXiv:2506.09042.
[5]  Qwen Team (2025). "Qwen2.5-VL Technical Report." arXiv:2502.13923.
[6]  Qwen Team (2025). "Qwen3-VL Technical Report." arXiv:2511.21631.
[7]  DeepSeek-AI (2025). "DeepSeek-R1: Incentivizing Reasoning Capability in LLMs via
     Reinforcement Learning." arXiv:2501.12948.
[8]  Shao et al. (2024). "DeepSeekMath: Pushing the Limits of Mathematical Reasoning in
     Open Language Models." arXiv:2402.03300.
[9]  Hwang et al. (2024). "EMMA: End-to-End Multimodal Model for Autonomous Driving."
     arXiv:2410.23262.
[10] Black et al. (2024). "π₀: A Vision-Language-Action Flow Model for General Robot
     Control." arXiv:2410.24164.
[11] Zhou et al. (2025). "OpenDriveVLA: Towards End-to-end Autonomous Driving with Large
     Vision Language Action Model." arXiv:2503.23463.
[12] Zhou et al. (2025). "AutoVLA: A VLA Model for End-to-End Autonomous Driving with
     Adaptive Reasoning and Reinforcement Fine-Tuning."
[13] Renz et al. (2025). "SimLingo: Vision-Only Closed-Loop Autonomous Driving with
     Language-Action Alignment." CVPR 2025.
[14] Luo et al. (2025). "AdaThinkDrive: Adaptive Thinking via Reinforcement Learning for
     Autonomous Driving." arXiv:2509.13769.
[15] Yuan et al. (2025). "AutoDrive-R2: Incentivizing Reasoning and Self-Reflection Capacity
     for VLA Model in Autonomous Driving." arXiv:2509.01944.
[16] Rowe et al. (2025). "Poutine: Vision-Language-Trajectory Pre-Training and Reinforcement
     Learning Post-Training Enable Robust End-to-End Autonomous Driving." arXiv:2506.11234.
[17] Li et al. (2025). "Drive-R1: Bridging Reasoning and Planning in VLMs for Autonomous
     Driving with Reinforcement Learning." arXiv:2506.18234.
[18] Zheng et al. (2025). "ReasonPlan: Unified Scene Prediction and Decision Reasoning for
     Closed-loop Autonomous Driving." arXiv:2505.20024.
[19] Mao et al. (2024). "Agent-Driver: A Language Agent for Autonomous Driving."
     arXiv:2311.10813.
[20] Cho et al. (2024). "Language-Image Models with 3D Understanding (Cube-LLM)."
     arXiv:2405.03685.
[21] Caesar et al. (2020). "nuScenes: A multimodal dataset for autonomous driving." CVPR.
[22] Wei et al. (2022). "Chain-of-Thought Prompting Elicits Reasoning in Large Language
     Models." NeurIPS.
[23] Bojarski et al. (2016). "End to End Learning for Self-Driving Cars." arXiv:1604.07316.
[24] Jiang et al. (2025). "A Survey on Vision-Language-Action Models for Autonomous
     Driving." arXiv:2506.24044. ICCV 2025 Workshop.
[25] Hu et al. (2025). "Vision-Language-Action Models for Autonomous Driving: Past,
     Present, and Future." arXiv:2512.16760.
```

---

> **참고**: 이 문서는 2026년 2월 25일 기준으로 웹 검색, arXiv, HuggingFace, GitHub, NVIDIA 공식 자료를 기반으로 작성되었습니다. Alpamayo-R1의 완전한 참고문헌 목록(60+ 편)은 원본 논문 PDF(https://arxiv.org/pdf/2511.00088)에서 확인할 수 있습니다.
