# HF Transformers `GemmaForCausalLM`을 직접 사용하는 자체 완결형 SFT 파이프라인을 작성했습니다.

## 최종 파일 구성

| 파일 | 역할 |
| ---- | ---- |
| config.py | `GemmaSFTConfig` — 모델/데이터/학습/FSDP 옵션 dataclass |
| model.py | `GemmaModel` — `GemmaForCausalLM` 래퍼 (SDPA, gradient checkpointing) |
| tokenizer.py | `build_tokenizer` — PAD/EOS 설정 + Gemma chat template 자동 주입 |
| sftdataset.py | Alpaca 형식 `SFTDataset` + collator, 응답 부분만 loss 계산 (`IGNORE_INDEX=-100`) |
| parallelize.py | `parallelize_gemma` — FSDP2 + `MixedPrecisionPolicy` (bf16 param / fp32 reduce) |
| train.py | 메인 `torchrun` 엔트리 (dist init, AdamW, warmup+linear decay, grad clip, HF 포맷 저장) |
| run_sft.sh | 편의 런처 (기본적으로 `./assets/hf/gemma-7b/main` 로컬 가중치 사용) |
| README.md | 사용법 |
| __init__.py | 패키지 exports |

## 실행 방법

기본 실행 (Alpaca):
```bash
NGPU=8 ./torchtitan/models/gemma/run_sft.sh \
    --dataset_name tatsu-lab/alpaca \
    --output_dir outputs/gemma-7b-alpaca
```

로컬 JSONL:
```bash
NGPU=8 ./torchtitan/models/gemma/run_sft.sh \
    --dataset_local_path ./my_sft.jsonl \
    --output_dir outputs/gemma-7b-custom \
    --num_epochs 3 --lr 2e-5 --per_device_batch_size 1 \
    --gradient_accumulation_steps 8
```

모든 `GemmaSFTConfig` 필드는 `tyro`를 통해 자동으로 CLI 플래그로 노출됩니다. 저장 결과물은 표준 HF 디렉터리(`config.json`, `model.safetensors*`, tokenizer 파일)라서 이후 `GemmaForCausalLM.from_pretrained(save_dir)`로 바로 로딩 가능합니다.

## 주요 설계 포인트

- **응답 전용 loss**: chat template로 user turn까지의 토큰을 렌더링한 뒤 길이를 재서, 해당 prefix 토큰의 label을 `-100`으로 마스킹합니다.
- **Chat template**: `google/gemma-7b`(base)에는 chat template이 없어 gemma-7b-it과 동일한 템플릿을 tokenizer에 자동으로 심어 downstream 호환성을 유지합니다.
- **FSDP2**: 각 `GemmaDecoderLayer` → HF 모델 → 래퍼 순서로 `fully_shard`. bf16 param / fp32 grad reduce.
- **디바이스**: CUDA/XPU 자동 감지 (NCCL/CCL 백엔드 자동 선택) — Aurora XPU에서도 그대로 동작.
- **체크포인트 저장**: FSDP2의 `get_model_state_dict(full_state_dict=True, cpu_offload=True)`로 rank 0에 전체 가중치를 모아 `model.` prefix를 벗겨 HF 표준 포맷으로 저장.



# alpaca와 my_sft.jsonl 둘 다 모두 학습 데이터 소스이며, 둘 중 **하나만** 지정하면 됩니다.

## 두 가지 데이터 소스

### 1. `--dataset_name tatsu-lab/alpaca` (HuggingFace Hub)
- HF Hub에서 스트리밍 다운로드하는 데이터셋 이름
- 기본값: [`tatsu-lab/alpaca`](https://huggingface.co/datasets/tatsu-lab/alpaca) — Stanford Alpaca 52K개 instruction 데이터
- 다른 예: `yahma/alpaca-cleaned`, `databricks/databricks-dolly-15k` 등

### 2. `--dataset_local_path ./my_sft.jsonl` (로컬 파일)
- 직접 준비한 JSON 또는 JSONL 파일 경로
- `my_sft.jsonl`은 **예시 파일명**일 뿐 — 실제 파일이 아니라 사용자가 만들어야 하는 파일입니다
- 형식은 Alpaca 스타일 dict의 리스트:
  ```json
  {"instruction": "다음 문장을 요약하세요.", "input": "긴 텍스트...", "output": "요약된 결과"}
  {"instruction": "파이썬으로 피보나치 함수를 짜줘", "input": "", "output": "def fib(n): ..."}
  ```

## 우선순위

sftdataset.py의 `load_sft_records`를 보면:

```python
if dataset_local_path:
    return _load_local_records(dataset_local_path)   # 로컬 파일이 있으면 이걸 우선
...
ds = load_dataset(dataset_name, dataset_config_name, split=dataset_split)  # 없으면 HF
```

즉 `--dataset_local_path`가 설정되면 `--dataset_name`은 무시됩니다.

## 필수 필드

두 경로 모두 각 샘플이 아래 필드를 가져야 합니다 (키 이름은 config에서 변경 가능):

| 필드 | 필수? | config 옵션 |
| ---- | ---- | ---- |
| `instruction` | O | `--instruction_key` |
| `input`       | X (없어도 됨) | `--input_key` |
| `output`      | O | `--output_key` |

만약 데이터셋의 필드명이 다르면 (예: `question`/`answer`) 이렇게 매핑할 수 있습니다:

```bash
--dataset_name your/dataset \
--instruction_key question \
--output_key answer
```



# **일부는 확실하고, 일부는 확인이 필요**합니다. 그리고 HF에 있어도 **필드명이 제각각**이라 `--instruction_key`/`--input_key`/`--output_key` 매핑이 거의 항상 필요합니다.

## 카테고리별 정리

### ✅ Math — 대부분 HF에 있음, SFT 바로 가능

| 이름 | HF ID | 필드 매핑 |
| --- | --- | --- |
| MetaMathQA | `meta-math/MetaMathQA` | `--instruction_key query --output_key response` |
| NuminaMath (CoT) | `AI-MO/NuminaMath-CoT` | `--instruction_key problem --output_key solution` |
| OpenMathInstruct-2 | `nvidia/OpenMathInstruct-2` | `--instruction_key problem --output_key generated_solution` |
| GSM8K | `openai/gsm8k` (config `main`) | `--instruction_key question --output_key answer`, `--dataset_config_name main` |
| MATH | `hendrycks/competition_math` 또는 `lighteval/MATH` | `--instruction_key problem --output_key solution` |

### ⚠️ Chemistry — 상당수는 확인/변환 필요

| 이름 | HF 상태 | 비고 |
| --- | --- | --- |
| ChemLLM Instruction | `AI4Chem/ChemData700K` 계열이 유사 | 이름이 정확히 "ChemLLM Instruction"인 데이터셋은 배포처마다 다름 — 실제 존재 여부 확인 필요 |
| ChemBench | 주로 **벤치마크(eval)** 용도 | 평가용이라 SFT에는 부적합할 수 있음 |
| PubChemQA | 그 이름 그대로 HF에 있는지는 **불확실** | 유사한 QA 계열: `alexandrainst/m_pubchemqa` 같은 mirror 존재 여부 확인 필요 |
| SciBench-Chem | `xw27/scibench`의 chemistry subset | SFT 형식으로 재가공 필요 (필드가 `problem_text`, `answer_number` 등) |

### ⚠️ Physics — 이름은 맞지만 형식 정리 필요

| 이름 | HF 상태 | 비고 |
| --- | --- | --- |
| SciBench-Physics | `xw27/scibench` (subset) | 위와 동일, subset 필터링 필요 |
| SciQ | `allenai/sciq` | ✅ 있음. 단 필드가 `question`, `correct_answer`, `support` — 매핑 필요 |
| OpenBookQA | `allenai/openbookqa` | ✅ 있음. 단 **Physics subset은 별도 제공 안 됨** — MCQ라 SFT용으로는 재가공 필요 |
| PhysicsQA | 그 이름 그대로는 **불확실** | 여러 커뮤니티 업로드 존재, 정식 canonical 없음 |

### ⚠️ Psychology — 대부분 확인/재가공 필요

| 이름 | HF 상태 | 비고 |
| --- | --- | --- |
| PsyQA | 원본은 GitHub(중국어), HF mirror 존재 여부 확인 필요 | 중국어 데이터임을 주의 |
| MentalChat | `ShenLab/MentalChat16K` 등 유사 배포 존재 | 정확한 canonical ID는 확인 필요 |
| Psychology QA | 일반명 — 특정 canonical 데이터셋이 아님 | 실체 없음, 검색 필요 |
| MedQA (Psych subset) | `bigbio/med_qa` 존재 | 그러나 **psychology subset 자체는 별도 제공 X** — 스스로 필터링해야 함 |

## 실전 팁 3개

**1. 필드 매핑 예시 (GSM8K 학습):**
```bash
NGPU=8 ./torchtitan/models/gemma/run_sft.sh \
    --dataset_name openai/gsm8k \
    --dataset_config_name main \
    --dataset_split train \
    --instruction_key question \
    --output_key answer \
    --output_dir outputs/gemma-7b-gsm8k
```

**2. 여러 도메인 섞으려면 로컬 JSONL로 병합**  
현재 sftdataset.py는 dataset을 **한 개**만 로드합니다. 여러 도메인을 섞으려면 미리 각 HF 데이터셋을 다운받아서 `{"instruction", "input", "output"}` 형태로 통일한 JSONL로 만든 뒤 `--dataset_local_path`로 넘기는 게 가장 간단합니다.

**3. 이름 확인은 항상 HF Hub에서 직접**  
정확한 dataset ID는 위 표를 그대로 믿지 말고 https://huggingface.co/datasets 에서 검색해 확인하세요 (같은 이름이 여러 uploader에게 있을 수 있고, canonical 배포처가 명확하지 않은 데이터셋들이 위 목록에 섞여 있습니다).

원하시면 **여러 HF 데이터셋을 자동으로 다운받아 통합 JSONL로 만드는 스크립트**를 추가로 짜드릴 수 있습니다.


# **바이오/의학 도메인은 HF에서 가장 잘 커버되는 도메인 중 하나**입니다. 화학·물리보다 훨씬 정리가 잘 되어 있어요. 다만 "bio"의 범위가 넓어서 목적별로 나눠서 보는 게 좋습니다.

## 1. 의학 QA / Instruction (SFT 바로 가능)

임상·의료 지식 기반 instruction 튜닝에 가장 많이 쓰입니다.

| 데이터셋 | HF ID | 특징 |
| --- | --- | --- |
| MedQA (USMLE) | `bigbio/med_qa` | 미국 의사국시 스타일 MCQ |
| MedMCQA | `openlifescienceai/medmcqa` | 인도 의사시험 MCQ, 194K |
| PubMedQA | `qiaojin/PubMedQA` (config `pqa_labeled` 등) | 논문 abstract 기반 yes/no/maybe QA |
| MMLU (medical subsets) | `cais/mmlu` | anatomy, clinical_knowledge, college_medicine, professional_medicine 등 subset |
| MedInstruct-52K | `casey-martin/MedInstruct-52k` 등 여러 mirror | Alpaca 형식 의학 instruction |
| Medical Meadow | `medalpaca/medical_meadow_*` (여러 subset) | Med-Alpaca 학습에 쓴 데이터 모음 |
| ChatDoctor | `LinhDuong/chatdoctor-200k` 계열 | 환자 상담 대화 |
| HealthCareMagic | `lavita/ChatDoctor-HealthCareMagic-100k` | 환자-의사 실제 대화 |

⚠️ 필드명은 제각각입니다. 예: MedQA는 `question`/`answer_idx`/`options` 구조라서 SFT 학습 전 스크립트로 `{"instruction": question + options, "output": correct_option_text}` 형태로 변환 필요.

## 2. 생명과학 / 논문 / 리서치

| 데이터셋 | HF ID | 용도 |
| --- | --- | --- |
| PubMed abstracts | `pubmed` / `armanc/scientific_papers` (pubmed config) | 사전학습·continue pretrain |
| BioASQ | `enelpol/bioasq` 등 mirror | 바이오 QA (task별 config) |
| BioMRC | `alexpolymerus/biomrc` 유사 | 생물의학 독해 |
| SciFact | `allenai/scifact` | 과학 주장 검증 |
| MedNLI | 원본은 라이선스 제한, HF에는 derivative만 | 임상 자연어 추론 |

## 3. 단백질 / 유전체 / 분자 (Bio-ML 계열, "instruction 튜닝"은 아님)

이건 **LLM SFT용이 아니라 특수 모델(ESM, Enformer, DNABERT 등)용** 데이터라 gemma 튜닝에는 부적합합니다:

- `InstaDeepAI/genomic-benchmarks`
- `agemagician/uniref50` (단백질 시퀀스)
- `InstaDeepAI/nucleotide_transformer_downstream_tasks`
- `zpn/human_proteome` 등

## 4. 벤치마크 모음 (참고)

목적을 잡을 때 좋은 통합 벤치마크:
- `bigbio/*` — 30+ 개 biomedical NLP 데이터셋의 표준화된 HF 미러 (**여기가 사실상 허브**)
- `openlifescienceai/*` — MedMCQA 저자 그룹의 큐레이션
- `medalpaca/*` — Med-Alpaca 학습 데이터 전체
- `MedARC/*` — 의료 리서치 큐레이션

## 실전 팁

**의학 SFT라면 이 조합이 정석적:**
```
medalpaca/medical_meadow_medqa
+ medalpaca/medical_meadow_medical_flashcards
+ lavita/ChatDoctor-HealthCareMagic-100k
+ openlifescienceai/medmcqa (변환 후)
```

이걸 미리 통합 JSONL로 만들어 `--dataset_local_path`로 넘기는 방식이 sftdataset.py과 가장 잘 맞습니다.

**주의사항 3가지:**
1. **라이선스**: 의료 데이터는 라이선스가 까다로운 경우 많음 (MIMIC 계열은 별도 인증 필요, PhysioNet 요구). HF에 올라와 있어도 사용 조건 확인 필수.
2. **한국어 의학 데이터는 매우 적음**. 대부분 영어. 한국어가 필요하면 자체 구축이 현실적.
3. **환각 위험**: 의학 도메인 SFT 후에도 hallucination이 심할 수 있으므로 실제 배포 전 별도 평가 필수 (`bigbio/med_qa`, `MedQA-USMLE-4-options` 같은 held-out set).

원하시면 위 목록 중 **선별한 데이터셋을 자동 다운로드 + 통합 JSONL 생성하는 스크립트**를 짜드릴 수 있습니다.