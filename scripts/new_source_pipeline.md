# FastMSS 新音源接入 Pipeline

从只有 `.wav` + 转录文本（无字/词级时间戳）的干净单说话人音频，到 FastMSS 可用的 Lhotse CutSet（含 word-level alignment）。

## 统一中转格式

Pipeline 的核心设计：**alignment JSON** 是所有上游对齐结果汇入的统一格式，`alignment_to_lhotse.py` 只消费这一种格式。

```json
{"utt_id": "utt_001", "words": [{"word": "HELLO", "start": 0.52, "end": 0.94}]}
```

上游无论来自 Qwen3 对齐还是外部 TextGrid/CTM/STM，一律先转为这个 JSON，再输入下游。

## Pipeline 概览

```
.wav + transcript                 外部对齐 (TextGrid/CTM/STM/...)
     │                                       │
     ▼                                       ▼
run_alignment.py (Qwen3)          *_to_alignment_json.py
     │                              (桥接脚本)
     │                                       │
     └──────────────┬────────────────────────┘
                    ▼
        alignment JSON ({utt_id, words[{word, start, end}]})
                    │
                    ▼
          alignment_to_lhotse.py
                    │
                    ▼
          Lhotse CutSet .jsonl.gz
                    │
                    ▼
              FastMSS sim.py
```

## 文件清单

| 脚本 | 用途 | 环境 |
|---|---|---|
| `run_alignment.py` | wav2transcript.scp → Qwen3 batch 对齐 → alignment JSON | `qwen3-asr` |
| `textgrid_to_alignment_json.py` | Praat TextGrid → alignment JSON（外部对齐桥接） | `nemo` |
| `alignment_to_lhotse.py` | alignment JSON → Lhotse CutSet (.jsonl.gz) | `nemo` |

## 输入格式

### wav2transcript.scp

```
/path/to/utt_001.wav hello world this is a test utterance
/path/to/utt_002.wav another example transcript here
```

每行: `<wav路径> <转录文本>`，按第一个空格切分。

### wav2spk.scp (推荐)

```
/path/to/utt_001.wav speaker_A
/path/to/utt_002.wav speaker_A
/path/to/utt_003.wav speaker_B
```

在 `alignment_to_lhotse.py` 中通过 `--spk-scp` 注入 speaker 信息。

## Step 1: 获取 alignment JSON

### 路径 A: Qwen3 强制对齐

```bash
conda activate qwen3-asr

python run_alignment.py \
    --scp wav2transcript.scp \
    --output-dir alignments/ \
    --batch-size 8
```

参数:

| 参数 | 说明 |
|---|---|
| `--scp` | wav2transcript.scp |
| `--output-dir` | 输出 alignment JSON 目录 |
| `--model` | HF model ID 或本地路径，默认 `Qwen/Qwen3-ForcedAligner-0.6B` |
| `--device` | 设备，默认 `cuda:0` |
| `--language` | `auto` / `Chinese` / `English` |
| `--batch-size` | 每批 utterance 数，默认 1（设大加速） |

### 路径 B: 外部对齐 → JSON (桥接)

```bash
conda activate nemo

python textgrid_to_alignment_json.py \
    --tg-dir external_textgrids/ \
    --output-dir alignments/
```

目前提供 TextGrid → JSON 桥接。CTM/STM/自定义格式的桥接脚本可按同样模式补写。

## Step 2: alignment JSON → Lhotse CutSet

```bash
conda activate nemo

python alignment_to_lhotse.py \
    --align-dir alignments/ \
    --wav-dir wavs/ \
    --spk-scp wav2spk.scp \
    --output lhotse_cutsets/ \
    --prefix my_corpus \
    --splits train,dev,test \
    --train-ratio 0.8
```

参数:

| 参数 | 说明 |
|---|---|
| `--align-dir` | alignment JSON 目录 |
| `--wav-dir` | wav 文件目录 |
| `--spk-scp` | wav2spk.scp，可选但推荐 |
| `--output` | Lhotse CutSet 输出目录 |
| `--prefix` | 文件名前缀 |
| `--splits` | `all` 或 `train,dev,test` |
| `--train-ratio` / `--dev-ratio` | 划分比例 |
| `--speaker` | 无 spk-scp 时的回退 speaker |

Speaker 优先级: spk-scp > word 级 speaker 字段 > `--speaker`

## Step 3: FastMSS 模拟

```bash
conda activate nemo
cd src/third_party/FastMSS

python recipes/sim.py \
    manifest_dir=../../data/dump/lhotse_cutsets/my_corpus \
    manifest_prefix=my_corpus \
    dset_splits=[train,dev,test] \
    n_meetings=5000 seed=42 \
    output_dir=data/dump_fastmss/simulated_sessions/my_corpus_v1
```

## 目录结构

```
data/dump/
├── unified/{dataset}/
│   ├── wav/{utt_id}.wav
│   ├── wav2transcript.scp          ← Qwen3 对齐输入
│   └── wav2spk.scp                 ← speaker 映射 (推荐)
├── force_align/{dataset}/
│   └── alignments/{utt_id}.json    ← 统一 alignment JSON
├── lhotse_cutsets/{dataset}/
│   └── {prefix}_{split}.jsonl.gz   ← Lhotse CutSet
└── fastmss/
    └── simulated_sessions/{data_version}/
```

## 依赖

| 环境 | 用途 | 关键包 |
|---|---|---|
| `qwen3-asr` | Qwen3 对齐 | `qwen-asr`, `transformers`, `torch`, `soundfile` |
| `nemo` | 桥接转换 + Lhotse + FastMSS | `lhotse`, torchaudio |
