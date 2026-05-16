# 扩展 FastMSS 音源: 从转录音频到 Lhotse CutSet

将仅有 `.wav` + ASR 转录文本（无字/词级时间戳）的单说话人音频接入 FastMSS。

## 核心设计

**Alignment JSON** 是统一桥接格式，`alignment_to_lhotse.py` 只消费这一种格式。
所有上游（Qwen3 对齐 / 外部 TextGrid / CTM / STM）一律先转成 alignment JSON，再汇入下游。

```
.wav + txt transcript          外部对齐 (TextGrid / CTM / STM / ...)
     │                                        │
     ▼                                        ▼
run_alignment.py (Qwen3)       *_to_alignment_json.py
     │                            (桥接脚本: textgrid → json, ctm → json, ...)
     │                                        │
     └────────────────┬───────────────────────┘
                      ▼
   alignment JSON: {utt_id, words: [{word, start, end}]}
                      │
                      ▼
            alignment_to_lhotse.py
                      │
                      ▼
            Lhotse CutSet .jsonl.gz
                      │
                      ▼
            FastMSS recipes/sim.py
```

## Alignment JSON 格式

```json
{
  "utt_id": "utt_001",
  "words": [
    {"word": "HELLO",   "start": 0.52, "end": 0.94},
    {"word": "WORLD",   "start": 1.50, "end": 2.05}
  ]
}
```

- `word`: 对齐后的词文本
- `start` / `end`: 绝对时间戳（秒），相对于录音开头
- 可选 `speaker` 字段，在 `alignment_to_lhotse.py` 中会被识别（优先级低于 `--spk-scp`）

对应到 Lhotse `AlignmentItem`:

| JSON 字段 | `AlignmentItem` 字段 | 转换 |
|---|---|---|
| `word` | `symbol` | 直接复制 |
| `start` | `start` | 直接复制 |
| `end - start` | `duration` | 计算 |

## 脚本清单

| 脚本 | 位置 | 用途 | 环境 |
|---|---|---|---|
| `run_alignment.py` | `src/third_party/FastMSS/scripts/` | wav2transcript.scp → Qwen3 批量对齐 → alignment JSON | `qwen3-asr` |
| `textgrid_to_alignment_json.py` | 同上 | Praat TextGrid → alignment JSON (桥接) | `nemo` |
| `alignment_to_lhotse.py` | 同上 | alignment JSON → Lhotse CutSet (.jsonl.gz) | `nemo` |

## 输入文件格式

### wav2transcript.scp (`run_alignment.py`)

```
/path/to/utt_001.wav hello world this is a test utterance
/path/to/utt_002.wav another example transcript here
```

每行按第一个空格切分: `<wav路径> <转录文本>`。

### wav2spk.scp (`alignment_to_lhotse.py`, 推荐)

```
/path/to/utt_001.wav speaker_A
/path/to/utt_001.wav speaker_A
/path/to/utt_002.wav speaker_B
```

每行: `<wav路径> <speaker_id>`。FastMSS 需要区分不同说话人做 turn-taking，
所有 utt 共用 spk0 会导致模拟器出问题。

### wav2textgrid.scp (`textgrid_to_alignment_json.py`)

```
/path/to/utt_001.wav /path/to/utt_001.TextGrid
/path/to/utt_002.wav /path/to/utt_002.TextGrid
```

TextGrid 格式: Praat 短文本格式，含一个名为 `"words"` 的 IntervalTier。

## Step 1: 获取 Alignment JSON

### 路径 A: Qwen3 强制对齐

```bash
conda activate qwen3-asr

python src/third_party/FastMSS/scripts/run_alignment.py \
    --scp data/dump/unified/my_corpus/wav2transcript.scp \
    --output-dir data/dump/force_align/my_corpus/alignments \
    --model Qwen/Qwen3-ForcedAligner-0.6B \
    --batch-size 8
```

参数:

| 参数 | 默认值 | 说明 |
|---|---|---|
| `--scp` | (必填) | wav2transcript.scp 路径 |
| `--output-dir` | (必填) | alignment JSON 输出目录 |
| `--model` | `Qwen/Qwen3-ForcedAligner-0.6B` | HF model ID 或本地路径；本地路径存在则加载本地，否则自动下载 |
| `--device` | `cuda:0` | 设备 |
| `--language` | `auto` | `auto` / `Chinese` / `English` |
| `--batch-size` | `1` | 每批 utterance 数，设大加速推理 |

约束: 单条音频时长 ≤ 300s（超时自动跳过并报错）。

输出: `{output_dir}/{utt_id}.json`

### 路径 B: 外部对齐桥接

当音源已有 word 级对齐结果（如 Praat TextGrid），直接用桥接脚本转入 alignment JSON。
目前提供 TextGrid 桥接:

```bash
conda activate nemo

# 从目录
python src/third_party/FastMSS/scripts/textgrid_to_alignment_json.py \
    --tg-dir external_textgrids/ \
    --output-dir data/dump/force_align/my_corpus/alignments

# 从 scp
python src/third_party/FastMSS/scripts/textgrid_to_alignment_json.py \
    --scp wav2textgrid.scp \
    --output-dir data/dump/force_align/my_corpus/alignments
```

其他格式 (CTM, STM, 自定义 JSON) 按同样模式补写桥接脚本，
核心逻辑：解析源格式 → 提取 `(word, start, end)` 三元组 → 输出 alignment JSON。

## Step 2: Alignment JSON → Lhotse CutSet

```bash
conda activate nemo

python src/third_party/FastMSS/scripts/alignment_to_lhotse.py \
    --align-dir data/dump/force_align/my_corpus/alignments \
    --wav-dir data/dump/unified/my_corpus/wav \
    --spk-scp data/dump/unified/my_corpus/wav2spk.scp \
    --output data/dump/lhotse_cutsets/my_corpus \
    --prefix my_corpus \
    --splits train,dev,test \
    --train-ratio 0.8
```

参数:

| 参数 | 默认值 | 说明 |
|---|---|---|
| `--align-dir` | (必填) | alignment JSON 目录 |
| `--wav-dir` | (必填) | wav 文件目录 (`{utt_id}.wav`) |
| `--spk-scp` | — | 可选，wav2spk.scp |
| `--output` | (必填) | 输出目录 |
| `--prefix` | `cutset` | 输出文件前缀 |
| `--splits` | `all` | `all` 或 `train,dev,test` |
| `--train-ratio` | `0.8` | train 比例 |
| `--dev-ratio` | `0.1` | dev 比例 |
| `--seed` | `42` | 随机种子 |
| `--speaker` | `spk0` | 无 spk-scp 且 alignment 无 speaker 时的回退 speaker |

**Speaker 优先级**:
1. `--spk-scp` 中的映射（按 `utt_id` 匹配）
2. Alignment JSON 中 word 级的 `speaker` 字段（取 dominant speaker）
3. `--speaker` 默认值

**输出**: `{output}/{prefix}_{split}.jsonl.gz`

### 序列化后的 Lhotse CutSet 结构

```json
{
  "id": "utt_001",
  "start": 0.0, "duration": 5.23, "channel": 0,
  "recording": {
    "id": "utt_001",
    "sources": [{"type": "file", "channels": [0], "source": "/path/to/utt_001.wav"}],
    "sampling_rate": 16000, "num_samples": 83680, "duration": 5.23
  },
  "supervisions": [{
    "id": "utt_001-sup000", "recording_id": "utt_001",
    "start": 0.0, "duration": 5.23, "channel": 0,
    "speaker": "speaker_A",
    "text": "HELLO WORLD",
    "alignment": {
      "word": [
        ["HELLO", 0.52, 0.42, null],
        ["WORLD", 1.50, 0.55, null]
      ]
    }
  }]
}
```

FastMSS 要求 `alignment["word"]` 必须存在且非空，否则 cut 会在 `split_monocuts_at_pauses()` 和
`ConversationalMeetingSimulator.__init__()` 中被静默丢弃。

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

## 完整目录结构

```
finetune_pipeline/
├── docs/
│   ├── extend_fastmss_sources.md                  ← 本文档
│   └── fastmss_data_source_extension_pipeline.md   ← 详细设计 (旧)
├── src/third_party/FastMSS/scripts/
│   ├── run_alignment.py                           ← Qwen3 批量对齐
│   ├── textgrid_to_alignment_json.py              ← TextGrid 桥接
│   ├── alignment_to_lhotse.py                     ← JSON → Lhotse
│   └── new_source_pipeline.md                     ← 该目录下的快速参考
├── data/
│   └── dump/
│       ├── unified/{dataset}/
│       │   ├── wav/{utt_id}.wav
│       │   ├── wav2transcript.scp
│       │   └── wav2spk.scp
│       ├── force_align/{dataset}/
│       │   └── alignments/{utt_id}.json
│       ├── lhotse_cutsets/{dataset}/
│       │   └── {prefix}_{split}.jsonl.gz
│       └── fastmss/simulated_sessions/{version}/
```

## 添加新桥接格式

以 CTM 为例，补写 `ctm_to_alignment_json.py`:

```python
# CTM 格式: <file> <ch> <start> <dur> <word> [<conf>]
# 转换逻辑
def parse_ctm_to_words(ctm_path):
    with open(ctm_path) as f:
        for line in f:
            _, _, start, dur, word = line.strip().split()[:5]
            yield {"word": word, "start": float(start), "end": float(start) + float(dur)}

# 输出
data = {"utt_id": utt_id, "words": list(parse_ctm_to_words(ctm_path))}
```

任何格式桥接都只需要这层薄壳：解析 → 提取 `(word, start, end)` → 写 JSON。

## 环境依赖

| 环境 | 用途 | 关键包 |
|---|---|---|
| `qwen3-asr` | Qwen3 对齐 (`run_alignment.py`) | `qwen-asr`, `transformers`, `torch`, `soundfile` |
| `nemo` | 桥接脚本 + Lhotse 转换 + FastMSS | `lhotse`, torchaudio |

两个 conda 环境相互独立，按 Step 手动切换。`path.sh` 中定义了 `qwen3-asr` 环境名。

## 约束与注意事项

| 项 | 说明 |
|---|---|
| 音频时长 ≤ 300s | Qwen3 对齐上限，超时跳过。用户保证输入满足此约束 |
| `alignment["word"]` 必须非空 | 否则 FastMSS 静默丢弃 cut |
| MonoCut 单说话人 | FastMSS 要求每个 cut 一个 speaker |
| 时间戳绝对性 | `start`/`end` 相对于 recording 开头 (与 `MonoCut.start=0` 一致) |
| spk-scp 推荐使用 | 避免全数据集共享 spk0 导致 turn-taking 退化 |
| 采样率 16kHz | 统一使用，FastMSS 默认 |
| bf16 精度 | Qwen3 对齐使用 bf16；7-bit mantissa 可能引起 PIT 浮点平局，参见 `docs/precision_loss_bf16.md` |

## 参考

- [Qwen3-ForcedAligner](https://huggingface.co/Qwen/Qwen3-ForcedAligner-0.6B)
- [FastMSS](https://github.com/popcornell/FastMSS)
- `docs/fastmss_data_source_extension_pipeline.md` — 原始详细设计 (含 G-STAR pipeline 集成细节)
- `src/third_party/FastMSS/scripts/new_source_pipeline.md` — 该目录下的快速参考
