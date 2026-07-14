# Parent–Child RAG 本地构建运行手册

## Generated portable runtime configuration

Do not manually edit the ignored runtime YAML. Generate it from the tracked,
strict local template; provider, model, endpoint, policy, and retry values are
copied only after source validation, while local locations remain explicit:

```powershell
python scripts/init_rag_runtime_config.py `
  --project-root . `
  --source-config config/rag/index.local.yaml `
  --data-root data `
  --index-root indexes/parent_child `
  --registry-path generation_registry.sqlite `
  --output config/rag/index.runtime.yaml
```

The generated YAML stores portable relative paths and is Git-ignored. The
initializer validates that every location remains inside `--project-root`, that
the source catalog agrees exactly with its strict `subject_policy_map`, and that
`evaluation`, `_needs_ocr`, `unclassified`, hidden, and cache directories are
excluded. It does not call a provider or create an index.

## Local experimental build orchestration

Use the one-key build entrypoint only with explicit paths and identifiers. It
never reads `chroma_store`, never resolves an active generation as a candidate,
and never invokes `activate`, `set-primary`, `set-shadow`, or `rollback`.

Run the real loader/splitter and write only an experimental, provider-free
report first. This mode creates no Chroma collection, BM25 artifact, Parent
Store, registry, or generation:

```powershell
python scripts/run_rag_local_build.py `
  --project-root . `
  --index-config config/rag/index.runtime.yaml `
  --benchmark-config config/rag/benchmark.yaml `
  --gold-dataset data/evaluation/<reviewed-gold-dataset>.json `
  --build-id <flat-build-id> `
  --generation-id <parent-child-generation-id> `
  --code-revision <current-git-sha> `
  --run-id <unique-run-id> `
  --no-embedding-cache `
  --embedding-cache-busy-timeout-seconds 10 `
  --offline-dry-run
```

`--execute` is the only mode that can contact configured providers or build
isolated local artifacts. It requires explicit chat-provider coordinates in
addition to the inputs above; the values must come from an approved provider
configuration, never from a hidden default:

```powershell
python scripts/run_rag_local_build.py `
  --project-root . `
  --index-config config/rag/index.runtime.yaml `
  --benchmark-config config/rag/benchmark.yaml `
  --gold-dataset data/evaluation/<reviewed-gold-dataset>.json `
  --build-id <flat-build-id> `
  --generation-id <parent-child-generation-id> `
  --code-revision <current-git-sha> `
  --run-id <unique-run-id> `
  --embedding-cache artifacts/rag/embedding_cache/<fingerprint>.sqlite `
  --embedding-cache-busy-timeout-seconds 10 `
  --llm-provider <provider> `
  --llm-protocol <explicit-chat-protocol> `
  --llm-model <model> `
  --llm-base-url <https-base-url> `
  --llm-endpoint-path <endpoint-path> `
  --llm-api-key-env <environment-variable-name> `
  --llm-timeout-seconds <positive-seconds> `
  --execute
```

The command first validates local dependencies, strict paths, catalog and
source groups; then probes Embedding, Reranker, and the explicit chat LLM. A
failure writes a redacted report in `reports/rag_build/<run-id>/`, returns
non-zero, and stops later stages. It never substitutes Flat Baseline output for
a failed candidate. A completed local build is always marked
`experimental_only=true` and `activation_prohibited=true`; a blocked Gold gate
does not become a passing formal validation result.

Passing a dataset's JSON schema alone does not make it formal Gold. In
particular, select a dataset version only after its spans, source groups, and
human or historical-query semantics have been reviewed; never infer that from a
filename such as `v2`.

本手册只覆盖可审计的本地构建、评估和部署前验证。它不会下载课程资料、不会修改现有 `chroma_store`，也不会因为某个依赖失败改用旧链路。所有命令从项目根目录执行，并且所有路径都必须位于项目根目录内且不得经过符号链接或 Windows reparse point。

在正式 validation 通过、数据 gate 通过且有明确发布批准以前，**不得执行 activate**。

## 1. 数据目录与 source groups

- `data/` 下每个未排除的一级目录由 `SubjectCatalog` 规范化后成为一个学科；不要在脚本或配置中手写学科清单。
- 只把已获准、可抽取的原始资料放入配置的 `catalog.data_root`。扫描件应留在配置的 `_needs_ocr` 隔离目录，不能进入候选构建。
- `catalog.supported_extensions` 是唯一允许的资料扩展名来源。`data/evaluation` 不是课程学科：当 `data_root=data` 时，生成 index config 时必须把 `evaluation` 放入 `--excluded-exact-names`。
- 每个可用 `source_relpath` 都必须在 `config/rag/source_groups.json` 的 `source_groups_v1` 中人工映射到独立来源组。上下册、同系列教材等是否独立由数据负责人确定；工具不会猜测或补全 source group。

提交资料或 source group 变更后，必须重新做 Gold span 校验、readiness audit 和正式 validation；旧结论不能复用。

## 2. 生成严格 index config

先准备一个或多个只含 `ChunkPolicyConfig` 的本地 policy fragment，例如 `artifacts/rag/policies/standard.yaml`。随后运行：

```powershell
python scripts/init_rag_index_config.py --help
```

按帮助文本显式传入全部参数，输出到 `config/rag/index.local.yaml`。其中必须包括：

- catalog 的 data root、扩展名、每项排除规则、normalization 和 `reject` symlink policy；
- storage 路径、collection 名称和所有 schema/timeout/retention 值；
- embedding 与 reranker 的 provider、model、base URL、endpoint、协议、input type、dimension、timeout、retry、batch 和 `api_key_env` 名称；
- BM25 artifact format 和 tokenizer；脚本会读取实际 Jieba 版本与内置字典并写入 SHA-256；
- 每一个 `--chunk-policy NAME=PATH` 以及每一个 `--subject-policy SUBJECT=NAME`；SubjectCatalog 发现的学科和映射必须精确一致；
- 所有 retrieval、context、multi-subject 参数。

严格配置还要求 `reranker.batch_size >= vector_top_k + bm25_top_k`。Flat
Baseline 会按既有语义一次 rerank 两路去重后的完整候选集；不得通过截断候选、
拆分后拼接不可比分数或运行时缩小 batch 来绕过该约束。

示意（省略其余**必填**参数）：

```powershell
python scripts/init_rag_index_config.py `
  --project-root . `
  --output config/rag/index.local.yaml `
  --schema-version rag_index_config_v1 `
  --data-root data `
  --supported-extensions '[".pdf", ".md", ".txt"]' `
  --excluded-exact-names '["evaluation"]' `
  --chunk-policy standard=artifacts/rag/policies/standard.yaml `
  --subject-policy <catalog-subject-id>=standard `
  --embedding-api-key-env RAG_EMBEDDING_API_KEY `
  --reranker-api-key-env RAG_RERANKER_API_KEY
```

不要传入 API Key 值；配置只能保存环境变量名称。脚本会用 `RagIndexConfig` 回读验证、重算 policy ID，并在默认情况下拒绝覆盖已有文件。只有确认配置可替换时才加 `--overwrite`。

## 3. 制作唯一 GoldDataset

GoldDataset 的最终唯一事实来源是 `data/evaluation/gold_dataset_v1.json`；不要把 child ID 或 parent ID 作为 gold 坐标。

1. 初始化一个空、可审查的草稿：

   ```powershell
   python scripts/prepare_rag_gold_dataset.py init `
     --project-root . --dataset-id local_gold_v1 `
     --output data/evaluation/gold_dataset_draft.json
   ```

2. 人工定位证据前，以和 Candidate 完全相同的 page-aware loader 检查资料。输出含 cleaned text、每页物理/逻辑页号、cleaned character offset、章节路径，仅供本地人工标注：

   ```powershell
   python scripts/prepare_rag_gold_dataset.py inspect-source `
     --project-root . --index-config config/rag/index.local.yaml `
     --source-relpath <subject/path/to/source.pdf> `
     --output artifacts/rag/gold-inspection.json
   ```

3. 人工在 draft 中填写 `human_gold`、`historical_annotated` 或 `synthetic_smoke` query。每个 evidence 必须明确提供 `source_group_id`、`source_relpath`、`doc_id`、`pagination_kind`、页范围、`[start_char,end_char)`、`section_path` 和 `relevance_grade`。不能自动猜测 group，不能自动生成 rollout eligible query。

4. 用 source group 和 page-aware loader 证明每个非空 cleaned span，封存最终数据集：

   ```powershell
   python scripts/prepare_rag_gold_dataset.py validate `
     --project-root . --index-config config/rag/index.local.yaml `
     --source-groups config/rag/source_groups.json `
     --input data/evaluation/gold_dataset_draft.json `
     --output data/evaluation/gold_dataset_v1.json
   ```

5. 仅从该 GoldDataset 导出 readiness inventories；synthetic 永远 `eligible_for_rollout=false`：

   ```powershell
   python scripts/prepare_rag_gold_dataset.py export-readiness-jsonl `
     --project-root . --gold-dataset data/evaluation/gold_dataset_v1.json `
     --human-output data/evaluation/human_gold.jsonl `
     --historical-output data/evaluation/historical_annotated.jsonl `
     --synthetic-output data/evaluation/synthetic_smoke.jsonl
   ```

默认不覆盖任何既有 Gold、inspection 或 JSONL 文件。

## 4. Doctor 与 readiness audit

先确认依赖、严格配置、环境变量名称所指向的 secret 是否存在，以及 catalog/benchmark/rollout 学科是否一致。doctor 不访问网络：

```powershell
python scripts/doctor_rag_env.py `
  --project-root . --pipeline parent-child `
  --index-config config/rag/index.local.yaml `
  --benchmark-config config/rag/benchmark.yaml `
  --rollout-config config/rag/rollout.yaml `
  --output reports/rag_doctor.json
```

随后运行只读 readiness audit。`--fail-on-blocked` 让资料或人工 Gold 不足成为非零退出；这表示需要补数据，不表示工具故障。

```powershell
python scripts/audit_rag_readiness.py `
  --project-root . --index-config config/rag/index.local.yaml `
  --benchmark-config config/rag/benchmark.yaml `
  --gold-dataset data/evaluation/gold_dataset_v1.json `
  --output reports/rag_readiness.json --fail-on-blocked
```

## 5. 构建隔离 Flat Baseline

先在项目内但 `index_root` 以外的新目录构建 Flat Baseline。它不会读取或覆盖现有 `chroma_store`，不会打开 deployment pointer，也不会创建 Parent–Child generation：

```powershell
python scripts/build_flat_baseline.py `
  --project-root . --pipeline flat-baseline `
  --index-config config/rag/index.local.yaml `
  --persist-dir artifacts/rag/flat-baseline-<build-id>/chroma `
  --manifest-output artifacts/rag/flat-baseline-<build-id>/manifest.json `
  --collection-name flat_baseline_<build-id> `
  --flat-build-id <build-id>
```

保留该 manifest；它把 baseline 的 source/policy、embedding、BM25 tokenizer 和 chunk ID 集绑定到后续 benchmark。

## 6. 构建 Parent–Child generation（不激活）

只有 readiness gate 不再 blocked 且已具备可用 provider 配置时，才可构建一个显式 generation：

长时间本地构建建议先从 embedding 身份完全一致的 Flat artifact 建立
exact-content cache。缓存只保存文本 SHA-256/长度和向量，不保存正文；历史同文
本但向量不一致的条目会标记为 ambiguous，并在 Candidate 构建时重新请求当前
provider：

```powershell
python scripts/seed_rag_embedding_cache.py `
  --project-root . --index-config config/rag/index.local.yaml `
  --flat-persist-dir artifacts/rag/<flat-build-id>/chroma `
  --flat-manifest artifacts/rag/<flat-build-id>/manifest.json `
  --cache-path artifacts/rag/embedding_cache/<fingerprint>.sqlite `
  --output reports/rag_build/<run-id>/embedding_cache_seed.json `
  --read-page-size 128 --busy-timeout-seconds 10
```

```powershell
python scripts/build_parent_child_generation.py `
  --project-root . --pipeline parent-child `
  --index-config config/rag/index.local.yaml `
  --generation-id <immutable-generation-id> `
  --code-revision <git-commit> `
  --registry-mode existing `
  --embedding-cache artifacts/rag/embedding_cache/<fingerprint>.sqlite `
  --embedding-cache-busy-timeout-seconds 10
```

不使用 cache 时也必须显式传 `--no-embedding-cache`，不能由异常触发 cache 或
其他 provider。cache miss 只调用 index config 中同一个 provider/model；provider
失败仍使 generation 失败，成功批次留在 cache 供全新 generation 重跑。

### Sealed Chroma safety

- `page_clean_v2` requires an explicit `nul_character_policy`. Local Chroma
  builds should use `replace_with_space_v1`, which is deterministic and
  length-preserving; `reject` is available when any extracted NUL must stop the
  build. Child persistence independently rejects remaining NUL characters.
- Chroma 1.5.x writes internal coordination state whenever a
  `PersistentClient` is opened. Validators and retrieval runtimes therefore
  open a marker-owned copy below the configured index root and remove it on
  close. They never open the canonical sealed `chroma_children` directory.
- A digest mismatch is not repairable by editing `manifest.json`. Keep the
  generation inactive and build a new immutable generation after fixing the
  source, cleaning policy, or runtime defect.

成功只代表该 generation 为 `READY`（已密封、完整性验证完成），**不代表已服务流量，更不代表已通过效果验证**。本命令不调用 activate。

## 7. 同 GoldDataset benchmark 与正式 validation

对同一个 canonical GoldDataset 显式提供 baseline artifact 和 candidate generation ID：

```powershell
python scripts/run_parent_child_benchmark.py `
  --project-root . --index-config config/rag/index.local.yaml `
  --gold-dataset data/evaluation/gold_dataset_v1.json `
  --baseline-persist-dir artifacts/rag/flat-baseline-<build-id>/chroma `
  --baseline-manifest artifacts/rag/flat-baseline-<build-id>/manifest.json `
  --candidate-generation-id <immutable-generation-id> `
  --output-dir artifacts/rag/benchmark/<run-id>
```

该 benchmark 从指定 ID 加载 READY candidate，绝不读取 active generation。Vector、BM25、reranker、Parent Store 任一失败都会失败；成功目录只有安全的 `baseline_retrieval_input.json`、`candidate_retrieval_input.json`、`operational_outcome.json`、包含各阶段 P50/P95 的 `operational_details.json`，以及不含 query/正文的诊断 JSONL。

对同一对 retrieval 输入，准备由相同回答模型和同一评审协议产出的外部 `AnswerRun`。本工具不生成答案、不自动评分：

```powershell
python scripts/run_rag_end_to_end_evaluation.py export-template `
  --project-root . --gold-dataset data/evaluation/gold_dataset_v1.json `
  --baseline-retrieval-input artifacts/rag/benchmark/<run-id>/baseline_retrieval_input.json `
  --candidate-retrieval-input artifacts/rag/benchmark/<run-id>/candidate_retrieval_input.json `
  --baseline-answer-run artifacts/rag/baseline_answers.json `
  --candidate-answer-run artifacts/rag/candidate_answers.json `
  --assessment-protocol artifacts/rag/assessment_protocol.json `
  --output artifacts/rag/human_score_template.json
```

人工完成全部评分后导入，缺任何一项都应失败：

```powershell
python scripts/run_rag_end_to_end_evaluation.py import-scores `
  --project-root . --gold-dataset data/evaluation/gold_dataset_v1.json `
  --baseline-retrieval-input artifacts/rag/benchmark/<run-id>/baseline_retrieval_input.json `
  --candidate-retrieval-input artifacts/rag/benchmark/<run-id>/candidate_retrieval_input.json `
  --baseline-answer-run artifacts/rag/baseline_answers.json `
  --candidate-answer-run artifacts/rag/candidate_answers.json `
  --assessment-protocol artifacts/rag/assessment_protocol.json `
  --scored-template artifacts/rag/human_score_template_completed.json `
  --output artifacts/rag/end_to_end_outcome.json
```

正式 validation 计算既有 validator 的指标与 gates；它会拒绝 dataset、digest、embedding、run、generation 或 artifact manifest 混用：

```powershell
python scripts/validate_parent_child_candidate.py `
  --project-root . --benchmark-config config/rag/benchmark.yaml `
  --gold-dataset data/evaluation/gold_dataset_v1.json `
  --baseline-input artifacts/rag/benchmark/<run-id>/baseline_retrieval_input.json `
  --candidate-input artifacts/rag/benchmark/<run-id>/candidate_retrieval_input.json `
  --operational-outcome artifacts/rag/benchmark/<run-id>/operational_outcome.json `
  --end-to-end-outcome artifacts/rag/end_to_end_outcome.json `
  --functional-tests-passed true `
  --output artifacts/rag/candidate_validation.json
```

## 8. READY、Shadow、activate 与 rollback

| 状态/操作 | 含义 | 是否改变用户服务路径 |
| --- | --- | --- |
| `READY` | generation 通过构建完整性校验，尚未部署。 | 否 |
| Shadow | baseline 正常服务，candidate 仅按明确控制面并行观察。 | 否 |
| activate | Registry 原子地把一个已验证的 READY generation 设为 primary。 | 是 |
| rollback | Registry 显式把 previous READY generation 重新设为 primary。 | 是 |

`scripts/manage_rag_generation.py` 是控制面工具；它不会因请求异常自动切换。只有所有离线、正式 validation、Shadow 和灰度门槛通过，并且 `rollout.yaml` 已由产品/数据负责人明确启用时，才可执行 `set-shadow` 或 `activate`。在本手册所述 validation 通过前，不得运行这些操作；若要回退，只能显式使用 `rollback`，不能把 candidate 失败伪装成 baseline 成功。
