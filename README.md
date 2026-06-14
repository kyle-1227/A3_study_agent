# A3 Study Agent

楂樻牎涓€у寲瀛︿範璧勬簮鐢熸垚鏅鸿兘浣撱€?

<p align="center">
  <a href="README_en.md">English README</a> |
  <a href="docs/architecture/v0.3.0/diagram_design.md">Architecture Diagrams</a> |
  <a href="CHANGELOG.md">Changelog</a>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/version-v0.3.0-orange?style=flat-square" alt="version" />
  <img src="https://img.shields.io/badge/python-3.11%2B-blue?style=flat-square" alt="python" />
  <a href="https://github.com/langchain-ai/langgraph">
    <img src="https://img.shields.io/badge/langgraph-v1.1.1-7C3AED?style=flat-square&logo=diagram-next&logoColor=white" alt="langgraph" />
  </a>
  <a href="./LICENSE">
    <img src="https://img.shields.io/badge/license-MIT-green?style=flat-square" alt="license" />
  </a>
</p>

## 鍏充簬椤圭洰

A3 Study Agent 鏄竴涓潰鍚戦珮鏍¤绋嬪涔犲満鏅殑澶氭櫤鑳戒綋瀛︿範璧勬簮鐢熸垚绯荤粺銆傚畠鍩轰簬 **LangGraph**銆?*FastAPI** 鍜?**Next.js** 鏋勫缓锛屽洿缁曞涔犺€呯殑璇剧▼闂銆佸涔犵洰鏍囧拰璧勬簮闇€姹傦紝鐢熸垚璇剧▼绛旂枒銆佸垎灞傜粌涔犮€佹€濈淮瀵煎浘鍜屽涔犺鍒掔瓑涓€у寲瀛︿範璧勬簮銆?

绯荤粺缁撳悎鏈湴璇剧▼璧勬枡 RAG銆丅M25銆丷eranker銆乀avily Web Search銆佺粨鏋勫寲 LLM 杈撳嚭鍜?OpenTelemetry 鍙娴嬫€э紝鏀寔鐪熷疄浜や簰閾捐矾涓殑妫€绱€佽瘉鎹鍐炽€佺敓鎴愬拰璇婃柇銆?

> 褰撳墠 React 鍓嶇涓昏鐢ㄤ簬婕旂ず澶嶆潅 Agent 浜や簰銆丼SE 娴佸紡杈撳嚭銆佽祫婧愮敓鎴愬拰杩愯杞ㄨ抗銆傚悗缁鍒掓ā鍧椾細缁х画鎵╁睍锛屼絾绗竴闃舵鏂囨。涓嶅啀瑕嗙洊涓撻」瑙勫垝椤甸潰銆?

## 鏍稿績鑳藉姏

- **璇剧▼绛旂枒**锛氬熀浜庢湰鍦拌绋嬭祫鏂欏拰 Web evidence 鐨勫弻婧愯瘉鎹瀺鍚堬紝鐢熸垚闈㈠悜楂樻牎瀛︿範鑰呯殑瑙ｉ噴涓庣ず渚嬨€?
- **涓€у寲瀛︿範璧勬簮鐢熸垚**锛氱敓鎴愬垎灞傜粌涔犻銆佹€濈淮瀵煎浘銆侀」鐩渚嬪拰瀛︿範鏉愭枡鎽樿銆?
- **瀛︿範瑙勫垝**锛氶€氳繃澶?Agent 璧疯崏銆佸鏌ュ拰浜哄伐鍙嶉锛屾敮鎸侀樁娈靛寲瀛︿範瀹夋帓銆?
- **鎯呯华涓庡涓氭敮鎸?*锛氫互楂樻牎瀛︿範瀵煎笀 / 瀛︿笟鏀寔瀵煎笀鐨勮姘旓紝鎻愪緵娓╂殩涓斿彲鎵ц鐨勫缓璁€?
- **鍙娴嬫€?*锛氶€氳繃 A3_TRACE銆丱penTelemetry銆丼SE 鑺傜偣浜嬩欢鍜岀粨鏋勫寲璇婃柇鏃ュ織鎺掓煡鐪熷疄浜や簰閾捐矾銆?
- **閰嶇疆椹卞姩**锛氶€氳繃 YAML 閰嶇疆鍜?XML prompt 绠＄悊杩愯鍙傛暟涓庢ā鍨嬭涓恒€?

## 绯荤粺鏋舵瀯

```mermaid
graph TD
  START([瀛︿範鑰呰緭鍏) --> supervisor[鎰忓浘璇嗗埆]

  supervisor -->|academic| academic_router[瀛︽湳瀛︿範璺敱]
  supervisor -->|academic study_plan| study_plan_emotional_intel[瑙勫垝涓婁笅鏂囨绱
  supervisor -->|emotional| emotional_response[瀛︿笟鏀寔鍥炲簲]
  supervisor -->|unknown| handle_unknown[鏈煡鎰忓浘澶勭悊]

  academic_router --> rag_retrieve[RAG / Web Evidence 妫€绱
  rag_retrieve --> generate_answer[鍥炵瓟鐢熸垚]
  generate_answer --> evaluate_hallucination[蹇犲疄鎬ц瘎浼癩
  evaluate_hallucination -->|閫氳繃| END_A([缁撴潫])
  evaluate_hallucination -->|閲嶈瘯| rewrite_query[鏌ヨ鏀瑰啓]
  rewrite_query --> academic_router

  study_plan_emotional_intel --> study_plan_planner[瑙勫垝淇℃伅鏀堕泦]
  study_plan_planner --> study_plan_agent[璁″垝璧疯崏]
  study_plan_agent --> study_plan_reviewer_academic[瀛︽湳瀹℃煡]
  study_plan_agent --> study_plan_reviewer_emotional[鎯呯华瀹℃煡]
  study_plan_reviewer_academic --> study_plan_consensus[鍏辫瘑妫€鏌
  study_plan_reviewer_emotional --> study_plan_consensus
  study_plan_consensus -->|閫氳繃| study_plan_output[璁″垝杈撳嚭 + HIL]
  study_plan_consensus -->|鎵撳洖| study_plan_rewrite[璁″垝淇]
  study_plan_rewrite --> study_plan_agent

  study_plan_output -->|纭| END_P([缁撴潫])
  study_plan_output -->|鍙嶉| study_plan_rewrite[鍙嶉鍒嗙被]
  study_plan_rewrite -->|寰皟| study_plan_rewrite[璁″垝寰皟]
  study_plan_rewrite -->|閲嶅啓| study_plan_agent
  study_plan_rewrite --> study_plan_output

  emotional_response --> END_E([缁撴潫])
  handle_unknown --> END_U([缁撴潫])
```

璇︾粏鏋舵瀯鍥捐 [`docs/architecture/v0.3.0/diagram_design.md`](docs/architecture/v0.3.0/diagram_design.md)銆?

## 鎶€鏈爤

| 灞傜骇 | 缁勪欢 |
| ---- | ---- |
| 鍓嶇 | Next.js 16銆丷eact銆乀ailwind CSS銆丷eact Flow |
| 鍚庣 API | FastAPI銆乁vicorn銆丼SE |
| 缂栨帓 | LangGraph |
| 鏈湴鐭ヨ瘑搴?| ChromaDB銆丅M25銆丷eranker |
| Web Search | Tavily |
| 鐘舵€佸揩鐓?| LangGraph Checkpointer锛岄粯璁?MemorySaver锛屽彲閫?PostgreSQL |
| 鍙娴嬫€?| A3_TRACE銆丱penTelemetry銆丣aeger銆丼QLite fallback |
| 閰嶇疆 | YAML settings銆乆ML prompts |

## 蹇€熷惎鍔?

### Docker Compose

```bash
git clone https://github.com/kyle-1227/A3_study_agent.git
cd A3_study_agent

cp .env.example .env
# 缂栬緫 .env锛屽～鍏ユ墍闇€妯″瀷銆佹悳绱㈠拰瑙傛祴閰嶇疆

docker compose up -d

# 鍙€夛細鍚敤 Jaeger tracing
docker compose --profile observability up -d
```

鍓嶇锛歚http://localhost:3000`
鍚庣 API锛歚http://localhost:8000`
Jaeger锛歚http://localhost:16686`

### 鏈湴寮€鍙?

```bash
python -m venv .venv
.\.venv\Scripts\Activate.ps1

python -m pip install --upgrade pip
pip install -r requirements.txt
pip install -e .

cp .env.example .env
# 缂栬緫 .env锛屽～鍏?API keys
```

#### 鏋勫缓鐭ヨ瘑搴?

灏?PDF / MD / TXT 璇剧▼璧勬枡鏀惧叆浠ヤ笅鐩綍涓殑涓€涓垨澶氫釜锛?

- `data/big_data`
- `data/computer`
- `data/machine_learning`
- `data/math`
- `data/python`

鐒跺悗杩愯锛?

```bash
python scripts/build_index.py
```

#### 鍚姩鏈嶅姟

```bash
# 缁堢 1锛氬悗绔?
uvicorn app:app --reload --port 8000

# 缁堢 2锛氬墠绔?
cd frontend
npm install
npm run dev
```

## 椤圭洰缁撴瀯

```text
A3_study_agent/
鈹溾攢鈹€ app.py                         # FastAPI SSE endpoints + lifespan
鈹溾攢鈹€ docker-compose.yml             # Backend + PostgreSQL + Jaeger
鈹溾攢鈹€ config/
鈹?  鈹溾攢鈹€ settings.yaml              # Runtime parameters
鈹?  鈹斺攢鈹€ prompts/                   # XML prompt templates
鈹溾攢鈹€ src/
鈹?  鈹溾攢鈹€ graph/                     # LangGraph nodes and state flow
鈹?  鈹溾攢鈹€ rag/                       # Local retrieval and indexing
鈹?  鈹溾攢鈹€ llm/                       # LLM factory and structured output runtime
鈹?  鈹溾攢鈹€ database/                  # Checkpointer management
鈹?  鈹溾攢鈹€ tracing/                   # OpenTelemetry setup
鈹?  鈹斺攢鈹€ tools/                     # Web search and resource tools
鈹溾攢鈹€ frontend/                      # Next.js UI
鈹溾攢鈹€ data/                          # University course materials
鈹溾攢鈹€ scripts/                       # Indexing and debug scripts
鈹斺攢鈹€ tests/                         # Test suite
```

## 娴嬭瘯

```bash
python -m pytest tests/test_config.py tests/test_app.py tests/test_rag.py tests/test_tracing.py -v

# 鐜鍏佽鏃?
python -m pytest -q
cd frontend && npm run build
```

## License

[MIT](./LICENSE)

