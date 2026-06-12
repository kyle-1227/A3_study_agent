# v0.2.0 鏋舵瀯鍥?

鏈枃妗ｅ寘鍚?v0.2.0 绯荤粺鏋舵瀯鐨?Mermaid 鍥捐В銆侻ermaid 鑺傜偣鏍囩鍙婁唬鐮佺墖娈典繚鐣欒嫳鏂囷紝浠ヤ究涓庝唬鐮佸簱淇濇寔涓€鑷淬€?

---

## 1. 鍏ㄧ郴缁熸灦鏋勬€昏

```mermaid
flowchart TD
    User(["馃懁 User"])

    subgraph Frontend["Frontend 鈥?Next.js 16"]
        Chat["Chat Area\n(SSE consumer\nMarkdown renderer)"]
        RightPanel["Right Panel\n(DAG viz / Node Trail\nSystem Logs\nToken Usage)"]
        Sidebar["Left Sidebar\n(Chat History)"]
    end

    subgraph Backend["Backend 鈥?FastAPI"]
        API["POST /stream\nStreamingResponse\n(text/event-stream)"]
        SSE["generate_sse()\nastream_events v2"]
    end

    subgraph Graph["LangGraph StateGraph 鈥?LearningState"]
        Supervisor["supervisor\nQwen2.5-7B\nSiliconFlow\ntemperature=0.0"]
        subgraph Academic["Academic Branch"]
            AR["academic_router"]
            RAG["rag_retrieve\nHybrid RAG"]
            WS["web_search\nDuckDuckGo"]
            GA["generate_answer\nDeepSeek-V3"]
            EH["evaluate_hallucination\nDeepSeek-V3 structured"]
        end
        subgraph Planner["Planner Branch"]
            SP["study_plan_emotional_intel\nPlanning context retrieval"]
            GP["study_plan_agent\nDeepSeek-V3"]
        end
        ER["emotional_response\nDeepSeek-V3"]
    end

    subgraph RAGStack["RAG Stack"]
        ChromaDB[("ChromaDB\nvector store")]
        BM25["BM25 Index\njieba tokenizer"]
        Reranker["BGE Reranker\nSiliconFlow API\nbge-reranker-v2-m3"]
    end

    subgraph Infra["Infrastructure"]
        PG[("PostgreSQL\nLangGraph Checkpointer")]
        Jaeger["Jaeger UI\nlocalhost:16686"]
        SQLite[("SQLite\ntraces.db")]
        OTel["OpenTelemetry\nTracerProvider"]
    end

    User -- "HTTP POST /stream" --> API
    API --> SSE
    SSE -- "astream_events" --> Graph
    SSE -- "SSE: token / node_event / usage" --> Chat
    Chat --> RightPanel

    Supervisor -->|academic| AR
    Supervisor -->|academic study_plan| SP
    Supervisor -->|emotional| ER

    AR --> RAG
    AR --> WS
    RAG --> GA
    WS --> GA
    GA --> EH
    EH -->|"retry (count 鈮?max_retries)"| AR
    EH -->|end| DONE1(["END"])

    SP --> GP
    GP --> DONE2(["END"])
    ER --> DONE3(["END"])

    RAG --> ChromaDB
    RAG --> BM25
    RAG --> Reranker

    Graph -. "checkpoint" .-> PG
    Graph -- "@traced_node" --> OTel
    OTel --> Jaeger
    OTel --> SQLite
```

---

## 2. LangGraph 鑺傜偣鎷撴墤锛堢姸鎬佹祦杞級

```mermaid
flowchart TD
    START(["START"])
    END1(["END"])
    END2(["END"])
    END3(["END"])

    START --> supervisor

    supervisor -->|"intent = academic"| academic_router
    supervisor -->|"requested_resource_type = study_plan"| study_plan_emotional_intel
    supervisor -->|"intent = emotional"| emotional_response

    subgraph fan_out["Fan-out / Fan-in (parallel)"]
        academic_router --> rag_retrieve
        academic_router --> web_search
        rag_retrieve --> generate_answer
        web_search --> generate_answer
    end

    generate_answer --> evaluate_hallucination

    evaluate_hallucination -->|"hallucination_detected=True\nretry_count 鈮?max_retries"| academic_router
    evaluate_hallucination -->|"faithful OR retries exhausted"| END1

    study_plan_emotional_intel --> study_plan_agent
    study_plan_agent --> END2

    emotional_response --> END3

    style fan_out fill:#f0f4f0,stroke:#7a9e7e
```

**`LearningState` 鍏抽敭瀛楁涓庡啓鍏ユ柟褰掑睘锛?*

| 瀛楁 | 鍐欏叆鏂?| 娑堣垂鏂?|
|------|--------|--------|
| `messages` | supervisor锛堝垵濮嬪寲锛夈€乬enerate_answer銆乬enerate_plan銆乪motional_response | 鎵€鏈夎妭鐐?|
| `intent` | supervisor | builder锛堟潯浠惰竟锛?|
| `subject` | supervisor | rag_retrieve锛堝厓鏁版嵁杩囨护锛?|
| `keypoints` | supervisor | rag_retrieve锛堟煡璇㈡瀯閫狅級 |
| `context` | rag_retrieve銆亀eb_search锛堥€氳繃 `operator.add` 鍚堝苟锛?| generate_answer |
| `study_plan_artifact` | study_plan_emotional_intel | study_plan_agent |
| `retry_count` | evaluate_hallucination | should_retry_or_end |
| `hallucination_detected` | evaluate_hallucination | should_retry_or_end |

---

## 3. 娣峰悎 RAG 娴佹按绾?

```mermaid
flowchart LR
    Q["User Query\n(joined keypoints)"]

    subgraph Stage1["Stage 1 鈥?Retrieval (parallel)"]
        V["Vector Search\nChromaDB + BGE-M3\ntop_k = 10\nsubject filter applied"]
        B["BM25 Search\njieba tokenize\ntop_k = 10\nno subject filter"]
    end

    subgraph Stage2["Stage 2 鈥?Merge"]
        M["Merge + Dedup\n(MD5 content hash)\nvector results first"]
    end

    subgraph Stage3["Stage 3 鈥?Rerank"]
        R["BGE Reranker\nSiliconFlow API\nbge-reranker-v2-m3\ntop_n = 5"]
    end

    OUT["Top-N docs\n{content, source, score,\nrerank_score, metadata}"]

    FALLBACK["Graceful Degradation:\nreranker API fails 鈫?sorted by original score\nBM25 empty 鈫?pure vector results\nChromaDB empty 鈫?empty result"]

    Q --> V
    Q --> B
    V --> M
    B --> M
    M --> R
    R --> OUT
    R -. "on failure" .-> FALLBACK
    FALLBACK --> OUT
```

**`config/settings.yaml` 閰嶇疆鍙傛暟璇存槑锛?*

```yaml
rag:
  vector_top_k: 10
  bm25_top_k: 10
  reranker_top_n: 5
  relevance_threshold: 0.3
  reranker_model: "BAAI/bge-reranker-v2-m3"
```

---

## 4. SSE 浜嬩欢娴佹牸寮忚鑼?

```mermaid
sequenceDiagram
    participant FE as Frontend (Next.js)
    participant BE as Backend (FastAPI)
    participant LG as LangGraph

    FE->>BE: POST /stream {"query": "...", "thread_id": "..."}
    BE->>LG: graph.astream_events(state_input, config, version="v2")

    loop for each graph node
        LG-->>BE: on_chain_start {name, metadata.langgraph_node}
        BE-->>FE: data: {"type":"node_event","status":"start","node":"supervisor"}

        alt LLM node (generate_answer / study_plan_agent / emotional_response)
            loop token streaming
                LG-->>BE: on_chat_model_stream {chunk.content}
                BE-->>FE: data: {"type":"token","content":"..."}
            end
            LG-->>BE: on_chat_model_end {output.usage_metadata}
            BE-->>FE: data: {"type":"usage","node":"generate_answer","input_tokens":N,"output_tokens":N,"total_tokens":N}
        end

        LG-->>BE: on_chain_end {name, metadata.langgraph_node}
        BE-->>FE: data: {"type":"node_event","status":"end","node":"supervisor","duration_ms":234,"error":null}
    end
```

**鍓嶇 SSE 浜嬩欢娑堣垂鏄犲皠锛?*

| SSE 浜嬩欢 | 鍓嶇澶勭悊閫昏緫 |
|----------|------------|
| `node_event` start | `nodeEvents` 鐘舵€侊細杩藉姞 `{node, status: "running", ts}` |
| `node_event` end | `nodeEvents`锛氭爣璁?`status: "done"`锛岄檮鍔?`durationMs`锛涘悜鏃ュ織杩藉姞 `[PERF]` 鏉＄洰 |
| `node_event` end with error | `nodeEvents`锛氭爣璁板畬鎴愶紱鍚戞棩蹇楄拷鍔?`[ERROR]` 鏉＄洰 |
| `token` | 灏?`content` 杩藉姞鍒板綋鍓嶅姪鎵嬫秷鎭紙娴佸紡鎵撳瓧鏈烘晥鏋滐級 |
| `usage` | 绱姞鍒?`tokenUsage` 鐘舵€侊紱鍚戞棩蹇楄拷鍔?`[USAGE]` 鏉＄洰 |

---

## 5. LLM 閰嶇疆鏋舵瀯

```mermaid
flowchart TD
    subgraph Settings["config/settings.yaml"]
        SUP_CFG["supervisor:\n  model: Qwen/Qwen2.5-7B-Instruct\n  base_url: siliconflow\n  api_key_env: SILICONFLOW_API_KEY\n  temperature: 0.0"]
        AC_CFG["academic:\n  temperature: 0.7\n  (no model override 鈫?DEEPSEEK_*)"]
        PL_CFG["planner:\n  temperature: 0.7\n  (no model override 鈫?DEEPSEEK_*)"]
        EM_CFG["emotional:\n  temperature: 0.8\n  (no model override 鈫?DEEPSEEK_*)"]
    end

    Factory["get_node_llm(node_name, **overrides)\nsrc/graph/llm.py"]

    SUP_CFG --> Factory
    AC_CFG --> Factory
    PL_CFG --> Factory
    EM_CFG --> Factory

    subgraph Env[".env"]
        DS["DEEPSEEK_API_KEY\nDEEPSEEK_BASE_URL\nDEEPSEEK_MODEL"]
        SF["SILICONFLOW_API_KEY\n(shared by: embedding,\nreranker, supervisor,\nfallback)"]
        FB["FALLBACK_MODEL\nFALLBACK_API_KEY\nFALLBACK_BASE_URL\n(鈫?SiliconFlow + Qwen2.5-7B)"]
    end

    DS --> Factory
    SF --> Factory

    Factory --> Supervisor["Supervisor ChatOpenAI\nQwen2.5-7B @ SiliconFlow"]
    Factory --> Academic["Academic ChatOpenAI\nDeepSeek-V3"]
    Factory --> Planner["Planner ChatOpenAI\nDeepSeek-V3"]
    Factory --> Emotional["Emotional ChatOpenAI\nDeepSeek-V3"]

    FB --> Fallback["Fallback ChatOpenAI\nQwen2.5-7B @ SiliconFlow\n(auto-triggered by invoke_with_fallback)"]
```

