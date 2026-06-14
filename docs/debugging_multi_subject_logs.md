# 澶?Subject 妫€绱?A3_TRACE 璋冭瘯鏃ュ織

鏈棩蹇楃敤浜庡紑鍙戦樁娈甸獙璇佸 subject 妫€绱㈤摼璺€傚紑鍚悗锛屽悗绔棩蹇椾細杈撳嚭缁熶竴鏍煎紡锛?
```text
A3_TRACE {"stage":"query_rewrite","request_id":"...","session_id":"...","thread_id":"..."}
```

姣忔潯鏃ュ織閮藉甫 `request_id`銆乣session_id`銆乣thread_id`锛屾帓鏌ュ杞璇濇椂鍙寜杩欎簺瀛楁杩囨护銆?
## 寮€鍏?
```env
# Master switch for all structured development trace logs
LOG_A3_TRACE=true

# Fine-grained trace switches
LOG_SUPERVISOR_RESULT=true
LOG_QUERY_REWRITE_RESULT=true
LOG_RETRIEVAL_PLAN=true
LOG_RAG_RESULT=true
LOG_CONTEXT_ASSEMBLY=true
LOG_WEB_SEARCH_RESULT=true
LOG_GENERATION_SUMMARY=true
LOG_PLANNING_INTEL=true
LOG_RETRY_TRACE=true
```

寮€鍙戦樁娈靛彲浠ュ叏閮ㄦ墦寮€銆傜ǔ瀹氬悗寤鸿鍙繚鐣?`LOG_RAG_RESULT=true` 鍜?`LOG_QUERY_REWRITE_RESULT=true`銆?
## 娴嬭瘯 1锛氬 subject 鏅€氱瓟鐤?
杈撳叆锛?
```text
鐢?Python 鍋氫竴涓満鍣ㄥ涔犺繃鎷熷悎妫€娴嬫渚?```

棰勬湡锛?
- `query_rewrite.retrieval_plan_count >= 2`
- subjects 鍖呭惈 `python` 鍜?`machine_learning`
- `machine_learning` role 鎺ヨ繎 `core_concept`
- `python` role 鎺ヨ繎 `implementation_tool`
- `rag_retrieve_plan_item.subject_mismatch_count = 0`
- `context_assembly.subject_doc_distribution` 鍚屾椂鍖呭惈 `python` 鍜?`machine_learning`

## 娴嬭瘯 2锛氬 subject 鎬濈淮瀵煎浘

杈撳叆锛?
```text
缁欐垜鐢熸垚涓€涓?Python 瀹炵幇鏈哄櫒瀛︿範杩囨嫙鍚堟娴嬬殑鎬濈淮瀵煎浘
```

棰勬湡锛?
- `supervisor.requested_resource_type = mindmap`
- `retrieval_plan` 鍖呭惈 `python` / `machine_learning`
- `mindmap_planner.subjects_used` 鍖呭惈涓や釜 subject

## 娴嬭瘯 3锛歱lanning 澶?subject

杈撳叆锛?
```text
甯垜鍒跺畾涓€涓?Python + 鏈哄櫒瀛︿範 4 鍛ㄥ叆闂ㄨ鍒?```

棰勬湡锛?
- `supervisor.requested_resource_type = study_plan`
- 璺緞缁忚繃 `query_rewrite`
- `planning_study_plan_planner.mode = multi_subject`

## 娴嬭瘯 4锛歊AG subject filter

杈撳叆锛?
```text
Python 鍑芥暟 鍙傛暟 杩斿洖鍊?浣滅敤鍩?```

棰勬湡锛?
- `rag_retrieve_single_subject.subject = python`
- `subject_mismatch_count = 0`
- `top_docs.metadata_subject` 鍏ㄩ儴涓?`python`

## Build Check

```bash
python -m py_compile \
  src/observability/a3_trace.py \
  src/graph/supervisor.py \
  src/graph/academic.py \
  src/graph/mindmap.py \
  src/graph/exercises.py \
  src/graph/planner.py

pytest
```

## Web Search 鎺掓煡

寮€鍚細
```env
LOG_WEB_SEARCH_RESULT=true
```

閲嶇偣鐪?`stage=web_search` 鐨勫瓧娈碉細
- `query_source`锛氭湰娆℃悳绱?query 鏉ヨ嚜 `rewritten_query`銆乣search_web_query`銆乣retrieval_plan_top_priority` 杩樻槸鍘熷闂銆?- `provider`锛氬綋鍓嶄负 `duckduckgo`銆?- `ok`锛氭悳绱㈠伐鍏锋槸鍚﹁涓鸿皟鐢ㄦ垚鍔熴€?- `result_count`锛氭渶缁堝彲鐢ㄧ粨鏋滄暟閲忋€?- `raw_type` / `raw_count`锛氬簳灞傚伐鍏疯繑鍥炵殑鏄?`list`銆乣str`銆乣str_empty_or_error` 绛夈€?- `error_type` / `error_message`锛氶敊璇被鍨嬩笌鑴辨晱鍚庣殑鐭敊璇俊鎭€?- `elapsed_ms`锛氭悳绱㈣€楁椂銆?
濡傛灉 `raw_type=str_empty_or_error` 涓?`result_count=0`锛岄€氬父琛ㄧず DuckDuckGo 杩斿洖浜嗏€滄棤鏈夋晥缁撴灉鈥濇垨闄愭祦/閿欒鏂囨湰锛岃繖绫诲瓧绗︿覆涓嶄細琚綋鎴愮湡瀹炴悳绱㈢粨鏋滃啓鍏ヤ笂涓嬫枃銆?
涔熷彲浠ヨ劚绂?Agent 鐩存帴杩愯锛?```bash
python scripts/debug_web_search.py
```

## Hallucination Evaluation 鎺掓煡

寮€鍚細
```env
LOG_RETRY_TRACE=true
```

閲嶇偣鐪?`stage=hallucination_eval` 鐨勫瓧娈碉細
- `success=true`锛氭渶缁堟嬁鍒颁簡鍙В鏋愮殑 `HallucinationEvaluation`銆?- `success=false` 涓?`defaulted_to_valid=true`锛歱rimary 鍜?fallback 閮芥病鏈夊緱鍒板彲瑙ｆ瀽缁撴灉锛屽洜姝ゆ寜鐜版湁涓氬姟瑙勫垯榛樿閫氳繃銆?- `primary_called` / `fallback_called` / `fallback_used`锛氭槸鍚﹁皟鐢?primary銆佹槸鍚﹀皾璇?fallback銆佹渶缁堟槸鍚﹂噰鐢?fallback 缁撴灉銆?- `failure_phase`锛氬け璐ラ樁娈碉紝渚嬪 `structured_parsing_error`銆乣parsed_none`銆乣fallback_structured_parsing_error`銆乣fallback_parsed_none`銆乣primary_call_failed`銆?- `parsing_error`锛氱粨鏋勫寲杈撳嚭瑙ｆ瀽閿欒鎽樿銆?- `raw_preview`锛歀LM 鍘熷杩斿洖鐨勭煭棰勮锛屼笉鍖呭惈瀹屾暣 raw銆?- `context_rag_count` / `context_web_count`锛氳瘎浼版椂浣跨敤鐨勪笂涓嬫枃鏉ユ簮鏁伴噺銆?
娉ㄦ剰锛氳瘖鏂棩蹇楀彧鍐欏叆鍚庣 logger锛屼笉浼氳繘鍏?`messages`銆佺敤鎴峰洖绛斻€丷AG context 鎴栧墠绔皵娉°€?
