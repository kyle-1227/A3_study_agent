import json
import pathlib
import sys

sys.path.insert(0, r'D:\software competition\A3_study_agent')
from src.rag.gold_dataset import load_gold_dataset_draft_or_final, validate_gold_dataset
from src.config.rag_index_config import load_rag_index_config
from src.rag.readiness import load_source_group_manifest

project = pathlib.Path(r'D:\software competition\A3_study_agent')
config = load_rag_index_config(project / 'config/rag/index.local.yaml')
source_groups = load_source_group_manifest(project / 'config/rag/source_groups.json')
path = project / 'data/evaluation/gold_dataset_draft.json'
dataset = load_gold_dataset_draft_or_final(path)
try:
    validate_gold_dataset(dataset=dataset, index_config=config, source_groups=source_groups)
    print('validated ok')
except Exception as exc:
    print(type(exc).__name__, exc)
    import traceback
    traceback.print_exc()
