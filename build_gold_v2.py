import json
import pathlib
import sys
from typing import Iterable

ROOT = pathlib.Path(r'D:\software competition\A3_study_agent')
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.config.rag_index_config import load_rag_index_config
from src.rag.chunking.structure_detector import detect_document_sections
from src.rag.gold_dataset import _catalog_snapshot, _catalog_sources
from src.rag.parent_child.config_adapter import resolve_subject_chunk_policy
from src.rag.parent_child.loader import load_cleaned_source, page_range_for_span
from src.rag.parent_child.models import SourceEntry
from src.rag.readiness import load_source_group_manifest


def _ensure_source_group_aliases() -> None:
    path = ROOT / 'config/rag/source_groups.json'
    payload = json.loads(path.read_text(encoding='utf-8'))
    manifest = payload['source_groups']
    for source_key, group_id in list(manifest.items()):
        if source_key.startswith('data/'):
            alias = source_key[len('data/'):]
            if alias not in manifest:
                manifest[alias] = group_id
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')


def _load_document(source_relpath: str):
    config = load_rag_index_config(ROOT / 'config/rag/index.local.yaml')
    snapshot = _catalog_snapshot(config)
    sources = _catalog_sources(snapshot)
    source = sources[source_relpath]
    policy = resolve_subject_chunk_policy(config, source.subject_id)
    document = load_cleaned_source(
        SourceEntry(
            schema_version='source_entry_v1',
            source_path=source.source_path,
            data_root=snapshot.data_root,
            subject=source.subject_id,
            doc_type='pdf' if source.extension == '.pdf' else 'markdown',
        ),
        policy.loader_config,
    )
    return document, config


def _question_for(topic: str, title: str) -> str:
    title_l = title.lower()
    if 'continuity' in title_l or 'continuous' in title_l:
        return 'How is continuity at a point defined using limits and function values?'
    if 'limit' in title_l:
        return 'What conditions are required for a limit to exist and how is it related to continuity?'
    if 'derivative' in title_l or 'differenti' in title_l:
        return 'What does the derivative represent and how is it connected to rates of change?'
    if 'integral' in title_l:
        return 'How is a definite integral related to accumulation and area?'
    if 'mean value' in title_l or 'mvt' in title_l:
        return 'What conditions must hold for the Mean Value Theorem to apply?'
    if 'chain' in title_l:
        return 'How does the chain rule apply to composite functions?'
    if 'stack' in title_l:
        return 'Why does an array stack achieve amortized O(1) time for push and pop operations?'
    if 'hash' in title_l or 'collision' in title_l:
        return 'How does a hash table handle collisions?'
    if 'bfs' in title_l or 'breadth' in title_l:
        return 'Why is breadth-first search appropriate for finding shortest paths in unweighted graphs?'
    if 'thread' in title_l or 'process' in title_l:
        return 'What are the key differences between processes and threads?'
    if 'tree' in title_l or 'red' in title_l:
        return 'How do the tree properties control the height of a balanced search tree?'
    if 'mapreduce' in title_l or 'combiner' in title_l:
        return 'What is the role of a combiner in MapReduce?'
    if 'spark' in title_l:
        return 'How does Spark lazy evaluation reduce unnecessary computation?'
    if 'warehouse' in title_l or 'fact' in title_l or 'dimension' in title_l:
        return 'How do fact tables and dimension tables differ in a data warehouse?'
    if 'stream' in title_l or 'event' in title_l:
        return 'How do event time and processing time differ in stream processing?'
    if 'partition' in title_l:
        return 'How do partitions affect throughput and message ordering?'
    if 'overfit' in title_l:
        return 'What is overfitting and how does regularization help reduce it?'
    if 'cross' in title_l or 'validation' in title_l:
        return 'Why is cross-validation useful for estimating generalization performance?'
    if 'random forest' in title_l:
        return 'How does a random forest reduce variance compared with a single decision tree?'
    if 'gradient' in title_l:
        return 'What happens when the learning rate is too large in gradient descent?'
    if 'precision' in title_l or 'recall' in title_l:
        return 'How do precision and recall differ in evaluation?'
    if 'mutable' in title_l or 'immutable' in title_l:
        return 'What is the difference between mutable and immutable objects in Python?'
    if 'copy' in title_l:
        return 'What is the difference between shallow copy and deep copy in Python?'
    if 'generator' in title_l:
        return 'Why do generators save memory compared with building a full list?'
    if 'exception' in title_l or 'finally' in title_l or 'else' in title_l:
        return 'When do else and finally execute in exception handling?'
    if 'class' in title_l or 'method' in title_l:
        return 'How do class methods, static methods, and instance methods differ?'
    return f'What are the main ideas explained in this section about {title}?'


def _extract_span(document, section) -> tuple[int, int]:
    content = document.content
    start_char = section.start_char
    end_char = section.end_char
    available = end_char - start_char
    if available < 220:
        raise ValueError('section too short')
    start = start_char + min(60, available // 8)
    end = min(end_char, start + min(500, max(220, available // 2)))
    while start < end and content[start:end].strip() == '':
        start += 1
    while start < end and content[start:end].strip() == '':
        end -= 1
    if end - start < 180:
        raise ValueError('span too short')
    return start, end


def build_dataset() -> None:
    _ensure_source_group_aliases()
    source_groups = load_source_group_manifest(ROOT / 'config/rag/source_groups.json')
    draft_path = ROOT / 'data/evaluation/gold_dataset_draft_v2.json'
    draft_payload = json.loads(draft_path.read_text(encoding='utf-8'))

    subject_plan = {
        'math': [
            ('math/openstax_calculus_volume_1.pdf', ['limit', 'continuity', 'derivative', 'integral', 'mean value', 'chain']),
            ('math/高等数学（上册） (同济大学数学系) (z-library.sk, 1lib.sk, z-lib.sk).pdf', ['导数', '积分', '极限', '连续']),
            ('math/clp1/clp_1_dc_text.pdf', ['limit', 'derivative', 'integral']),
        ],
        'computer': [
            ('computer/open_data_structures.pdf', ['stack', 'hash', 'tree', 'bfs', 'queue']),
            ('computer/ostep/intro.pdf', ['thread', 'process', 'cpu', 'concurrency']),
            ('computer/计算机学科基础知识 (计算机学科基础知识) (z-library.sk, 1lib.sk, z-lib.sk).pdf', ['算法', '堆栈', '树', '并发']),
        ],
        'big_data': [
            ('big_data/大数据导论 (林子雨) (z-library.sk, 1lib.sk, z-lib.sk).pdf', ['MapReduce', 'hadoop', 'spark', '分布式']),
            ('big_data/大数据开发工程师系列：Hadoop  Spark大数据开发实战 (大数据开发工程师系列：Hadoop  Spark大数据开发实战) (z-library.sk, 1lib.sk, z-lib.sk).pdf', ['spark', 'hadoop', 'mapreduce']),
            ('big_data/data_intensive_text_processing_mapreduce.pdf', ['mapreduce', 'combiner', 'warehouse']),
        ],
        'machine_learning': [
            ('machine_learning/机器学习 Machine Learning (Chinese Edition) (Zhou Zhihua 周志华) (z-library.sk, 1lib.sk, z-lib.sk).pdf', ['overfitting', 'cross', 'forest', 'gradient', 'precision', 'recall']),
            ('machine_learning/mathematics_for_machine_learning.pdf', ['gradient', 'probability', 'matrix', 'optimization']),
            ('machine_learning/dive_into_deep_learning_zh.pdf', ['gradient', 'overfitting', 'regularization']),
        ],
        'python': [
            ('python/python_for_everybody.pdf', ['variable', 'list', 'loop', 'exception', 'class']),
            ('python/think_python_2e.pdf', ['copy', 'generator', 'class', 'exception']),
            ('python/Python Basics A Practical Introduction to Python 3 (Real Python) (z-library.sk, 1lib.sk, z-lib.sk).pdf', ['mutable', 'immutable', 'class', 'exception']),
        ],
    }

    queries = []
    counter = 1
    for subject in ['math', 'computer', 'big_data', 'machine_learning', 'python']:
        for source_relpath, keywords in subject_plan[subject]:
            document, _ = _load_document(source_relpath)
            sections = tuple(section for section in detect_document_sections(document.content) if section.section_path)
            if not sections:
                continue
            used = 0
            for section in sections:
                if used >= 8:
                    break
                title = ' > '.join(section.section_path)
                title_l = title.lower()
                content_l = document.content[section.start_char:section.end_char].lower()
                if not any(keyword.lower() in title_l or keyword.lower() in content_l for keyword in keywords):
                    continue
                try:
                    start_char, end_char = _extract_span(document, section)
                except ValueError:
                    continue
                span_text = document.content[start_char:end_char]
                if not span_text.strip():
                    continue
                page_start, page_end = page_range_for_span(document, start_char=start_char, end_char=end_char)
                source_group_id = source_groups.source_groups.get(source_relpath)
                if source_group_id is None:
                    source_group_id = source_groups.source_groups.get('data/' + source_relpath)
                if source_group_id is None:
                    continue
                query_id = f'{subject}-q{counter:03d}'
                gold_span_id = f'gold_{subject}_{counter:03d}'
                queries.append({
                    'schema_version': 'gold_query_v1',
                    'query_id': query_id,
                    'subject': subject,
                    'query': _question_for(' '.join(keywords), title),
                    'dataset_kind': 'human_gold',
                    'eligible_for_rollout': True,
                    'gold_spans': [{
                        'schema_version': 'gold_evidence_span_v1',
                        'gold_span_id': gold_span_id,
                        'source_group_id': source_group_id,
                        'source_relpath': source_relpath,
                        'doc_id': document.doc_id,
                        'pagination_kind': document.pagination_kind,
                        'page_start': page_start,
                        'page_end': page_end,
                        'start_char': start_char,
                        'end_char': end_char,
                        'section_path': section.section_path,
                        'relevance_grade': 3,
                    }],
                })
                counter += 1
                used += 1
                if counter > 100:
                    break
            if counter > 100:
                break

    if len(queries) < 100:
        raise RuntimeError(f'expected at least 100 queries, got {len(queries)}')

    draft_payload['queries'] = queries[:100]
    draft_path.write_text(json.dumps(draft_payload, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    print(f'wrote {len(queries[:100])} queries to {draft_path}')


if __name__ == '__main__':
    build_dataset()
