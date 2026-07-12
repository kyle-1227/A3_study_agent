import json
import pathlib

root = pathlib.Path(r'D:\software competition\A3_study_agent')
path = root / 'data/evaluation/gold_dataset_draft.json'
data = json.loads(path.read_text(encoding='utf-8'))
for query in data['queries']:
    for span in query['gold_spans']:
        rel = span['source_relpath']
        if rel == 'math/openstax_calculus_volume_1.pdf':
            span['source_relpath'] = 'data/math/openstax_calculus_volume_1.pdf'
            span['source_group_id'] = 'calculus_openstax'
        elif rel == 'computer/open_data_structures.pdf':
            span['source_relpath'] = 'data/computer/open_data_structures.pdf'
            span['source_group_id'] = 'computer_open_data_structures'
path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
print('updated', len(data['queries']))
