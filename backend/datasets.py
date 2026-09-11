import hashlib
import json
import random
import re
from collections import Counter

from .storage import ROOT, save

CATEGORIES = {'general', 'accuracy', 'hallucinations', 'sycophancy', 'abstention'}


def normalize(text):
    return ' '.join(re.findall(r'\w+', text.casefold()))


def parse_dataset(content, seed=42):
    if len(content) > 20 * 1024 * 1024:
        raise ValueError('Dataset must be at most 20 MB.')
    try:
        text = content.decode('utf-8-sig')
    except UnicodeError as error:
        raise ValueError('Dataset must be UTF-8 JSONL.') from error
    records, seen, duplicates = [], {}, 0
    for line_number, line in enumerate(text.splitlines(), 1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
            if not isinstance(row, dict):
                raise ValueError('Each line must be a JSON object.')  # noqa: TRY004
            if 'messages' in row:
                messages = row['messages']
                if not isinstance(messages, list) or len(messages) < 2:
                    raise ValueError('Messages must include a user and final assistant answer.')
                expected = 'user'
                for index, message in enumerate(messages):
                    if not isinstance(message, dict) or not isinstance(message.get('content'), str) or not message['content'].strip():
                        raise ValueError('Every message requires nonempty text content.')
                    role = message.get('role')
                    if index == 0 and role == 'system':
                        continue
                    if role != expected:
                        raise ValueError('Messages must alternate user and assistant after an optional system message.')
                    expected = 'assistant' if role == 'user' else 'user'
                if messages[-1]['role'] != 'assistant' or not any(m['role'] == 'user' for m in messages):
                    raise ValueError('The last message must be the assistant reference answer.')
                prompt_messages = [{'role': m['role'], 'content': m['content'].strip()} for m in messages[:-1]]
                response = messages[-1]['content'].strip()
            else:
                prompt, response = row.get('prompt'), row.get('response')
                if not isinstance(prompt, str) or not prompt.strip() or not isinstance(response, str) or not response.strip():
                    raise ValueError('Provide prompt and response strings, or chat messages.')
                prompt_messages = [{'role': 'user', 'content': prompt.strip()}]
                response = response.strip()
            if sum(len(m['content']) for m in prompt_messages) + len(response) > 16000:
                raise ValueError('Example exceeds 16,000 characters.')
            category = row.get('category', 'general')
            if category not in CATEGORIES:
                raise ValueError('Unknown evaluation category.')
            answers = row.get('answers', [response])
            if not isinstance(answers, list) or not answers or any(not isinstance(a, str) or not a.strip() for a in answers):
                raise ValueError('Answers must be a nonempty list of reference strings.')
            # Group identical normalized prompts to prevent exact prompt leakage.
            key = json.dumps([(m['role'], normalize(m['content'])) for m in prompt_messages])
            if key in seen:
                if seen[key] != response:
                    raise ValueError('Conflicting responses for the same prompt.')
                duplicates += 1
                continue
            seen[key] = response
            records.append({'messages': prompt_messages + [{'role': 'assistant', 'content': response}], 'answers': answers, 'category': category})
            if len(records) > 100000:
                raise ValueError('Dataset exceeds 100,000 examples.')
        except (ValueError, KeyError, TypeError) as error:
            raise ValueError(f'Line {line_number}: {error}') from error
    if len(records) < 5:
        raise ValueError('Provide at least five unique examples for train, validation and test splits.')
    random.Random(seed).shuffle(records)
    test_count = max(1, int(len(records) * .1))
    validation_count = max(1, int(len(records) * .1))
    splits = {'test': records[:test_count], 'validation': records[test_count:test_count + validation_count], 'train': records[test_count + validation_count:]}
    return splits, duplicates


def import_dataset(content, name, seed=42):
    splits, duplicates = parse_dataset(content, seed)
    fingerprint = hashlib.sha256(content + str(seed).encode()).hexdigest()[:24]
    directory = ROOT / 'datasets' / fingerprint
    directory.mkdir(parents=True, exist_ok=True)
    for split, records in splits.items():
        filename = 'valid.jsonl' if split == 'validation' else f'{split}.jsonl'
        training_records = []
        for record in records:
            text = '\n'.join(f"{message['role']}: {message['content']}" for message in record['messages'])
            training_records.append({'text': text})
        (directory / filename).write_text(''.join(json.dumps(r, ensure_ascii=False) + '\n' for r in training_records))
        if split == 'test':
            (directory / 'evaluation.jsonl').write_text(''.join(json.dumps(r, ensure_ascii=False) + '\n' for r in records))
    return save('dataset', {'name': name, 'path': str(directory), 'seed': seed, 'duplicates_removed': duplicates, 'counts': {k: len(v) for k, v in splits.items()}, 'sha256': hashlib.sha256(content).hexdigest()}, fingerprint)


def score(prediction, references):
    predicted = normalize(prediction)
    exact = float(any(predicted == normalize(answer) for answer in references))
    def f1(answer):
        tokens = normalize(answer).split()
        output = predicted.split()
        common = sum((Counter(tokens) & Counter(output)).values())
        return 2 * common / (len(tokens) + len(output)) if tokens or output else 1.0
    return {'exact_match': exact, 'token_f1': max(f1(a) for a in references)}
