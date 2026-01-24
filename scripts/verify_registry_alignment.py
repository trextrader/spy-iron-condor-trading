import re
from pathlib import Path
import yaml

ROOT = Path('c:/SPYOptionTrader_test')
DSL_PATH = ROOT / 'docs' / 'Complete_Ruleset_DSL.yaml'
EXEC_PATH = ROOT / 'intelligence' / 'rule_engine' / 'executor.py'


def load_dsl_ids():
    data = yaml.safe_load(DSL_PATH.read_text(encoding='utf-8'))
    ids = set()
    for key, rule_def in data.items():
        if not isinstance(rule_def, dict):
            continue
        if not key.startswith('RULE_'):
            continue
        for p in rule_def.get('primitives', []):
            ids.add(p.get('id'))
        for g in rule_def.get('gates', []):
            ids.add(g.get('id'))
    return {i for i in ids if i}


def load_registry_ids():
    text = EXEC_PATH.read_text(encoding='utf-8')
    keys = re.findall(r"[\"']([A-Z]\d{3})[\"']\s*:", text)
    return keys


def find_duplicate_keys(keys):
    seen = set()
    dupes = set()
    for k in keys:
        if k in seen:
            dupes.add(k)
        seen.add(k)
    return sorted(dupes)


def main():
    dsl_ids = load_dsl_ids()
    reg_keys = load_registry_ids()
    reg_set = set(reg_keys)

    missing = sorted([i for i in dsl_ids if i not in reg_set])
    extra = sorted([i for i in reg_set if i not in dsl_ids])
    dupes = find_duplicate_keys(reg_keys)

    print('DSL IDs:', len(dsl_ids))
    print('Registry IDs:', len(reg_set))
    print('Missing in registry:', missing)
    print('Extra in registry:', extra)
    print('Duplicate keys in registry:', dupes)

    if missing or dupes:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
