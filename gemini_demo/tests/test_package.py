from pathlib import Path
import importlib.util
import yaml
ROOT=Path(__file__).resolve().parents[1]

def test_demo_data():
    spec=importlib.util.spec_from_file_location('preflight',ROOT/'scripts/preflight.py')
    mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod)
    mod.check((ROOT/'data').resolve())

def test_configs():
    for name in ['smoke','one_epoch']:
        cfg=yaml.safe_load((ROOT/'configs'/f'{name}.yaml').read_text())
        assert cfg['finetuning_type']=='lora'
        assert cfg['template']=='qwen2_vl'
        assert cfg['video_maxlen']<=8
        assert cfg['dataset']=='demo_train'
        assert cfg['bf16'] is True
