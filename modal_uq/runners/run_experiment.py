
from ..utils.io import read_json
from ..registry import build, register
from ..utils.seed import set_seed

from ..experiments.selective import SelectivePrediction
from ..experiments.ood import OODExperiment
from ..experiments.active_learning import ActiveLearning

register('experiment','selective')(SelectivePrediction)
register('experiment','ood')(OODExperiment)
register('experiment','active_learning')(ActiveLearning)

def run_from_config(config_path: str):
    cfg = read_json(config_path)
    set_seed(cfg.get('experiment',{}).get('seed', 42))

    ds_cfg = cfg['dataset']
    ds = build('dataset', ds_cfg['name'], **ds_cfg)

    pgt_cfg = cfg['pseudo_ground_truth']
    pgt = build('pgt', pgt_cfg['name'], **pgt_cfg.get('params', {}))

    model_cfg = cfg['model']
    model = build('model', model_cfg['name'], **model_cfg.get('params', {}))

    exp = build('experiment', cfg['experiment']['type'], ds=ds, pgt=pgt, model=model, metrics=cfg.get('metrics',{}), cfg=cfg)
    exp.run(); exp.report()
