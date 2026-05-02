from ..utils.io import read_json
from ..registry import build, register
from ..utils.seed import set_seed

from ..experiments.selective import SelectivePrediction
from ..experiments.ood import OODExperiment
from ..experiments.active_learning import ActiveLearning

from ..datasets.faithful import Faithful
# from ..datasets.synthetic import SyntheticMultiModalDataset
from ..datasets.synthetic_conditional import SyntheticMultiModalConditionalDataset
from ..datasets.synthetic_constant_var import SyntheticConstantVarDataset

from ..pgt.conditional_kde import ConditionalKDE

# from ..models.mdn import MixtureDensityModel
from ..models.ensemble import Ensemble
from ..models.condGMM import CondGMM
# from ..models.bnn_vi import BayesianNNVI
# from ..models.gp import GaussianProcessModel

from ..uncertainty.variance import PredictiveVariance
from ..uncertainty.differential_entropy import DifferentialEntropy
from ..uncertainty.quest import QUESTUncertainty

register('experiment','selective')(SelectivePrediction)
register('experiment','ood')(OODExperiment)
register('experiment','active_learning')(ActiveLearning)

def _split_name(cfg: dict, key: str = 'name'):
    cfg = dict(cfg)              # shallow copy to avoid mutating original
    name = cfg.pop(key)          # remove 'name' so it won't be duplicated
    return name, cfg

def run_from_config(config_path: str):
    cfg = read_json(config_path)
    set_seed(cfg.get('experiment',{}).get('seed', 42))

    # ---- Dataset ----
    ds_cfg = cfg['dataset']
    ds_name, ds_params = _split_name(ds_cfg, 'name')
    # For use with sythentic conditional data set only. Todo: Implement a more general mechanism for this.
    if ds_cfg.get('parameters', not None):
        ds = build('dataset', ds_name, **ds_cfg.get('params', {}))
    else:
        ds = build('dataset', ds_name, **ds_params)
    
    # ---- Pseudo Ground Truth ----
    pgt_cfg = cfg['pseudo_ground_truth']
    pgt = build('pgt', pgt_cfg['name'], **pgt_cfg.get('params', {}))

    # ---- Model ----
    model_cfg = cfg['model']
    model_params = dict(model_cfg.get('params', {}))
    if 'inferential_choice' in model_cfg:
        model_params['inferential_choice'] = model_cfg['inferential_choice']
    model = build('model', model_cfg['name'], **model_params)

    # ---- Experiment ----
    exp = build(
        'experiment',
        cfg['experiment']['type'],
        ds=ds, pgt=pgt, model=model, metrics=cfg.get('metrics',{}), cfg=cfg
    )
    exp.run(); exp.report()