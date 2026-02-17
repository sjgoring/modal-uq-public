
from typing import Callable, Dict

_REGISTRIES: Dict[str, Dict[str, Callable]] = {
    'dataset': {}, 'pgt': {}, 'model': {}, 'acquisition': {}, 'experiment': {}, 'uncertainty': {}
}

def register(kind: str, name: str):
    def deco(cls_or_fn: Callable):
        if kind not in _REGISTRIES:
            raise ValueError(f'Unknown registry kind: {kind}')
        _REGISTRIES[kind][name] = cls_or_fn
        return cls_or_fn
    return deco

def build(kind: str, name: str, **kwargs):
    try:
        ctor = _REGISTRIES[kind][name]
    except KeyError as e:
        have = list(_REGISTRIES.get(kind, {}).keys())
        raise ValueError(f"Unknown {kind} '{name}'. Available: {have}") from e
    return ctor(**kwargs)
