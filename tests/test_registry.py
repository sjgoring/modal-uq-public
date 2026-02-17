
from modal_uq.registry import register, build

def test_registry_roundtrip():
    @register('dataset','_dummy')
    class D: 
        def __init__(self, **kwargs): pass
    obj = build('dataset','_dummy')
    assert obj is not None
