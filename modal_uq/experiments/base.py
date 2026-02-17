
class ExperimentBase:
    def __init__(self, ds, pgt, model, metrics, cfg):
        self.ds, self.pgt, self.model, self.metrics, self.cfg = ds, pgt, model, metrics, cfg
    def run(self):
        raise NotImplementedError
    def report(self):
        pass
