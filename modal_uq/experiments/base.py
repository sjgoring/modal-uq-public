
class ExperimentBase:
    def __init__(self, ds, pgt, model, metrics, cfg, n_jobs=None):
        self.ds, self.pgt, self.model, self.metrics, self.cfg = ds, pgt, model, metrics, cfg
        # Number of parallel jobs to use for compute-heavy operations. None means library defaults.
        self.n_jobs = n_jobs
    def run(self):
        raise NotImplementedError
    def report(self):
        pass
