# risk.py

class RiskModel:
    def __init__(self, config):
        self.max_gross_exposure = config.get("max_gross_exposure", 20)

    def apply_constraints(self, signals, portfolio):
        return signals
