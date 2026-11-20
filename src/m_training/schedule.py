# SPDX-License-Identifier: GPL-3.0-or-later
"""Learning-rate scheduling driven by complete held-out validation loss."""
import math

import torch


V3_LEARNING_RATES = (1e-4, 7e-5, 4e-5, 1e-5, 7e-6, 4e-6, 1e-6, 7e-7, 4e-7, 1e-7)


class V3ValidationStaircase:
    """Lower one fixed rung after exactly twenty non-improving full validations."""
    def __init__(self, optimizer):
        self.optimizer = optimizer
        self.stage, self.last_epoch, self.num_bad_epochs = 0, 0, 0
        self.best = math.inf
        self._last_lr = [float(g["lr"]) for g in optimizer.param_groups]
        if any(not math.isclose(lr, V3_LEARNING_RATES[0], rel_tol=1e-12) for lr in self._last_lr):
            raise ValueError("A new V3 schedule must start at 1e-4")

    def step(self, metrics, *, validation_complete=True):
        if not validation_complete:
            raise ValueError("V3 learning rate requires complete validation")
        value = float(metrics)
        if not math.isfinite(value):
            raise ValueError("Nonfinite validation loss cannot advance the V3 schedule")
        self.last_epoch += 1
        if value < self.best:
            self.best, self.num_bad_epochs = value, 0
        else:
            self.num_bad_epochs += 1
            if self.num_bad_epochs >= 20 and self.stage < len(V3_LEARNING_RATES)-1:
                self.stage += 1
                for group in self.optimizer.param_groups:
                    group["lr"] = V3_LEARNING_RATES[self.stage]
                self.num_bad_epochs = 0
        self._last_lr = [float(g["lr"]) for g in self.optimizer.param_groups]

    def get_last_lr(self):
        return list(self._last_lr)

    def state_dict(self):
        return {"type":"v3_validation_staircase", "rates":list(V3_LEARNING_RATES), "patience":20,
                "stage":self.stage,"last_epoch":self.last_epoch,"num_bad_epochs":self.num_bad_epochs,
                "best":self.best,"_last_lr":self.get_last_lr()}

    def load_state_dict(self, state):
        if state["type"] != "v3_validation_staircase" or tuple(state["rates"]) != V3_LEARNING_RATES or state["patience"] != 20:
            raise ValueError("Checkpoint V3 schedule definition differs")
        if (type(state["stage"]) is not int or not 0 <= state["stage"] < len(V3_LEARNING_RATES)
                or type(state["last_epoch"]) is not int or state["last_epoch"] < 0
                or type(state["num_bad_epochs"]) is not int or state["num_bad_epochs"] < 0
                or math.isnan(state["best"])):
            raise ValueError("Invalid V3 scheduler checkpoint")
        rate = V3_LEARNING_RATES[state["stage"]]
        if (len(state["_last_lr"]) != len(self.optimizer.param_groups)
                or any(not math.isclose(lr,rate,rel_tol=1e-12) for lr in state["_last_lr"])
                or any(not math.isclose(g["lr"],rate,rel_tol=1e-12) for g in self.optimizer.param_groups)):
            raise ValueError("Optimizer and V3 scheduler checkpoint rates differ")
        for name in ("stage","last_epoch","num_bad_epochs","best"):
            setattr(self,name,state[name])
        self._last_lr = list(state["_last_lr"])

