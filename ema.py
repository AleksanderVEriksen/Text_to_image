import torch

class ExponentialMovingAverage:
    """Simple EMA for model parameters."""
    def __init__(self, model, decay=0.9999):
        self.decay = decay
        self.shadow = {}
        self.backup = {}
        # register initial shadow weights
        for name, p in model.named_parameters():
            if p.requires_grad:
                self.shadow[name] = p.detach().clone()

    def update(self, model):
        """Call after optimizer.step() to update EMA with current model params."""
        for name, p in model.named_parameters():
            if not p.requires_grad:
                continue
            new_val = p.detach().clone()
            old = self.shadow[name]
            # keep on same device as model param
            self.shadow[name] = old.to(new_val.device) * self.decay + (1.0 - self.decay) * new_val

    def apply_shadow(self, model):
        """Replace model params with EMA (save originals to backup)."""
        self.backup = {}
        for name, p in model.named_parameters():
            if not p.requires_grad:
                continue
            self.backup[name] = p.detach().clone()
            p.data.copy_(self.shadow[name].to(p.device))

    def restore(self, model):
        """Restore original model params from backup."""
        for name, p in model.named_parameters():
            if not p.requires_grad:
                continue
            p.data.copy_(self.backup[name].to(p.device))
        self.backup = {}

    def state_dict(self):
        # move shadow to CPU for checkpointing
        return {k: v.detach().cpu() for k, v in self.shadow.items()}

    def load_state_dict(self, state_dict):
        # load shadow (assumes keys match)
        for k, v in state_dict.items():
            self.shadow[k] = v.clone()