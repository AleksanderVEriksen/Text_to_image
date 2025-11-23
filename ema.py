import torch

class ExponentialMovingAverage:
    """Simple EMA for model parameters."""
    def __init__(self, model, decay=0.9999):
        self.decay = decay
        self.model = model
        # Initialize shadow (EMA) weights
        self.shadow_params = [p.detach().clone() for p in model.parameters() if p.requires_grad]

    @torch.no_grad()
    def update(self):
        """Update shadow params after each optimizer step."""
        for s, p in zip(self.shadow_params, self.model.parameters()):
            if not p.requires_grad:
                continue
            s.mul_(self.decay).add_(p.detach(), alpha=1.0 - self.decay)

    @torch.no_grad()
    def apply_shadow(self, model=None):
        """Copy EMA weights into the model (for evaluation/saving)."""
        ref_model = model or self.model
        for s, p in zip(self.shadow_params, ref_model.parameters()):
            if p.requires_grad:
                p.copy_(s)

    @torch.no_grad()
    def restore(self, model=None):
        """(No-op unless you backed up original weights before apply_shadow)."""
        # Implement backup/restore if needed; currently apply_shadow overwrites directly.

    def state_dict(self):
        """Serialize EMA state."""
        return {
            "decay": self.decay,
            "shadow_params": [p.clone() for p in self.shadow_params]
        }

    def load_state_dict(self, state):
        """Load EMA state."""
        self.decay = state.get("decay", self.decay)
        sp = state["shadow_params"]
        assert len(sp) == len(self.shadow_params)
        for dst, src in zip(self.shadow_params, sp):
            dst.copy_(src)