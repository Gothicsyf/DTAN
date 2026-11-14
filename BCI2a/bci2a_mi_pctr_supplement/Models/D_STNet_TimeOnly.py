"""R026 model: temporal branch only; fixed-TAT training remains unchanged."""
from .D_STNet import Net as FullNet


class Net(FullNet):
    def __init__(self, *args, **kwargs):
        kwargs["use_gasl"] = False
        super().__init__(*args, **kwargs)
