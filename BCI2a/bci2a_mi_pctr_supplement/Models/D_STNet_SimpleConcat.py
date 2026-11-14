"""R028 model: both branches with naive concatenation instead of DBPF."""
from .D_STNet import Net as FullNet


class Net(FullNet):
    def __init__(self, *args, **kwargs):
        kwargs["use_dbp_fusion"] = False
        super().__init__(*args, **kwargs)
