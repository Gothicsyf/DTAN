"""R027 model: learnable Gaussian spectral branch only."""
from .D_STNet import Net as FullNet


class Net(FullNet):
    def forward(self, x):
        x = self.dynamic_ea(x) if self.use_dynamic_ea else x
        x_freq = self.torch_psd(x)
        x_freq_out = self.fre_conv(x_freq)
        x_fused = self.fusion.freq_projector(x_freq_out)
        features = self.feature_extractor(x_fused)
        if self.use_feature_align:
            features = self.feature_align(features)
        logits = self.classifier_head(features)
        return logits, features
