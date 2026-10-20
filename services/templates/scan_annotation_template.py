import torch
import torch.nn as nn
import numpy as np


class RobustScale1d(nn.Module):
    """Odpowiednik robust_scale_per_sample napisany w PyTorchu."""
    def __init__(self, eps: float = 1e-8):
        super().__init__()
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape: (B, C, L) -> statystyki liczymy wzdłuż wymiaru L (dim=-1)
        median = x.median(dim=-1, keepdim=True).values
        q75 = torch.quantile(x, 0.75, dim=-1, keepdim=True)
        q25 = torch.quantile(x, 0.25, dim=-1, keepdim=True)
        
        iqr = q75 - q25
        iqr = torch.where(iqr == 0, torch.tensor(1.0, device=x.device, dtype=x.dtype), iqr)
        
        return (x - median) / iqr


class DoubleConv1d(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, dilation: int = 1):
        super().__init__()
        padding = 3 * dilation
        self.net = nn.Sequential(
            nn.Conv1d(
                in_channels,
                out_channels,
                kernel_size=7,
                padding=padding,
                dilation=dilation,
                bias=False,
            ),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv1d(
                out_channels,
                out_channels,
                kernel_size=3,
                padding=1,
                bias=False,
            ),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class DownBlock1d(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.double_conv = DoubleConv1d(in_channels, out_channels)
        self.down_conv = nn.Conv1d(
            out_channels,
            out_channels,
            kernel_size=3,
            stride=2,
            padding=1,
            bias=False,
        )

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        skip = self.double_conv(x)
        down = self.down_conv(skip)
        return skip, down


class Bottleneck1d(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.conv1 = DoubleConv1d(in_channels, out_channels, dilation=1)
        self.conv2 = DoubleConv1d(out_channels, out_channels, dilation=2)
        self.conv3 = DoubleConv1d(out_channels, out_channels, dilation=4)
        self.conv4 = DoubleConv1d(out_channels, out_channels, dilation=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        return x


class UNet1D(nn.Module):
    def __init__(
        self,
        in_channels: int = 1,
        num_classes: int = 3,
        features: list[int] | None = None,
        use_robust_scaling: bool = True,
    ):
        super().__init__()
        if features is None:
            features = [32, 64, 128, 256]
        else:
            features = list(features)

        self.in_channels = in_channels
        self.num_classes = num_classes
        self.features = features
        self.use_robust_scaling = use_robust_scaling

        if self.use_robust_scaling:
            self.scaler = RobustScale1d()

        self.downs = nn.ModuleList()
        self.ups = nn.ModuleList()

        # Encoder
        in_c = in_channels
        for feature in features:
            self.downs.append(DownBlock1d(in_c, feature))
            in_c = feature

        # Bottleneck
        self.bottleneck = Bottleneck1d(features[-1], features[-1] * 2)

        # Decoder
        for feature in reversed(features):
            self.ups.append(
                nn.ConvTranspose1d(
                    feature * 2, feature, kernel_size=2, stride=2
                )
            )
            self.ups.append(DoubleConv1d(feature * 2, feature))

        # Head
        self.final_conv = nn.Conv1d(features[0], num_classes, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.use_robust_scaling:
            x = self.scaler(x)

        skip_connections = []

        for down in self.downs:
            skip, x = down(x)
            skip_connections.append(skip)

        x = self.bottleneck(x)
        skip_connections = skip_connections[::-1]

        for idx in range(0, len(self.ups), 2):
            x = self.ups[idx](x)
            skip = skip_connections[idx // 2]

            if x.shape[2:] != skip.shape[2:]:
                x = torch.nn.functional.interpolate(
                    x, size=skip.shape[2:], mode="nearest"
                )

            x = torch.cat((skip, x), dim=1)
            x = self.ups[idx + 1](x)

        return self.final_conv(x)

    def predict(
        self,
        X: np.ndarray,
        batch_size: int = 32,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
    ) -> np.ndarray:
        """Inferencja bezpośrednio z surowych danych (skalowanie odbywa się wewnątrz modelu)."""
        self.eval()
        self.to(device)

        if X.ndim == 2:
            X = X[:, np.newaxis, :]

        num_samples = len(X)
        probs_list = []

        with torch.no_grad():
            for i in range(0, num_samples, batch_size):
                batch = torch.from_numpy(X[i : i + batch_size]).float().to(device)
                logits = self(batch)
                probs = torch.softmax(logits, dim=1)
                probs = probs.permute(0, 2, 1)
                probs_list.append(probs.cpu().numpy())

        return np.concatenate(probs_list, axis=0)

    def save_to_file(self, filepath: str) -> None:
        checkpoint = {
            "state_dict": self.state_dict(),
            "init_kwargs": {
                "in_channels": self.in_channels,
                "num_classes": self.num_classes,
                "features": self.features,
                "use_robust_scaling": self.use_robust_scaling,
            },
        }
        torch.save(checkpoint, filepath)

    @classmethod
    def from_file(cls, filepath: str, device: str = "cuda") -> "UNet1D":
        checkpoint = torch.load(filepath, map_location=device, weights_only=True)
        model = cls(**checkpoint["init_kwargs"])
        model.load_state_dict(checkpoint["state_dict"])
        model.to(device)
        return model


    def robust_scale_per_sample(self, X: np.ndarray) -> np.ndarray:
        median = np.median(X, axis=1, keepdims=True)
        p75 = np.percentile(X, 75, axis=1, keepdims=True)
        p25 = np.percentile(X, 25, axis=1, keepdims=True)
        iqr = p75 - p25
        iqr = np.where(iqr == 0, 1.0, iqr) # Zabezpieczenie przed zerowym IQR
        return (X - median) / iqr
