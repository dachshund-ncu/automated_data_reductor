import torch
import torch.nn as nn
import numpy as np


class DoubleConv1d(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv1d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class UNet1D(nn.Module):
    def __init__(
            self,
            in_channels: int = 1,
            num_classes: int = 3,
            features: list[int] | None = None
        ):
            super().__init__()
            # -- parameters --
            if features is None:
                features = [32, 64, 128, 256]
            else:
                features = list(features)
            self.in_channels = in_channels
            self.num_classes = num_classes
            self.features = features

            # -- model blocks --
            self.downs = nn.ModuleList()
            self.ups = nn.ModuleList()
            self.pool = nn.MaxPool1d(kernel_size=2, stride=2)

            # Encoder
            in_c = in_channels
            for feature in features:
                self.downs.append(DoubleConv1d(in_c, feature))
                in_c = feature

            # Bottleneck
            self.bottleneck = DoubleConv1d(features[-1], features[-1] * 2)

            # Decoder
            for feature in reversed(features):
                self.ups.append(
                    nn.ConvTranspose1d(feature * 2, feature, kernel_size=2, stride=2)
                )
                self.ups.append(DoubleConv1d(feature * 2, feature))

            # Classification head
            self.final_conv = nn.Conv1d(features[0], num_classes, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        skip_connections = []

        for down in self.downs:
            x = down(x)
            skip_connections.append(x)
            x = self.pool(x)

        x = self.bottleneck(x)
        skip_connections = skip_connections[::-1]

        for idx in range(0, len(self.ups), 2):
            x = self.ups[idx](x)
            skip = skip_connections[idx // 2]
            
            if x.shape != skip.shape:
                x = torch.nn.functional.interpolate(x, size=skip.shape[2:], mode="nearest")

            x = torch.cat((skip, x), dim=1)
            x = self.ups[idx + 1](x)

        return self.final_conv(x)

    def summary(self, input_size: tuple[int, int] = (1, 4096), device: str = "cpu") -> None:
        """Basic summary method"""
        self.to(device)
        x = torch.randn(1, *input_size, device=device)
        
        print("=" * 65)
        print(f"{'UNet1D Model Summary':^65}")
        print("=" * 65)
        print(f"{'Block / Layer':<30} | {'Output dimension (B, C, L)':<30}")
        print("-" * 65)
        print(f"{'Input':<30} | {str(list(x.shape)):<30}")

        skips = []
        for i, down in enumerate(self.downs):
            x = down(x)
            skips.append(x)
            print(f"{f'Encoder Block {i+1}':<30} | {str(list(x.shape)):<30}")
            x = self.pool(x)

        x = self.bottleneck(x)
        print(f"{'Bottleneck':<30} | {str(list(x.shape)):<30}")
        skips = skips[::-1]

        for idx in range(0, len(self.ups), 2):
            x = self.ups[idx](x)
            skip = skips[idx // 2]
            if x.shape != skip.shape:
                x = torch.nn.functional.interpolate(x, size=skip.shape[2:], mode="nearest")
            x = torch.cat((skip, x), dim=1)
            x = self.ups[idx + 1](x)
            print(f"{f'Decoder Block {idx//2+1}':<30} | {str(list(x.shape)):<30}")

        out = self.final_conv(x)
        print(f"{'Final Conv1D (Output)':<30} | {str(list(out.shape)):<30}")
        print("=" * 65)
        
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(f"No. of params:    {total:,}")
        print(f"Trainable params: {trainable:,}")
        print("=" * 65)

    def predict(
            self,
            X: np.ndarray,
            batch_size: int = 32,
            device: str = "cuda" if torch.cuda.is_available() else "cpu"
        ) -> np.ndarray:
            self.eval()
            self.to(device)
            
            if X.ndim == 2:
                X = X[:, np.newaxis, :]
                
            num_samples = len(X)
            probs_list = []
            
            with torch.no_grad():
                for i in range(0, num_samples, batch_size):
                    batch = torch.from_numpy(X[i:i + batch_size]).float().to(device)
                    logits = self(batch)
                    
                    # Softmax po wymiarze klas (dim=1)
                    probs = torch.softmax(logits, dim=1)
                    
                    # Zmiana układu z (B, C, L) na (B, L, C)
                    probs = probs.permute(0, 2, 1)
                    
                    probs_list.append(probs.cpu().numpy())
                    
            return np.concatenate(probs_list, axis=0)

    def save_to_file(self, filepath: str) -> None:
        """
        Save model weights and its init params
        Args:
            filepath (str): absolute path to the saved weights filename
        """
        checkpoint = {
            "state_dict": self.state_dict(),
            "init_kwargs": {
                "in_channels": self.in_channels,
                "num_classes": self.num_classes,
                "features": self.features,
            },
        }
        torch.save(checkpoint, filepath)

    @classmethod
    def from_file(cls, filepath: str, device: str = "cuda") -> "UNet1D":
        """
        Simply creates a model class, using saved model
        Args:
            filepath (str): path to a .pt file with classes
            device (str, optional): Device used to infer on a model. Defaults to "cuda"
        Returns:
            UNet1D: A scan segmentation model
        """
        checkpoint = torch.load(filepath, map_location=device, weights_only=True)
        model = cls(**checkpoint["init_kwargs"])
        model.load_state_dict(checkpoint["state_dict"])
        model.to(device)
        return model