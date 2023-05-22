import torch
from torch import nn

from super_segmenter.utils.helpers import padding_same
from super_segmenter.params import UnetModelParams


class UNetConvblock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self._block = nn.Sequential(
            nn.Conv2d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=3,
                stride=1,
                padding=padding_same(kernel_size=3),
            ),
            nn.ReLU(),
            nn.BatchNorm2d(num_features=out_channels),
            nn.Conv2d(
                in_channels=out_channels,
                out_channels=out_channels,
                kernel_size=3,
                stride=1,
                padding=padding_same(kernel_size=3),
            ),
            nn.ReLU(),
            nn.BatchNorm2d(num_features=out_channels),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self._block(x)


class UnetEncoder(nn.Module):
    def __init__(self, in_channels: int, channels: list) -> None:
        super().__init__()
        self._layers = nn.ModuleList()
        self._max_pool = nn.MaxPool2d(kernel_size=2, stride=2)
        for out_ch in channels:
            self._layers.append(
                UNetConvblock(in_channels=in_channels, out_channels=out_ch)
            )
            in_channels = out_ch

    def forward(
        self, x: torch.Tensor
    ) -> tuple[torch.Tensor, list[torch.Tensor]]:
        outs = []
        for layer in self._layers:
            x = layer(x)
            outs.append(x)
            x = self._max_pool(x)
        return x, outs


class UnetDecoder(nn.Module):
    def __init__(self, in_channels: int, channels: list) -> None:
        super().__init__()
        self._strided_conv_layers = nn.ModuleList()
        self._layers = nn.ModuleList()
        in_ch = in_channels
        for out_ch in channels:
            self._strided_conv_layers.append(
                nn.ConvTranspose2d(
                    in_channels=in_ch,
                    out_channels=out_ch,
                    kernel_size=3,
                    stride=2,
                    padding=1,
                    output_padding=1,
                ),
            )
            in_ch = out_ch

        in_ch = in_channels
        for out_ch in channels:
            self._layers.append(
                UNetConvblock(in_channels=in_channels, out_channels=out_ch),
            )
            in_channels = out_ch

    def forward(
        self, x: torch.Tensor, encoder_outs: list[torch.Tensor]
    ) -> torch.Tensor:
        cur_encoder_idx = -1
        # x: [b_s, middle_channels, H, W]
        for strided_conv, conv in zip(self._strided_conv_layers, self._layers):
            x = strided_conv(x)
            x = torch.cat([x, encoder_outs[cur_encoder_idx]], dim=1)
            x = conv(x)
            cur_encoder_idx -= 1
        # x: [b_s, encoder_ch_0, H, W]
        return x


class UNet(nn.Module):
    def __init__(self, params: UnetModelParams) -> None:
        super(UNet, self).__init__()
        num_classes: int = params.num_classes
        assert len(params.encoder_channels) == len(params.decoder_channels)
        # Contraction
        self._encoder = UnetEncoder(
            in_channels=params.img_channels, channels=params.encoder_channels
        )
        # Middle
        self._middle = UNetConvblock(
            in_channels=params.encoder_channels[-1],
            out_channels=params.middle_channels,
        )
        # Decoder
        self._decoder = UnetDecoder(
            in_channels=params.middle_channels,
            channels=params.decoder_channels
        )
        # Output
        self.output = nn.Conv2d(
            in_channels=params.decoder_channels[-1],
            out_channels=num_classes,
            kernel_size=3,
            stride=1,
            padding=1,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # [-1, #in_channels, 256, 256]
        enc_out, encoder_outs = self._encoder(x)
        # [-1, 512, 16, 16]
        middle_out = self._middle(enc_out)
        # [-1, 1024, 16, 16]
        decoder_out = self._decoder(x=middle_out, encoder_outs=encoder_outs)
        output_out = self.output(decoder_out)
        # [-1, num_classes, 256, 256]
        return output_out
