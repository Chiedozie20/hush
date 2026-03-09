import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


class Conv1d(nn.Conv1d):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size,
        stride=1,
        padding=0,
        dilation=1,
        groups: int = 1,
        bias: bool = True,
        padding_mode: str = 'zeros',
        use_hardware: bool = False,
    ):
        super().__init__(
            in_channels,
            out_channels,
            kernel_size,
            stride,
            padding,
            dilation,
            groups,
            bias,
            padding_mode,
        )
        self.use_hardware = use_hardware
        self._hardware_available = self._check_hardware_available()

    def _check_hardware_available(self) -> bool:
        # TODO: implement hw interface
        return False

    def _conv_forward_hardware(self, x: torch.Tensor, weight: torch.Tensor, bias: Optional[torch.Tensor]) -> torch.Tensor:
        raise NotImplementedError("Hardware acceleration not yet implemented")

    def _conv_forward(
        self, x: torch.Tensor, weight: torch.Tensor, bias: Optional[torch.Tensor]
    ) -> torch.Tensor:
        if self.use_hardware and self._hardware_available:
            return self._conv_forward_hardware(x, weight, bias)
        else:
            return super()._conv_forward(
                x, weight.to(x.dtype), None if bias is None else bias.to(x.dtype)
            )

    def extra_repr(self) -> str:
        """String representation with hardware acceleration status."""
        base_repr = super().extra_repr()
        hw_status = "enabled" if self.use_hardware else "disabled"
        hw_available = "available" if self._hardware_available else "unavailable"
        return f"{base_repr}, hardware_accel={hw_status} ({hw_available})"
