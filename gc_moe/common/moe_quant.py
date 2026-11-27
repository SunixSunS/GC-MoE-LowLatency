"""Quantization and SF-layout helpers shared with the Mega MoE baselines.

These mirror the pure-Python paths used by DeepGEMM's reference utilities so
that the GC path does not require the DeepGEMM extension at import time.
"""

import torch


def ceil_div(x: int, y: int) -> int:
    return (x + y - 1) // y


def align(x: int, y: int) -> int:
    return ceil_div(x, y) * y


def ceil_to_ue8m0(x: torch.Tensor):
    bits = x.abs().float().view(torch.int)
    exp = ((bits >> 23) & 0xFF) + (bits & 0x7FFFFF).bool().int()
    return (exp.clamp(1, 254) << 23).view(torch.float)


def pack_ue8m0_to_int(x: torch.Tensor):
    assert x.dtype == torch.float and x.size(-1) % 4 == 0
    bits = x.view(torch.int32)
    assert (bits >= 0).all() and (bits & 0x7fffff == 0).all()
    return (bits >> 23).to(torch.uint8).view(torch.int32)


def per_token_cast_to_fp8(x: torch.Tensor, use_ue8m0: bool, gran_k: int = 128,
                          use_packed_ue8m0: bool = False):
    assert x.dim() == 2
    m, n = x.shape
    padded_n = align(n, gran_k)
    x_padded = torch.empty((m, padded_n), dtype=x.dtype, device=x.device).fill_(0)
    x_padded[:, :n] = x
    x_view = x_padded.view(m, padded_n // gran_k, gran_k)
    x_amax = x_view.abs().float().amax(dim=2).view(m, padded_n // gran_k).clamp(1e-4)
    sf = x_amax / 448.0
    sf = ceil_to_ue8m0(sf) if use_ue8m0 else sf
    x_fp8 = (x_view * (1.0 / sf.unsqueeze(2))).to(torch.float8_e4m3fn).view(
        m, padded_n)[:, :n].contiguous()
    if use_packed_ue8m0:
        num_sf = sf.size(-1)
        if num_sf % 4:
            sf = torch.nn.functional.pad(sf, (0, align(num_sf, 4) - num_sf), value=1.0)
        return x_fp8, pack_ue8m0_to_int(sf)
    return x_fp8, sf


def per_block_cast_to_fp8(x: torch.Tensor, use_ue8m0: bool, gran_k: int = 128):
    assert x.dim() == 2
    m, n = x.shape
    x_padded = torch.zeros((align(m, gran_k), align(n, gran_k)),
                           dtype=x.dtype, device=x.device)
    x_padded[:m, :n] = x
    x_view = x_padded.view(-1, gran_k, x_padded.size(1) // gran_k, gran_k)
    x_amax = x_view.abs().float().amax(dim=(1, 3), keepdim=True).clamp(1e-4)
    sf = x_amax / 448.0
    sf = ceil_to_ue8m0(sf) if use_ue8m0 else sf
    x_scaled = (x_view * (1.0 / sf)).to(torch.float8_e4m3fn)
    return x_scaled.view_as(x_padded)[:m, :n].contiguous(), sf.view(
        x_view.size(0), x_view.size(2))


def _quantize_to_fp4_e2m1(x: torch.Tensor) -> torch.Tensor:
    ax = x.abs()
    code = torch.zeros_like(x, dtype=torch.uint8)
    for boundary, round_up in zip(
            (0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5.0),
            (False, True, False, True, False, True, False)):
        code += (ax >= boundary if round_up else ax > boundary).to(torch.uint8)
    sign = (x < 0) & (code != 0)
    code = code | (sign.to(torch.uint8) << 3)
    return code.view(torch.int8)


def per_token_cast_to_fp4(x: torch.Tensor, use_ue8m0: bool, gran_k: int = 128):
    m, n = x.shape
    assert n % 2 == 0
    padded_n = align(n, gran_k)
    x_padded = torch.zeros((m, padded_n), dtype=x.dtype, device=x.device)
    x_padded[:, :n] = x
    x_view = x_padded.view(m, -1, gran_k)
    x_amax = x_view.abs().float().amax(dim=2).clamp_min(1e-4)
    sf = x_amax / 6.0
    sf = ceil_to_ue8m0(sf) if use_ue8m0 else sf
    x_scaled = x_view * (1.0 / sf.unsqueeze(2))
    codes = _quantize_to_fp4_e2m1(x_scaled).view(m, padded_n)
    codes2 = codes.view(m, padded_n // 2, 2)
    packed = (codes2[:, :, 0] & 0x0F) | ((codes2[:, :, 1] & 0x0F) << 4)
    return packed[:, :n // 2].contiguous(), sf


def transform_sf_into_required_layout(sf: torch.Tensor, mn: int, k: int,
                                      recipe, num_groups=None):
    """Pack FP32/UE8M0 scales into the SM100 MN-major TMA layout.

    Mirrors DeepGEMM's torch fallback for (FP32, gran_mn, gran_k) inputs with
    gran_k in (32, 128), where each four UE8M0 bytes become one int32 word.
    """
    gran_mn, gran_k = recipe
    if gran_mn != 1:
        sf = sf.index_select(
            -2, torch.arange(mn, device=sf.device).floor_divide_(gran_mn))
    if sf.dtype != torch.float:
        raise ValueError('only FP32 scales are supported here')
    if gran_k not in (32, 128):
        raise ValueError(f'unsupported scale gran_k {gran_k}')
    batched = sf.unsqueeze(0) if sf.dim() == 2 else sf
    ue8m0 = batched.contiguous().view(torch.int32).bitwise_right_shift(23).to(torch.uint8)
    batches, rows, cols = ue8m0.shape
    aligned_rows = align(rows, 4)
    aligned_cols = align(cols, 4)
    padded = torch.zeros((batches, aligned_rows, aligned_cols),
                         dtype=torch.uint8, device=sf.device)
    padded[:, :rows, :cols] = ue8m0
    packed = padded.view(-1).view(torch.int32).view(
        batches, aligned_rows, aligned_cols // 4)
    out = torch.empty_strided(
        (batches, aligned_rows, aligned_cols // 4),
        (aligned_rows * (aligned_cols // 4), 1, aligned_rows),
        dtype=torch.int32, device=sf.device)
    out.copy_(packed)
    out = out[:, :rows]
    return out.squeeze(0) if sf.dim() == 2 else out
