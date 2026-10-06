"""Byte-level CPU storage semantics, separate from explicit saturation policy."""

import numpy as np
import pytest

from attention import cast, raw_storage_cast

torch = pytest.importorskip("torch")


def _bf16_inputs():
    bits = np.arange(65536, dtype=np.uint16)
    # Independent bit expansion, including every signed NaN payload.
    expanded = (bits.astype(np.uint32) << 16).view(np.float32)
    native = torch.from_numpy(bits.view(np.int16)).view(torch.bfloat16)
    return bits, expanded, native


def test_all_65536_bf16_raw_storage_cast_bytes_match_torch_cpu():
    bits, expanded, native = _bf16_inputs()
    expected = native.to(torch.float8_e4m3fn).view(torch.uint8).numpy()
    actual = raw_storage_cast(expanded, "e4m3").view(np.uint8)
    bad = np.flatnonzero(actual != expected)
    assert not len(bad), [
        (hex(int(bits[i])), hex(int(actual[i])), hex(int(expected[i]))) for i in bad[:16]
    ]


def test_all_65536_bf16_explicit_saturating_cast_bytes_match_torch_cpu():
    _, expanded, native = _bf16_inputs()
    expected = native.float().clamp(-448, 448).to(torch.float8_e4m3fn).view(torch.uint8).numpy()
    stored = raw_storage_cast(np.clip(expanded, -448, 448), "e4m3")
    np.testing.assert_array_equal(stored.view(np.uint8), expected)
    # Expanded project outputs retain signed NaNs and negative zero as well.
    repacked = raw_storage_cast(cast(expanded, "e4m3"), "e4m3")
    np.testing.assert_array_equal(repacked.view(np.uint8), expected)


def test_raw_overflow_infinity_nan_signs_and_negative_zero():
    # Float32 signed signaling/quiet payloads, not just Python's positive nan.
    x = np.array(
        [
            0x00000000,
            0x80000000,
            0x7F800000,
            0xFF800000,
            0x7F800001,
            0xFF800001,
            0x7FC00000,
            0xFFC00000,
        ],
        dtype=np.uint32,
    ).view(np.float32)
    actual = raw_storage_cast(x, "e4m3").view(np.uint8)
    expected = np.array([0x00, 0x80, 0x7F, 0xFF, 0x7F, 0xFF, 0x7F, 0xFF], dtype=np.uint8)
    np.testing.assert_array_equal(actual, expected)
    np.testing.assert_array_equal(
        actual, torch.from_numpy(x).to(torch.float8_e4m3fn).view(torch.uint8).numpy()
    )
    np.testing.assert_array_equal(
        raw_storage_cast([448, 464, 480, 1000, -448, -464, -480, -1000], "e4m3").view(np.uint8),
        [0x7E, 0x7E, 0x7F, 0x7F, 0xFE, 0xFE, 0xFF, 0xFF],
    )


def test_explicit_saturation_is_not_raw_cast():
    x = np.array([1000, -1000, np.inf, -np.inf, 0, -0.0], dtype=np.float32)
    np.testing.assert_array_equal(cast(x, "e4m3"), [448, -448, 448, -448, 0, 0])
    assert np.signbit(cast(x, "e4m3")[-1])
    stored = raw_storage_cast(cast(x, "e4m3"), "e4m3").view(np.uint8)
    np.testing.assert_array_equal(stored, [0x7E, 0xFE, 0x7E, 0xFE, 0x00, 0x80])
    assert np.isnan(raw_storage_cast(x[:4], "e4m3")).all()


def test_subnormal_nearest_even_and_signed_underflow_bytes():
    values = [2**-11, -(2**-11), 2**-10, -(2**-10), 2**-9, -(2**-9), 3 * 2**-10]
    np.testing.assert_array_equal(
        raw_storage_cast(values, "e4m3").view(np.uint8),
        [0x00, 0x80, 0x00, 0x80, 0x01, 0x81, 0x02],
    )
