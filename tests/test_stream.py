"""Offline real-model parity and optional local pinned-Qwen validation.

Run the small suite with ``pytest -q tests/test_stream.py``. To also compare a
locally downloaded, revision-pinned Qwen2.5-0.5B snapshot, set
``ATTENTION_NUMERICS_QWEN_SNAPSHOT`` to that directory and select
``-k pinned_qwen``. No test downloads weights or sets global thread counts.
"""

import os
import weakref
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")
transformers = pytest.importorskip("transformers")
pytest.importorskip("safetensors")
modeling_llama = pytest.importorskip("transformers.models.llama.modeling_llama")
modeling_olmo2 = pytest.importorskip("transformers.models.olmo2.modeling_olmo2")
modeling_qwen2 = pytest.importorskip("transformers.models.qwen2.modeling_qwen2")
LayerStream = pytest.importorskip("study.stream").LayerStream
SafeTensorWeights = pytest.importorskip("study.weights").SafeTensorWeights
AutoModelForCausalLM = transformers.AutoModelForCausalLM
LlamaConfig = transformers.LlamaConfig
Olmo2Config = transformers.Olmo2Config
Qwen2Config = transformers.Qwen2Config

ARCHITECTURES = {
    "qwen2": (Qwen2Config, modeling_qwen2, modeling_qwen2.Qwen2ForCausalLM),
    "llama": (LlamaConfig, modeling_llama, modeling_llama.LlamaForCausalLM),
    "olmo2": (Olmo2Config, modeling_olmo2, modeling_olmo2.Olmo2ForCausalLM),
}
VARIANTS = ["tile", "rotate", "smooth_k", "rotate_smooth_k", "smooth_kq"]


def save_small(snapshot, architecture, tied=False, sharded=False, sliding=False):
    config_class, _, model_class = ARCHITECTURES[architecture]
    kwargs = {}
    if sliding:
        kwargs = dict(use_sliding_window=True, sliding_window=3, max_window_layers=1)
    config = config_class(
        vocab_size=97,
        hidden_size=32,
        intermediate_size=48,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        max_position_embeddings=64,
        pad_token_id=0,
        bos_token_id=1,
        eos_token_id=2,
        tie_word_embeddings=tied,
        attention_dropout=0.0,
        **kwargs,
    )
    config._attn_implementation = "sdpa"
    torch.manual_seed(901)
    model = model_class(config).eval()
    model.save_pretrained(
        snapshot, safe_serialization=True, max_shard_size="2KB" if sharded else "1GB"
    )
    return model_class


def assert_close(actual, expected, dtype):
    if dtype == torch.float32:
        torch.testing.assert_close(actual.float(), expected.float(), rtol=2e-5, atol=2e-6)
    else:
        torch.testing.assert_close(actual.float(), expected.float(), rtol=2e-2, atol=4e-3)


def compare_native(snapshot, native, ids, dtype, monkeypatch, architecture):
    module = ARCHITECTURES[architecture][1]
    captures = []
    layer_outputs = []
    original = module.ALL_ATTENTION_FUNCTIONS["sdpa"]

    def observed(module, query, key, value, attention_mask, scaling, **kwargs):
        captures.append((query.clone(), key.clone(), value.clone(), scaling))
        return original(module, query, key, value, attention_mask, scaling=scaling, **kwargs)

    with monkeypatch.context() as context:
        context.setitem(module.ALL_ATTENTION_FUNCTIONS, "sdpa", observed)
        handles = [
            layer.register_forward_hook(lambda module, inputs, output: layer_outputs.append(output))
            for layer in native.model.layers
        ]
        try:
            with torch.inference_mode():
                expected = native(ids, use_cache=False, output_hidden_states=True)
        finally:
            for handle in handles:
                handle.remove()

    stream = LayerStream(snapshot, dtype=dtype)
    calls = []
    load_module = stream.weights.load_module

    def observed_load(module, prefix, dtype):
        assert all(parameter.is_meta for parameter in module.parameters())
        calls.append(prefix)
        load_module(module, prefix, dtype)
        assert all(not parameter.is_meta for parameter in module.parameters())

    monkeypatch.setattr(stream.weights, "load_module", observed_load)
    actual_captures = []

    def capture(index, query, key, value, scale):
        assert index == len(actual_captures)
        assert query.shape == (ids.shape[0], native.config.num_attention_heads, ids.shape[1], 8)
        assert (
            key.shape
            == value.shape
            == (ids.shape[0], native.config.num_key_value_heads, ids.shape[1], 8)
        )
        assert query.dtype == key.dtype == value.dtype == dtype
        actual_captures.append((query.clone(), key.clone(), value.clone(), scale))

    hidden = stream.embed(ids)
    assert_close(hidden, expected.hidden_states[0], dtype)
    for index in range(stream.layer_count):
        incoming = hidden
        hidden = stream.forward_layer(index, incoming, capture=capture)
        assert_close(hidden, layer_outputs[index], dtype)
        cached = stream._layer
        again = stream.forward_layer(index, incoming)
        assert stream._layer is cached
        assert "forward" not in cached.self_attn.__dict__
        assert_close(again, hidden, dtype)
        del cached
    assert calls == [f"model.layers.{index}." for index in range(stream.layer_count)]
    assert len(captures) == len(actual_captures) == stream.layer_count
    for actual, reference in zip(actual_captures, captures, strict=True):
        for tensor, native_tensor in zip(actual[:3], reference[:3], strict=True):
            assert_close(tensor, native_tensor, dtype)
        assert actual[3] == reference[3]
    old_layer = weakref.ref(stream._layer)
    stream.release_current()
    assert stream._layer is None and stream._layer_index is None
    assert old_layer() is None
    hidden = stream.finish(hidden)
    assert_close(hidden, expected.hidden_states[-1], dtype)
    pieces = []
    for start in range(0, native.config.vocab_size, 29):
        stop = min(start + 29, native.config.vocab_size)
        logits = stream.project_logits(hidden, start, stop)
        assert logits.dtype == torch.float32
        assert_close(logits, expected.logits[..., start:stop], dtype)
        pieces.append(logits)
    assert_close(torch.cat(pieces, dim=-1), expected.logits, dtype)
    stream.close()


@pytest.mark.parametrize("architecture", ARCHITECTURES)
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("tied", [False, True])
@pytest.mark.parametrize("sharded", [False, True])
def test_native_hidden_states_logits_and_captures(
    tmp_path, monkeypatch, architecture, dtype, tied, sharded
):
    model_class = save_small(tmp_path, architecture, tied=tied, sharded=sharded)
    native = model_class.from_pretrained(tmp_path, dtype=dtype, attn_implementation="sdpa").eval()
    ids = torch.tensor([[1, 21, 3, 21, 96, 4, 7, 8, 5], [1, 43, 6, 4, 2, 10, 12, 13, 14]])
    compare_native(tmp_path, native, ids, dtype, monkeypatch, architecture)


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_native_qwen_sliding_window_mask(tmp_path, monkeypatch, dtype):
    model_class = save_small(tmp_path, "qwen2", sliding=True)
    native = model_class.from_pretrained(tmp_path, dtype=dtype, attn_implementation="sdpa").eval()
    ids = torch.tensor([[1, 11, 12, 13, 14, 15, 16, 17, 18]])
    compare_native(tmp_path, native, ids, dtype, monkeypatch, "qwen2")
    stream = LayerStream(tmp_path, dtype=dtype)
    with pytest.raises(ValueError, match="sliding windows"):
        stream.forward_layer(1, stream.embed(ids), variant="tile")
    stream.close()


@pytest.mark.parametrize("architecture", ARCHITECTURES)
@pytest.mark.parametrize("variant", VARIANTS)
def test_real_variant_native_operations_and_batch_isolation(
    tmp_path, monkeypatch, architecture, variant
):
    from study.attention import apply_attention

    dtype = torch.bfloat16
    model_class = save_small(tmp_path, architecture)
    native = model_class.from_pretrained(tmp_path, dtype=dtype, attn_implementation="sdpa").eval()
    module = ARCHITECTURES[architecture][1]
    original = module.ALL_ATTENTION_FUNCTIONS["sdpa"]

    def variant_attention(module, query, key, value, attention_mask, scaling, **kwargs):
        result = apply_attention(query, key, value, variant, scale=scaling)
        return result.transpose(1, 2).contiguous().to(query.dtype), None

    ids = torch.tensor([[1, 21, 3, 21, 96, 4, 7], [1, 43, 6, 4, 2, 10, 12]])
    with monkeypatch.context() as context:
        context.setitem(module.ALL_ATTENTION_FUNCTIONS, "sdpa", variant_attention)
        with torch.inference_mode():
            expected = native(ids, use_cache=False, output_hidden_states=True)
    assert module.ALL_ATTENTION_FUNCTIONS["sdpa"] is original
    stream = LayerStream(tmp_path)
    hidden = stream.embed(ids)
    singles = [stream.embed(ids[index : index + 1]) for index in range(ids.shape[0])]
    for index in range(stream.layer_count):
        hidden = stream.forward_layer(index, hidden, variant=variant)
        singles = [stream.forward_layer(index, item, variant=variant) for item in singles]
        assert_close(hidden, torch.cat(singles, dim=0), dtype)
        assert module.ALL_ATTENTION_FUNCTIONS["sdpa"] is original
    hidden = stream.finish(hidden)
    assert_close(hidden, expected.hidden_states[-1], dtype)
    assert_close(stream.project_logits(hidden, 0, 97), expected.logits, dtype)
    stream.close()


def test_embedding_and_head_only_read_selected_source_slices(tmp_path, monkeypatch):
    import study.weights as weights_module

    save_small(tmp_path, "olmo2", tied=True)
    weights = SafeTensorWeights(tmp_path)
    selected_slices = []
    original = weights_module.safe_open

    class ObservedSlice:
        def __init__(self, view, name):
            self.view, self.name = view, name

        def __getitem__(self, selection):
            selected_slices.append((self.name, selection))
            return self.view[selection]

    class ObservedOpen:
        def __init__(self, *args, **kwargs):
            self.handle = original(*args, **kwargs)

        def __enter__(self):
            self.handle.__enter__()
            return self

        def __exit__(self, *args):
            return self.handle.__exit__(*args)

        def get_tensor(self, name):
            assert name not in {"model.embed_tokens.weight", "lm_head.weight"}
            return self.handle.get_tensor(name)

        def get_slice(self, name):
            return ObservedSlice(self.handle.get_slice(name), name)

        def keys(self):
            return self.handle.keys()

    monkeypatch.setattr(weights_module, "safe_open", ObservedOpen)
    stream = LayerStream(tmp_path)
    ids = torch.tensor([[3, 4, 3, 90], [90, 5, 4, 5]])
    embedded = stream.embed(ids)
    assert embedded.shape == (2, 4, 32) and embedded.dtype == torch.bfloat16
    assert torch.equal(embedded[0, 0], embedded[0, 2])
    assert torch.equal(embedded[0, 3], embedded[1, 0])
    logits = stream.project_logits(embedded, 7, 11)
    assert logits.shape == (2, 4, 4) and logits.dtype == torch.float32
    assert selected_slices == [
        ("model.embed_tokens.weight", slice(0, 0)),
        ("model.embed_tokens.weight", slice(3, 6)),
        ("model.embed_tokens.weight", slice(90, 91)),
        ("model.embed_tokens.weight", slice(7, 11)),
    ]
    assert "lm_head.weight" not in weights
    stream.close()


@pytest.mark.skipif(
    not os.environ.get("ATTENTION_NUMERICS_QWEN_SNAPSHOT"),
    reason="Optional validation requires an existing pinned local Qwen snapshot",
)
def test_pinned_qwen_snapshot():
    snapshot = Path(os.environ["ATTENTION_NUMERICS_QWEN_SNAPSHOT"])
    native = AutoModelForCausalLM.from_pretrained(
        snapshot, dtype=torch.bfloat16, attn_implementation="sdpa", local_files_only=True
    ).eval()
    assert native.config.model_type == "qwen2"
    assert native.config.hidden_size == 896 and native.config.num_hidden_layers == 24
    ids = torch.arange(1, 65, dtype=torch.long).unsqueeze(0)
    with torch.inference_mode():
        expected = native(ids, use_cache=False, output_hidden_states=True)
    # Drop the complete model before streaming, keeping only reference outputs.
    del native
    stream = LayerStream(snapshot)
    hidden = stream.embed(ids)
    assert_close(hidden, expected.hidden_states[0], torch.bfloat16)
    for index in range(stream.layer_count):
        hidden = stream.forward_layer(index, hidden)
        if index + 1 < stream.layer_count:
            assert_close(hidden, expected.hidden_states[index + 1], torch.bfloat16)
    stream.release_current()
    hidden = stream.finish(hidden)
    assert_close(hidden, expected.hidden_states[-1], torch.bfloat16)
    for start, stop in [(0, 127), (500, 691), (150000, stream.config.vocab_size)]:
        actual = stream.project_logits(hidden, start, stop)
        assert_close(actual, expected.logits[..., start:stop], torch.bfloat16)
    stream.close()
