"""冻结教师的 CPU 注意力实现，复用每个已拟合成员的固定源投影。"""
from contextvars import ContextVar
from types import MethodType


class _CpuSingleKeyAttention:
    """冻结教师用广播视图共享同一份 K/V。仅接管该教师发出的 CPU 推理。"""
    name = 'zhixun_frozen_regression_cpu_single_key'

    def __init__(self):
        self.active = ContextVar(self.name, default=False)

    def is_preferred(self, spec):
        return (self.active.get() and spec.device.type == 'cpu' and not spec.is_grad_enabled
                and spec.quantized_kv_dtype is not None and spec.num_kv_heads == 1
                and spec.num_heads > 1)

    def run(self, q_BSHD, k_BSJD, v_BSJD, *, quantized_kv=None):
        # 作者在调用后端前完成与原实现相同的缩放和类型转换。
        from tabpfn.architectures.shared.scaled_dot_product_attention import _torch_sdpa
        heads = q_BSHD.shape[2]
        return _torch_sdpa(q_BSHD, k_BSJD.expand(-1, -1, heads, -1),
                           v_BSJD.expand(-1, -1, heads, -1))


def _cached_induced_attention(layer, x, single_eval_pos=None, cached_hidden=None, *,
                              return_hidden=False, enable_torch_compile=False):
    """作者提供的固定源上下文可复用投影；每批传入的查询仍经过原注意力和前馈计算。"""
    import torch
    from tabpfn.architectures.tabpfn_v3_5 import _batched_scaled_dot_product_attention
    if (cached_hidden is None or return_hidden or enable_torch_compile or x.device.type != 'cpu'
            or not torch.is_inference_mode_enabled()
            or cached_hidden.numel() * x.element_size() * 2 > 128 * 1024**2):
        return layer._original_induced_attention(x, single_eval_pos, cached_hidden, return_hidden=return_hidden,
                                                enable_torch_compile=enable_torch_compile)
    block = layer.cross_attn_block2
    attention = block.attn
    identity = (cached_hidden.data_ptr(), cached_hidden.shape, cached_hidden.stride(), cached_hidden.dtype, x.dtype)
    saved = getattr(layer, '_source_projections', None)
    if saved is None or saved[0] != identity:
        source = block.layernorm_kv(cached_hidden.to(x.dtype))
        batch, length, _ = source.shape
        keys = attention.k_norm(attention.k_projection(source).view(batch, length, -1, attention.head_dim))
        values = attention.v_projection(source).view(batch, length, -1, attention.head_dim)
        # 强引用源张量保证旧地址不会被另一个上下文复用。每层只保留最后一个上下文。
        saved = (identity, cached_hidden, keys, values)
        layer._source_projections = saved
    batch, count, _ = x.shape
    query = attention.q_norm(attention.q_projection(block.layernorm_q(x)).view(batch, count, -1, attention.head_dim))
    result = _batched_scaled_dot_product_attention(query, saved[2], saved[3],
                                                 softmax_scaling_layer=attention.softmax_scaling_layer)
    x = x + attention.out_projection(result.reshape(batch, count, attention.head_dim * attention.num_heads))
    return x + block.mlp(block.layernorm2(x))


def _enable_frozen_cpu_inference(teacher):
    # 自定义读出也可实现公开 predict 契约，只有作者缓存执行器提供可优化的投影层。
    caches = getattr(getattr(teacher, 'executor_', None), 'model_caches', None)
    if caches is None:
        return
    import torch
    from tabpfn.architectures.shared.attention_backends import register_attention_backend, registered_attention_backends
    from tabpfn.architectures.tabpfn_v3_5 import InducedSelfAttentionBlock

    backend = next((b for b in registered_attention_backends() if b.name == _CpuSingleKeyAttention.name), None)
    if backend is None:
        backend = _CpuSingleKeyAttention()
        register_attention_backend(backend)
    for entry in caches:
        for layer in entry.get(torch.device('cpu')).modules():
            if isinstance(layer, InducedSelfAttentionBlock) and not hasattr(layer, '_original_induced_attention'):
                layer._original_induced_attention = layer._induced_attention
                layer._induced_attention = MethodType(_cached_induced_attention, layer)
    teacher._zhixun_cpu_backend = backend
