"""Addressing for the opt-in SM100 decode timeline."""

import cutlass
import cutlass.cute as cute
from cutlass import Uint32, Uint64
from cutlass._mlir.dialects import llvm
from cutlass.cutlass_dsl import T, dsl_user_op

TRACE_CTAS = 320
TRACE_TASKS = 32
TRACE_TASK_FIELDS = (
    'a_task', 'a_k0_ready', 'a_k0_issued',
    'b_task', 'b_k0_issued',
    'mma_task', 'mma_k0_ready', 'mma_last_k_ready', 'mma_last_k_issued',
    'epilogue_begin', 'epilogue_end', 'mma_operand_wait', 'mma_operand_readiness',
    'mma_accum_credit_wait', 'b_last_k_issued', 'a_last_k_issued',
)
TRACE_STARTUP_FIELDS = 16
TRACE_TASK_BASE = 4 + TRACE_STARTUP_FIELDS
TRACE_CTA_FIELDS = TRACE_TASK_BASE + len(TRACE_TASK_FIELDS) * TRACE_TASKS
TRACE_BYTES = 3 * TRACE_CTAS * TRACE_CTA_FIELDS * 16


@dsl_user_op
def global_timer(*, loc=None, ip=None) -> Uint64:
    value = llvm.inline_asm(
        T.i64(), [], 'mov.u64 $0, %globaltimer;', '=l,~{memory}',
        has_side_effects=True, asm_dialect=llvm.AsmDialect.AD_ATT, loc=loc, ip=ip)
    return Uint64(value)


@dsl_user_op
def sm_id(*, loc=None, ip=None) -> Uint32:
    value = llvm.inline_asm(
        T.i32(), [], 'mov.u32 $0, %smid;', '=r', has_side_effects=True,
        asm_dialect=llvm.AsmDialect.AD_ATT, loc=loc, ip=ip)
    return Uint32(value)


@dsl_user_op
def _stamp(address: cute.Pointer, tag: Uint64, *, loc=None, ip=None):
    llvm.inline_asm(
        None, [address.toint(loc=loc, ip=ip).ir_value(), tag.ir_value()],
        '{ .reg .b64 t; mov.u64 t, %globaltimer; st.global.v2.u64 [$0], {t, $1}; }',
        'l,l,~{memory}', has_side_effects=True,
        asm_dialect=llvm.AsmDialect.AD_ATT, loc=loc, ip=ip)


@cute.jit
def cta_mark(signals: cute.Tensor, offset: cutlass.Constexpr, role: cutlass.Constexpr,
             cta: Uint32, slot: Uint32, tag: Uint64):
    if cta < Uint32(TRACE_CTAS):
        words = signals.iterator + offset // 4
        index = (Uint32((role - 1) * TRACE_CTAS) + cta) * Uint32(TRACE_CTA_FIELDS) + slot
        _stamp(words + index * Uint32(4), tag)


@cute.jit
def task_mark(signals: cute.Tensor, offset: cutlass.Constexpr, cta: Uint32,
              task_idx: Uint32, field: cutlass.Constexpr, tag: Uint64,
              sample_stride: cutlass.Constexpr = 1):
    if cutlass.const_expr(sample_stride != 0):
        if (task_idx < Uint32(TRACE_TASKS) and
                (cta // Uint32(2)) % Uint32(sample_stride) == Uint32(0)):
            cta_mark(signals, offset, 2, cta,
                     Uint32(TRACE_TASK_BASE + field * TRACE_TASKS) + task_idx, tag)


@cute.jit
def task_tag(task_words: cute.Tensor, task_stage: Uint32, phase: Uint32) -> Uint64:
    base = task_stage * Uint32(8)
    return (Uint64(phase) | (Uint64(task_words[base + 1]) << Uint64(8)) |
            (Uint64(task_words[base + 3]) << Uint64(20)) |
            ((Uint64(task_words[base + 4]) & Uint64(0xffffff)) << Uint64(32)) |
            (Uint64(task_words[base + 5]) << Uint64(56)))
