"""Register-staged traffic for the SM100 small-batch worker graph."""

import cutlass
import cutlass.cute as cute
from cutlass import Float32, Uint16, Uint32, Uint64

from ..common.math import pack_bf16_pair
from ..ptx.ld_st import (ld_acquire_sys_u32, ld_global_u32x4, ld_volatile_u32, ld_volatile_u64,
                         st_global_u32x4, st_global_u64, st_release_u32)
from ..ptx.utils import accumulate_bf16_into_f32, device_trap, read_clock64, sleep_ns


VECTOR_BYTES = 16
WARP_BYTES = 32 * VECTOR_BYTES
W2_FLAG_BYTES = WARP_BYTES // 2


@cute.jit
def combine_flag_index(token: Uint32, slot: Uint32, segment: Uint32,
                       hidden: cutlass.Constexpr, slots: cutlass.Constexpr) -> Uint64:
    return ((Uint64(token) * Uint64(hidden * 2 // W2_FLAG_BYTES) + Uint64(segment)) *
            Uint64(slots) + Uint64(slot))


@cute.jit
def copy_input_vector(dst: cute.Pointer, src: cute.Pointer, lane: Uint32,
                      hidden: cutlass.Constexpr, load_group: cutlass.Constexpr = 1) -> None:
    """Copy one FP8 row through registers between global allocations."""
    if cutlass.const_expr(load_group == 1):
        for part in cutlass.range_constexpr(hidden // WARP_BYTES):
            offset = (Uint32(part * 32) + lane) * Uint32(VECTOR_BYTES // 4)
            words = ld_global_u32x4(src + offset)
            st_global_u32x4(dst + offset, words[0], words[1], words[2], words[3])
    else:
        pending = cute.make_rmem_tensor((load_group, 4), Uint32)
        for group in cutlass.range_constexpr(hidden // (WARP_BYTES * load_group)):
            for part in cutlass.range_constexpr(load_group):
                offset = Uint32((group * load_group + part) * WARP_BYTES // 4) + lane * Uint32(4)
                words = ld_global_u32x4(src + offset)
                for word in cutlass.range_constexpr(4):
                    pending[part, word] = words[word]
            for part in cutlass.range_constexpr(load_group):
                offset = Uint32((group * load_group + part) * WARP_BYTES // 4) + lane * Uint32(4)
                st_global_u32x4(dst + offset, pending[part, 0], pending[part, 1],
                                pending[part, 2], pending[part, 3])


@cute.jit
def _accumulate_loaded_words(accum: cute.Tensor, words) -> None:
    for index in cutlass.range_constexpr(len(words)):
        packed = words[index]
        accum[index * 2] = accumulate_bf16_into_f32(
            accum[index * 2], Uint16(packed & Uint32(0xffff)))
        accum[index * 2 + 1] = accumulate_bf16_into_f32(
            accum[index * 2 + 1], Uint16(packed >> Uint32(16)))


@cute.jit
def reduce_output_vector(combine: cute.Pointer, output: cute.Pointer,
                         token: Uint32, chunk: Uint32, lane: Uint32,
                         mask: Uint32, h: cutlass.Constexpr,
                         capacity: cutlass.Constexpr, topk: cutlass.Constexpr,
                         has_shared: cutlass.Constexpr,
                         load_group: cutlass.Constexpr = 1,
                         token_major: cutlass.Constexpr = False) -> None:
    """Read ready BF16 slots into registers and write the reduced output."""
    offset = Uint64(token) * Uint64(h * 2) + \
        Uint64(chunk * Uint32(WARP_BYTES) + lane * Uint32(VECTOR_BYTES))
    token_base = offset
    slot_bytes = capacity * h * 2
    if cutlass.const_expr(token_major):
        token_base += Uint64(token) * Uint64((topk + has_shared - 1) * h * 2)
        slot_bytes = h * 2
    accum = cute.make_rmem_tensor((8,), Float32)
    for index in cutlass.range_constexpr(8):
        accum[index] = Float32(0)
    # Preserve the accumulator's shared-first rounding order.
    order = ([topk] if has_shared else []) + list(range(topk))
    if cutlass.const_expr(load_group == 1):
        for index in cutlass.range_constexpr(len(order)):
            slot = order[index]
            if ((mask >> Uint32(slot)) & Uint32(1)) != Uint32(0):
                src = cute.recast_ptr(combine + Uint64(slot * slot_bytes) + token_base,
                                       dtype=Uint32)
                _accumulate_loaded_words(accum, ld_global_u32x4(src))
    else:
        for first in cutlass.range_constexpr(0, len(order), load_group):
            loaded = cute.make_rmem_tensor((load_group, 4), Uint32)
            count = min(load_group, len(order) - first)
            for group_slot in cutlass.range_constexpr(count):
                slot = order[first + group_slot]
                if ((mask >> Uint32(slot)) & Uint32(1)) != Uint32(0):
                    src = cute.recast_ptr(combine + Uint64(slot * slot_bytes) + token_base,
                                           dtype=Uint32)
                    words = ld_global_u32x4(src)
                    for word in cutlass.range_constexpr(4):
                        loaded[group_slot, word] = words[word]
            for group_slot in cutlass.range_constexpr(count):
                slot = order[first + group_slot]
                if ((mask >> Uint32(slot)) & Uint32(1)) != Uint32(0):
                    values = (loaded[group_slot, 0], loaded[group_slot, 1],
                              loaded[group_slot, 2], loaded[group_slot, 3])
                    _accumulate_loaded_words(accum, values)
    dst = cute.recast_ptr(output + offset, dtype=Uint32)
    st_global_u32x4(dst, pack_bf16_pair(accum[0], accum[1]),
                    pack_bf16_pair(accum[2], accum[3]),
                    pack_bf16_pair(accum[4], accum[5]),
                    pack_bf16_pair(accum[6], accum[7]))


@cute.jit
def reduce_output_sentinel(combine: cute.Pointer, output: cute.Pointer,
                           token: Uint32, chunk: Uint32, lane: Uint32,
                           mask: Uint32, rank: Uint32, h: cutlass.Constexpr,
                           capacity: cutlass.Constexpr, topk: cutlass.Constexpr,
                           has_shared: cutlass.Constexpr,
                           vectors: cutlass.Constexpr = 1,
                           poll: cutlass.Constexpr = 'full',
                           token_major: cutlass.Constexpr = False,
                           chunk_major: cutlass.Constexpr = False,
                           poll_delay_ns: cutlass.Constexpr = 0,
                           ready_flags=None,
                           adaptive_poll: cutlass.Constexpr = False,
                           ready_mask: cutlass.Constexpr = False,
                           half_vector: cutlass.Constexpr = False) -> None:
    """Observe and retire each W2 result vector as its words arrive."""
    vector_bytes = VECTOR_BYTES // 2 if half_vector else VECTOR_BYTES
    vector_words = vector_bytes // 4
    warp_bytes = 32 * vector_bytes
    chunk_base = (Uint64(token) * Uint64(h * 2) +
                  Uint64(chunk * Uint32(vectors * warp_bytes)))
    offset = chunk_base + Uint64(lane * Uint32(vector_bytes))
    order = ([topk] if has_shared else []) + list(range(topk))
    mailbox_chunk = chunk_base
    slot_bytes = capacity * h * 2
    if cutlass.const_expr(token_major):
        mailbox_chunk += Uint64(token) * Uint64((len(order) - 1) * h * 2)
        slot_bytes = h * 2
    elif cutlass.const_expr(chunk_major):
        mailbox_chunk = chunk_base * Uint64(len(order))
        slot_bytes = vectors * WARP_BYTES
    mailbox_offset = mailbox_chunk + Uint64(lane * Uint32(vector_bytes))
    inline_prefix = poll in ('inline-prefix', 'shared-prefix')
    if cutlass.const_expr(poll in ('ordered', 'prefix') or inline_prefix):
        ordered_accum = cute.make_rmem_tensor((8,), Float32)
        for index in cutlass.range_constexpr(8):
            ordered_accum[index] = Float32(0)
    if cutlass.const_expr(poll != 'ordered'):
        loaded = cute.make_rmem_tensor((len(order), vector_words * vectors), Uint32)
    if cutlass.const_expr(poll == 'prefix' or inline_prefix):
        reduced = Uint32(0)
    sentinel = Uint32(0xffffffff)
    arrived = Uint32(0)
    start = read_clock64()
    flag_segment = chunk * Uint32(vectors * 2) + lane // Uint32(16)
    if cutlass.const_expr(ready_mask):
        mask_pointer = ready_flags + combine_flag_index(token, Uint32(0), flag_segment, h, 1)
    waiting = arrived != mask
    if cutlass.const_expr(poll == 'warp-probe'):
        probed_slots = Uint32(0)
    while waiting:
        if cutlass.const_expr(ready_mask):
            visible = Uint32(0)
            if lane % Uint32(16) == Uint32(0):
                visible = ld_acquire_sys_u32(mask_pointer)
            visible = cute.arch.shuffle_sync(visible, lane & Uint32(16))
        if cutlass.const_expr(poll_delay_ns or poll == 'prefix'):
            before = arrived
        if cutlass.const_expr(poll == 'warp-probe'):
            # Each active peer slot gets one probe lane. Latch discoveries so
            # a consumer resetting its own vector cannot hide readiness again.
            if probed_slots != mask:
                probe_bit = Uint32(1) << lane
                slot_visible = False
                if (mask & probe_bit) != Uint32(0) and (probed_slots & probe_bit) == Uint32(0):
                    probe_offset = Uint64(lane) * Uint64(slot_bytes) + mailbox_chunk
                    probe_ptr = cute.recast_ptr(combine + probe_offset, dtype=Uint32)
                    slot_visible = ld_volatile_u32(probe_ptr + (vectors * WARP_BYTES // 4 - 1)) != sentinel
                probed_slots |= cute.arch.vote_ballot_sync(slot_visible)
        earlier_slots = Uint32(0)
        for index in cutlass.range_constexpr(len(order)):
            slot = order[index]
            bit = Uint32(1 << slot)
            next_in_order = True
            if cutlass.const_expr(poll == 'ordered'):
                next_in_order = (arrived & earlier_slots) == (mask & earlier_slots)
                earlier_slots |= bit
            if cutlass.const_expr(poll in ('shared-first', 'shared-prefix') and has_shared and index > 0):
                # Shared W2 runs first. While it is still pending, avoid
                # polling all routed mailboxes during the weight reads.
                shared_slot = Uint32(1 << topk)
                next_in_order = (arrived & shared_slot) == (mask & shared_slot)
            if next_in_order and (mask & bit) != Uint32(0) and (arrived & bit) == Uint32(0):
                src = cute.recast_ptr(combine + Uint64(slot * slot_bytes) + mailbox_offset,
                                       dtype=Uint32)
                probe_ready = True
                if cutlass.const_expr(ready_mask):
                    probe_ready = (visible & bit) != Uint32(0)
                elif cutlass.const_expr(ready_flags is not None):
                    for vector in cutlass.range_constexpr(vectors):
                        index_of_flag = combine_flag_index(token, Uint32(slot),
                            flag_segment + Uint32(2 * vector), h, len(order))
                        slot_flag = ld_acquire_sys_u32(ready_flags + index_of_flag)
                        probe_ready = probe_ready and slot_flag == Uint32(1)
                if cutlass.const_expr(poll == 'probe'):
                    probe_ready = ld_volatile_u32(src + 3) != sentinel
                if cutlass.const_expr(poll == 'warp-probe'):
                    probe_ready = (probed_slots & bit) != Uint32(0)
                if probe_ready:
                    valid = True
                    words_by_vector = ()
                    for vector in cutlass.range_constexpr(vectors):
                        if cutlass.const_expr(half_vector):
                            packed = ld_volatile_u64(src)
                            words = (Uint32(packed), Uint32(packed >> Uint64(32)))
                        else:
                            words = ld_global_u32x4(
                                src + vector * (WARP_BYTES // 4),
                                volatile=ready_flags is None and poll not in ('relaxed-sys', 'ordered'),
                                relaxed_sys=ready_flags is None and poll in ('relaxed-sys', 'ordered'))
                        words_by_vector += (words,)
                        if cutlass.const_expr(ready_flags is None):
                            for word in cutlass.range_constexpr(vector_words):
                                valid = valid and words[word] != sentinel
                    if valid:
                        if cutlass.const_expr(poll == 'ordered'):
                            _accumulate_loaded_words(ordered_accum, words_by_vector[0])
                        for vector in cutlass.range_constexpr(vectors):
                            if cutlass.const_expr(poll != 'ordered'):
                                for word in cutlass.range_constexpr(vector_words):
                                    loaded[index, vector * vector_words + word] = words_by_vector[vector][word]
                            if cutlass.const_expr(ready_flags is None and poll != 'defer-reset'):
                                if cutlass.const_expr(half_vector):
                                    st_global_u64(src, Uint64(0xffffffffffffffff))
                                else:
                                    st_global_u32x4(src + vector * (WARP_BYTES // 4),
                                                    sentinel, sentinel, sentinel, sentinel)
                        arrived |= bit
            if cutlass.const_expr(inline_prefix):
                # Drain the ready prefix during the probe pass. Any later
                # arrival stays in registers until earlier slots are reduced.
                if ((reduced & earlier_slots) == (mask & earlier_slots) and
                        (arrived & ~reduced & bit) != Uint32(0)):
                    words = (loaded[index, 0], loaded[index, 1],
                             loaded[index, 2], loaded[index, 3])
                    _accumulate_loaded_words(ordered_accum, words)
                    reduced |= bit
                earlier_slots |= bit
        if cutlass.const_expr(poll == 'prefix'):
            # Observe all slots independently. Advance the arithmetic
            # only across the contiguous prefix in the reference order.
            if arrived != before:
                prior = Uint32(0)
                for index in cutlass.range_constexpr(len(order)):
                    bit = Uint32(1 << order[index])
                    if (reduced & prior) == (mask & prior) and (arrived & ~reduced & bit) != Uint32(0):
                        words = (loaded[index, 0], loaded[index, 1],
                                 loaded[index, 2], loaded[index, 3])
                        _accumulate_loaded_words(ordered_accum, words)
                        reduced |= bit
                    prior |= bit
        if Uint64(read_clock64() - start) >= Uint64(60 * 2000000000):
            if arrived != mask:
                cute.printf('GC sentinel Combine timeout rank={} token={} chunk={}', rank, token, chunk)
                device_trap()
        waiting = arrived != mask
        if cutlass.const_expr(poll_delay_ns):
            if waiting and arrived == before:
                delay = Uint32(poll_delay_ns)
                if cutlass.const_expr(adaptive_poll):
                    delay >>= Uint32(arrived != Uint32(0))
                    delay >>= Uint32(cute.arch.popc(mask & ~arrived) <= Uint32(2))
                sleep_ns(delay)
        if cutlass.const_expr(poll == 'warp-probe' or ready_mask):
            waiting = cute.arch.vote_ballot_sync(waiting) != Uint32(0)
    if cutlass.const_expr(ready_flags is not None):
        # A half-warp may clear a flag only after all consumers took the data.
        cute.arch.sync_warp()
        if lane % Uint32(16) == Uint32(0):
            if cutlass.const_expr(ready_mask):
                st_release_u32(mask_pointer, Uint32(0), system=True)
            else:
                for slot in cutlass.range_constexpr(len(order)):
                    if (mask & Uint32(1 << slot)) != Uint32(0):
                        for vector in cutlass.range_constexpr(vectors):
                            index_of_flag = combine_flag_index(token, Uint32(slot),
                                flag_segment + Uint32(2 * vector), h, len(order))
                            st_release_u32(ready_flags + index_of_flag, Uint32(0), system=True)
    for vector in cutlass.range_constexpr(vectors):
        if cutlass.const_expr(poll in ('ordered', 'prefix') or inline_prefix):
            accum = ordered_accum
        else:
            accum = cute.make_rmem_tensor((vector_words * 2,), Float32)
            for index in cutlass.range_constexpr(vector_words * 2):
                accum[index] = Float32(0)
            for index in cutlass.range_constexpr(len(order)):
                if (mask & Uint32(1 << order[index])) != Uint32(0):
                    if cutlass.const_expr(poll == 'defer-reset'):
                        src = cute.recast_ptr(combine + Uint64(order[index] * slot_bytes) + mailbox_offset,
                                               dtype=Uint32) + vector * (WARP_BYTES // 4)
                        st_global_u32x4(src, sentinel, sentinel, sentinel, sentinel)
                    if cutlass.const_expr(half_vector):
                        words = (loaded[index, 0], loaded[index, 1])
                    else:
                        words = (loaded[index, vector * 4], loaded[index, vector * 4 + 1],
                                 loaded[index, vector * 4 + 2], loaded[index, vector * 4 + 3])
                    _accumulate_loaded_words(accum, words)
        dst = cute.recast_ptr(output + offset, dtype=Uint32) + vector * (WARP_BYTES // 4)
        first = pack_bf16_pair(accum[0], accum[1])
        second = pack_bf16_pair(accum[2], accum[3])
        if cutlass.const_expr(half_vector):
            st_global_u64(dst, Uint64(first) | (Uint64(second) << Uint64(32)))
        else:
            st_global_u32x4(dst, first, second, pack_bf16_pair(accum[4], accum[5]),
                            pack_bf16_pair(accum[6], accum[7]))


@cute.jit
def combine_after_shared_w2(mailbox: cute.Pointer, output: cute.Pointer,
                            routes: cute.Pointer, shared_words,
                            token: Uint32, slot_stride_words: cutlass.Constexpr,
                            topk: cutlass.Constexpr, experts: cutlass.Constexpr) -> None:
    """Reduce one W2 output vector with its shared partial already in registers."""
    sentinel = Uint32(0xffffffff)
    accum = cute.make_rmem_tensor((8,), Float32)
    for index in cutlass.range_constexpr(8):
        accum[index] = Float32(0)
    escaped = ()
    for index in cutlass.range_constexpr(4):
        word = shared_words[index]
        escaped += (Uint32(0x7fff7fff) if word == sentinel else word,)
    _accumulate_loaded_words(accum, escaped)
    start = read_clock64()
    for slot in cutlass.range_constexpr(topk):
        if routes[token * Uint32(topk) + Uint32(slot)] < Uint64(experts):
            src = mailbox + slot * slot_stride_words
            pending = True
            while pending:
                words = ld_global_u32x4(src, volatile=True)
                complete = True
                for index in cutlass.range_constexpr(4):
                    complete = complete and words[index] != sentinel
                if complete:
                    _accumulate_loaded_words(accum, words)
                    st_global_u32x4(src, sentinel, sentinel, sentinel, sentinel)
                    pending = False
                else:
                    if Uint64(read_clock64() - start) >= Uint64(60 * 2000000000):
                        device_trap()
                    sleep_ns(Uint32(64))
    st_global_u32x4(output, pack_bf16_pair(accum[0], accum[1]),
                    pack_bf16_pair(accum[2], accum[3]),
                    pack_bf16_pair(accum[4], accum[5]),
                    pack_bf16_pair(accum[6], accum[7]))
