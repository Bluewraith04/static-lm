use std::alloc::{Layout, alloc_zeroed, dealloc, handle_alloc_error};
use std::ptr::NonNull;

use crate::quant::blocks::BlockQ8_0;

pub const ARENA_ALIGN: usize = 64;
const ELEMENTS_PER_CACHE_LINE: usize = ARENA_ALIGN / std::mem::size_of::<f32>();

/// Typed offset handle into an [`AlignedArena`].
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub struct ArenaSlice {
    pub offset: usize,
    pub len: usize,
}

impl ArenaSlice {
    #[inline(always)]
    pub const fn end(&self) -> usize {
        self.offset + self.len
    }

    #[inline(always)]
    pub const fn overlaps(&self, other: &ArenaSlice) -> bool {
        self.offset < other.end() && other.offset < self.end()
    }

    #[inline(always)]
    pub const fn subslice(&self, rel_offset: usize, len: usize) -> ArenaSlice {
        assert!(rel_offset + len <= self.len, "ArenaSlice::subslice out of bounds");
        ArenaSlice {
            offset: self.offset + rel_offset,
            len,
        }
    }
}

/// Pre-allocated 64-byte aligned contiguous `f32` activation scratchpad.
pub struct AlignedArena {
    ptr: NonNull<f32>,
    layout: Layout,
    capacity: usize,
    cursor: usize,
}

// SAFETY: AlignedArena uniquely owns its heap allocation.
unsafe impl Send for AlignedArena {}
unsafe impl Sync for AlignedArena {}

impl AlignedArena {
    /// Allocates a zeroed, 64-byte aligned `f32` arena with at least `capacity` elements.
    pub fn with_capacity(capacity: usize) -> Self {
        crate::quant::kernels::init_gemv_pool();
        let aligned_cap = capacity
            .max(ELEMENTS_PER_CACHE_LINE)
            .next_multiple_of(ELEMENTS_PER_CACHE_LINE);
        let byte_size = aligned_cap
            .checked_mul(std::mem::size_of::<f32>())
            .expect("arena byte size overflow");
        let layout = Layout::from_size_align(byte_size, ARENA_ALIGN)
            .expect("invalid arena memory layout");

        // SAFETY: byte_size > 0 and ARENA_ALIGN is a power of two.
        let raw_ptr = unsafe { alloc_zeroed(layout) as *mut f32 };
        let ptr = NonNull::new(raw_ptr).unwrap_or_else(|| handle_alloc_error(layout));

        Self {
            ptr,
            layout,
            capacity: aligned_cap,
            cursor: 0,
        }
    }

    /// Returns total capacity in `f32` elements.
    #[inline(always)]
    pub fn capacity(&self) -> usize {
        self.capacity
    }

    /// Returns current bump cursor offset in `f32` elements.
    #[inline(always)]
    pub fn used(&self) -> usize {
        self.cursor
    }

    /// Resets the bump allocation cursor to the beginning of the arena without deallocating.
    #[inline(always)]
    pub fn reset(&mut self) {
        self.cursor = 0;
    }

    /// Bump-allocates a 64-byte aligned [`ArenaSlice`] of `len` `f32` elements.
    pub fn alloc_slice(&mut self, len: usize) -> ArenaSlice {
        let aligned_offset = self.cursor.next_multiple_of(ELEMENTS_PER_CACHE_LINE);
        let next_cursor = aligned_offset
            .checked_add(len)
            .expect("arena cursor overflow");
        assert!(
            next_cursor <= self.capacity,
            "AlignedArena out of capacity: requested {} elements at offset {}, capacity {}",
            len,
            aligned_offset,
            self.capacity
        );
        self.cursor = next_cursor;
        ArenaSlice {
            offset: aligned_offset,
            len,
        }
    }

    /// Bump-allocates a 64-byte aligned [`ArenaSlice`] sized to hold `num_blocks` [`BlockQ8_0`] structs.
    pub fn alloc_q8_scratch(&mut self, num_blocks: usize) -> ArenaSlice {
        let bytes_needed = num_blocks
            .checked_mul(std::mem::size_of::<BlockQ8_0>())
            .expect("q8 scratch byte size overflow");
        let f32_len = bytes_needed.div_ceil(std::mem::size_of::<f32>());
        self.alloc_slice(f32_len)
    }

    /// Returns a view of the entire arena capacity.
    #[inline(always)]
    pub fn as_slice(&self) -> &[f32] {
        // SAFETY: `ptr` is valid for `capacity` initialized `f32` elements.
        unsafe { std::slice::from_raw_parts(self.ptr.as_ptr(), self.capacity) }
    }

    /// Returns a mutable view of the entire arena capacity.
    #[inline(always)]
    pub fn as_mut_slice(&mut self) -> &mut [f32] {
        // SAFETY: `ptr` is uniquely owned and valid for `capacity` initialized `f32` elements.
        unsafe { std::slice::from_raw_parts_mut(self.ptr.as_ptr(), self.capacity) }
    }

    /// Returns an immutable slice for a given [`ArenaSlice`] handle.
    #[inline(always)]
    pub fn get(&self, slice: ArenaSlice) -> &[f32] {
        assert!(slice.end() <= self.capacity, "ArenaSlice out of bounds");
        &self.as_slice()[slice.offset..slice.end()]
    }

    /// Returns a mutable slice for a given [`ArenaSlice`] handle.
    #[inline(always)]
    pub fn get_mut(&mut self, slice: ArenaSlice) -> &mut [f32] {
        assert!(slice.end() <= self.capacity, "ArenaSlice out of bounds");
        &mut self.as_mut_slice()[slice.offset..slice.end()]
    }

    /// Simultaneously borrows one immutable input slice and one disjoint mutable output slice.
    #[inline(always)]
    pub fn io_slices(&mut self, src: ArenaSlice, dst: ArenaSlice) -> (&[f32], &mut [f32]) {
        assert!(
            src.end() <= self.capacity && dst.end() <= self.capacity,
            "ArenaSlice out of bounds"
        );
        assert!(
            !src.overlaps(&dst),
            "io_slices: src {:?} overlaps dst {:?}",
            src,
            dst
        );
        let base = self.ptr.as_ptr();
        // SAFETY: Bounds and non-overlap verified above.
        unsafe {
            (
                std::slice::from_raw_parts(base.add(src.offset), src.len),
                std::slice::from_raw_parts_mut(base.add(dst.offset), dst.len),
            )
        }
    }

    /// Simultaneously borrows two immutable input slices and one disjoint mutable output slice.
    #[inline(always)]
    pub fn io2_slices(
        &mut self,
        src_a: ArenaSlice,
        src_b: ArenaSlice,
        dst: ArenaSlice,
    ) -> (&[f32], &[f32], &mut [f32]) {
        assert!(
            src_a.end() <= self.capacity
                && src_b.end() <= self.capacity
                && dst.end() <= self.capacity,
            "ArenaSlice out of bounds"
        );
        assert!(
            !src_a.overlaps(&dst) && !src_b.overlaps(&dst),
            "io2_slices: dst {:?} overlaps input ({:?} or {:?})",
            dst,
            src_a,
            src_b
        );
        let base = self.ptr.as_ptr();
        // SAFETY: Bounds and non-overlap with `dst` verified above.
        unsafe {
            (
                std::slice::from_raw_parts(base.add(src_a.offset), src_a.len),
                std::slice::from_raw_parts(base.add(src_b.offset), src_b.len),
                std::slice::from_raw_parts_mut(base.add(dst.offset), dst.len),
            )
        }
    }

    /// Simultaneously borrows an input `f32` slice, a mutable `BlockQ8_0` quantization scratchpad,
    /// and a mutable output `f32` slice.
    #[inline(always)]
    pub fn gemv_slices(
        &mut self,
        src: ArenaSlice,
        q8_scratch: ArenaSlice,
        dst: ArenaSlice,
        num_blocks: usize,
    ) -> (&[f32], &mut [BlockQ8_0], &mut [f32]) {
        assert!(
            src.end() <= self.capacity
                && q8_scratch.end() <= self.capacity
                && dst.end() <= self.capacity,
            "ArenaSlice out of bounds"
        );
        assert!(
            !src.overlaps(&q8_scratch) && !src.overlaps(&dst) && !q8_scratch.overlaps(&dst),
            "gemv_slices: slices must be pairwise disjoint ({:?}, {:?}, {:?})",
            src,
            q8_scratch,
            dst
        );
        let q8_bytes = num_blocks
            .checked_mul(std::mem::size_of::<BlockQ8_0>())
            .expect("q8 byte len overflow");
        let scratch_bytes = q8_scratch.len * std::mem::size_of::<f32>();
        assert!(
            q8_bytes <= scratch_bytes,
            "q8_scratch slice too small: need {} bytes, have {} bytes",
            q8_bytes,
            scratch_bytes
        );

        let base = self.ptr.as_ptr();
        // SAFETY: Pairwise disjointness, alignment (4-byte f32 >= 2-byte BlockQ8_0), and bounds verified above.
        unsafe {
            let src_slice = std::slice::from_raw_parts(base.add(src.offset), src.len);
            let q8_raw_bytes =
                std::slice::from_raw_parts_mut(base.add(q8_scratch.offset) as *mut u8, q8_bytes);
            let q8_slice: &mut [BlockQ8_0] = bytemuck::cast_slice_mut(q8_raw_bytes);
            let dst_slice = std::slice::from_raw_parts_mut(base.add(dst.offset), dst.len);
            (src_slice, q8_slice, dst_slice)
        }
    }

    /// Simultaneously borrows `q`, `k_cache`, and `v_cache` as immutable slices, and
    /// `att` and `out` as disjoint mutable slices for causal multi-head attention.
    #[inline(always)]
    pub fn attention_slices(
        &mut self,
        q: ArenaSlice,
        k_cache: ArenaSlice,
        v_cache: ArenaSlice,
        att: ArenaSlice,
        out: ArenaSlice,
    ) -> (&[f32], &[f32], &[f32], &mut [f32], &mut [f32]) {
        assert!(
            q.end() <= self.capacity
                && k_cache.end() <= self.capacity
                && v_cache.end() <= self.capacity
                && att.end() <= self.capacity
                && out.end() <= self.capacity,
            "ArenaSlice out of bounds"
        );
        assert!(
            !att.overlaps(&out)
                && !att.overlaps(&q)
                && !att.overlaps(&k_cache)
                && !att.overlaps(&v_cache)
                && !out.overlaps(&q)
                && !out.overlaps(&k_cache)
                && !out.overlaps(&v_cache),
            "attention_slices: mutable outputs must not overlap inputs or each other"
        );
        let base = self.ptr.as_ptr();
        // SAFETY: Bounds and non-overlap of mutable slices verified above.
        unsafe {
            (
                std::slice::from_raw_parts(base.add(q.offset), q.len),
                std::slice::from_raw_parts(base.add(k_cache.offset), k_cache.len),
                std::slice::from_raw_parts(base.add(v_cache.offset), v_cache.len),
                std::slice::from_raw_parts_mut(base.add(att.offset), att.len),
                std::slice::from_raw_parts_mut(base.add(out.offset), out.len),
            )
        }
    }
}

impl Drop for AlignedArena {
    fn drop(&mut self) {
        // SAFETY: `self.ptr` was allocated with `self.layout` via `alloc_zeroed`.
        unsafe {
            dealloc(self.ptr.as_ptr() as *mut u8, self.layout);
        }
    }
}

pub struct ThreadCountingAllocator;

thread_local! {
    static ALLOC_COUNT: std::cell::Cell<usize> = const { std::cell::Cell::new(0) };
    static DEALLOC_COUNT: std::cell::Cell<usize> = const { std::cell::Cell::new(0) };
    static ALLOC_BYTES: std::cell::Cell<usize> = const { std::cell::Cell::new(0) };
}

unsafe impl std::alloc::GlobalAlloc for ThreadCountingAllocator {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        ALLOC_COUNT.with(|c| c.set(c.get() + 1));
        ALLOC_BYTES.with(|b| b.set(b.get().saturating_add(layout.size())));
        unsafe { std::alloc::System.alloc(layout) }
    }

    unsafe fn alloc_zeroed(&self, layout: Layout) -> *mut u8 {
        ALLOC_COUNT.with(|c| c.set(c.get() + 1));
        ALLOC_BYTES.with(|b| b.set(b.get().saturating_add(layout.size())));
        unsafe { std::alloc::System.alloc_zeroed(layout) }
    }

    unsafe fn realloc(&self, ptr: *mut u8, layout: Layout, new_size: usize) -> *mut u8 {
        ALLOC_COUNT.with(|c| c.set(c.get() + 1));
        ALLOC_BYTES.with(|b| b.set(b.get().saturating_add(new_size)));
        unsafe { std::alloc::System.realloc(ptr, layout, new_size) }
    }

    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        DEALLOC_COUNT.with(|c| c.set(c.get() + 1));
        unsafe { std::alloc::System.dealloc(ptr, layout) }
    }
}

#[global_allocator]
static GLOBAL_ALLOC: ThreadCountingAllocator = ThreadCountingAllocator;

/// Returns `(alloc_count, dealloc_count, allocated_bytes)` on the calling thread.
pub fn current_thread_heap_stats() -> (usize, usize, usize) {
    (
        ALLOC_COUNT.with(|c| c.get()),
        DEALLOC_COUNT.with(|c| c.get()),
        ALLOC_BYTES.with(|b| b.get()),
    )
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;
    use crate::quant::blocks::{BlockQ4_0, QK4_0, quantize_row_q4_0, quantize_row_q8_0};
    use crate::quant::kernels::{gemv_q4_0, gemv_q4_0_arena};
    use bytemuck::Zeroable;

    pub(crate) fn current_thread_heap_ops() -> (usize, usize) {
        let (a, d, _) = current_thread_heap_stats();
        (a, d)
    }

    #[test]
    fn test_arena_alignment_and_sub_slices() {
        let mut arena = AlignedArena::with_capacity(512);
        assert_eq!((arena.as_slice().as_ptr() as usize) % ARENA_ALIGN, 0);

        let s1 = arena.alloc_slice(33);
        let s2 = arena.alloc_slice(17);
        let s3 = arena.alloc_q8_scratch(8);
        let s4 = arena.alloc_slice(64);

        for s in [s1, s2, s3, s4] {
            let ptr = arena.get(s).as_ptr() as usize;
            assert_eq!(
                ptr % ARENA_ALIGN,
                0,
                "ArenaSlice {:?} base pointer {:#x} is not 64-byte aligned",
                s,
                ptr
            );
        }
    }

    #[test]
    #[should_panic(expected = "overlaps")]
    fn test_disjoint_borrow_overlap_detection() {
        let mut arena = AlignedArena::with_capacity(128);
        let s1 = arena.alloc_slice(32);
        let overlapping = ArenaSlice {
            offset: s1.offset + 16,
            len: 32,
        };
        let _ = arena.io_slices(s1, overlapping);
    }

    #[test]
    fn test_zero_heap_allocations_in_multi_layer_gemv() {
        let dim = 256;
        let num_layers = 4;
        let blocks_per_row = dim / QK4_0;
        let blocks_per_layer = dim * blocks_per_row;

        // Pre-allocate and quantize 4 layer matrices
        let mut layers = Vec::with_capacity(num_layers);
        for layer_idx in 0..num_layers {
            let f32_weights: Vec<f32> = (0..(dim * dim))
                .map(|i| (((i + layer_idx * 97) as f32) * 0.013).sin() * 0.25)
                .collect();
            let mut q4_weights = vec![BlockQ4_0::zeroed(); blocks_per_layer];
            quantize_row_q4_0(&f32_weights, &mut q4_weights);
            layers.push(q4_weights);
        }

        let init_input: Vec<f32> = (0..dim).map(|i| ((i as f32) * 0.03).cos()).collect();

        // Pre-allocate static activation arena and slices
        let mut arena = AlignedArena::with_capacity(1024);
        let ping = arena.alloc_slice(dim);
        let pong = arena.alloc_slice(dim);
        let q8_scratch = arena.alloc_q8_scratch(blocks_per_row);

        arena.get_mut(ping).copy_from_slice(&init_input);

        // Warm up TLS counter and snapshot heap operation counts before the hot loop
        let ops_before = current_thread_heap_ops();

        // Execute 10 tokens x 4-layer GEMV forward pass strictly inside the arena
        for _step in 0..10 {
            gemv_q4_0_arena(&mut arena, &layers[0], ping, q8_scratch, pong, dim, dim);
            gemv_q4_0_arena(&mut arena, &layers[1], pong, q8_scratch, ping, dim, dim);
            gemv_q4_0_arena(&mut arena, &layers[2], ping, q8_scratch, pong, dim, dim);
            gemv_q4_0_arena(&mut arena, &layers[3], pong, q8_scratch, ping, dim, dim);
        }

        let ops_after = current_thread_heap_ops();
        assert_eq!(
            ops_after, ops_before,
            "Expected 0 heap allocs/deallocs during hot GEMV loop, found {:?}",
            (ops_after.0 - ops_before.0, ops_after.1 - ops_before.1)
        );

        // Verify numerical equivalence against a standalone reference pass for 1 full 4-layer step
        arena.get_mut(ping).copy_from_slice(&init_input);
        gemv_q4_0_arena(&mut arena, &layers[0], ping, q8_scratch, pong, dim, dim);
        gemv_q4_0_arena(&mut arena, &layers[1], pong, q8_scratch, ping, dim, dim);
        gemv_q4_0_arena(&mut arena, &layers[2], ping, q8_scratch, pong, dim, dim);
        gemv_q4_0_arena(&mut arena, &layers[3], pong, q8_scratch, ping, dim, dim);

        let mut ref_a = init_input.clone();
        let mut ref_b = vec![0.0f32; dim];
        let mut ref_q8 = vec![BlockQ8_0::zeroed(); blocks_per_row];
        for layer in &layers {
            quantize_row_q8_0(&ref_a, &mut ref_q8);
            gemv_q4_0(layer, &ref_q8, &mut ref_b, dim, dim);
            ref_a.copy_from_slice(&ref_b);
        }

        assert_eq!(arena.get(ping), ref_a.as_slice());
    }
}
