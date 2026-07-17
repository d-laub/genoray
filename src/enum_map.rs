//! One array-backed fixed-size map keyed by a small enum. Shared by `dense.rs`
//! (DenseClass) and `streams.rs` (StreamTag) — see EnumKey impls there.

use std::marker::PhantomData;

/// A small closed enum usable as a fixed-size map key. `ALL` lists every
/// variant in `index()` order; `COUNT == ALL.len()`.
pub trait EnumKey: Copy + 'static {
    const COUNT: usize;
    const ALL: &'static [Self];
    fn index(self) -> usize;
}

/// Fixed-size map keyed by `K`, backed by an array (O(1), no hashing).
/// `N` is pinned to `K::COUNT` by the concrete type aliases.
#[derive(Debug, PartialEq)]
pub struct EnumMap<K: EnumKey, T, const N: usize> {
    slots: [T; N],
    _k: PhantomData<K>,
}

impl<K: EnumKey, T, const N: usize> EnumMap<K, T, N> {
    pub fn from_fn(mut f: impl FnMut(K) -> T) -> Self {
        Self {
            slots: std::array::from_fn(|i| f(K::ALL[i])),
            _k: PhantomData,
        }
    }
    #[inline]
    pub fn get(&self, k: K) -> &T {
        &self.slots[k.index()]
    }
    #[inline]
    pub fn get_mut(&mut self, k: K) -> &mut T {
        &mut self.slots[k.index()]
    }
    pub fn iter(&self) -> impl Iterator<Item = (K, &T)> {
        K::ALL.iter().copied().zip(self.slots.iter())
    }
    pub fn iter_mut(&mut self) -> impl Iterator<Item = (K, &mut T)> {
        K::ALL.iter().copied().zip(self.slots.iter_mut())
    }
    pub fn into_iter_tagged(self) -> impl Iterator<Item = (K, T)> {
        K::ALL.iter().copied().zip(self.slots)
    }
}
