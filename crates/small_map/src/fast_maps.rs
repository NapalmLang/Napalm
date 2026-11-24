//! Centralized FastMap/FastSet backend used by the Napalm runtime and SmallMap.
//! The default hasher is Napalm's TLS-seeded AES builder, not hashbrown's default.

use crate::runtime_hash::AesBuildHasher;

/// The engine's keyed FastMap. `S` remains configurable for compatibility with
/// existing SmallMap APIs; omitting it uses Napalm's keyed AES/aHash policy.
pub type FastMap<K, V, S = AesBuildHasher> = hashbrown::HashMap<K, V, S>;

/// The engine's keyed FastSet, with the same builder policy.
pub type FastSet<K, S = AesBuildHasher> = hashbrown::HashSet<K, S>;

pub use hashbrown::Equivalent;
pub type FastMapIter<'a, K, V> = hashbrown::hash_map::Iter<'a, K, V>;
pub type FastMapIntoIter<K, V> = hashbrown::hash_map::IntoIter<K, V>;
