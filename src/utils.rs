/// Converts a 3D index [v, s, p] into a flat 1D memory pointer
#[macro_export]
macro_rules! ravel {
    ($shape:expr, $idx:expr) => {{
        ($idx[0] * $shape[1] * $shape[2]) + ($idx[1] * $shape[2]) + $idx[2]
    }};
}

/// Converts a flat 1D memory pointer back into a 3D index [v, s, p]
#[macro_export]
macro_rules! unravel {
    ($shape:expr, $flat_idx:expr) => {{
        let p = $flat_idx % $shape[2];
        let mut temp = $flat_idx / $shape[2];
        let s = temp % $shape[1];
        let v = temp / $shape[1];
        [v, s, p]
    }};
}