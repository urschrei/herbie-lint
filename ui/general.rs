// Test for herbie lint: numerically unstable expressions

#![allow(dead_code)]
#![allow(unused_variables)]

fn main() {
    let (a, b) = (1.0_f64, 2.0_f64);

    // sqrt(a*a + b*b) should become a.hypot(b)
    let _ = (a * a + b * b).sqrt();

    // (a + 1.0).ln() should become a.ln_1p()
    let _ = (a + 1.0).ln();

    // a.exp() - 1.0 should become a.exp_m1()
    let _ = a.exp() - 1.0;
}
