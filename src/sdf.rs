#![allow(non_snake_case)]
#![allow(dead_code)]

use core::f64;

use num_traits::AsPrimitive;
use partial_application::partial;

#[inline]
fn tuple4map<T, U>(f: impl Fn(T) -> U, (a, b, c, d): (T, T, T, T)) -> (U, U, U, U) {
    (f(a), f(b), f(c), f(d))
}

fn triangleNumber(i: i64) -> i64 {
    if 0x8000000 < i as u64 {
        -1
    } else {
        ((i * (i - 1)) as u64 >> 1) as i64
    }
} // keep it in the domain where f64 math can take the triangleRoot correctly
fn triangleRoot(i: i64) -> i64 {
    // the accurate range could be extended with newton iteration in u64 math but the range f64 gives us should be plenty if sqrt is decent
    if 0x20000003ffffff <= i as u64 {
        -1
    } else {
        // bound is correct on hardware where f64.sqrt is correctly-rounded; might be smaller elsewhere
        (0.5 + ((2 * i) as f64 + 0.25).sqrt()) as i64
    } // assumes i64.f64 always rounds positive numbers down
}
fn triangleIndex<A: Copy>(xs: &[A], i: i64) -> (A, A) {
    let j = triangleRoot(i);
    (xs[(i - triangleNumber(j)) as usize], xs[j as usize])
}

trait FloatCbrt {
    fn ilog2AbsNormal(self) -> i32;
    fn premulDepressedCubic_computeScaleExponent(self, b: Self) -> i32;
    fn rcbrtPositiveNormalApprox(self) -> Self;
    fn rcbrtPositiveNormal(self) -> Self;
    fn pow2i(q: i32) -> Self;
}

fn from_fraction<T: Copy + 'static + std::ops::Div<Output = T>>(n: i64, d: i64) -> T
where
    i64: num_traits::AsPrimitive<T>,
{
    n.as_() / d.as_()
}

impl FloatCbrt for f32 {
    fn ilog2AbsNormal(self) -> i32 {
        (255 & (self.to_bits() >> 23) as i32) - 127
    }

    fn premulDepressedCubic_computeScaleExponent(self, c1divn3: Self) -> i32 {
        let a = (127 & (self.to_bits() >> 24) as i32) - 82;
        let b = (((255 << 23) & c1divn3.to_bits()) / (3 << 23)) as i32 - 61;
        a.max(b)
    }

    fn rcbrtPositiveNormalApprox(self) -> Self {
        // u32 division by constant 3 is compiled to multiply-then-shift by many compilers, so this should be division-free.
        // If you care about ulp error instead of relative error, use 0x54a21d29 instead.
        let x = self;
        let y = f32::from_bits(0x54a208f8 - x.to_bits() / 3);
        let p = (x * y).mul_add(y * y, -1.0); // p = x * y**3 - 1 is the expression we're finding a zero of.
        // Using that definition of p, the true answer x**(-1/3) == ((p+1)**(-1/3)-1) * y + y in exact arithmetic.
        // Our initial guess is ok enough that p is close-ish to 0, so we use a Taylor approximation of (p+1)**(-1/3)-1 centered at 0.
        // The order n Taylor approximation gives order n+1 convergence. We use n=4 here.
        // More accurate than trisectApprox. -20.44ulp < err < 17.34ulp; -2**-19.38 < relerr < 2**-19.38.
        // 4 fewer flops than the |err|<1ulp version. 3 fewer flops than the |err|<2ulp version.
        // A minimax polynomial over the appropriate interval would be better & might allow saving a flop, but this is good enough.
        p.mul_add(
            p.mul_add(
                p.mul_add(from_fraction(35, 243), from_fraction(-14, 81)),
                from_fraction(2, 9),
            ),
            from_fraction(-1, 3),
        )
        .mul_add(p * y, y)
    }

    fn rcbrtPositiveNormal(self) -> Self {
        let x = self;
        let y = f32::from_bits(0x54a232a8 - x.to_bits() / 3);
        let p = (x * y).mul_add(y * y, -1.0);
        let y = p
            .mul_add(from_fraction(2, 9), from_fraction(-1, 3))
            .mul_add(p * y, y);
        let p = (x * y).mul_add(y * y, -1.0);
        //f32.(fma p (y * from_fraction (-1) 3) y) // Slightly biased toward 0. -1.77ulp < err < +0.96ulp. -2**-22.68 < relerr < 2**-23.51.
        p.mul_add(from_fraction(2, 9), from_fraction(-1, 3))
            .mul_add(p * y, y) // -0.99ulp < err < +0.99ulp. -2**-23.44 < relerr < 2**-23.44.
    }

    fn pow2i(q: i32) -> Self {
        f32::from_bits(((q + 0x7f) as u32) << 23)
    }
}

const fn rsqrtPosNormal(x: f32) -> f32 {
    let y = f32::from_bits(0x5f375405u32 - (x.to_bits() >> 1));
    let p = (x * y).mul_add(y, -1.0);
    p.mul_add(0.2746582_f32, -0.31393552_f32)
        .mul_add(p, 0.3749987_f32)
        .mul_add(p, -0.49999833_f32)
        .mul_add(p * y, y)
}

impl FloatCbrt for f64 {
    fn ilog2AbsNormal(self) -> i32 {
        (2047 & (self.to_bits() >> 52) as i32) - 1023
    }

    fn premulDepressedCubic_computeScaleExponent(self, c1divn3: Self) -> i32 {
        let a = (1023 & (self.to_bits() >> 53) as i32) - 679;
        let b = ((c1divn3.to_bits() >> 52) & 2047) as i32 / 3 - 509;
        a.max(b)
    }

    fn rcbrtPositiveNormalApprox(self) -> Self {
        let x = self;
        // Magic number found by bisection search using a bunch of sample points. Might not be optimal but it's close enough.
        // My compiler also converts this division into a multiply that keeps the high bits; if yours doesn't, consider using mul_hi or mad_hi.
        let y = f64::from_bits(0x553eebe7c0a5ceb0 - x.to_bits() / 3);
        let p = (x * y).mul_add(y * y, -1.0);
        // Again, only trying to be better than trisectApprox here.
        p.mul_add(
            p.mul_add(
                p.mul_add(from_fraction(35, 243), from_fraction(-14, 81)),
                from_fraction(2, 9),
            ),
            from_fraction(-1, 3),
        )
        .mul_add(p * y, y)
    }
    fn rcbrtPositiveNormal(self) -> Self {
        let y = self.rcbrtPositiveNormalApprox();
        let p = (self * y).mul_add(y * y, -1.0);
        // As far as I can easily tell, the following might be within 1 ulp of correct, but I have not checked it against higher precision yet.
        p.mul_add(
            p.mul_add(
                p.mul_add(from_fraction(35, 243), from_fraction(-14, 81)),
                from_fraction(2, 9),
            ),
            from_fraction(-1, 3),
        )
        .mul_add(p * y, y)
    }

    fn pow2i(q: i32) -> Self {
        Self::from_bits(((q + 0x3ff) as u64) << 52)
    }
}

/*
module Complex = {
  fn conj (z: Complex) = (z.0, Real.neg z.1)
  fn neg = tuple2map Real.neg
  fn (+) = tuple2map2 (Real.+)
  fn (-) = tuple2map2 (Real.-)
  fn mulAccurate (a: Complex) (b: Complex) = (prodDiffAccurate(a.0, b.0, a.1, b.1), dot2dAccurate a (b.1, b.0))
  fn (*) (a: Complex) (b: Complex) = (dot2d a (conj b), dot2d a (b.1, b.0))
  fn scaleBy x = tuple2map ((Real.*) x)
  fn squaredMag z = dot2d z z
  fn mag z = Real.sqrt (squaredMag z)
  fn recipMag z = rsqrtPosNormal(z.squaredMag())
  fn normalize z = scaleBy (recipMag z) z
  fn mulAdd (a: Complex) (b: Complex) (c: Complex) : Complex = // I would not grant this the title of without.mul_add(more, effort) to ensure tight error bounds.
    Real.((fma (neg a.1) b.1 (fma a.0 b.0 c.0), fma a.1 b.0 (fma a.0 b.1 c.1)))
  fn nan : Complex = (Real.nan, Real.nan)
}*/

type Real = f32;

#[allow(clippy::derive_ord_xor_partial_ord)]
#[derive(
    Clone,
    Copy,
    Debug,
    PartialEq,
    PartialOrd,
    bytemuck::Pod,
    bytemuck::Zeroable,
    zerocopy::IntoBytes,
    zerocopy::Immutable,
)]
#[repr(C)]
pub struct Complex {
    pub x: f32,
    pub y: f32,
}

impl Eq for Complex {}

impl Ord for Complex {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        self.partial_cmp(other).unwrap()
    }
}

impl From<[f32; 2]> for Complex {
    fn from([x, y]: [f32; 2]) -> Self {
        Self { x, y }
    }
}

impl From<(f32, f32)> for Complex {
    fn from((x, y): (f32, f32)) -> Self {
        Self { x, y }
    }
}

impl std::ops::Sub for Complex {
    type Output = Self;

    fn sub(self, rhs: Self) -> Self::Output {
        Self {
            x: self.x - rhs.x,
            y: self.y - rhs.y,
        }
    }
}

impl std::ops::Add for Complex {
    type Output = Self;

    fn add(self, rhs: Self) -> Self::Output {
        Self {
            x: self.x + rhs.x,
            y: self.y + rhs.y,
        }
    }
}

pub const BOUNDARY_THRESHOLD: Real = 0.25; // Max distance from boundary to be considered "on the boundary". One could imagine choosing this differently or even dynamically, but here's a guess.

const fn keepFinite(backup: Real, preferred: Real) -> Real {
    if preferred.is_finite() {
        preferred
    } else {
        backup
    }
}

// see https://pharr.org/matt/blog/2019/11/03/difference-of-floats or "Further Analysis of Kahan's Algorithm for the Accurate Computation of 2x2 Determinants"
const fn dot2dAccurate(a: Complex, b: Complex) -> Real {
    let t = a.x * b.x;
    a.y.mul_add(b.y, t) + a.x.mul_add(b.x, -t)
}

const fn dot2d(a: Complex, b: Complex) -> Real {
    a.y.mul_add(b.y, a.x * b.x)
}
const fn prodDiffAccurate(a: Real, b: Real, c: Real, d: Real) -> Real {
    let t = -(c * d);
    a.mul_add(b, t) - c.mul_add(d, t)
}
const fn det2x2Accurate(a: Complex, b: Complex) -> Real {
    prodDiffAccurate(a.x, b.y, a.y, b.x)
}
const fn det2x2(a: Complex, b: Complex) -> Real {
    a.x.mul_add(b.y, -(a.y * b.x))
}

const COMPLEX_NAN: Complex = Complex::new(Real::NAN, Real::NAN);

impl Complex {
    pub const fn new(x: f32, y: f32) -> Self {
        Self { x, y }
    }
    pub const fn conj(self) -> Self {
        Self::new(self.x, -self.y)
    }
    pub const fn mulAccurate(self, b: Self) -> Self {
        Self::new(
            prodDiffAccurate(self.x, b.x, self.y, b.y),
            dot2dAccurate(self, Self::new(b.y, b.x)),
        )
    }
    pub const fn scaleBy(self, x: Real) -> Self {
        Self::new(self.x * x, self.y * x)
    }
    pub const fn squaredMag(self) -> Real {
        dot2d(self, self)
    }
    pub fn mag(self) -> Real {
        self.squaredMag().sqrt()
    }
    pub const fn recipMag(self) -> Real {
        rsqrtPosNormal(self.squaredMag())
    }
    pub const fn normalize(self) -> Self {
        self.scaleBy(self.recipMag())
    }
    pub const fn mulAdd(self, b: Self, c: Self) -> Self {
        Self::new(
            (-self.y).mul_add(b.y, self.x.mul_add(b.x, c.x)),
            self.y.mul_add(b.x, self.x.mul_add(b.y, c.y)),
        )
    }
    pub const fn mul(self, b: Self) -> Self {
        Self::new(dot2d(self, b.conj()), dot2d(self, Self::new(b.y, b.x)))
    }
}

pub trait AsArrayRef<const N: usize>
where
    Self: bytemuck::Pod,
{
    fn as_array(&self) -> &[f32; N] {
        bytemuck::cast_ref(self)
    }
}

#[derive(Clone, Copy, PartialEq, PartialOrd, Debug, bytemuck::Pod, bytemuck::Zeroable)]
#[repr(C)]
pub struct Circle {
    pub center: Complex,
    pub radius: Real,
}

impl AsArrayRef<3> for Circle {}

#[derive(Clone, Copy, PartialEq, PartialOrd, Debug, bytemuck::Pod, bytemuck::Zeroable)]
#[repr(C)]
pub struct HalfPlane {
    pub normal: Complex,
    pub shift: Real,
} // NOTICE: some algoithms here assume the normal has magnitude 1. If the magnitude is significantly different from 1, you will get wrong results!

impl AsArrayRef<3> for HalfPlane {}

#[derive(Clone, Copy, PartialEq, PartialOrd, Debug, bytemuck::Pod, bytemuck::Zeroable)]
#[repr(C)]
struct Bezier2o1d(pub Real, pub Real, pub Real);

impl AsArrayRef<3> for Bezier2o1d {}

#[derive(Clone, Copy, PartialEq, PartialOrd, Debug, bytemuck::Pod, bytemuck::Zeroable)]
#[repr(C)]
pub struct Bezier2o2d(pub Complex, pub Complex, pub Complex);

impl AsArrayRef<6> for Bezier2o2d {}

// degree n polynomial in n fma operations
// degree n polynomial in n fma operations
pub fn linear_eval(c0: Real, c1: Real, t: Real) -> Real {
    t.mul_add(c1, c0)
}

pub fn quadratic_eval(c0: Real, c1: Real, c2: Real, t: Real) -> Real {
    t.mul_add(t.mul_add(c2, c1), c0)
}

pub fn cubic_eval(c0: Real, c1: Real, c2: Real, c3: Real, t: Real) -> Real {
    t.mul_add(t.mul_add(t.mul_add(c3, c2), c1), c0)
}

pub fn quartic_eval(c0: Real, c1: Real, c2: Real, c3: Real, c4: Real, t: Real) -> Real {
    t.mul_add(t.mul_add(t.mul_add(t.mul_add(c4, c3), c2), c1), c0)
}

// monic version replaces 1 with.mul_add(addition, and) doesn't load the highest-degree coefficient of 1 but overall same number of flops and same accuracy
fn monicLinear_eval(x: Real, y: Real) -> Real {
    x + y
}
pub fn monic_quadratic_eval(c0: Real, c1: Real, t: Real) -> Real {
    t.mul_add(t + c1, c0)
}

pub fn monic_cubic_eval(c0: Real, c1: Real, c2: Real, t: Real) -> Real {
    t.mul_add(t.mul_add(t + c2, c1), c0)
}

pub fn monic_quartic_eval(c0: Real, c1: Real, c2: Real, c3: Real, t: Real) -> Real {
    t.mul_add(t.mul_add(t.mul_add(t + c3, c2), c1), c0)
}

// depressed polynomials have highest-degree coefficient 1, next-highest-degree coefficient 0, and evaluate in 1 less flop with the same accuracy
pub fn depressed_quadratic_eval(c0: Real, t: Real) -> Real {
    t.mul_add(t, c0)
}

pub fn depressed_cubic_eval(c0: Real, c1: Real, t: Real) -> Real {
    t.mul_add(t.mul_add(t, c1), c0)
}

pub fn depressed_quartic_eval(c0: Real, c1: Real, c2: Real, t: Real) -> Real {
    t.mul_add(t.mul_add(t.mul_add(t, c2), c1), c0)
}

// degree n polynomial and its first derivative in 2*n - 1 fma operations
fn next_coef1(t: Real, (p, v): (Real, Real), c: Real) -> (Real, Real) {
    (t.mul_add(p, c), t.mul_add(v, p))
}
pub fn linear_eval1(c0: Real, c1: Real, t: Real) -> (Real, Real) {
    (t.mul_add(c1, c0), c1)
}

// computing the derivative as (t.mul_add(c2 + c2, c1)) would spend an extra operation to guarantee a correctly-rounded result
fn quadratic_eval1(c0: Real, c1: Real, c2: Real, t: Real) -> (Real, Real) {
    next_coef1(t, linear_eval1(c1, c2, t), c0)
}

pub fn cubic_eval1(c0: Real, c1: Real, c2: Real, c3: Real, t: Real) -> (Real, Real) {
    next_coef1(t, quadratic_eval1(c1, c2, c3, t), c0)
}

pub fn quartic_eval1(c0: Real, c1: Real, c2: Real, c3: Real, c4: Real, t: Real) -> (Real, Real) {
    next_coef1(t, cubic_eval1(c1, c2, c3, c4, t), c0)
}

// computing the derivative as (t + t) + c1 would spend an extra operation to guarantee a correctly-rounded result
pub fn monic_quadratic_eval1(c0: Real, c1: Real, t: Real) -> (Real, Real) {
    let q = t + c1;
    (t.mul_add(q, c0), t + q)
}

//fn monicCubic_eval1(c0: Real, c1: Real, c2: Real, t: Real) -> Real { nextCoef1 t (monicQuadratic_eval1 c1 c2 t) c0
// loads a constant, then uses 6 ops instead of 5, but less rounding error
pub fn monic_cubic_eval1(c0: Real, c1: Real, c2: Real, t: Real) -> (Real, Real) {
    (
        t.mul_add(t.mul_add(t + c2, c1), c0),
        t.mul_add(t.mul_add(3.0, c2 + c2), c1),
    )
}

pub fn monic_quartic_eval1(c0: Real, c1: Real, c2: Real, c3: Real, t: Real) -> (Real, Real) {
    next_coef1(t, monic_cubic_eval1(c1, c2, c3, t), c0)
}

pub fn depressed_quadratic_eval1(c0: Real, t: Real) -> (Real, Real) {
    (t.mul_add(t, c0), t + t)
}

pub fn depressed_cubic_eval1(c0: Real, c1: Real, t: Real) -> (Real, Real) {
    (t.mul_add(t.mul_add(t, c1), c0), t.mul_add(t * 3.0, c1))
} // 4 ops instead of 5 for cubic_eval1, same accuracy

// 6 ops instead of 7 for quartic_eval1
pub fn depressed_quartic_eval1(c0: Real, c1: Real, c2: Real, t: Real) -> (Real, Real) {
    (
        t.mul_add(t.mul_add(t.mul_add(t, c2), c1), c0), // same accuracy
        {
            let u = t + t;
            u.mul_add(u.mul_add(t, c2), c1) // better accuracy
        },
    )
}

// coefficients for degree 2 taylor expansion centered at t of degree n polynomial in 3*n - 3 fma operations
pub fn next_coef2(t: Real, (p, v, adiv2): (Real, Real, Real), c: Real) -> (Real, Real, Real) {
    (t.mul_add(p, c), t.mul_add(v, p), t.mul_add(adiv2, v))
}
pub fn quadratic_eval2(c0: Real, c1: Real, c2: Real, t: Real) -> (Real, Real, Real) {
    let q = t.mul_add(c2, c1);
    (t.mul_add(q, c0), t.mul_add(c2, q), c2)
}

pub fn cubic_eval2(c0: Real, c1: Real, c2: Real, c3: Real, t: Real) -> (Real, Real, Real) {
    next_coef2(t, quadratic_eval2(c1, c2, c3, t), c0)
}

pub fn quartic_eval2(
    c0: Real,
    c1: Real,
    c2: Real,
    c3: Real,
    c4: Real,
    t: Real,
) -> (Real, Real, Real) {
    next_coef2(t, cubic_eval2(c1, c2, c3, c4, t), c0)
}

fn monicCubic_eval2(c0: Real, c1: Real, c2: Real, t: Real) -> (Real, Real, Real) {
    let g = t + c2;
    let h = t.mul_add(g, c1);
    let s = t + g;
    (t.mul_add(h, c0), t.mul_add(s, h), t + s)
}

//fn monicCubic_eval2(c0: Real, c1: Real, c2: Real, t: Real) -> Real {(fma t (t.mul_add(t + c2, c1)) c0, fma t (fma t (f64 3) (c2 + c2)) c1, t.mul_add(f64 3, c2))) // loads a constant, uses 7 ops instead of 6, but less rounding error in the derivatives
fn monicQuartic_eval2(c0: Real, c1: Real, c2: Real, c3: Real, t: Real) -> (Real, Real, Real) {
    next_coef2(t, monicCubic_eval2(c1, c2, c3, t), c0)
}

// 4 ops instead of 6 for cubic_eval2, same accuracy
pub fn depressed_cubic_eval2(c0: Real, c1: Real, t: Real) -> (Real, Real, Real) {
    let adiv2 = 3.0 * t;
    (t.mul_add(t.mul_add(t, c1), c0), t.mul_add(adiv2, c1), adiv2)
}

// 8 ops instead of 9 for quartic_eval2
pub fn depressed_quartic_eval2(c0: Real, c1: Real, c2: Real, t: Real) -> (Real, Real, Real) {
    let u = t + t;
    (
        t.mul_add(t.mul_add(t.mul_add(t, c2), c1), c0), // same accuracy
        u.mul_add(u.mul_add(t, c2), c1),                // better accuracy
        u.mul_add(u + t, c2),                           // better accuracy
    )
}

// | Approximates cos ((acos x) / 3.0) using a shifted-and-scaled sqrt followed by a quartic polynomial.
// | Coefficients chosen using several iterations of the Remez algorithm subject to the restriction 0.5 <= trisectApprox x <= 1.0 whenever abs x <= 1.
// | (Some bits of the f64 coefficients had not converged when I stopped improving them, but these are certainly the best coefficients to within better than f32 precision.)
// | Undershooting 0.5 would cause some roots to swap order.
// | Exceeding 1.0 would make NaNs because this function's output is fed to something like (\x -> sqrt (1 - x*x)).
// | This definition stays in the appropriate range on IEEE754 compliant hardware for both f32 and f64, both with and without fma.
pub fn trisectApprox(x: Real) -> Real {
    let c = (
        0.5,
        0.576_978_9,
        -0.107_102_93,
        0.039_126_46,
        -0.009_002_428,
    );
    quartic_eval(c.0, c.1, c.2, c.3, c.4, x.mul_add(c.0, c.0).sqrt())
}

// | Solves c2*t*t - 2*c1divn2*t + c0 == 0 for real t.
// | Works even when c2 == 0 (as long as c1divn2 != 0), putting the result in slot 0.
// | Should be numerically stable even with small discriminant on platforms with fma.
// | Intermediate overflow to inf was deemed not worth the computation time to fix.
fn premulQuadratic_findRoots(c0: Real, c1divn2: Real, c2: Real) -> (Real, Real) {
    let q = c1divn2
        + prodDiffAccurate(c1divn2, c1divn2, c0, c2)
            .sqrt()
            .copysign(c1divn2);
    (c0 / q, q / c2)
}

fn quadratic_findRoots(c0: Real, c1: Real, c2: Real) -> (Real, Real) {
    premulQuadratic_findRoots(c0, c1 * -0.5, c2)
}

// | If c2 == 1, then there are fewer places where precision-loss can happen, so the discriminant calculation can be shorter.
fn premulMonicQuadratic_findRoots(c0: Real, c1divn2: Real) -> (Real, Real) {
    let q = c1divn2 + c1divn2.mul_add(c1divn2, -c0).sqrt().copysign(c1divn2);
    (c0 / q, q)
}
fn monicQuadratic_findRoots(c0: Real, c1: Real) -> (Real, Real) {
    premulMonicQuadratic_findRoots(c0, c1 * -0.5)
}

fn newtonOnceIfBetter(
    eval1: impl Fn(Real) -> (Real, Real),
    eval: impl Fn(Real) -> Real,
    t: Real,
) -> Real {
    let (p, v) = eval1(t);
    let t1 = t - p / v;
    if eval(t1).abs() <= p.abs() { t1 } else { t }
}

fn halleyOnceIfBetter(
    eval2: impl Fn(Real) -> (Real, Real, Real),
    eval: impl Fn(Real) -> Real,
    t: Real,
) -> Real {
    let (p, v, adiv2) = eval2(t);
    let t1 = p.mul_add(v / (-v).mul_add(v, p * adiv2), t);
    if eval(t1).abs() <= p.abs() { t1 } else { t }
}

fn cubic_newtonOnceIfBetter(c0: Real, c1: Real, c2: Real, c3: Real, t: Real) -> Real {
    newtonOnceIfBetter(
        partial!(move cubic_eval1 => c0, c1, c2, c3, _),
        partial!(move cubic_eval => c0, c1, c2, c3, _),
        t,
    )
}
fn cubic_halleyOnceIfBetter(c0: Real, c1: Real, c2: Real, c3: Real, t: Real) -> Real {
    halleyOnceIfBetter(
        partial!(move cubic_eval2 =>  c0, c1, c2, c3, _),
        partial!(move cubic_eval =>  c0, c1, c2, c3, _),
        t,
    )
}
fn monicCubic_newtonOnceIfBetter(c0: Real, c1: Real, c2: Real, t: Real) -> Real {
    newtonOnceIfBetter(
        partial!(move monic_cubic_eval1 =>  c0, c1, c2, _),
        partial!(move monic_cubic_eval =>  c0, c1, c2, _),
        t,
    )
}
fn monicCubic_halleyOnceIfBetter(c0: Real, c1: Real, c2: Real, t: Real) -> Real {
    halleyOnceIfBetter(
        partial!(move monicCubic_eval2 =>  c0, c1, c2, _),
        partial!(move monic_cubic_eval =>  c0, c1, c2, _),
        t,
    )
}
fn depressedCubic_newtonOnceIfBetter(c0: Real, c1: Real, t: Real) -> Real {
    newtonOnceIfBetter(
        partial!(move depressed_cubic_eval1 =>  c0, c1, _),
        partial!(move depressed_cubic_eval =>  c0, c1, _),
        t,
    )
}
fn depressedCubic_halleyOnceIfBetter(c0: Real, c1: Real, t: Real) -> Real {
    halleyOnceIfBetter(
        partial!(move depressed_cubic_eval2 =>  c0, c1, _),
        partial!(move depressed_cubic_eval =>  c0, c1, _),
        t,
    )
}
fn quartic_newtonOnceIfBetter(c0: Real, c1: Real, c2: Real, c3: Real, c4: Real, t: Real) -> Real {
    newtonOnceIfBetter(
        partial!(move quartic_eval1 =>  c0, c1, c2, c3, c4, _),
        partial!(move quartic_eval =>  c0, c1, c2, c3, c4, _),
        t,
    )
}
fn quartic_halleyOnceIfBetter(c0: Real, c1: Real, c2: Real, c3: Real, c4: Real, t: Real) -> Real {
    halleyOnceIfBetter(
        partial!(move quartic_eval2 =>  c0, c1, c2, c3, c4, _),
        partial!(move quartic_eval =>  c0, c1, c2, c3, c4, _),
        t,
    )
}
fn depressedQuartic_newtonOnceIfBetter(c0: Real, c1: Real, c2: Real, t: Real) -> Real {
    newtonOnceIfBetter(
        partial!(move depressed_quartic_eval1 =>  c0, c1, c2, _),
        partial!(move depressed_quartic_eval =>  c0, c1, c2, _),
        t,
    )
}
fn depressedQuartic_halleyOnceIfBetter(c0: Real, c1: Real, c2: Real, t: Real) -> Real {
    halleyOnceIfBetter(
        partial!(move depressed_quartic_eval2 =>  c0, c1, c2, _),
        partial!(move depressed_quartic_eval =>  c0, c1, c2, _),
        t,
    )
}

// | Solves t**3 - 3*c1divn3*t - 2*c0divn2 == 0 for real t.
// | No scaling is attempted; if it overflows, it overflows.
// | In the 3-root case, the roots are sorted (most-positive, middle, most-negative).
// | The middle root has the fewest significant bits. The most-positive root has the most.
// | If any of the roots are good, the root in slot 0 is.
#[allow(clippy::manual_clamp)]
fn premulDepressedCubic_findRoots_fast(c0divn2: Real, c1divn3: Real) -> (Real, Real, Real) {
    let d = prodDiffAccurate(c0divn2, c0divn2, c1divn3 * c1divn3, c1divn3);
    if !d.is_finite() {
        (d, d, d)
    } else {
        if 0.0 < d {
            // 1 real root
            let q = c0divn2.abs() + d.sqrt();
            // sqrt of a positive finite number is normal in all supported formats. No need for rcbrt to handle subnormals.
            let u = q.rcbrtPositiveNormalApprox();
            (
                u.copysign(c0divn2) * q.mul_add(u, c1divn3),
                Real::NAN,
                Real::NAN,
            )
            // If you're on a platform that provides a fast rcbrt that preserves sign, you can skip a step with this:
            //let q = c0divn2 + copysign (sqrt d) c0divn2;
            //let u = rcbrt q in (u * q.mul_add(u, c1divn3), Real::NAN, Real::NAN)
        } else {
            // 3 real roots counting multiplicity, unless underflow
            let h = rsqrtPosNormal(c1divn3);
            let s = c1divn3 * h;
            let u = c0divn2 * h * h * h;
            if !u.is_finite() {
                // Mostly happens when c1divn3 ** 1.5 is subnormal or 0, AND c0divn2 is too close to 0 to make d positive. In those cases, the equation is very close to x**3 == 0.
                //(cbrt (c0divn2 + c0divn2), Real::NAN, Real::NAN) // Even in f32 with subnormal-flushing and no fma, abs(exact answer) in this codepath is always < 2 ** -20.6, so we just round to 0.
                (0.0, Real::NAN, Real::NAN) // If you're finding a quadratic Bezier parameter without rescaling, you need > 2 ** 17.6 px between control points before this produces 1/4 px of error.
            } else {
                let u = u.max(-1.0).min(1.0); // This line only does anything if u was computed inaccurately AND 2 roots were close together, but I don't know a way to rule that out.
                let a = trisectApprox(u);
                let b = (-a).mul_add(a * 3.0, 3.0).sqrt(); // Not needed for the most-positive root so can be skipped if you only need that.
                //let v = acos u * from_fraction 1 3;
                //let a = cos v in // Use this instead if your hardware makes acos, cos, and sin particularly fast & accurate
                //let b = sin v * sqrt (f64 3);
                let (sa, sb) = (s * a, s * b);
                (sa + sa, sb - sa, -sb - sa)
            }
        }
    }
}

// | Like premulDepressedCubic_findRoots_fast but does a bit of extra work to avoid overflow and underflow.
fn premulDepressedCubic_findRoots_stopOverflow(c0divn2: Real, c1divn3: Real) -> (Real, Real, Real) {
    let scaleExponent = c0divn2.premulDepressedCubic_computeScaleExponent(c1divn3);
    let scale = Real::pow2i(-scaleExponent);
    let sc0divn2 = (c0divn2 * scale) * (scale * scale);
    let sc1divn3 = c1divn3 * (scale * scale);
    let unscale = Real::pow2i(scaleExponent);
    let (a, b, c) = premulDepressedCubic_findRoots_fast(sc0divn2, sc1divn3);
    (a * unscale, b * unscale, c * unscale)
}

// | Solves c3*t**3 + c2*t**2 + c1*t + c0 == 0 for real t.
// | If there's only 1 root, it's in slot 0, but the other root-order guarantees may be broken.
fn cubic_findRoots(c0: Real, c1: Real, c2: Real, c3: Real) -> (Real, Real, Real) {
    //if isnan c3 { (c3, c3, c3) } else {// Without this line, c3=nan ends up acting like c3=0. Maybe that's fine.
    let n3c3 = c3 * -3.0;
    let m = (c0 / (c3 * -2.0), c1 / n3c3, c2 / n3c3);
    if !(m.0.is_finite() && m.1.is_finite() && m.2.is_finite()) {
        // In this function, we mostly care about roots with magnitude <= 1.
        // In this branch, c3 is so much smaller than another coefficient that replacing
        // c3 with 0 should barely change anything about this function inside that unit disk.
        // FIXME: Not currently trying to handle potential overflow in the quadratic discriminant.
        let q = quadratic_findRoots(c0, c1, c2);
        (q.0, q.1, Real::NAN)
    } else {
        let d0divn2 = m.2.mul_add(m.2.mul_add(m.2, m.1 * 1.5), m.0);
        let d1divn3 = m.2.mul_add(m.2, m.1);
        let (a, b, c) = premulDepressedCubic_findRoots_stopOverflow(d0divn2, d1divn3);
        (a + m.2, b + m.2, c + m.2)
    }
}

// | Solves t**4 + c2*t**2 + c1*t + c0 == 0 for real t.
fn depressedQuartic_findRoots(c0: Real, c1: Real, c2: Real) -> (Real, Real, Real, Real) {
    // Before feeding the resolvent cubic to the analytic solver, it is depressed and rescaled.
    // This scaling factor was chosen to make these constants be exactly-representable values close to 1 on a log scale
    // with lots of trailing 0 bits in the mantissa to minimize the impact of rounding error.
    let c2div4 = c2 * 0.25;
    let d0divn2 = c2div4.mul_add(c2div4.mul_add(c2div4, -2.25 * c0), 0.2109375 * c1 * c1);
    let d1divn3 = c2div4.mul_add(c2div4, c0 * 0.75);
    // Since we only want the most positive root, might it be worth inlining the following function 2 layers deep to skip computing the other roots?
    let drcMostPositiveRoot = premulDepressedCubic_findRoots_stopOverflow(d0divn2, d1divn3).0;
    let r = (drcMostPositiveRoot - c2) * from_fraction::<f32>(4, 3); // undepress and rescale
    // Even the existence of real solutions is very sensitive to imprecision in the value of r. We need to recover some precision.
    // It may look tempting to move these Halley corrections to the depressed cubic to save time.
    // If you do that, your precision gains will too often be wiped out by catastrophic cancellation in the undepressing step.
    // 1 Halley step is not enough to even detect the existence of solutions in too many cases.
    // We need at least 2 here. I put in a third just in case.
    let rc0 = -(c1 * c1);
    let rc1 = c2.mul_add(c2, -4.0 * c0);
    let rc2 = c2 + c2;
    let g = r + rc2;
    let h = r.mul_add(g, rc1);
    let p = r.mul_add(h, rc0);
    let s = r + g;
    let v = r.mul_add(s, h);
    let r1 = p.mul_add(v / (-v).mul_add(v, (r + s) * p), r);
    let g = r1 + rc2;
    let h = r1.mul_add(g, rc1);
    let p1 = r1.mul_add(h, rc0);
    let (r, p) = if p1.abs() <= p.abs() {
        (r1, p1)
    } else {
        (r, p)
    }; // In the rare failed-to-improve case it is equivalent to skip to the end of Halley corrections.
    let s = r + g;
    let v = r.mul_add(s, h);
    let r1 = p.mul_add(v / (-v).mul_add(v, (r + s) * p), r);
    let g = r1 + rc2;
    let h = r1.mul_add(g, rc1);
    let p1 = r1.mul_add(h, rc0);
    let (r, p) = if p1.abs() <= p.abs() {
        (r1, p1)
    } else {
        (r, p)
    }; // In the rare failed-to-improve case it is equivalent to skip to the end of Halley corrections.
    let s = r + g;
    let v = r.mul_add(s, h);
    let r1 = p.mul_add(v / (-v).mul_add(v, (r + s) * p), r);
    let g = r1 + rc2;
    let h = r1.mul_add(g, rc1);
    let p1 = r1.mul_add(h, rc0);
    let r = if p1.abs() <= p.abs() { r1 } else { r }; // END OF HALLEY CORRECTIONS
    if r <= 0.0 {
        // c1 is so small that this quartic is effectively a quadratic in t**2
        // FIXME: Not currently trying to handle potential overflow in the quadratic discriminant.
        let (t0, t1) = monicQuadratic_findRoots(c0, c2);
        let t = (t0.sqrt(), t1.sqrt());
        (t.0, t.1, -t.0, -t.1)
    } else {
        let q = 0.5 * rsqrtPosNormal(r);
        let g = 0.5 * (c2 + r);
        let h = q * c1;
        let s = q * r;
        // FIXME: Not currently trying to handle potential overflow in the quadratic discriminants.
        let (t0, t1) = premulMonicQuadratic_findRoots(g + h, s);
        let (t2, t3) = premulMonicQuadratic_findRoots(g - h, -s);
        (t0, t1, t2, t3)
    }
}

// | Solves c4*t**4 + c3*t**3 + c2*t**2 + c1*t + c0 == 0 for real t.
pub fn quartic_findRoots(
    c0: Real,
    c1: Real,
    c2: Real,
    c3: Real,
    c4: Real,
) -> (Real, Real, Real, Real) {
    //if isnan c4 { (c4, c4, c4) } else {// Without this line, c4=nan ends up acting like c4=0. Maybe that's fine.
    let m = (c0 / c4, c1 / c4, c2 / c4, c3 / c4);
    if !(m.0.is_finite() && m.1.is_finite() && m.2.is_finite() && m.3.is_finite()) {
        // In this function, we mostly care about roots with magnitude <= 1.
        // In this branch, c4 is so much smaller than another coefficient that replacing
        // c4 with 0 should barely change anything about this function inside that unit disk.
        let t = cubic_findRoots(c0, c1, c2, c3);
        (t.0, t.1, t.2, Real::NAN)
    } else {
        let m3div2 = m.3 * 0.5;
        let m3divn4 = m.3 * (-0.25);
        let m3by3div4 = m.3 + m3divn4;
        let d0 = m3divn4.mul_add(m3divn4.mul_add(m3divn4.mul_add(m3by3div4, m.2), m.1), m.0);
        let d1 = m3div2.mul_add(m3div2.mul_add(m3div2, -m.2), m.1);
        let d2 = m3div2.mul_add(-m3by3div4, m.2);
        let t = depressedQuartic_findRoots(d0, d1, d2);
        (t.0 + m3divn4, t.1 + m3divn4, t.2 + m3divn4, t.3 + m3divn4)
    }
}

#[inline(always)]
pub fn sdf_empty() -> impl Fn(Complex) -> Real {
    move |_: Complex| Real::MAX
}
#[inline(always)]
pub fn sdf_point(p: Complex) -> impl Fn(Complex) -> Real {
    move |pos: Complex| (pos - p).mag()
}
#[inline(always)]
pub fn sdf_disk(d: Circle) -> impl Fn(Complex) -> Real {
    move |pos: Complex| (pos - d.center).mag() - d.radius
}
#[inline(always)]
pub fn sdf_halfPlane(hp: HalfPlane) -> impl Fn(Complex) -> Real {
    move |pos: Complex| {
        pos.y
            .mul_add(hp.normal.y, pos.x.mul_add(hp.normal.x, -hp.shift))
    }
}

pub trait SDF
where
    Self: Fn(Complex) -> Real + Sized,
{
    #[inline]
    fn negate(self) -> impl Fn(Complex) -> Real {
        move |pos: Complex| -self(pos)
    }
    #[inline]
    fn intersection(self, g: impl Fn(Complex) -> Real) -> impl Fn(Complex) -> Real {
        move |pos: Complex| self(pos).max(g(pos))
    }
    #[inline]
    fn union(self, g: impl Fn(Complex) -> Real) -> impl Fn(Complex) -> Real {
        move |pos: Complex| self(pos).min(g(pos))
    }
    #[inline]
    fn hollowOut(self) -> impl Fn(Complex) -> Real {
        move |pos: Complex| self(pos).abs()
    }
    fn into_boolf(self) -> impl Fn(Complex) -> bool {
        move |pos: Complex| self(pos) < 0.0
    }
}

impl<T: Fn(Complex) -> Real> SDF for T {}

#[inline]
fn boolf_empty() -> impl Fn(Complex) -> bool {
    move |_: Complex| false
}
#[inline]
fn boolf_hollowOut() -> impl Fn(Complex) -> bool {
    move |_: Complex| false
}
#[inline]
fn boolf_point() -> impl Fn(Complex) -> bool {
    move |_: Complex| false
}
#[inline]
fn boolf_disk(d: Circle) -> impl Fn(Complex) -> bool {
    move |pos: Complex| (pos - d.center).mag() < d.radius
}
#[inline]
fn boolf_halfPlane(hp: HalfPlane) -> impl Fn(Complex) -> bool {
    sdf_halfPlane(hp).into_boolf()
}

trait BoolF
where
    Self: Fn(Complex) -> bool + Sized,
{
    #[inline]
    fn negate(self) -> impl Fn(Complex) -> bool {
        move |pos: Complex| !self(pos)
    }
    #[inline]
    fn intersection(self, g: Self) -> impl Fn(Complex) -> bool {
        move |pos: Complex| self(pos) && g(pos)
    }
    #[inline]
    fn union(self, g: Self) -> impl Fn(Complex) -> bool {
        move |pos: Complex| self(pos) || g(pos)
    }
}

impl<T: Fn(Complex) -> bool> BoolF for T {}

#[inline]
fn scale_add(scale: Real, p: Complex, c: Complex) -> Complex {
    Complex::new(scale.mul_add(p.x, c.x), scale.mul_add(p.y, c.y))
}

pub fn diskNBP(disk: Circle) -> impl Fn(Complex) -> Complex {
    move |pos: Complex| {
        let v = pos - disk.center;
        let q = disk.radius * v.recipMag();
        if q.is_finite() {
            scale_add(q, v, disk.center)
        }
        // close to the center, all boundary points are about the same distance away, so just pick one
        else {
            Complex::new(disk.center.x + disk.radius, disk.center.y)
        }
    }
}

pub fn halfPlaneNBP(hp: HalfPlane) -> impl Fn(Complex) -> Complex {
    move |pos: Complex| scale_add(-sdf_halfPlane(hp)(pos), hp.normal, pos)
}

impl Bezier2o1d {
    fn eval(self, t: Real) -> Real {
        let v0div2 = self.1 - self.0;
        t.mul_add(t.mul_add(self.2 - self.1 - v0div2, v0div2 + v0div2), self.0)
    }
    fn solve(self, target: Real) -> (Real, Real) {
        let v0divn2 = self.0 - self.1;
        premulQuadratic_findRoots(self.0 - target, v0divn2, self.2 - self.1 + v0divn2)
    }
}

// Note that our bezier solvers can exclude the endpoints because our algorithm handles the endpoints separately anyway.
impl Bezier2o2d {
    fn evalPreproc(p0: Complex, v0: Complex, adiv2: Complex, t: Real) -> Complex {
        scale_add(t, scale_add(t, adiv2, v0), p0)
    }
    fn eval(self, t: Real) -> Complex {
        let v0div2 = self.1 - self.0;
        Self::evalPreproc(self.0, v0div2 + v0div2, self.2 - self.1 - v0div2, t)
    }
    fn filteredEval(self, t: Real) -> Complex {
        if 0.0 < t && t < 1.0 {
            self.eval(t)
        } else {
            Complex::new(Real::NAN, Real::NAN)
        }
    }

    fn filteredEvalPreproc(p0: Complex, v0: Complex, adiv2: Complex, t: Real) -> Complex {
        if 0.0 < t && t < 1.0 {
            Self::evalPreproc(p0, v0, adiv2, t)
        } else {
            Complex::new(Real::NAN, Real::NAN)
        }
    }
    pub fn twoBezier2o2dsIntersect(self, bez1: Bezier2o2d) -> [Complex; 4] {
        // find t vals such that (Bezier2o2d.eval bez0 t) gives a point on bez1
        let a1div2 = bez1.1 - bez1.0;
        let a2 = bez1.2 - bez1.1 - a1div2;
        if a2.x == 0.0 && a2.y == 0.0 {
            // bez1 has no acceleration. In this case, the usual quartic equation collapses to 0 == 0.
            // There are 2 options for handling this: nudge bez1.1 towards an endpoint and proceed with the quartic, or use our quadratic-bezier-line intersector.
            let bez1Normal = Complex::new(-a1div2.y, a1div2.x).normalize();
            let z = self.bezier2o2dLineIntersect(HalfPlane {
                normal: bez1Normal,
                shift: dot2dAccurate(bez1Normal, bez1.0),
            });
            [
                z[0],
                z[1],
                Complex::new(Real::NAN, Real::NAN),
                Complex::new(Real::NAN, Real::NAN),
            ]
        } else {
            let a0 = bez1.0 - self.0;
            let a1 = a1div2 + a1div2;
            let p1div2 = self.1 - self.0;
            let p1 = p1div2 + p1div2;
            let p2 = self.2 - self.1 - p1div2;
            // Solve a2 s^2 + a1 s + a0 - p2 t^2 - p1 t == 0 for t without computing s
            // matrix([[a2x, a1x, a0x-p2x*t^2-p1x*t, 0], [0, a2x, a1x, a0x-p2x*t^2-p1x*t], [a2y, a1y, a0y-p2y*t^2-p1y*t, 0], [0, a2y, a1y, a0y-p2y*t^2-p1y*t]]).det() == 0
            let a0a1 = det2x2Accurate(a0, a1);
            let a0a2 = det2x2Accurate(a0, a2);
            let a1a2 = det2x2Accurate(a1, a2);
            let a1p1 = det2x2Accurate(a1, p1);
            let a1p2 = det2x2Accurate(a1, p2);
            let a2p1 = det2x2Accurate(a2, p1);
            let a2p2 = det2x2Accurate(a2, p2);
            // If all control points are at integer coordinates whose max abs component is 2**12.5 or less in f32, 2**26 or less in f64,
            // or if all coordinates could be uniformly scaled up or down by a power of 2 to achieve the above constraint without any
            // overflow or underflow, then I think the above 7 determinants can be regular det2x2.
            let a0a2times2 = a0a2 + a0a2;
            let c0 = prodDiffAccurate(a0a2, a0a2, a0a1, a1a2);
            let c1 = prodDiffAccurate(a0a2times2, a2p1, a1a2, a1p1);
            let c2 = a2p1.mul_add(a2p1, prodDiffAccurate(a0a2times2, a2p2, a1a2, a1p2));
            let c3 = (a2p1 + a2p1) * a2p2;
            let c4 = a2p2 * a2p2;
            let ts = quartic_findRoots(c0, c1, c2, c3, c4);
            let ts = tuple4map(
                partial!(move quartic_halleyOnceIfBetter => c0, c1, c2, c3, c4, _),
                ts,
            ); // might be unnecessary?
            let vdiv2 = self.1 - self.0;
            let adiv2 = self.2 - self.1 - vdiv2;
            let v = vdiv2 + vdiv2;
            //let bez1EndpointDiff = bez1.2 Complex.- bez1.0 in // for secondary filtering
            //let sideInfo = det2x2Accurate(bez1EndpointDiff, a1) in // for secondary filtering
            //let finish t = Real.(
            //  if ! (f64 0 < t && t < f64 1) then Complex::new(Real::NAN, Real::NAN) else
            //  let q = tuple2map2 (fma t) adiv2 v;
            //  let r = tuple2map2 (fma t) q (Complex.neg a0);
            //  if f64 0 <= sideInfo * det2x2Accurate(bez1EndpointDiff, r) // If this secondary filtering step incorrectly deletes any points (most likely with nearly-straight curves), it can be relaxed or removed.
            //    then tuple2map2 (fma t) q bez0.0
            //    else Complex::new(Real::NAN, Real::NAN)
            //);
            //let ps = tuple4map finish ts;
            let ps = tuple4map(
                partial!(move Self::filteredEvalPreproc => self.0,v,adiv2, _),
                ts,
            ); // The secondary filtering step was difficult to get right, so we're using this instead for now.
            [ps.0, ps.1, ps.2, ps.3]
        }
    }

    fn findLocallyNearestTVals(self, pos: Complex) -> (f32, f32) {
        let v0div2 = self.1 - self.0;
        let adiv2 = self.2 - self.1 - v0div2;
        let relstart = self.0 - pos;

        //let v0 = v0div2 Complex.+ v0div2
        // find roots of d/dt (Complex.squaredMag (pos - eval bez t))/4
        let c0 = dot2dAccurate(v0div2, relstart);
        let c1 = adiv2.y.mul_add(
            relstart.y,
            adiv2.x.mul_add(relstart.x, 2.0 * v0div2.squaredMag()),
        );
        let c2divn3 = -dot2dAccurate(adiv2, v0div2);
        let c3 = adiv2.squaredMag();
        let c0divc3 = c0 / c3;
        let c1divc3 = c1 / c3;
        let c2divn3c3 = c2divn3 / c3;
        let d0divn2 = c2divn3c3.mul_add(
            c2divn3c3.mul_add(c2divn3c3, c1divc3 * (-0.5)),
            c0divc3 * (-0.5),
        );
        let d1divn3 = c2divn3c3.mul_add(c2divn3c3, c1divc3 * from_fraction::<f32>(-1, 3));

        let tdep = premulDepressedCubic_findRoots_fast(d0divn2, d1divn3);
        let t0 = tdep.0 + c2divn3c3;
        let t1 = tdep.2 + c2divn3c3; // Even if there are multiple real roots, the middle root is never a local min, so we skip it.
        let t0 = if t0.is_finite() { t0 } else { -c0 / c1 }; // fallback to what the solution would be if the curve's acceleration were 0

        // Halley's method would give more accuracy per operation but we might only need 1 Newton step anyway.
        // We cannot use the depressed cubic for this correction because this needs to work when c3 is near 0.
        (
            cubic_newtonOnceIfBetter(c0, c1, c2divn3 * -3.0, c3, t0),
            cubic_newtonOnceIfBetter(c0, c1, c2divn3 * -3.0, c3, t1),
        )
    }

    pub fn findLocallyNearestPoints(self, pos: Complex) -> [Complex; 2] {
        let (t0, t1) = self.findLocallyNearestTVals(pos);
        [self.filteredEval(t0), self.filteredEval(t1)]
    }

    #[inline]
    #[allow(clippy::manual_clamp)]
    pub fn sdf(self) -> impl Fn(Complex) -> f32 {
        move |pos: Complex| {
            let (t0, t1) = self.findLocallyNearestTVals(pos);

            let p0 = self.eval(t0.max(0.0).min(1.0));
            let p1 = self.eval(t1.max(0.0).min(1.0));

            sdf_point(p0)(pos).min(sdf_point(p1)(pos))
        }
    }

    pub fn bezier2o2dLineIntersect(self, hp: HalfPlane) -> [Complex; 2] {
        let proj = Bezier2o1d(
            dot2dAccurate(hp.normal, self.0),
            dot2dAccurate(hp.normal, self.1),
            dot2dAccurate(hp.normal, self.2),
        );
        let p = proj.solve(hp.shift);
        [self.filteredEval(p.0), self.filteredEval(p.1)]
    }

    pub fn bezier2o2dCircleIntersect(self, circ: Circle) -> [Complex; 4] {
        let v0div2 = self.1 - self.0;
        let adiv2 = self.2 - self.1 - v0div2;
        let v0 = v0div2 + v0div2;
        let a = adiv2 + adiv2;
        let relstart = self.0 - circ.center;
        let negRadiusSquared = -(circ.radius * circ.radius);
        let c0 = relstart
            .y
            .mul_add(relstart.y, relstart.x.mul_add(relstart.x, negRadiusSquared))
            - circ.radius.mul_add(circ.radius, negRadiusSquared);
        let c1 = 2.0 * dot2dAccurate(v0, relstart);
        let c2 =
            a.y.mul_add(relstart.y, a.x.mul_add(relstart.x, v0.squaredMag()));
        let c3 = dot2dAccurate(a, v0);
        let c4 = adiv2.squaredMag();
        let t = quartic_findRoots(c0, c1, c2, c3, c4);
        let t = tuple4map(
            partial!(move quartic_halleyOnceIfBetter => c0, c1, c2, c3, c4, _),
            t,
        ); // might be unnecessary?
        let p = tuple4map(
            partial!(move Self::filteredEvalPreproc => self.0, v0, adiv2, _),
            t,
        );
        [p.0, p.1, p.2, p.3]
    }
}

pub fn twoCirclesIntersect(c0: Circle, c1: Circle) -> [Complex; 2] {
    let centerDiff = c1.center - c0.center;
    let rcd = centerDiff.recipMag();
    let r0 = c0.radius * rcd;
    let r1 = c1.radius * rcd;
    let a = prodDiffAccurate(r0, r0, r1, r1).mul_add(0.5, 0.5);
    let z = Complex::new(a, (prodDiffAccurate(r0, r0, a, a)).sqrt());
    [
        z.mulAdd(centerDiff, c0.center),
        z.conj().mulAdd(centerDiff, c0.center),
    ]
}

pub fn circleLineIntersect(circ: Circle, hp: HalfPlane) -> [Complex; 2] {
    let perpOffset = -sdf_halfPlane(hp)(circ.center);
    let z = Complex::new(
        perpOffset,
        (prodDiffAccurate(circ.radius, circ.radius, perpOffset, perpOffset)).sqrt(),
    );
    [
        z.mulAdd(hp.normal, circ.center),
        z.conj().mulAdd(hp.normal, circ.center),
    ]
}

fn halfPlaneToLineFunc(hp: HalfPlane, x: Real) -> Real {
    hp.shift.mul_add(
        hp.normal.y,
        hp.shift
            .mul_add(hp.normal.x, (-x) * hp.normal.x / hp.normal.y),
    )
}

/*pub fn twoLinesIntersect(a: HalfPlane, b: HalfPlane) -> Complex {
    a.normal.mul(Complex::new(
        a.shift,
        halfPlaneToLineFunc(
            HalfPlane {
                normal: b.normal.mul(a.normal.conj()),
                shift: b.shift,
            },
            a.shift,
        ),
    ))
}*/

pub fn twoLinesIntersect(a: HalfPlane, b: HalfPlane) -> Complex {
    let det = det2x2Accurate(a.normal, b.normal);
    let x = a.shift.mul_add(b.normal.y, -b.shift * a.normal.y) / det;
    let y = a.normal.x.mul_add(b.shift, -b.normal.x * a.shift) / det;
    Complex::new(x, y)
}

pub fn isBoundaryPoint(f: &impl Fn(Complex) -> Real, pos: Complex) -> bool {
    f(pos).abs() <= BOUNDARY_THRESHOLD
}

fn selectNearer(pos: Complex, a: Complex, b: Complex) -> Complex {
    let db = (b - pos).squaredMag();
    if db.is_nan() || (a - pos).squaredMag() <= db {
        a
    } else {
        b
    }
}

pub fn impliedPoints(
    bezs: &[Bezier2o2d],
    circles: &[Circle],
    lines: &[HalfPlane],
) -> impl Iterator<Item = Complex> {
    use itertools::Itertools;
    let cpoints = circles
        .iter()
        .cartesian_product(circles.iter())
        .flat_map(|(a, b)| twoCirclesIntersect(*a, *b));

    let bezcirc = bezs
        .iter()
        .cartesian_product(circles.iter())
        .flat_map(|(a, b)| a.bezier2o2dCircleIntersect(*b));

    let bezline = bezs
        .iter()
        .cartesian_product(lines.iter())
        .flat_map(|(a, b)| a.bezier2o2dLineIntersect(*b));

    let circline = circles
        .iter()
        .cartesian_product(lines.iter())
        .flat_map(|(a, b)| circleLineIntersect(*a, *b));

    bezs.iter()
        .map(|b| b.0)
        .chain(bezs.iter().map(|b| b.2))
        .chain(
            lines
                .iter()
                .cartesian_product(lines.iter())
                .map(|(a, b)| twoLinesIntersect(*a, *b)),
        )
        .chain(cpoints)
        .chain(
            bezs.iter()
                .cartesian_product(bezs.iter())
                .flat_map(|(a, b)| a.twoBezier2o2dsIntersect(*b)),
        )
        .chain(bezcirc)
        .chain(bezline)
        .chain(circline)
}
