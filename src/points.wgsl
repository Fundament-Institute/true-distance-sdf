const OP_UNION: u32 = 0;
const OP_INTERSECT: u32 = 1;
const SHAPE_CIRCLE: u32 = 2;
const SHAPE_LINE: u32 = 3;
const SHAPE_BEZIER: u32 = 4;
const SHAPE_POLYGON: u32 = 5;
const OP_NEGATE: u32 = 8;
const OP_HOLLOW: u32 = 16;
const OP_MASK: u32 = OP_NEGATE - 1;

const MAX_F32: f32 = 3.402823466e+38;
const MAX_VEC: vec2f = vec2f(MAX_F32, MAX_F32);

fn mag_sq(v: vec2f) -> f32 {
    return dot(v, v);
}

fn mag(v: vec2f) -> f32 {
    return sqrt(mag_sq(v));
}

fn cubic_eval(c0: f32, c1: f32, c2: f32, c3: f32, t: f32) -> f32 {
    return fma(t, fma(t, fma(t, c3, c2), c1), c0);
}

// degree n polynomial and its first derivative in 2*n - 1 fma operations
fn next_coef1(t: f32, pv: vec2f, c: f32) -> vec2f {
    return vec2f(fma(t, pv.x, c), fma(t, pv.y, pv.x));
}

fn linear_eval1(c0: f32, c1: f32, t: f32) -> vec2f {
    return vec2f(fma(t, c1, c0), c1);
}

// computing the derivative as (t.mul_add(c2 + c2, c1)) would spend an extra operation to guarantee a correctly-rounded result
fn quadratic_eval1(c0: f32, c1: f32, c2: f32, t: f32) -> vec2f {
    return next_coef1(t, linear_eval1(c1, c2, t), c0);
}

fn cubic_eval1(c0: f32, c1: f32, c2: f32, c3: f32, t: f32) -> vec2f {
    return next_coef1(t, quadratic_eval1(c1, c2, c3, t), c0);
}

fn cubic_newtonOnceIfBetter(c0: f32, c1: f32, c2: f32, c3: f32, t: f32) -> f32 {
    let pv = cubic_eval1(c0, c1, c2, c3, t);
    let t1 = t - pv.x / pv.y;
    return select(t, t1, abs(cubic_eval(c0, c1, c2, c3, t1)) <= abs(pv.x));
}

fn findLocallyNearestTVals(bez: Bezier2o2d, pos: vec2f) -> vec2f {
    let v0div2 = bez[1] - bez[0];
    let adiv2 = (bez[2] - bez[1] - v0div2);
    let relstart = bez[0] - pos;

    //let v0 = v0div2 + v0div2
    // find roots of d/dt (mag_sq (pos - eval self t))/4
    let c0 = dot2dAccurate(v0div2, relstart);
    let c1 = fma(adiv2.y, relstart.y, fma(adiv2.x, relstart.x, 2.0 * mag_sq(v0div2)));
    let c2divn3 = - dot2dAccurate(adiv2, v0div2);
    let c3 = mag_sq(adiv2);
    let c0divc3 = c0 / c3;
    let c1divc3 = c1 / c3;
    let c2divn3c3 = c2divn3 / c3;
    let d0divn2 = fma(c2divn3c3, fma(c2divn3c3, c2divn3c3, c1divc3 * (- 0.5)), c0divc3 * (- 0.5));
    let d1divn3 = fma(c2divn3c3, c2divn3c3, c1divc3 * (- 1.0 / 3.0));
    let tdep = premulDepressedCubic_findRoots_fast(d0divn2, d1divn3);
    var t0 = tdep.x + c2divn3c3;
    let t1 = tdep.z + c2divn3c3;
    // Even if there are multiple real roots, the middle root is never a local min, so we skip it.

    // fallback to what the solution would be if the curve's acceleration were 0
    if !isFinite(t0) {
        t0 = - c0 / c1;
    }

    // Halley's method would give more accuracy per operation but we probably only need 1 Newton step anyway.
    // We cannot use the depressed cubic for this correction because this needs to work when c3 is near 0.
    let tn0 = cubic_newtonOnceIfBetter(c0, c1, c2divn3 * - 3.0, c3, t0);
    let tn1 = cubic_newtonOnceIfBetter(c0, c1, c2divn3 * - 3.0, c3, t1);

    return vec2f(tn0, tn1);
}

fn psdf_halfplane(normal: vec2f, shift: f32, pos: vec2f) -> f32 {
    return fma(pos.y, normal.y, fma(pos.x, normal.x, (- shift)));
}

fn psdf_point(p: vec2f, pos: vec2f) -> f32 {
    return mag(pos - p);
}

fn psdf_disk(center: vec2f, radius: f32, pos: vec2f) -> f32 {
    return psdf_point(center, pos) - radius;
}

fn psdf_bez(bez: Bezier2o2d, pos: vec2f) -> f32 {
    let t = findLocallyNearestTVals(bez, pos);
    let p0 = eval_bezier(bez, min(max(t.x, 0.0), 1.0));
    let p1 = eval_bezier(bez, min(max(t.y, 0.0), 1.0));

    return min(psdf_point(p0, pos), psdf_point(p1, pos));
}

fn op_negate(x: f32) -> f32 {
    return - x;
}

fn op_hollow(x: f32) -> f32 {
    return abs(x);
}

fn op_union(l: f32, r: f32) -> f32 {
    return min(l, r);
}

fn op_intersect(l: f32, r: f32) -> f32 {
    return max(l, r);
}

fn unary_op(x: f32, op: u32) -> f32 {
    let r = select(x, op_negate(x), (op & OP_NEGATE) != 0);
    return select(r, op_hollow(r), (op & OP_HOLLOW) != 0);
}

fn shapeField(pos: vec2f) -> f32 {
    var stack = array<f32, 32>();
    var len = 0;

    for (var i = 0u; i < arrayLength(&shapes);) {
        let op = bitcast<u32>(shapes[i]);
        switch op & OP_MASK {
            case OP_UNION, OP_INTERSECT: {
                let r = stack[len - 1];
                let l = stack[len - 2];
                stack[len - 2] = unary_op(select(op_intersect(l, r), op_union(l, r), (op & OP_MASK) == OP_UNION), op);
                len -= 1;
                i += 1;
            }
            case SHAPE_CIRCLE: {
                stack[len] = unary_op(psdf_disk(vec2f(shapes[i + 1], shapes[i + 2]), shapes[i + 3], pos), op);
                len += 1;
                i += 4;
            }
            case SHAPE_BEZIER: {
                stack[len] = unary_op(psdf_bez(Bezier2o2d(vec2f(shapes[i + 1], shapes[i + 2]), vec2f(shapes[i + 3], shapes[i + 4]), vec2f(shapes[i + 5], shapes[i + 6])), pos), op);
                len += 1;
                i += 7;
            }
            case SHAPE_LINE: {
                stack[len] = unary_op(psdf_halfplane(vec2f(shapes[i + 1], shapes[i + 2]), shapes[i + 3], pos), op);
                len += 1;
                i += 4;
            }
            default : {
                return 0.0;
            }
        }
    }

    return stack[0];
}

const BOUNDARY_THRESHOLD: f32 = 0.25;

fn isFinite(x: f32) -> bool {
    return abs(x) < MAX_F32;
}

fn isFinite2(v: vec2f) -> bool {
    return abs(v.x) < MAX_F32 && abs(v.y) < MAX_F32;
}

alias Bezier2o2d = array<vec2f, 3>;

fn evalPreproc(p0: vec2f, v0: vec2f, adiv2: vec2f, t: f32) -> vec2f {
    return scale_add(t, scale_add(t, adiv2, v0), p0);
}

fn eval_bezier(bez: Bezier2o2d, t: f32) -> vec2f {
    let v0div2 = bez[1] - bez[0];
    return evalPreproc(bez[0], (v0div2 + v0div2), (bez[2] - bez[1] - v0div2), t);
}

fn scale_add(scale: f32, p: vec2f, c: vec2f) -> vec2f {
    return vec2f(fma(scale, p.x, c.x), fma(scale, p.y, c.y));
}

// see https://pharr.org/matt/blog/2019/11/03/difference-of-floats or "Further Analysis of Kahan's Algorithm for the Accurate Computation of 2x2 Determinants"
fn dot2dAccurate(a: vec2f, b: vec2f) -> f32 {
    let t = a.x * b.x;
    return fma(a.y, b.y, t) + fma(a.x, b.x, - t);
}

fn prodDiffAccurate(a: f32, b: f32, c: f32, d: f32) -> f32 {
    let t = - (c * d);
    return fma(a, b, t) - fma(c, d, t);
}

// Simplified copysign implementation that assumes input is non-negative
fn fakecopysign(input: f32, other: f32) -> f32 {
    // sign returns 0 if other is zero, which doesn't work for this
    //return input * sign(other);
    return select(input, - input, other < 0);
}

fn from_fraction(n: i32, d: i32) -> f32 {
    return f32(n) / f32(d);
}

fn quartic_eval(c0: f32, c1: f32, c2: f32, c3: f32, c4: f32, t: f32) -> f32 {
    return fma(t, fma(t, fma(t, fma(t, c4, c3), c2), c1), c0);
}

fn rcbrtPositiveNormalApprox(x: f32) -> f32 {
    // The extra performance here doesn't seem to be necessary
    return pow(x, - 1.0 / 3.0);

    // u32 division by constant 3 is compiled to multiply-then-shift by many compilers, so this should be division-free.
    // If you care about ulp error instead of relative error, use 0x54a21d29 instead.
    let y = bitcast<f32>(0x54a208f8u - (bitcast<u32>(x) / 3));
    let p = fma((x * y), y * y, - 1.0);
    // p = x * y**3 - 1 is the expression we're finding a zero of.
    // Using that definition of p, the true answer x**(-1/3) == ((p+1)**(-1/3)-1) * y + y in exact arithmetic.
    // Our initial guess is ok enough that p is close-ish to 0, so we use a Taylor approximation of (p+1)**(-1/3)-1 centered at 0.
    // The order n Taylor approximation gives order n+1 convergence. We use n=4 here.
    // More accurate than trisectApprox. -20.44ulp < err < 17.34ulp; -2**-19.38 < relerr < 2**-19.38.
    // 4 fewer flops than the |err|<1ulp version. 3 fewer flops than the |err|<2ulp version.
    // A minimax polynomial over the appropriate interval would be better & might allow saving a flop, but this is good enough.

    return fma(fma(p, fma(p, fma(p, 35.0 / 243.0, - 14.0 / 81.0), 2.0 / 9.0), - 1.0 / 3.0), p * y, y);
}

// | Approximates cos ((acos x) / 3.0) using a shifted-and-scaled sqrt followed by a quartic polynomial.
// | Coefficients chosen using several iterations of the Remez algorithm subject to the restriction 0.5 <= trisectApprox x <= 1.0 whenever abs x <= 1.
// | (Some bits of the f64 coefficients had not converged when I stopped improving them, but these are certainly the best coefficients to within better than f32 precision.)
// | Undershooting 0.5 would cause some roots to swap order.
// | Exceeding 1.0 would make NaNs because this function's output is fed to something like (\x -> sqrt (1 - x*x)).
// | This definition stays in the appropriate range on IEEE754 compliant hardware for both f32 and f64, both with and without fma.
fn trisectApprox(x: f32) -> f32 {
    let c0 = 0.5;
    let c1 = 0.5769789;
    let c2 = - 0.10710293;
    let c3 = 0.03912646;
    let c4 = - 0.009002428;
    return quartic_eval(c0, c1, c2, c3, c4, sqrt(fma(x, c0, c0)));
}

fn recipMag(v: vec2f) -> f32 {
    return inverseSqrt(mag_sq(v));
}

fn isBoundaryPoint(pos: vec2f) -> bool {
    return select(false, abs(shapeField(pos)) <= BOUNDARY_THRESHOLD, isFinite2(pos));
}

// | Solves t**3 - 3*c1divn3*t - 2*c0divn2 == 0 for real t.
// | No scaling is attempted; if it overflows, it overflows.
// | In the 3-root case, the roots are sorted (most-positive, middle, most-negative).
// | The middle root has the fewest significant bits. The most-positive root has the most.
// | If any of the roots are good, the root in slot 0 is.
fn premulDepressedCubic_findRoots_fast(c0divn2: f32, c1divn3: f32) -> vec3f {
    let d = prodDiffAccurate(c0divn2, c0divn2, c1divn3 * c1divn3, c1divn3);
    if !isFinite(d) {
        return vec3f(d, d, d);
    }

    if 0.0 < d {
        // 1 real root
        let q = abs(c0divn2) + sqrt(d);
        // sqrt of a positive finite number is normal in all supported formats. No need for rcbrt to handle subnormals.
        let u = rcbrtPositiveNormalApprox(q);
        return vec3f(fakecopysign(u, c0divn2) * fma(q, u, c1divn3), MAX_F32, MAX_F32);
        // If you're on a platform that provides a fast rcbrt that preserves sign, you can skip a step with this:
        //let q = c0divn2 + fakecopysign (sqrt d) c0divn2;
        //let u = rcbrt q in (u * q.mul_add(u, c1divn3), Real::NAN, Real::NAN)
    }
    else {
        // 3 real roots counting multiplicity, unless underflow
        let h = inverseSqrt(c1divn3);
        let s = c1divn3 * h;
        let u = c0divn2 * h * h * h;
        if !isFinite(u) {
            // Mostly happens when c1divn3 ** 1.5 is subnormal or 0, AND c0divn2 is too close to 0 to make d positive. In those cases, the equation is very close to x**3 == 0.
            //(cbrt (c0divn2 + c0divn2), Real::NAN, Real::NAN) // Even in f32 with subnormal-flushing and no fma, abs(exact answer) in this codepath is always < 2 ** -20.6, so we just round to 0.
            return vec3f(0.0, MAX_F32, MAX_F32);
            // If you're finding a quadratic Bezier parameter without rescaling, you need > 2 ** 17.6 px between control points before this produces 1/4 px of error.
        }
        else {
            let u = max(min(u, 1.0), - 1.0);
            // This line only does anything if u was computed inaccurately AND 2 roots were close together, but I don't know a way to rule that out.
            let a = trisectApprox(u);
            let b = sqrt(fma(- a, a * 3.0, 3.0));
            // Not needed for the most-positive root so can be skipped if you only need that.
            //let v = acos u * from_fraction 1 3;
            //let a = cos v in // Use this instead if your hardware makes acos, cos, and sin particularly fast & accurate
            //let b = sin v * sqrt (f64 3);
            let sa = s * a;
            let sb = s * b;
            return vec3f(sa + sa, sb - sa, - sb - sa);
        }
    }
}

/// ^^^ FUNCTIONS ABOVE DUPLICATED FROM draw.wgsl ^^^

fn dot2d(a: vec2f, b: vec2f) -> f32 {
    return fma(a.y, b.y, a.x * b.x);
}

fn complex_mul(a: vec2f, b: vec2f) -> vec2f {
    return vec2f(dot2d(a, conj(b)), dot2d(a, vec2f(b.y, b.x)));
}

fn conj(v: vec2f) -> vec2f {
    return vec2f(v.x, - v.y);
}

fn pow2i(q: i32) -> f32 {
    return bitcast<f32>((u32(q + 0x7f)) << 23);
}

fn premulDepressedCubic_computeScaleExponent(x: f32, c1divn3: f32) -> i32 {
    let a = (127 & i32(bitcast<u32>(x) >> 24)) - 82;
    let b = i32(((255 << 23) & bitcast<u32>(c1divn3)) / (3 << 23)) - 61;
    return max(a, b);
}

// | Like premulDepressedCubic_findRoots_fast but does a bit of extra work to avoid overflow and underflow.
fn premulDepressedCubic_findRoots_stopOverflow(c0divn2: f32, c1divn3: f32) -> vec3f {
    let scaleExponent = premulDepressedCubic_computeScaleExponent(c0divn2, c1divn3);
    let scale = pow2i(- scaleExponent);
    let sc0divn2 = (c0divn2 * scale) * (scale * scale);
    let sc1divn3 = c1divn3 * (scale * scale);
    let unscale = pow2i(scaleExponent);
    let abc = premulDepressedCubic_findRoots_fast(sc0divn2, sc1divn3);
    return abc * unscale;
}

// | If c2 == 1, then there are fewer places where precision-loss can happen, so the discriminant calculation can be shorter.
fn premulMonicQuadratic_findRoots(c0: f32, c1divn2: f32) -> vec2f {
    let q = c1divn2 + fakecopysign(sqrt(fma(c1divn2, c1divn2, - c0)), c1divn2);
    return vec2f(c0 / q, q);
}

fn monicQuadratic_findRoots(c0: f32, c1: f32) -> vec2f {
    return premulMonicQuadratic_findRoots(c0, c1 * - 0.5);
}

// | Solves t**4 + c2*t**2 + c1*t + c0 == 0 for real t.
fn depressedQuartic_findRoots(c0: f32, c1: f32, c2: f32) -> vec4f {
    // Before feeding the resolvent cubic to the analytic solver, it is depressed and rescaled.
    // This scaling factor was chosen to make these constants be exactly-representable values close to 1 on a log scale
    // with lots of trailing 0 bits in the mantissa to minimize the impact of rounding error.
    let c2div4 = c2 * 0.25;
    let d0divn2 = fma(c2div4, fma(c2div4, c2div4, - 2.25 * c0), 0.2109375 * c1 * c1);
    let d1divn3 = fma(c2div4, c2div4, c0 * 0.75);
    // Since we only want the most positive root, might it be worth inlining the following function 2 layers deep to skip computing the other roots?
    let drcMostPositiveRoot = premulDepressedCubic_findRoots_stopOverflow(d0divn2, d1divn3).x;
    let r = (drcMostPositiveRoot - c2) * from_fraction(4, 3);
    // undepress and rescale
    // Even the existence of real solutions is very sensitive to imprecision in the value of r. We need to recover some precision.
    // It may look tempting to move these Halley corrections to the depressed cubic to save time.
    // If you do that, your precision gains will too often be wiped out by catastrophic cancellation in the undepressing step.
    // 1 Halley step is not enough to even detect the existence of solutions in too many cases.
    // We need at least 2 here. I put in a third just in case.
    let rc0 = - (c1 * c1);
    let rc1 = fma(c2, c2, - 4.0 * c0);
    let rc2 = c2 + c2;
    let g0 = r + rc2;
    let h0 = fma(r, g0, rc1);
    let p0 = fma(r, h0, rc0);
    let s0 = r + g0;
    let v0 = fma(r, s0, h0);
    let r1_0 = fma(p0, v0 / fma((- v0), v0, (r + s0) * p0), r);
    let g1 = r1_0 + rc2;
    let h1 = fma(r1_0, g1, rc1);
    let p1_1 = fma(r1_0, h1, rc0);
    // In the rare failed-to-improve case it is equivalent to skip to the end of Halley corrections.
    let r_a = select(r, r1_0, abs(p1_1) <= abs(p0));
    let p_a = select(p0, p1_1, abs(p1_1) <= abs(p0));
    let s1 = r_a + g1;
    let v1 = fma(r_a, s1, h1);
    let r1_1 = fma(p_a, v1 / fma((- v1), v1, (r_a + s1) * p_a), r_a);
    let g2 = r1_1 + rc2;
    let h2 = fma(r1_1, g2, rc1);
    let p1_2 = fma(r1_1, h2, rc0);
    // In the rare failed-to-improve case it is equivalent to skip to the end of Halley corrections.
    let r_b = select(r_a, r1_1, abs(p1_2) <= abs(p_a));
    let p_b = select(p_a, p1_2, abs(p1_2) <= abs(p_a));
    let s2 = r_b + g2;
    let v2 = fma(r_b, s2, h2);
    let r1_2 = fma(p_b, v2 / fma((- v2), v2, (r_b + s2) * p_b), r_b);
    let g3 = r1_2 + rc2;
    let h3 = fma(r1_2, g3, rc1);
    let p1_3 = fma(r1_2, h3, rc0);
    let r_final = select(r_b, r1_2, abs(p1_3) <= abs(p_b));
    // END OF HALLEY CORRECTIONS
    if r_final <= 0.0 {
        // c1 is so small that this quartic is effectively a quadratic in t**2
        // FIXME: Not currently trying to handle potential overflow in the quadratic discriminant.
        let t = sqrt(monicQuadratic_findRoots(c0, c2));
        return vec4f(t.x, t.y, - t.x, - t.y);
    }
    else {
        let q = 0.5 * inverseSqrt(r_final);
        let g = 0.5 * (c2 + r_final);
        let h = q * c1;
        let s = q * r_final;
        // FIXME: Not currently trying to handle potential overflow in the quadratic discriminants.
        let t01 = premulMonicQuadratic_findRoots(g + h, s);
        let t23 = premulMonicQuadratic_findRoots(g - h, - s);
        return vec4f(t01.x, t01.y, t23.x, t23.y);
    }
}

fn premulQuadratic_findRoots(c0: f32, c1divn2: f32, c2: f32) -> vec2f {
    let q = c1divn2 + fakecopysign(sqrt(prodDiffAccurate(c1divn2, c1divn2, c0, c2)), c1divn2);
    return vec2f(c0 / q, q / c2);
}

fn quadratic_findRoots(c0: f32, c1: f32, c2: f32) -> vec2f {
    return premulQuadratic_findRoots(c0, c1 * - 0.5, c2);
}

// | Solves c3*t**3 + c2*t**2 + c1*t + c0 == 0 for real t.
// | If there's only 1 root, it's in slot 0, but the other root-order guarantees may be broken.
fn cubic_findRoots(c0: f32, c1: f32, c2: f32, c3: f32) -> vec3f {
    //if isnan c3 { (c3, c3, c3) } else {// Without this line, c3=nan ends up acting like c3=0. Maybe that's fine.
    let n3c3 = c3 * - 3.0;
    let m = vec3f(c0 / (c3 * - 2.0), c1 / n3c3, c2 / n3c3);
    if !(isFinite(m.x) && isFinite(m.y) && isFinite(m.z)) {
        // In this function, we mostly care about roots with magnitude <= 1.
        // In this branch, c3 is so much smaller than another coefficient that replacing
        // c3 with 0 should barely change anything about this function inside that unit disk.
        // FIXME: Not currently trying to handle potential overflow in the quadratic discriminant.
        let q = quadratic_findRoots(c0, c1, c2);
        return vec3f(q.x, q.y, MAX_F32);
    }
    else {
        let d0divn2 = fma(m.z, fma(m.z, m.z, m.y * 1.5), m.x);
        let d1divn3 = fma(m.z, m.z, m.y);
        let abc = premulDepressedCubic_findRoots_stopOverflow(d0divn2, d1divn3);
        return abc + m.z;
    }
}

// | Solves c4*t**4 + c3*t**3 + c2*t**2 + c1*t + c0 == 0 for real t.
fn quartic_findRoots(c0: f32, c1: f32, c2: f32, c3: f32, c4: f32) -> vec4f {
    //if isnan c4 { (c4, c4, c4) } else {// Without this line, c4=nan ends up acting like c4=0. Maybe that's fine.
    let m = vec4f(c0 / c4, c1 / c4, c2 / c4, c3 / c4);
    if !(isFinite(m.x) && isFinite(m.y) && isFinite(m.z) && isFinite(m.w)) {
        // In this function, we mostly care about roots with magnitude <= 1.
        // In this branch, c4 is so much smaller than another coefficient that replacing
        // c4 with 0 should barely change anything about this function inside that unit disk.
        let t = cubic_findRoots(c0, c1, c2, c3);
        return vec4f(t.x, t.y, t.z, MAX_F32);
    }
    else {
        let m3div2 = m.w * 0.5;
        let m3divn4 = m.w * (- 0.25);
        let m3by3div4 = m.w + m3divn4;
        let d0 = fma(m3divn4, fma(m3divn4, fma(m3divn4, m3by3div4, m.z), m.y), m.x);
        let d1 = fma(m3div2, fma(m3div2, m3div2, - m.z), m.y);
        let d2 = fma(m3div2, - m3by3div4, m.z);
        let t = depressedQuartic_findRoots(d0, d1, d2);
        return t + m3divn4;
    }
}

fn b2o1d_solve(b: vec3f, t: f32) -> vec2f {
    let v0divn2 = b.x - b.y;
    return premulQuadratic_findRoots(b.x - t, v0divn2, b.z - b.y + v0divn2);
}

fn b2o2d_eval(b: Bezier2o2d, t: f32) -> vec2f {
    let v0div2 = b[1] - b[0];
    return evalPreproc(b[0], v0div2 + v0div2, b[2] - b[1] - v0div2, t);
}

fn filteredEval(b: Bezier2o2d, t: f32) -> vec2f {
    if 0.0 < t && t < 1.0 {
        return b2o2d_eval(b, t);
    }
    else {
        return MAX_VEC;
    }
}

fn filteredEvalPreproc(p0: vec2f, v0: vec2f, adiv2: vec2f, t: f32) -> vec2f {
    if 0.0 < t && t < 1.0 {
        return evalPreproc(p0, v0, adiv2, t);
    }
    else {
        return MAX_VEC;
    }
}

fn bezier2o2dLineIntersect(bez0: Bezier2o2d, normal: vec2f, shift: f32) -> array<vec2f, 2> {
    let proj = vec3f(dot2dAccurate(normal, bez0[0]), dot2dAccurate(normal, bez0[1]), dot2dAccurate(normal, bez0[2]));
    let p = b2o1d_solve(proj, shift);
    return array(filteredEval(bez0, p[0]), filteredEval(bez0, p[1]));
}

fn next_coef2(t: f32, pva: vec3f, c: f32) -> vec3f {
    return vec3f(fma(t, pva.x, c), fma(t, pva.y, pva.x), fma(t, pva.z, pva.y));
}

fn quadratic_eval2(c0: f32, c1: f32, c2: f32, t: f32) -> vec3f {
    let q = fma(t, c2, c1);
    return vec3f(fma(t, q, c0), fma(t, c2, q), c2);
}

fn cubic_eval2(c0: f32, c1: f32, c2: f32, c3: f32, t: f32) -> vec3f {
    return next_coef2(t, quadratic_eval2(c1, c2, c3, t), c0);
}

fn quartic_eval2(c0: f32, c1: f32, c2: f32, c3: f32, c4: f32, t: f32) -> vec3f {
    return next_coef2(t, cubic_eval2(c1, c2, c3, c4, t), c0);
}

fn quartic_halleyOnceIfBetter(c0: f32, c1: f32, c2: f32, c3: f32, c4: f32, t: f32) -> f32 {
    let pva = quartic_eval2(c0, c1, c2, c3, c4, t);
    let t1 = fma(pva.x, pva.y / fma(- pva.y, pva.y, pva.x * pva.z), t);
    return select(t, t1, abs(quartic_eval(c0, c1, c2, c3, c4, t1)) <= abs(pva.x));
}

fn bezier2o2dCircleIntersect(bez0: Bezier2o2d, center: vec2f, radius: f32) -> array<vec2f, 4> {
    let v0div2 = bez0[1] - bez0[0];
    let adiv2 = bez0[2] - bez0[1] - v0div2;
    let v0 = v0div2 + v0div2;
    let a = adiv2 + adiv2;
    let relstart = bez0[0] - center;
    let negRadiusSquared = - (radius * radius);
    let c0 = fma(relstart.y, relstart.y, fma(relstart.x, relstart.x, negRadiusSquared)) - fma(radius, radius, negRadiusSquared);
    let c1 = 2 * dot2dAccurate(v0, relstart);
    let c2 = fma(a.y, relstart.y, fma(a.x, relstart.x, mag_sq(v0)));
    let c3 = dot2dAccurate(a, v0);
    let c4 = mag_sq(adiv2);
    let t = quartic_findRoots(c0, c1, c2, c3, c4);
    let t2 = vec4f(quartic_halleyOnceIfBetter(c0, c1, c2, c3, c4, t.x), quartic_halleyOnceIfBetter(c0, c1, c2, c3, c4, t.y), quartic_halleyOnceIfBetter(c0, c1, c2, c3, c4, t.z), quartic_halleyOnceIfBetter(c0, c1, c2, c3, c4, t.w));
    return array(filteredEvalPreproc(bez0[0], v0, adiv2, t2.x), filteredEvalPreproc(bez0[0], v0, adiv2, t2.y), filteredEvalPreproc(bez0[0], v0, adiv2, t2.z), filteredEvalPreproc(bez0[0], v0, adiv2, t2.w));
}

fn det2x2Accurate(a: vec2f, b: vec2f) -> f32 {
    return prodDiffAccurate(a.x, b.y, a.y, b.x);
}

fn det2x2(a: vec2f, b: vec2f) -> f32 {
    return fma(a.x, b.y, - (a.y * b.x));
}

fn twoBezier2o2dsIntersect(bez0: Bezier2o2d, bez1: Bezier2o2d) -> array<vec2f, 4> {
    // find t vals such that (Bezier2o2d.eval bez0 t) gives a point on bez1
    let a1div2 = bez1[1] - bez1[0];
    let a2 = bez1[2] - bez1[1] - a1div2;
    if a2.x == 0 && a2.y == 0 {
        // bez1 has no acceleration. In this case, the usual quartic equation collapses to 0 == 0.
        // There are 2 options for handling this: nudge bez1[1] towards an endpoint and proceed with the quartic, or use our quadratic-bezier-line intersector.
        let bez1Normal = normalize(vec2f(- a1div2.y, a1div2.x));
        let z = bezier2o2dLineIntersect(bez0, bez1Normal, dot2dAccurate(bez1Normal, bez1[0]));
        return array(z[0], z[1], vec2f(MAX_F32, MAX_F32), vec2f(MAX_F32, MAX_F32));
    }
    else {
        let a0 = bez1[0] - bez0[0];
        let a1 = a1div2 + a1div2;
        let p1div2 = bez0[1] - bez0[0];
        let p1 = p1div2 + p1div2;
        let p2 = bez0[2] - bez0[1] - p1div2;
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
        let c2 = fma(a2p1, a2p1, prodDiffAccurate(a0a2times2, a2p2, a1a2, a1p2));
        let c3 = (a2p1 + a2p1) * a2p2;
        let c4 = a2p2 * a2p2;
        let ts2 = quartic_findRoots(c0, c1, c2, c3, c4);
        let ts = vec4f(quartic_halleyOnceIfBetter(c0, c1, c2, c3, c4, ts2.x), quartic_halleyOnceIfBetter(c0, c1, c2, c3, c4, ts2.y), quartic_halleyOnceIfBetter(c0, c1, c2, c3, c4, ts2.z), quartic_halleyOnceIfBetter(c0, c1, c2, c3, c4, ts2.w));
        let vdiv2 = bez0[1] - bez0[0];
        let adiv2 = bez0[2] - bez0[1] - vdiv2;
        let v = vdiv2 + vdiv2;
        //let bez1EndpointDiff = bez1[2] vec2f.- bez1[0] in // for secondary filtering
        //let sideInfo = det2x2Accurate(bez1EndpointDiff, a1) in // for secondary filtering
        //let finish t = f32.(
        //  if ! (f64 0 < t && t < f64 1) then vec2f(MAX_F32, MAX_F32) else
        //  let q = tuple2map2 (fma t) adiv2 v;
        //  let r = tuple2map2 (fma t) q (vec2f.neg a0);
        //  if f64 0 <= sideInfo * det2x2Accurate(bez1EndpointDiff, r) // If this secondary filtering step incorrectly deletes any points (most likely with nearly-straight curves), it can be relaxed or removed.
        //    then tuple2map2 (fma t) q bez0[0]
        //    else vec2f(MAX_F32, MAX_F32)
        //);
        //let ps = tuple4map finish ts;

        return array(filteredEvalPreproc(bez0[0], v, adiv2, ts.x), filteredEvalPreproc(bez0[0], v, adiv2, ts.y), filteredEvalPreproc(bez0[0], v, adiv2, ts.z), filteredEvalPreproc(bez0[0], v, adiv2, ts.w));
    }
}

fn complex_mulAdd(a: vec2f, b: vec2f, c: vec2f) -> vec2f {
    return vec2f(fma((- a.y), b.y, fma(a.x, b.x, c.x)), fma(a.y, b.x, fma(a.x, b.y, c.y)));
}

fn twoCirclesIntersect(c0_center: vec2f, c0_radius: f32, c1_center: vec2f, c1_radius: f32) -> array<vec2f, 2> {
    let centerDiff = c1_center - c0_center;
    let rcd = recipMag(centerDiff);
    let r0 = c0_radius * rcd;
    let r1 = c1_radius * rcd;
    let a = fma(prodDiffAccurate(r0, r0, r1, r1), 0.5, 0.5);
    let z = vec2f(a, sqrt(prodDiffAccurate(r0, r0, a, a)));
    return array(complex_mulAdd(z, centerDiff, c0_center), complex_mulAdd(conj(z), centerDiff, c0_center));
}

fn sdf_halfPlane(normal: vec2f, shift: f32, pos: vec2f) -> f32 {
    return fma(pos.y, normal.y, fma(pos.x, normal.x, - shift));
}

fn circleLineIntersect(center: vec2f, radius: f32, normal: vec2f, shift: f32) -> array<vec2f, 2> {
    let perpOffset = - sdf_halfPlane(normal, shift, center);
    let z = vec2f(perpOffset, sqrt(prodDiffAccurate(radius, radius, perpOffset, perpOffset)));
    return array(complex_mulAdd(z, normal, center), complex_mulAdd(conj(z), normal, center));
}

fn halfPlaneToLineFunc(normal: vec2f, shift: f32, x: f32) -> f32 {
    return fma(shift, normal.y, fma(shift, normal.x, (- x) * normal.x / normal.y));
}

fn twoLinesIntersect(a_normal: vec2f, a_shift: f32, b_normal: vec2f, b_shift: f32) -> vec2f {
    let det = det2x2Accurate(a_normal, b_normal);
    let x = fma(a_shift, b_normal.y, - b_shift * a_normal.y) / det;
    let y = fma(a_normal.x, b_shift, - b_normal.x * a_shift) / det;
    return vec2f(x, y);
}

@group(0) @binding(0)
var<storage, read> shapes: array<f32>;
// The size of this must be max_points(arrayLength(shape_idx))
@group(0) @binding(1)
var<storage, read_write> points: array<vec2f>;
/// Tells us where in the shapes expression tree each shape lives
@group(0) @binding(2)
var<storage, read> shape_idx: array<u32>;

@group(0) @binding(3)
var<storage, read_write> total_offset: atomic<u32>;

fn triangle_idx(i: u32) -> u32 {
    return u32(ceil((sqrt(f32(9 + 8 * i)) - 1.0) / 2.0)) - 1;
}

fn triangle_count(n: u32) -> u32 {
    return (n * (1 + n)) / 2;
}

fn rank(i: u32, j: u32) -> u32 {
    return triangle_count(i + j);
}

fn unrank(x: u32) -> vec2<u32> {
    let i = triangle_idx(x);
    let j = x - triangle_count(i);
    return vec2<u32>(i, j);
}

const WG_SIZE = 128u;

var<workgroup> sh_offsets: array<u32, WG_SIZE>;
var<workgroup> sh_offset: u32;

// Must be invoked with a number of workgroups equal to triangle_count(arrayLength(shape_idx)) / WG_SIZE
@compute @workgroup_size(128)
fn implied_points(@builtin(global_invocation_id) g_id: vec3<u32>, @builtin(local_invocation_index) id: u32, @builtin(workgroup_id) w_id: vec3<u32>) {
    let max_count = min(WG_SIZE, triangle_count(arrayLength(&shape_idx)) - w_id.x * WG_SIZE);
    if id >= max_count {
        return;
    }

    let idx = unrank(g_id.x);
    var intersections = array(MAX_VEC, MAX_VEC, MAX_VEC, MAX_VEC);

    // Calculate intersection points at our offset
    if idx.x == idx.y {
        let i = shape_idx[idx.x];
        let v = bitcast<u32>(shapes[i]) & OP_MASK;
        if v == SHAPE_BEZIER {
            // Output the first and last point of the bezier
            intersections[0] = vec2f(shapes[i + 1], shapes[i + 2]);
            intersections[1] = vec2f(shapes[i + 5], shapes[i + 6]);
        }
    }
    else {
        var i = shape_idx[idx.x];
        var j = shape_idx[idx.y];
        var l = bitcast<u32>(shapes[i]) & OP_MASK;
        var r = bitcast<u32>(shapes[j]) & OP_MASK;
        switch (r << 3) | l {
            case (SHAPE_CIRCLE << 3) | SHAPE_CIRCLE: {
                let pt = twoCirclesIntersect(vec2f(shapes[i + 1], shapes[i + 2]), shapes[i + 3], vec2f(shapes[j + 1], shapes[j + 2]), shapes[j + 3]);
                intersections[0] = select(MAX_VEC, pt[0], isBoundaryPoint(pt[0]));
                intersections[1] = select(MAX_VEC, pt[1], isBoundaryPoint(pt[1]));
            }
            case (SHAPE_LINE << 3) | SHAPE_LINE: {
                let pt = twoLinesIntersect(vec2f(shapes[i + 1], shapes[i + 2]), shapes[i + 3], vec2f(shapes[j + 1], shapes[j + 2]), shapes[j + 3]);
                intersections[0] = select(MAX_VEC, pt, isBoundaryPoint(pt));
            }
            case (SHAPE_LINE << 3) | SHAPE_CIRCLE, (SHAPE_CIRCLE << 3) | SHAPE_LINE: {
                if r == SHAPE_CIRCLE {
                    var t = i;
                    i = j;
                    j = t;
                }
                let pt = circleLineIntersect(vec2f(shapes[i + 1], shapes[i + 2]), shapes[i + 3], vec2f(shapes[j + 1], shapes[j + 2]), shapes[j + 3]);
                intersections[0] = select(MAX_VEC, pt[0], isBoundaryPoint(pt[0]));
                intersections[1] = select(MAX_VEC, pt[1], isBoundaryPoint(pt[1]));
            }
            case (SHAPE_BEZIER << 3) | SHAPE_BEZIER: {
                let bl = Bezier2o2d(vec2f(shapes[i + 1], shapes[i + 2]), vec2f(shapes[i + 3], shapes[i + 4]), vec2f(shapes[i + 5], shapes[i + 6]));
                let br = Bezier2o2d(vec2f(shapes[j + 1], shapes[j + 2]), vec2f(shapes[j + 3], shapes[j + 4]), vec2f(shapes[j + 5], shapes[j + 6]));
                let pt = twoBezier2o2dsIntersect(bl, br);
                intersections[0] = select(MAX_VEC, pt[0], isBoundaryPoint(pt[0]));
                intersections[1] = select(MAX_VEC, pt[1], isBoundaryPoint(pt[1]));
                intersections[2] = select(MAX_VEC, pt[2], isBoundaryPoint(pt[2]));
                intersections[3] = select(MAX_VEC, pt[3], isBoundaryPoint(pt[3]));
            }
            case (SHAPE_BEZIER << 3) | SHAPE_CIRCLE, (SHAPE_CIRCLE << 3) | SHAPE_BEZIER: {
                if r == SHAPE_BEZIER {
                    var t = i;
                    i = j;
                    j = t;
                }
                let bl = Bezier2o2d(vec2f(shapes[i + 1], shapes[i + 2]), vec2f(shapes[i + 3], shapes[i + 4]), vec2f(shapes[i + 5], shapes[i + 6]));
                let pt = bezier2o2dCircleIntersect(bl, vec2f(shapes[j + 1], shapes[j + 2]), shapes[j + 3]);
                intersections[0] = select(MAX_VEC, pt[0], isBoundaryPoint(pt[0]));
                intersections[1] = select(MAX_VEC, pt[1], isBoundaryPoint(pt[1]));
                intersections[2] = select(MAX_VEC, pt[2], isBoundaryPoint(pt[2]));
                intersections[3] = select(MAX_VEC, pt[3], isBoundaryPoint(pt[3]));
            }
            case (SHAPE_LINE << 3) | SHAPE_BEZIER, (SHAPE_BEZIER << 3) | SHAPE_LINE: {
                if r == SHAPE_BEZIER {
                    var t = i;
                    i = j;
                    j = t;
                }
                let bl = Bezier2o2d(vec2f(shapes[i + 1], shapes[i + 2]), vec2f(shapes[i + 3], shapes[i + 4]), vec2f(shapes[i + 5], shapes[i + 6]));

                let pt = bezier2o2dLineIntersect(bl, vec2f(shapes[j + 1], shapes[j + 2]), shapes[j + 3]);
                intersections[0] = select(MAX_VEC, pt[0], isBoundaryPoint(pt[0]));
                intersections[1] = select(MAX_VEC, pt[1], isBoundaryPoint(pt[1]));
            }
            default : { }
        }
    }

    var count = 0u;

    for (var i = 0u; i < 4u; i++) {
        count += select(0u, 1u, isFinite2(intersections[i]));
    }

    // Assemble a prefix sum of valid points
    sh_offsets[id] = count;
    if id == max_count - 1 {
        sh_offset = 0;
    }

    for (var i = 0u; i < firstLeadingBit(max_count) + 1u; i += 1u) {
        workgroupBarrier();
        if id >= (1u << i) {
            count += sh_offsets[id - (1u << i)];
        }
        workgroupBarrier();
        sh_offsets[id] = count;
    }

    // We perform an atomic add on our global offset to reserve a chunk for ourselves in the output, using
    // the maximum sum in our workgroup chunk. Only the last thread does this atomic operation.
    if id == max_count - 1 {
        sh_offset = atomicAdd(&total_offset, sh_offsets[max_count - 1]);
    }

    // We then need one last barrier to ensure the shared offset is written to before the rest of the
    // workgroup threads read it.
    workgroupBarrier();

    // Assign all valid points to sequential indices in the output using the cumulative sum
    var index = 0u;

    for (var i = 0u; i < 4u; i++) {
        if isFinite2(intersections[i]) {
            index += 1;
            points[sh_offset + count - index] = intersections[i];
        }
    }
    //points[g_id.x] = vec2f(bitcast<f32>(sh_offset), bitcast<f32>(sh_offsets[max_count - 1]));
}