fn linearstep(low: f32, high: f32, x: f32) -> f32 {
  return clamp((x - low) / (high - low), 0.0f, 1.0f);
}

fn u32_to_vec4(c: u32) -> vec4<f32> {
  return vec4<f32>(f32((c & 0xff000000u) >> 24u) / 255.0, f32((c & 0x00ff0000u) >> 16u) / 255.0, f32((c & 0x0000ff00u) >> 8u) / 255.0, f32(c & 0x000000ffu) / 255.0);
}

fn srgb_to_linear(c: f32) -> f32 {
  if c <= 0.04045 {
    return c / 12.92;
  }
  else {
    return pow((c + 0.055) / 1.055, 2.4);
  }
}

fn srgb_to_linear_vec4(c: vec4<f32>) -> vec4<f32> {
  return vec4f(srgb_to_linear(c.x), srgb_to_linear(c.y), srgb_to_linear(c.z), c.w);
}

const MAX_F32: f32 = 3.402823466e+38;

fn isFinite(x: f32) -> bool {
  return abs(x) < MAX_F32;
}

@group(0) @binding(0)
var<storage, read> shapes: array<f32>;

@group(0) @binding(1)
var<storage, read> points: array<vec2f>;

@group(0) @binding(2)
var<storage, read> quadtree: array<u32>;

struct Config {
  dim: vec2f,
  pos: vec2f,
}

var<immediate> config: Config;

struct VertexOutput {
  @invariant @builtin(position) position: vec4<f32>,
  @location(0) uv: vec2f,
  //@location(1) @interpolate(flat) index: u32,
}

struct Circle {
  center: vec2f,
  radius: f32,
}

struct HalfPlane {
  normal: vec2f,
  shift: f32,
}

@vertex
fn vs_main(@builtin(vertex_index) idx: u32) -> VertexOutput {
  var pos = array<vec2f, 3>(vec2f(- 1.0, 3.0), vec2f(3.0, - 1.0), vec2f(- 1.0, - 1.0));
  let outpos = vec4f(pos[idx], 0.0, 1.0);
  let uv = pos[idx] * 0.5 + 0.5;
  return VertexOutput(outpos, (uv - vec2f(0.5, 0.5)) * config.dim + config.pos);
}

// see https://pharr.org/matt/blog/2019/11/03/difference-of-floats or "Further Analysis of Kahan's Algorithm for the Accurate Computation of 2x2 Determinants"
fn dot2dAccurate(a: vec2f, b: vec2f) -> f32 {
  let t = a.x * b.x;
  return fma(a.y, b.y, t) + fma(a.x, b.x, - t);
}

fn mag_sq(v: vec2f) -> f32 {
  return dot(v, v);
}

fn mag(v: vec2f) -> f32 {
  return sqrt(mag_sq(v));
}

fn recipMag(v: vec2f) -> f32 {
  return inverseSqrt(mag_sq(v));
}

fn sdf_point(p: vec2f, pos: vec2f) -> f32 {
  return mag(pos - p);
}

fn sdf_disk(d: Circle, pos: vec2f) -> f32 {
  return mag(pos - d.center) - d.radius;
}

fn sdf_halfPlane(hp: HalfPlane, pos: vec2f) -> f32 {
  return fma(pos.y, hp.normal.y, fma(pos.x, hp.normal.x, - hp.shift));
}

fn scale_add(scale: f32, p: vec2f, c: vec2f) -> vec2f {
  return vec2f(fma(scale, p.x, c.x), fma(scale, p.y, c.y));
}

fn diskNBP(disk: Circle, pos: vec2f) -> vec2f {
  let v = (pos - disk.center);
  let q = disk.radius * recipMag(v);
  return select(vec2f(disk.center.x + disk.radius, disk.center.y), scale_add(q, v, disk.center), isFinite(q));
}

fn halfPlaneNBP(hp: HalfPlane, pos: vec2f) -> vec2f {
  return scale_add(- sdf_halfPlane(hp, pos), hp.normal, pos);
}

alias Bezier2o2d = array<vec2f, 3>;

fn evalPreproc(p0: vec2f, v0: vec2f, adiv2: vec2f, t: f32) -> vec2f {
  return scale_add(t, scale_add(t, adiv2, v0), p0);
}

fn eval_bezier(bez: Bezier2o2d, t: f32) -> vec2f {
  let v0div2 = bez[1] - bez[0];
  return evalPreproc(bez[0], (v0div2 + v0div2), (bez[2] - bez[1] - v0div2), t,);
}

fn filteredEval(bez: Bezier2o2d, t: f32) -> vec2f {
  return select(vec2f(MAX_F32, MAX_F32), eval_bezier(bez, t), 0.0 < t && t < 1.0);
}

fn from_fraction(n: i32, d: i32) -> f32 {
  return f32(n) / f32(d);
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

fn rcbrtPositiveNormal(x: f32) -> f32 {
  // The extra performance here doesn't seem to be necessary
  return pow(x, - 1.0 / 3.0);

  let y1 = bitcast<f32>(0x54a232a8u - (bitcast<u32>(x) / 3));
  let p1 = fma((x * y1), y1 * y1, - 1.0);
  let y = fma(fma(p1, 2.0 / 9.0, - 1.0 / 3.0), p1 * y1, y1);
  let p = fma(x * y, y * y, - 1.0);
  //f32.(fma p (y * from_fraction (-1) 3) y) // Slightly biased toward 0. -1.77ulp < err < +0.96ulp. -2**-22.68 < relerr < 2**-23.51.
  return fma(fma(p, 2.0 / 9.0, - 1.0 / 3.0), p * y, y);
  // -0.99ulp < err < +0.99ulp. -2**-23.44 < relerr < 2**-23.44.
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

fn quartic_eval(c0: f32, c1: f32, c2: f32, c3: f32, c4: f32, t: f32) -> f32 {
  return fma(t, fma(t, fma(t, fma(t, c4, c3), c2), c1), c0);
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
  let d0divn2 = fma(c2divn3c3, fma(c2divn3c3, c2divn3c3, c1divc3 * (- 0.5)), c0divc3 * (- 0.5),);
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

fn findLocallyNearestPoints(bez: Bezier2o2d, pos: vec2f) -> array<vec2f, 2> {
  let t = findLocallyNearestTVals(bez, pos);
  return array(filteredEval(bez, t.x), filteredEval(bez, t.y));
}

const OP_UNION: u32 = 0;
const OP_INTERSECT: u32 = 1;
const SHAPE_CIRCLE: u32 = 2;
const SHAPE_LINE: u32 = 3;
const SHAPE_BEZIER: u32 = 4;
const SHAPE_POLYGON: u32 = 5;
const OP_NEGATE: u32 = 8;
const OP_HOLLOW: u32 = 16;
const OP_MASK: u32 = OP_NEGATE - 1;

fn psdf_halfplane(hp: HalfPlane, pos: vec2f) -> f32 {
  return fma(pos.y, hp.normal.y, fma(pos.x, hp.normal.x, (- hp.shift)));
}

fn psdf_point(p: vec2f, pos: vec2f) -> f32 {
  return mag(pos - p);
}

fn psdf_disk(d: Circle, pos: vec2f) -> f32 {
  return psdf_point(d.center, pos) - d.radius;
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
        stack[len] = unary_op(psdf_disk(Circle(vec2f(shapes[i + 1], shapes[i + 2]), shapes[i + 3]), pos), op);
        len += 1;
        i += 4;
      }
      case SHAPE_BEZIER: {
        stack[len] = unary_op(psdf_bez(Bezier2o2d(vec2f(shapes[i + 1], shapes[i + 2]), vec2f(shapes[i + 3], shapes[i + 4]), vec2f(shapes[i + 5], shapes[i + 6])), pos), op);
        len += 1;
        i += 7;
      }
      case SHAPE_LINE: {
        stack[len] = unary_op(psdf_halfplane(HalfPlane(vec2f(shapes[i + 1], shapes[i + 2]), shapes[i + 3]), pos), op);
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

fn isBoundaryPoint(pos: vec2f) -> bool {
  return abs(shapeField(pos)) <= BOUNDARY_THRESHOLD;
}

fn shapefunc(pos: vec2f) -> vec2f {
  var nearest = vec2f(MAX_F32, MAX_F32);
  var lastdist_sq = MAX_F32;
  for (var i = 0u; i < arrayLength(&points); i++) {
    let dist = mag_sq(points[i] - pos);

    // We can skip isBoundaryPoint here because the intersection points are prefiltered
    if dist < lastdist_sq {
      lastdist_sq = dist;
      nearest = points[i];
    }
  }

  var p = vec2f(MAX_F32, MAX_F32);

  for (var i = 0u; i < arrayLength(&shapes);) {
    let opshape = bitcast<u32>(shapes[i]);

    switch opshape & OP_MASK {
      case OP_UNION, OP_INTERSECT: {
        i += 1;
        continue;
      }
      case SHAPE_CIRCLE: {
        p = diskNBP(Circle(vec2f(shapes[i + 1], shapes[i + 2]), shapes[i + 3]), pos);
      }
      case SHAPE_BEZIER: {
        let pt = findLocallyNearestPoints(Bezier2o2d(vec2f(shapes[i + 1], shapes[i + 2]), vec2f(shapes[i + 3], shapes[i + 4]), vec2f(shapes[i + 5], shapes[i + 6])), pos);
        p = pt[0];

        let dist = mag_sq(pt[1] - pos);
        if dist < lastdist_sq && isBoundaryPoint(pt[1]) {
          lastdist_sq = dist;
          nearest = pt[1];
        }
      }
      case SHAPE_LINE: {
        p = halfPlaneNBP(HalfPlane(vec2f(shapes[i + 1], shapes[i + 2]), shapes[i + 3]), pos);
      }
      default : {
        return vec2f(MAX_F32, MAX_F32);
      }
    }

    let dist = mag_sq(p - pos);
    if dist < lastdist_sq && isBoundaryPoint(p) {
      lastdist_sq = dist;
      nearest = p;
    }

    i += select(4u, 7u, (opshape & OP_MASK) == SHAPE_BEZIER);
  }

  return nearest;
}

fn indirect_shapeField(pos: vec2f, start: u32) -> f32 {
  let end = start + quadtree[start];
  var stack = array<f32, 32>();
  var len = 0;

  for (var idx = start + 1; idx < end; idx++) {
    let i = quadtree[idx];
    let op = bitcast<u32>(shapes[i]);
    switch op & OP_MASK {
      case OP_UNION, OP_INTERSECT: {
        let r = stack[len - 1];
        let l = stack[len - 2];
        stack[len - 2] = unary_op(select(op_intersect(l, r), op_union(l, r), (op & OP_MASK) == OP_UNION), op);
        len -= 1;
      }
      case SHAPE_CIRCLE: {
        stack[len] = unary_op(psdf_disk(Circle(vec2f(shapes[i + 1], shapes[i + 2]), shapes[i + 3]), pos), op);
        len += 1;
      }
      case SHAPE_BEZIER: {
        stack[len] = unary_op(psdf_bez(Bezier2o2d(vec2f(shapes[i + 1], shapes[i + 2]), vec2f(shapes[i + 3], shapes[i + 4]), vec2f(shapes[i + 5], shapes[i + 6])), pos), op);
        len += 1;
      }
      case SHAPE_LINE: {
        stack[len] = unary_op(psdf_halfplane(HalfPlane(vec2f(shapes[i + 1], shapes[i + 2]), shapes[i + 3]), pos), op);
        len += 1;
      }
      default : {
        return 0.0;
      }
    }
  }

  return stack[0];
}

fn indirect_isBoundaryPoint(pos: vec2f, start: u32) -> bool {
  return abs(indirect_shapeField(pos, start)) <= BOUNDARY_THRESHOLD;
}

fn indirect_shapefunc(pos: vec2f, start: u32) -> vec2f {
  let end = start + 1 + quadtree[start];

  var nearest = vec2f(MAX_F32, MAX_F32);
  var lastdist_sq = MAX_F32;
  for (var i = 0u; i < arrayLength(&points); i++) {
    let dist = mag_sq(points[i] - pos);

    // We can skip isBoundaryPoint here because the intersection points are prefiltered
    if dist < lastdist_sq {
      lastdist_sq = dist;
      nearest = points[i];
    }
  }

  var p = vec2f(MAX_F32, MAX_F32);

  for (var idx = start + 1; idx < end; idx++) {
    let i = quadtree[idx];
    let opshape = bitcast<u32>(shapes[i]);

    switch opshape & OP_MASK {
      case OP_UNION, OP_INTERSECT: {
        continue;
      }
      case SHAPE_CIRCLE: {
        p = diskNBP(Circle(vec2f(shapes[i + 1], shapes[i + 2]), shapes[i + 3]), pos);
      }
      case SHAPE_BEZIER: {
        let pt = findLocallyNearestPoints(Bezier2o2d(vec2f(shapes[i + 1], shapes[i + 2]), vec2f(shapes[i + 3], shapes[i + 4]), vec2f(shapes[i + 5], shapes[i + 6])), pos);
        p = pt[0];

        let dist = mag_sq(pt[1] - pos);
        if dist < lastdist_sq && isBoundaryPoint(pt[1]) {
          lastdist_sq = dist;
          nearest = pt[1];
        }
      }
      case SHAPE_LINE: {
        p = halfPlaneNBP(HalfPlane(vec2f(shapes[i + 1], shapes[i + 2]), shapes[i + 3]), pos);
      }
      default : {
        return vec2f(MAX_F32, MAX_F32);
      }
    }

    let dist = mag_sq(p - pos);
    if dist < lastdist_sq && isBoundaryPoint(p) {
      lastdist_sq = dist;
      nearest = p;
    }
  }

  return nearest;
}

const QUAD_CHILD: u32 = (1u << 31u);

fn debug_viz(start: u32) -> vec4f {
  let end = start + quadtree[start] + 1;
  var colors = array(vec3f(0.0, 0.0, 0.0), vec3f(0.0, 0.0, 0.0), vec3f(0.0, 0.0, 0.0));

  for (var idx = start + 1; idx < end; idx++) {
    let i = quadtree[idx];
    let opshape = bitcast<u32>(shapes[i]);
    if ((opshape & OP_MASK) > 1) {
      if i < 4 {
        colors[0].r = 1.0;
      }
      else if i < 8 {
        colors[1].g = 1.0;
      }
      else {
        colors[2].b = 1.0;
      }
    }
  }

  return vec4f(colors[0] + colors[1] + colors[2], 1.0);
}

fn traverse(pos: vec2f) -> vec2f {
  var dim = config.dim;
  var idx = 0u;
  var offset = config.dim * - 0.5;

  //debug
  let flip = vec2f(pos.x, pos.y);

  while (idx < arrayLength(&quadtree)) {
    dim *= 0.5;
    let p = flip;
    let child = select(0u, 1u, p.x > offset.x + dim.x) | select(0u, 2u, p.y > offset.y + dim.y);
    offset.x += dim.x * f32(child & 1);
    offset.y += dim.y * f32((child & 2) >> 1);

    if (quadtree[idx + child] & QUAD_CHILD) != 0 {
      //return debug_viz(quadtree[idx + child] & (~QUAD_CHILD));
      return indirect_shapefunc(pos, quadtree[idx + child] & (~QUAD_CHILD));
      //return shapefunc(pos);
    }

    idx = quadtree[idx + child];
    if idx == 0 {
      // shouldn't happen if quadtree is valid
      break;
    }
  }

  return vec2f(0.0);
}

@fragment
fn tdf(input: VertexOutput) -> @location(0) vec4f {
  let line_width: f32 = 1.25;
  let band_width: f32 = 8;
  let pos = input.uv;
  //return vec4f(pos.x / 400.0, pos.y / 300.0, 0, 1);
  //return traverse(pos);

  //let dist = mag(pos - shapefunc(pos));
  let dist = mag(pos - traverse(pos));
  let field = shapeField(pos);
  let inside = field < 0;
  let in_line = dist <= (line_width / 2);
  let in_band = dist % (band_width * 2) >= band_width;
  let fade = select(32 * log2(dist), 128, dist < 1);
  let primary_colorval = select(fade, 255.0, in_line || 255 < fade);
  var secondary_colorval = select(0.0, 255.0, in_line);
  if in_band {
    secondary_colorval = primary_colorval / 2;
  }
  let a = 255.0;
  let r = select(primary_colorval, secondary_colorval, inside);
  let b = select(secondary_colorval, primary_colorval, inside);
  let g = secondary_colorval;
  //return srgb_to_linear_vec4(vec4f(field, 0.0, 0.0, 1.0));
  return srgb_to_linear_vec4(vec4f(r / 255.0, g / 255.0, b / 255.0, a / 255.0));
}

@fragment
fn fs_main(input: VertexOutput) -> @location(0) vec4f {
  let line_width: f32 = 1.25;
  let band_width: f32 = 8;
  let pos = input.uv;
  return srgb_to_linear_vec4(vec4f(pos.x, pos.y, 0.0, 1.0));
  //return srgb_to_linear_vec4(vec4f(r / 255.0, g / 255.0, b / 255.0, a / 255.0));
}