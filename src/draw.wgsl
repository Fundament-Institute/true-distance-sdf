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
var<storage, read> kdtree: array<vec4<u32>>;
//var<storage, read> points: array<vec2f>;

@group(0) @binding(2)
var<storage, read> quadtree: array<u32>;

fn get_nearest_point(query: vec2f) -> vec3f {
  var nearest_item = vec2f(bitcast<f32>(kdtree[0].x), bitcast<f32>(kdtree[0].y));
  if nearest_item.x == MAX_F32 {
    return vec3f(MAX_F32, MAX_F32, MAX_F32);
  }
  var best_distance = mag_sq(query - nearest_item);
  var stack = array<vec3<u32>, 64>();

  stack[0] = vec3u(0, arrayLength(&kdtree), 0);
  var len = 1;

  while len > 0 {
    len -= 1;
    let begin = stack[len].x;
    let end = stack[len].y;
    let axis = stack[len].z;

    let mid_idx = begin + ((end - begin) / 2);
    let item = vec2f(bitcast<f32>(kdtree[mid_idx].x), bitcast<f32>(kdtree[mid_idx].y));

    let squared_distance = mag_sq(query - item);
    if squared_distance < best_distance {
      nearest_item = item;
      best_distance = squared_distance;
      if best_distance == 0 {
        continue;
      }
    }
    let mid_pos = select(item.y, item.x, axis == 0);
    var b1_start = begin;
    var b1_end = mid_idx;
    var b2_start = mid_idx + 1;
    var b2_end = end;

    if query[axis] >= mid_pos {
      b2_start = begin;
      b2_end = mid_idx;
      b1_start = mid_idx + 1;
      b1_end = end;
    }
    if b2_start != b2_end {
      let diff = query[axis] - mid_pos;
      if diff * diff < best_distance {
        stack[len] = vec3u(b2_start, b2_end, (axis + 1) % 2);
        len += 1;
      }
    }
    if b1_start != b1_end {
      stack[len] = vec3u(b1_start, b1_end, (axis + 1) % 2);
      len += 1;
    }
  }

  return vec3f(nearest_item, best_distance);
}

/*fn get_nearest_point(pos: vec2f) -> vec3f {
  var nearest = vec2f(MAX_F32, MAX_F32);
  var lastdist_sq = MAX_F32;
  for (var i = 0u; i < arrayLength(&points); i++) {
    if !isFinite(points[i].x) {
      break;
    }

    let dist = mag_sq(points[i] - pos);

    // We can skip isBoundaryPoint here because the intersection points are prefiltered
    if dist < lastdist_sq {
      lastdist_sq = dist;
      nearest = points[i];
    }
  }

  return vec3f(nearest, lastdist_sq);
}*/

struct Config {
  dim: vec2f,
  pos: vec2f,
  mouse: vec2f,
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
const SHAPE_CONSTANT: u32 = 5;
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
  var stack = array<f32, 64>();
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
  var nlast = get_nearest_point(pos);
  var nearest = nlast.xy;
  var lastdist_sq = nlast.z;

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
  let end = start + quadtree[start] + 1;
  var stack = array<f32, 32>();
  var len = 0;

  for (var idx = start + 1; idx < end; idx++) {
    var i = quadtree[idx];
    if i == 0xFFFFFFFF {
      stack[len] = MAX_F32;
      len += 1;
      continue;
    }
    else if i == 0xFFFFFFFE {
      stack[len] = - MAX_F32;
      len += 1;
      continue;
    }

    /*if (i & 1) != 0 {
      stack[len] = bitcast<f32>(i);
      len += 1;
      continue;
    }*/

    i = i >> 1;

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

  var nlast = get_nearest_point(pos);
  var nearest = nlast.xy;
  var lastdist_sq = nlast.z;

  var p = vec2f(MAX_F32, MAX_F32);

  for (var idx = start + 1; idx < end; idx++) {
    var i = quadtree[idx];
    if i == 0xFFFFFFFF || i == 0xFFFFFFFE {
      //if (i & 1) != 0 {
      continue;
    }
    i = i >> 1;

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
        if dist < lastdist_sq && indirect_isBoundaryPoint(pt[1], start) {
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
    if dist < lastdist_sq && indirect_isBoundaryPoint(p, start) {
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

  while (idx < arrayLength(&quadtree)) {
    dim *= 0.5;
    let child = select(0u, 1u, pos.x > offset.x + dim.x) | select(0u, 2u, pos.y > offset.y + dim.y);
    offset.x += dim.x * f32(child & 1);
    offset.y += dim.y * f32((child & 2) >> 1);

    if (quadtree[idx + child] & QUAD_CHILD) != 0 {
      return indirect_shapefunc(pos, quadtree[idx + child] & (~QUAD_CHILD));
    }

    idx = quadtree[idx + child];
    if idx == 0 {
      break;
    }
    // TODO: shouldn't happen if quadtree is valid
  }

  return vec2f(0.0);
}

fn boolf_halfplane(hp: HalfPlane, pos: vec2f) -> bool {
  return psdf_halfplane(hp, pos) < 0.0;
}

fn boolf_point(p: vec2f, pos: vec2f) -> bool {
  return false;
}

fn boolf_disk(d: Circle, pos: vec2f) -> bool {
  return mag_sq(pos - d.center) < d.radius * d.radius;
}

fn boolf_bez(bez: Bezier2o2d, pos: vec2f) -> bool {
  return false;
}

fn boolf_negate(x: bool) -> bool {
  return !x;
}

fn boolf_hollow(x: bool) -> bool {
  return false;
}

fn boolf_union(l: bool, r: bool) -> bool {
  return l || r;
}

fn boolf_intersect(l: bool, r: bool) -> bool {
  return l && r;
}

fn unary_bool(x: bool, op: u32) -> bool {
  let r = select(x, boolf_negate(x), (op & OP_NEGATE) != 0);
  return select(r, boolf_hollow(r), (op & OP_HOLLOW) != 0);
}

fn shape_boolF(pos: vec2f) -> bool {
  var stack: u32 = 0u;
  var len = 0;

  for (var i = 0u; i < arrayLength(&shapes);) {
    let op = bitcast<u32>(shapes[i]);
    switch op & OP_MASK {
      case OP_UNION, OP_INTERSECT: {
        let r = (stack & (1u << u32(len - 1))) != 0u;
        let l = (stack & (1u << u32(len - 2))) != 0u;
        let b = unary_bool(select(boolf_intersect(l, r), boolf_union(l, r), (op & OP_MASK) == OP_UNION), op);
        let bit = (1u << u32(len - 2));
        stack = (stack & (~bit)) | select(0u, bit, b);
        len -= 1;
        i += 1;
      }
      case SHAPE_CIRCLE: {
        let b = unary_bool(boolf_disk(Circle(vec2f(shapes[i + 1], shapes[i + 2]), shapes[i + 3]), pos), op);
        let bit = (1u << u32(len));
        stack = (stack & (~bit)) | select(0u, bit, b);
        len += 1;
        i += 4;
      }
      case SHAPE_BEZIER: {
        let b = unary_bool(boolf_bez(Bezier2o2d(vec2f(shapes[i + 1], shapes[i + 2]), vec2f(shapes[i + 3], shapes[i + 4]), vec2f(shapes[i + 5], shapes[i + 6])), pos), op);
        let bit = (1u << u32(len));
        stack = (stack & (~bit)) | select(0u, bit, b);
        len += 1;
        i += 7;
      }
      case SHAPE_LINE: {
        let b = unary_bool(boolf_halfplane(HalfPlane(vec2f(shapes[i + 1], shapes[i + 2]), shapes[i + 3]), pos), op);
        let bit = (1u << u32(len));
        stack = (stack & (~bit)) | select(0u, bit, b);
        len += 1;
        i += 4;
      }
      default : {
        return false;
      }
    }
  }

  return (stack & 1) != 0;
}

fn draw_sdf(dist: f32, inside: bool) -> vec4f {
  let d = dist * 0.005;
  // This coloring method taken from Inigo Quilez, used under the MIT license: https://www.shadertoy.com/view/MlKcDD
  var col = select(vec3f(0.9, 0.6, 0.3), vec3f(0.65, 0.85, 1.0), inside);
  col *= 1.0 - exp(- 4.0 * abs(d));
  col *= 0.8 + 0.2 * cos(110.0 * d);
  col = mix(col, vec3(1.0), 1.0 - smoothstep(0.0, 0.01, abs(d)));

  return srgb_to_linear_vec4(vec4f(col, 1.0));
}

@fragment
fn tdf(input: VertexOutput) -> @location(0) vec4f {
  let pos = input.uv;
  //return vec4f(pos.x / 400.0, pos.y / 300.0, 0, 1);
  //return traverse(pos);

  let nearest = traverse(pos);
  //let nearest = shapefunc(pos);
  let dist = mag(pos - nearest);
  //let field = shapeField(pos);
  //let inside = field < 0;
  let inside = shape_boolF(pos);

  var col = draw_sdf(dist, inside);
  /*{
    let m = (config.mouse - (config.dim * 0.5)) * vec2f(1.0, - 1.0) + config.pos;
    let q = traverse(m);
    let d = mag(pos - q);

    //col = vec4f(mix(col.rgb, vec3(1.0, 1.0, 0.0), 1.0 - smoothstep(0.0, 1.0, abs(length(pos - m) - (d / 2.0)))), col.a);
    col = vec4f(mix(col.rgb, vec3(1.0, 1.0, 0.0), 1.0 - smoothstep(0.0, 1.0, length(pos - m) - 5)), col.a);
    col = vec4f(mix(col.rgb, vec3(1.0, 1.0, 0.0), 1.0 - smoothstep(0.0, 1.0, length(pos - q) - 5)), col.a);
  }*/

  return col;
}

// Standard comparison SDFs from Inigo Quilez, used under the MIT license - https://iquilezles.org/articles/distfunctions2d/
fn sdRoundedBox(p: vec2f, b: vec2f, r: vec4f) -> f32 {
  let r2 = select(r.zw, r.xy, p.x > 0.0);
  let r3 = select(r2.y, r2.x, p.y > 0.0);
  let q: vec2f = abs(p) - b + r3;
  return min(max(q.x, q.y), 0.0) + length(max(q, vec2f(0.0, 0.0))) - r3;
}

fn sdCircle(p: vec2f, center: vec2f, r: f32) -> f32 {
  return length(p - center) - r;
}

fn dot2(v: vec2f) -> f32 {
  return dot(v, v);
}

fn sdBezier(pos: vec2f, A: vec2f, B: vec2f, C: vec2f) -> f32 {
  let a = B - A;
  let b = A - 2.0 * B + C;
  let c = a * 2.0;
  let d = A - pos;
  let kk = 1.0 / dot(b, b);
  let kx = kk * dot(a, b);
  let ky = kk * (2.0 * dot(a, a) + dot(d, b)) / 3.0;
  let kz = kk * dot(d, a);
  var res = 0.0;
  let p = ky - kx * kx;
  let p3 = p * p * p;
  let q = kx * (2.0 * kx * kx - 3.0 * ky) + kz;
  var h = q * q + 4.0 * p3;
  if (h >= 0.0) {
    h = sqrt(h);
    let x = (vec2(h, - h) - q) / 2.0;
    let uv = sign(x) * pow(abs(x), vec2(1.0 / 3.0));
    let t = clamp(uv.x + uv.y - kx, 0.0, 1.0);
    res = dot2(d + (c + b * t) * t);
  }
  else {
    let z = sqrt(- p);
    let v = acos(q / (p * z * 2.0)) / 3.0;
    let m = cos(v);
    let n = sin(v) * 1.732050808;
    let t = clamp(vec3f(m + m, - n - m, n - m) * z - kx, vec3f(0.0), vec3f(1.0));
    res = min(dot2(d + (c + b * t.x) * t.x), dot2(d + (c + b * t.y) * t.y));
    // the third root cannot be the closest
    // res = min(res,dot2(d+(c+b*t.z)*t.z));
  }
  return sqrt(res);
}

fn sdOrientedBox(p: vec2f, a: vec2f, b: vec2f, th: f32) -> f32 {
  let l = length(b - a);
  let d = (b - a) / l;
  var q = p - (a + b) * 0.5;
  q = mat2x2(d.x, d.y, - d.y, d.x) * q;
  q = abs(q) - vec2(l * 0.5, th);
  return length(max(q, vec2(0.0))) + min(max(q.x, q.y), 0.0);
}

fn sdTriangleIsosceles(pp: vec2f, q: vec2f) -> f32 {
  var p = pp;
  p.x = abs(p.x);
  let a = p - q * clamp(dot(p, q) / dot(q, q), 0.0, 1.0);
  let b = p - q * vec2(clamp(p.x / q.x, 0.0, 1.0), 1.0);
  let s = - sign(q.y);
  let d = min(vec2(dot(a, a), s * (p.x * q.y - p.y * q.x)), vec2(dot(b, b), s * (p.y - q.y)));
  return - sqrt(d.x) * sign(d.y);
}

// End standard SDFs

// This simulates access patterns from a shader that picks between a few single SDF options by
// reusing our shape storage buffer
@fragment
fn fs_sdf(input: VertexOutput) -> @location(0) vec4f {
  var d = 0.0;
  for (var i = 0u; i < arrayLength(&shapes);) {
    let kind = bitcast<u32>(shapes[0]);
    switch kind {
      case 0 : {
        d += sdRoundedBox(input.uv, vec2f(shapes[1], shapes[2]), vec4f(shapes[3], shapes[4], shapes[5], shapes[6]));
        i += 7;
      }
      case 1 : {
        d += min(sdOrientedBox(input.uv, vec2f(shapes[1], shapes[2]), vec2f(shapes[3], shapes[4]), shapes[5]), sdOrientedBox(input.uv, vec2f(shapes[3], shapes[2]), vec2f(shapes[1], shapes[4]), shapes[5]));
        i += 6;

      }
      case 7 : {
        d += sdTriangleIsosceles(input.uv, vec2f(shapes[1], shapes[2]));
        i += 3;
      }
      case SHAPE_BEZIER: {
        d += sdBezier(input.uv, vec2f(shapes[1], shapes[2]), vec2f(shapes[3], shapes[4]), vec2f(shapes[5], shapes[6]));
        //d += psdf_bez(array(vec2f(shapes[1], shapes[2]), vec2f(shapes[3], shapes[4]), vec2f(shapes[5], shapes[6])), input.uv);
        i += 7;
      }
      case SHAPE_CIRCLE: {
        d += sdCircle(input.uv, vec2f(shapes[1], shapes[2]), shapes[3]);
        i += 4;
      }
      default : {
        d += 0.0;
        i += 1;
      }
    }
  }

  return draw_sdf(d, d < 0.0);
}
