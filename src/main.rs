#![allow(dead_code)]

use core::f32;
use std::collections::{HashMap, HashSet};
use std::hash::Hash;
use std::num::NonZero;
use std::sync::Arc;
use std::thread::Thread;

use euclid::default::Vector2D;
use rand::rngs::Xoshiro128PlusPlus;
use wgpu::util::{BufferInitDescriptor, DeviceExt};
use winit::application::ApplicationHandler;
use winit::dpi::PhysicalPosition;
#[cfg(not(target_family = "wasm"))]
use winit::dpi::PhysicalSize;
use winit::event::{MouseButton, WindowEvent};
use winit::event_loop::{ActiveEventLoop, EventLoop};
#[cfg(target_family = "wasm")]
use winit::platform::web::WindowAttributesWeb;
use winit::window::{Window, WindowAttributes, WindowId};

use eyre::eyre;
use wgpu::{
    BindGroupEntry, BindGroupLayoutEntry, BufferUsages, CurrentSurfaceTexture,
    ExperimentalFeatures, InstanceDescriptor, InstanceFlags,
};

#[cfg(target_os = "windows")]
use winit::platform::windows::EventLoopBuilderExtWindows;

//  This logic is the same for both X11 and Wayland because the any_thread
// variable is the same on both
#[cfg(target_os = "linux")]
use winit::platform::x11::EventLoopBuilderExtX11;

use crate::sdf::{Circle, Complex};

pub mod sdf;

pub const fn triangle_count(n: i32) -> i32 {
    (n * (1 + n)) / 2
}

impl<U> From<euclid::Point2D<f32, U>> for Complex {
    fn from(value: euclid::Point2D<f32, U>) -> Self {
        Self::new(value.x, value.y)
    }
}

#[derive(Debug)]
struct Driver {
    adapter: wgpu::Adapter,
    draw_shader: wgpu::ShaderModule,
    compute_shader: wgpu::ShaderModule,
    queue: wgpu::Queue,
    device: wgpu::Device,
    qset: wgpu::QuerySet,
    resolve_buffer: wgpu::Buffer,
    destination_buffer: wgpu::Buffer,
    extract_points: wgpu::Buffer,
}

const NUM_QUERIES: u32 = 4;

impl Driver {
    pub async fn new(
        instance: &wgpu::Instance,
        surface: &wgpu::Surface<'static>,
    ) -> eyre::Result<Self> {
        let adapter = futures_lite::future::block_on(instance.request_adapter(
            &wgpu::RequestAdapterOptions {
                compatible_surface: Some(surface),
                ..Default::default()
            },
        ))?;

        if !adapter.features().contains(wgpu::Features::TIMESTAMP_QUERY) {
            eprintln!("Adapter does not support timestamps")
        }

        let required_limits = wgpu::Limits {
            max_immediate_size: size_of::<[f32; 6]>() as u32,
            ..Default::default()
        };

        // Create the logical device and command queue
        let (device, queue) = adapter
            .request_device(&wgpu::DeviceDescriptor {
                label: Some("wgpu Device"),
                required_features: wgpu::Features::TIMESTAMP_QUERY | wgpu::Features::IMMEDIATES,
                required_limits,
                memory_hints: wgpu::MemoryHints::MemoryUsage,
                trace: wgpu::Trace::Off,
                experimental_features: ExperimentalFeatures::disabled(),
            })
            .await?;

        let s = std::borrow::Cow::Borrowed(include_str!("draw.wgsl"));
        let cs = std::borrow::Cow::Borrowed(include_str!("points.wgsl"));

        let draw_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("draw.wgsl"),
            source: wgpu::ShaderSource::Wgsl(s),
        });

        let compute_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("points.wgsl"),
            source: wgpu::ShaderSource::Wgsl(cs),
        });

        let qset = device.create_query_set(&wgpu::QuerySetDescriptor {
            label: Some("Timestamps"),
            count: NUM_QUERIES,
            ty: wgpu::QueryType::Timestamp,
        });

        use zerocopy::IntoBytes;
        let resolve_buffer = device.create_buffer_init(&BufferInitDescriptor {
            label: Some("Timestamp resolve"),
            contents: [0u64; NUM_QUERIES as usize].as_bytes(),
            usage: wgpu::BufferUsages::COPY_SRC | wgpu::BufferUsages::QUERY_RESOLVE,
        });

        let destination_buffer = device.create_buffer_init(&BufferInitDescriptor {
            label: Some("Timestamp dest"),
            contents: [0u64; NUM_QUERIES as usize].as_bytes(),
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
        });

        let extract_points = device.create_buffer_init(&BufferInitDescriptor {
            label: Some("Debug Points buffer"),
            contents: [0.0f32; 1024 * 2].as_bytes(),
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
        });

        Ok(Self {
            adapter,
            device,
            queue,
            draw_shader,
            compute_shader,
            qset,
            resolve_buffer,
            destination_buffer,
            extract_points,
        })
    }

    fn get_timestamps(&self) -> Vec<u64> {
        self.destination_buffer
            .slice(..)
            .map_async(wgpu::MapMode::Read, |_| ());
        self.device
            .poll(wgpu::PollType::wait_indefinitely())
            .unwrap();

        let timestamps = {
            let timestamp_view = self
                .destination_buffer
                .slice(..self.destination_buffer.size())
                .get_mapped_range();
            bytemuck::cast_slice(&timestamp_view).to_vec()
        };

        self.destination_buffer.unmap();

        timestamps
    }

    fn get_gpu_points(&self) -> Vec<Complex> {
        self.extract_points
            .slice(..)
            .map_async(wgpu::MapMode::Read, |_| ());
        self.device
            .poll(wgpu::PollType::wait_indefinitely())
            .unwrap();

        let points = {
            let view = self
                .extract_points
                .slice(..self.extract_points.size())
                .get_mapped_range();
            bytemuck::cast_slice(&view).to_vec()
        };

        self.extract_points.unmap();

        points
    }

    pub fn get_compute_pipeline(&self) -> (wgpu::ComputePipeline, wgpu::BindGroupLayout) {
        let bind_group_layout =
            self.device
                .create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                    label: Some("Compute Bind Group"),
                    entries: &[
                        BindGroupLayoutEntry {
                            binding: 0,
                            visibility: wgpu::ShaderStages::COMPUTE,
                            ty: wgpu::BindingType::Buffer {
                                ty: wgpu::BufferBindingType::Storage { read_only: true },
                                has_dynamic_offset: false,
                                min_binding_size: None,
                            },
                            count: None,
                        },
                        BindGroupLayoutEntry {
                            binding: 1,
                            visibility: wgpu::ShaderStages::COMPUTE,
                            ty: wgpu::BindingType::Buffer {
                                ty: wgpu::BufferBindingType::Storage { read_only: false },
                                has_dynamic_offset: false,
                                min_binding_size: None,
                            },
                            count: None,
                        },
                        BindGroupLayoutEntry {
                            binding: 2,
                            visibility: wgpu::ShaderStages::COMPUTE,
                            ty: wgpu::BindingType::Buffer {
                                ty: wgpu::BufferBindingType::Storage { read_only: true },
                                has_dynamic_offset: false,
                                min_binding_size: None,
                            },
                            count: None,
                        },
                        BindGroupLayoutEntry {
                            binding: 3,
                            visibility: wgpu::ShaderStages::COMPUTE,
                            ty: wgpu::BindingType::Buffer {
                                ty: wgpu::BufferBindingType::Storage { read_only: false },
                                has_dynamic_offset: false,
                                min_binding_size: NonZero::new(size_of::<u32>() as u64),
                            },
                            count: None,
                        },
                    ],
                });

        let layout = self
            .device
            .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("Compute Pipeline"),
                bind_group_layouts: &[Some(&bind_group_layout)],
                immediate_size: 0,
            });

        let pipeline = self
            .device
            .create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: None,
                layout: Some(&layout),
                module: &self.compute_shader,
                entry_point: Some("implied_points"),
                compilation_options: Default::default(),
                cache: None,
            });

        (pipeline, bind_group_layout)
    }

    pub fn get_draw_pipeline(
        &self,
        format: wgpu::TextureFormat,
    ) -> (wgpu::RenderPipeline, wgpu::BindGroupLayout) {
        let bind_group_layout =
            self.device
                .create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                    label: Some("Draw Bind Group"),
                    entries: &[
                        BindGroupLayoutEntry {
                            binding: 0,
                            visibility: wgpu::ShaderStages::FRAGMENT,
                            ty: wgpu::BindingType::Buffer {
                                ty: wgpu::BufferBindingType::Storage { read_only: true },
                                has_dynamic_offset: false,
                                min_binding_size: None,
                            },
                            count: None,
                        },
                        BindGroupLayoutEntry {
                            binding: 1,
                            visibility: wgpu::ShaderStages::FRAGMENT,
                            ty: wgpu::BindingType::Buffer {
                                ty: wgpu::BufferBindingType::Storage { read_only: true },
                                has_dynamic_offset: false,
                                min_binding_size: None,
                            },
                            count: None,
                        },
                        BindGroupLayoutEntry {
                            binding: 2,
                            visibility: wgpu::ShaderStages::FRAGMENT,
                            ty: wgpu::BindingType::Buffer {
                                ty: wgpu::BufferBindingType::Storage { read_only: true },
                                has_dynamic_offset: false,
                                min_binding_size: None,
                            },
                            count: None,
                        },
                    ],
                });

        let layout = self
            .device
            .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("Draw Pipeline"),
                bind_group_layouts: &[Some(&bind_group_layout)],
                immediate_size: size_of::<[f32; 6]>() as u32,
            });

        let pipeline = self
            .device
            .create_render_pipeline(&wgpu::RenderPipelineDescriptor {
                label: None,
                layout: Some(&layout),
                vertex: wgpu::VertexState {
                    module: &self.draw_shader,
                    entry_point: Some("vs_main"),
                    buffers: &[],
                    compilation_options: Default::default(),
                },
                fragment: Some(wgpu::FragmentState {
                    module: &self.draw_shader,
                    //entry_point: Some("fs_sdf"),
                    entry_point: Some("tdf"),
                    compilation_options: Default::default(),
                    targets: &[Some(wgpu::ColorTargetState {
                        format,
                        blend: Some(wgpu::BlendState::PREMULTIPLIED_ALPHA_BLENDING),
                        write_mask: wgpu::ColorWrites::ALL,
                    })],
                }),
                primitive: wgpu::PrimitiveState {
                    front_face: wgpu::FrontFace::Cw,
                    topology: wgpu::PrimitiveTopology::TriangleList,
                    ..Default::default()
                },
                depth_stencil: None,
                multisample: wgpu::MultisampleState::default(),
                multiview_mask: None,
                cache: None,
            });

        (pipeline, bind_group_layout)
    }
}

#[derive(Debug)]
struct AppState {
    window: Arc<Window>,
    instance: wgpu::Instance,
    surface: wgpu::Surface<'static>,
    config: wgpu::wgt::SurfaceConfiguration<Vec<wgpu::TextureFormat>>,
    driver: Driver,
    draw_pipeline: wgpu::RenderPipeline,
    draw_group: wgpu::BindGroup,
    compute_pipeline: wgpu::ComputePipeline,
    compute_group: wgpu::BindGroup,
    shapes: wgpu::Buffer,
    points: wgpu::Buffer,
    atomic_offset: wgpu::Buffer,
    shape_idx: wgpu::Buffer,
    quadtree: wgpu::Buffer,
}

const BACKCOLOR: wgpu::Color = wgpu::Color {
    r: 0.1,
    g: 0.2,
    b: 0.3,
    a: 1.0,
};

fn fconv<T: num_traits::ToPrimitive, U: num_traits::NumCast>(v: T) -> U {
    U::from(v).unwrap()
}

pub fn srgb_to_linear<T: num_traits::Float>(c: T) -> T {
    if c <= fconv(0.04045) {
        c / fconv(12.92)
    } else {
        ((c + fconv(0.055)) / fconv(1.055)).powf(fconv(2.4))
    }
}

impl AppState {
    pub(crate) fn draw(
        &mut self,
        mut encoder: wgpu::CommandEncoder,
        offset: [f32; 2],
        mousepos: [f32; 2],
    ) -> eyre::Result<()> {
        let frame = match self.surface.get_current_texture() {
            CurrentSurfaceTexture::Success(t) | CurrentSurfaceTexture::Suboptimal(t) => Ok(t),
            CurrentSurfaceTexture::Timeout | CurrentSurfaceTexture::Occluded => return Ok(()),
            CurrentSurfaceTexture::Outdated => {
                self.surface.configure(&self.driver.device, &self.config);
                return Ok(());
            }
            CurrentSurfaceTexture::Validation => Err(eyre!("Unhandled validation error!")),
            _ => Err(eyre!("Failed to acquire surface texture")),
        }?;

        let view = frame
            .texture
            .create_view(&wgpu::TextureViewDescriptor::default());

        // driver.queue.write_buffer(
        //     &self.clip,
        //     0,
        //     bytemuck::cast_slice(self.clipdata.as_slice()),
        // );

        {
            encoder.clear_buffer(&self.atomic_offset, 0, Some(size_of::<u32>() as u64));
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("Implied Points Pass"),
                timestamp_writes: Some(wgpu::ComputePassTimestampWrites {
                    query_set: &self.driver.qset,
                    beginning_of_pass_write_index: Some(2),
                    end_of_pass_write_index: Some(3),
                }),
            });

            pass.set_pipeline(&self.compute_pipeline);
            pass.set_bind_group(0, &self.compute_group, &[]);
            pass.dispatch_workgroups(
                (triangle_count((self.shape_idx.size() / size_of::<f32>() as u64) as i32) / 128)
                    as u32
                    + 1,
                1,
                1,
            );
        }
        {
            let mut backcolor = BACKCOLOR;
            if frame.texture.format().is_srgb() {
                backcolor.r = srgb_to_linear(backcolor.r);
                backcolor.g = srgb_to_linear(backcolor.g);
                backcolor.b = srgb_to_linear(backcolor.b);
            }

            let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("Window Pass"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &view,
                    resolve_target: None,
                    depth_slice: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(backcolor),
                        store: wgpu::StoreOp::Store,
                    },
                })],
                depth_stencil_attachment: None,
                timestamp_writes: Some(wgpu::RenderPassTimestampWrites {
                    query_set: &self.driver.qset,
                    beginning_of_pass_write_index: Some(0),
                    end_of_pass_write_index: Some(1),
                }),
                occlusion_query_set: None,
                multiview_mask: None,
            });

            pass.set_viewport(
                0.0,
                0.0,
                self.config.width as f32,
                self.config.height as f32,
                0.0,
                1.0,
            );

            use zerocopy::IntoBytes;

            pass.set_pipeline(&self.draw_pipeline);
            pass.set_immediates(
                0,
                [
                    self.config.width as f32,
                    self.config.height as f32,
                    offset[0],
                    offset[1],
                    mousepos[0],
                    mousepos[1],
                ]
                .as_bytes(),
            );
            pass.set_bind_group(0, &self.draw_group, &[]);
            pass.draw(0..6, 0..1);
        }

        encoder.resolve_query_set(
            &self.driver.qset,
            0..NUM_QUERIES,
            &self.driver.resolve_buffer,
            0,
        );
        encoder.copy_buffer_to_buffer(
            &self.driver.resolve_buffer,
            0,
            &self.driver.destination_buffer,
            0,
            self.driver.resolve_buffer.size(),
        );
        /*encoder.copy_buffer_to_buffer(
            &self.points,
            0,
            &self.driver.extract_points,
            0,
            self.points.size(),
        );*/
        self.driver.queue.submit(Some(encoder.finish()));
        frame.present();

        Ok(())
    }
}

#[derive(Clone, Debug)]
pub enum Shape {
    Circle(sdf::Circle, u8, usize),
    HalfPlane(sdf::HalfPlane, u8, usize),
    Bezier(sdf::Bezier2o2d, u8, usize),
    Composite(Box<CompositeField>, usize),
    Constant(f32, usize),
}

impl From<sdf::Circle> for Shape {
    fn from(value: sdf::Circle) -> Self {
        Shape::Circle(value, 0, 0)
    }
}
impl From<sdf::HalfPlane> for Shape {
    fn from(value: sdf::HalfPlane) -> Self {
        Shape::HalfPlane(value, 0, 0)
    }
}
impl From<sdf::Bezier2o2d> for Shape {
    fn from(value: sdf::Bezier2o2d) -> Self {
        Shape::Bezier(value, 0, 0)
    }
}

#[repr(transparent)]
#[derive(Clone)]
pub struct ShapeIter<'a>(Vec<&'a Shape>);

impl<'a> ShapeIter<'a> {
    pub fn new(x: &'a Shape) -> Self {
        Self(vec![x])
    }
}

impl<'a> Iterator for ShapeIter<'a> {
    type Item = &'a Shape;

    fn next(&mut self) -> Option<Self::Item> {
        while let Some(item) = self.0.pop() {
            match item {
                x @ Shape::Circle(_, _, _)
                | x @ Shape::HalfPlane(_, _, _)
                | x @ Shape::Bezier(_, _, _)
                | x @ Shape::Constant(_, _) => {
                    return Some(x);
                }
                Shape::Composite(f, _) => {
                    self.0.push(&f.r);
                    self.0.push(&f.l);
                }
            }
        }
        None
    }
}

impl<'a> IntoIterator for &'a Shape {
    type Item = &'a Shape;
    type IntoIter = ShapeIter<'a>;

    fn into_iter(self) -> Self::IntoIter {
        self.iter()
    }
}

#[repr(u8)]
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Debug)]
enum ShapeOp {
    OpUnion = 0,
    OpIntersect = 1,
    Circle = 2,
    HalfPlane = 3,
    Bezier = 4,
    Constant = 5,
    OpNegate = 8,
    OpHollow = 16,
}

const OP_MASK: u8 = ShapeOp::OpNegate as u8 - 1;

#[derive(Clone, Debug)]
pub struct CompositeField {
    l: Shape,
    op: u8,
    r: Shape,
}

impl<T: Into<Self>> std::ops::BitOr<T> for Shape {
    type Output = Shape;

    fn bitor(self, rhs: T) -> Self::Output {
        Shape::Composite(
            Box::new(CompositeField {
                l: self,
                op: ShapeOp::OpUnion as u8,
                r: rhs.into(),
            }),
            0,
        )
    }
}

impl<T: Into<Self>> std::ops::BitAnd<T> for Shape {
    type Output = Shape;

    fn bitand(self, rhs: T) -> Self::Output {
        Shape::Composite(
            Box::new(CompositeField {
                l: self,
                op: ShapeOp::OpIntersect as u8,
                r: rhs.into(),
            }),
            0,
        )
    }
}

impl std::ops::Neg for Shape {
    type Output = Shape;

    fn neg(self) -> Self::Output {
        match self {
            Self::Circle(c, f, idx) => Self::Circle(c, f ^ ShapeOp::OpNegate as u8, idx),
            Self::HalfPlane(hp, f, idx) => Self::HalfPlane(hp, f ^ ShapeOp::OpNegate as u8, idx),
            Self::Bezier(b, f, idx) => Self::Bezier(b, f ^ ShapeOp::OpNegate as u8, idx),
            Self::Constant(v, idx) => Self::Constant(-v, idx),
            Self::Composite(mut f, idx) => {
                f.op ^= ShapeOp::OpNegate as u8;
                Self::Composite(f, idx)
            }
        }
    }
}

impl Shape {
    pub fn iter(&self) -> ShapeIter<'_> {
        ShapeIter::new(self)
    }

    pub fn demorgan(self, mut flip: bool) -> Self {
        match self {
            Self::Composite(mut f, idx) => {
                flip ^= (f.op & ShapeOp::OpNegate as u8) != 0;
                if flip {
                    f.op = if (f.op & OP_MASK) == ShapeOp::OpUnion as u8 {
                        ShapeOp::OpIntersect as u8
                    } else {
                        ShapeOp::OpUnion as u8
                    };
                }
                f.l = f.l.demorgan(flip);
                f.r = f.r.demorgan(flip);
                Self::Composite(f, idx)
            }
            x => {
                if flip {
                    -x
                } else {
                    x
                }
            }
        }
    }
    pub fn abs(self) -> Self {
        match self {
            Self::Circle(c, f, idx) => Self::Circle(c, f | ShapeOp::OpHollow as u8, idx),
            Self::HalfPlane(hp, f, idx) => Self::HalfPlane(hp, f | ShapeOp::OpHollow as u8, idx),
            Self::Bezier(b, f, idx) => Self::Bezier(b, f | ShapeOp::OpHollow as u8, idx),
            Self::Constant(v, idx) => Self::Constant(v.abs(), idx),
            Self::Composite(mut f, idx) => {
                f.op |= ShapeOp::OpHollow as u8;
                Self::Composite(f, idx)
            }
        }
    }
    pub fn circle(center: (f32, f32), radius: f32) -> Self {
        Self::Circle(
            sdf::Circle {
                center: center.into(),
                radius,
            },
            0,
            0,
        )
    }
    pub fn halfplane(normal: (f32, f32), shift: f32) -> Self {
        let mut hp = sdf::HalfPlane {
            normal: normal.into(),
            shift,
        };
        hp.normal = hp.normal.normalize();
        Self::HalfPlane(hp, 0, 0)
    }
    pub fn bezier(control: [(f32, f32); 3]) -> Self {
        Self::Bezier(
            sdf::Bezier2o2d(control[0].into(), control[1].into(), control[2].into()),
            0,
            0,
        )
    }
    fn recurse_array(&mut self, v: &mut Vec<f32>) {
        use sdf::AsArrayRef;

        match self {
            Self::Circle(c, f, idx) => {
                *idx = v.len();
                v.push(bytemuck::cast((ShapeOp::Circle as u8 | *f) as u32));
                v.extend_from_slice(c.as_array());
            }
            Self::HalfPlane(hp, f, idx) => {
                *idx = v.len();
                v.push(bytemuck::cast((ShapeOp::HalfPlane as u8 | *f) as u32));
                v.extend_from_slice(hp.as_array())
            }
            Self::Bezier(b, f, idx) => {
                *idx = v.len();
                v.push(bytemuck::cast((ShapeOp::Bezier as u8 | *f) as u32));
                v.extend_from_slice(b.as_array())
            }
            Self::Constant(val, idx) => {
                panic!("shouldn't happen");
                *idx = v.len();
                v.push(bytemuck::cast((ShapeOp::Constant as u8) as u32));
                v.push(*val);
            }
            Self::Composite(f, idx) => {
                *idx = f.recurse_array(v);
            }
        }
    }
    /// Converts an expression tree to reverse polish notation using postfix traversal
    pub fn to_array(&mut self) -> Vec<f32> {
        let mut v = Vec::new();
        self.recurse_array(&mut v);
        v
    }

    #[inline]
    fn unary_op(op: u8, mut x: f32) -> f32 {
        if (op & ShapeOp::OpNegate as u8) != 0 {
            x = -x;
        }
        if (op & ShapeOp::OpHollow as u8) != 0 {
            x = x.abs();
        }
        x
    }

    fn drop_unary(self) -> Self {
        match self {
            Self::Circle(c, f, idx) => Self::Circle(c, 0, idx),
            Self::HalfPlane(hp, f, idx) => Self::HalfPlane(hp, 0, idx),
            Self::Bezier(b, f, idx) => Self::Bezier(b, 0, idx),
            x => x,
        }
    }

    pub fn eval(&self, pos: Complex) -> f32 {
        match self {
            Self::Circle(circle, f, _) => Self::unary_op(*f, sdf::sdf_disk(*circle)(pos)),
            Self::HalfPlane(half_plane, f, _) => {
                Self::unary_op(*f, sdf::sdf_halfPlane(*half_plane)(pos))
            }
            Self::Bezier(b, f, _) => Self::unary_op(*f, b.sdf()(pos)),
            Self::Constant(v, _) => *v,
            Self::Composite(f, _) => {
                let l = f.l.eval(pos);
                let r = f.r.eval(pos);
                Self::unary_op(
                    f.op,
                    if f.op & OP_MASK == ShapeOp::OpUnion as u8 {
                        l.min(r)
                    } else {
                        l.max(r)
                    },
                )
            }
        }
    }

    pub fn nbp(
        &self,
        root: &Shape,
        pos: Complex,
        nearest: &mut Complex,
        points: &[(sdf::Complex, (usize, usize))],
    ) {
        for (p, _) in points {
            let dist = (*p - pos).squaredMag();

            // We can skip isBoundaryPoint here because the intersection points are prefiltered
            if dist < (*nearest - pos).squaredMag() {
                *nearest = *p;
            }
        }

        match self {
            Self::Circle(circle, _, _) => {
                let p = sdf::diskNBP(*circle)(pos);
                if (p - pos).squaredMag() < (*nearest - pos).squaredMag()
                    && is_boundary_point(root, p)
                {
                    *nearest = p;
                }
            }
            Self::HalfPlane(half_plane, _, _) => {
                let p = sdf::halfPlaneNBP(*half_plane)(pos);
                if (p - pos).squaredMag() < (*nearest - pos).squaredMag()
                    && is_boundary_point(root, p)
                {
                    *nearest = p;
                }
            }
            Self::Bezier(b, _, _) => {
                for p in b.findLocallyNearestPoints(pos) {
                    if (p - pos).squaredMag() < (*nearest - pos).squaredMag()
                        && is_boundary_point(root, p)
                    {
                        *nearest = p;
                    }
                }
            }
            Self::Constant(_, _) => (),
            Self::Composite(f, _) => {
                f.l.nbp(root, pos, nearest, points);
                f.r.nbp(root, pos, nearest, points);
            }
        }
    }

    pub fn trim_shape(
        &self,
        centroid: sdf::Complex,
        nearest: sdf::Complex,
        diagonal: f32,
        points: &[(sdf::Complex, (usize, usize))],
    ) -> Self {
        let mut marked: HashSet<usize> = HashSet::new();
        let r2 = diagonal * diagonal;

        for (pos, (l, r)) in points {
            let dist_sq = (*pos - nearest).squaredMag();
            if dist_sq <= r2 {
                marked.insert(*l);
                marked.insert(*r);
            }
        }

        let r = diagonal + (centroid - nearest).mag();

        self.candidate_points(self, centroid, r * r, &mut marked);

        // Now we have marked all SDFs that participate in this region, so we go through our Shape and delete parts with unused SDFs
        self.clone().trim_marked(&marked, centroid)
    }

    fn constant_fold(vl: f32, vr: f32, op: u8) -> f32 {
        Self::unary_op(
            op,
            if op & OP_MASK == ShapeOp::OpUnion as u8 {
                vl.min(vr)
            } else {
                vl.max(vr)
            },
        )
    }

    fn trim_marked(self, marked: &HashSet<usize>, centroid: Complex) -> Self {
        match self {
            x @ Self::Circle(_, _, idx)
            | x @ Self::HalfPlane(_, _, idx)
            | x @ Self::Bezier(_, _, idx) => {
                if marked.contains(&idx) {
                    x
                } else {
                    Self::Constant(x.eval(centroid), idx)
                    //Self::Constant(x.eval(centroid), idx)
                }
            }
            x @ Self::Constant(_, _) => x,
            Self::Composite(mut f, flag) => {
                let l = f.l.trim_marked(marked, centroid);
                let r = f.r.trim_marked(marked, centroid);

                /*match (l, r, f.op & OP_MASK) {
                    (Shape::Constant(vl, _), Shape::Constant(vr, _), _) => Shape::Constant(
                        Self::constant_fold(vl, vr, f.op),
                        usize::MAX,
                    ),
                    (Shape::Constant(_, _), x, o) | (x, Shape::Constant(_, _), o) => {
                        if o == ShapeOp::OpUnion as u8 {
                            f.l = x;
                            f.r = Shape::Constant(f32::MAX, usize::MAX);
                        } else if o == ShapeOp::OpIntersect as u8 {
                            f.l = Shape::Constant(f32::MIN, usize::MAX);
                            f.r = x;
                        } else {
                            panic!("invalid op");
                        }
                        Self::Composite(f, flag)
                    }
                    (l, r, _) => {
                        f.l = l;
                        f.r = r;
                        Self::Composite(f, flag)
                    }
                }*/
                match (l, r) {
                    (Shape::Constant(vl, _), Shape::Constant(vr, _)) => {
                        Shape::Constant(Self::constant_fold(vl, vr, f.op), usize::MAX)
                    }
                    (Shape::Constant(vl, _), Shape::Composite(b, _))
                    | (Shape::Composite(b, _), Shape::Constant(vl, _))
                        if (matches!(b.l, Shape::Constant(_, _))
                            || matches!(b.r, Shape::Constant(_, _)))
                            && b.op == f.op =>
                    {
                        match (b.l, b.r) {
                            (Shape::Constant(vr, _), x) | (x, Shape::Constant(vr, _)) => {
                                f.l = x;
                                f.r =
                                    Shape::Constant(Self::constant_fold(vl, vr, f.op), usize::MAX);
                                /*f.r = Shape::Constant(
                                    if (f.op & OP_MASK) == ShapeOp::OpUnion as u8 {
                                        f32::MAX
                                    } else {
                                        f32::MIN
                                    },
                                    usize::MAX,
                                );*/
                                Self::Composite(f, flag)
                            }
                            _ => panic!("invalid match"),
                        }
                    }
                    (l, r) => {
                        f.l = l;
                        f.r = r;
                        Self::Composite(f, flag)
                    }
                }
                /*if let Shape::Constant(vl, _) = &l
                    && let Shape::Constant(vr, _) = &r
                {
                    // Do constant folding
                    Shape::Constant(Self::constant_fold(vl, vr, f.op), usize::MAX)
                } else {
                    // Put everything back into the composite and return it
                    f.l = l;
                    f.r = r;
                    Self::Composite(f, flag)
                }*/

                /*f.l = l;
                f.r = r;
                Self::Composite(f, flag)*/
            }
        }
    }

    fn candidate_points(
        &self,
        root: &Self,
        centroid: sdf::Complex,
        r2: f32,
        marked: &mut HashSet<usize>,
    ) {
        match self {
            Self::Circle(c, _, idx) if !marked.contains(idx) => {
                let pos = sdf::diskNBP(*c)(centroid);
                if (pos - centroid).squaredMag() < r2 && is_boundary_point(root, pos) {
                    marked.insert(*idx);
                }
            }
            Self::HalfPlane(hp, _, idx) if !marked.contains(idx) => {
                let pos = sdf::halfPlaneNBP(*hp)(centroid);
                if (pos - centroid).squaredMag() < r2 && is_boundary_point(root, pos) {
                    marked.insert(*idx);
                }
            }
            Self::Bezier(b, _, idx) if !marked.contains(idx) => {
                for pos in b.findLocallyNearestPoints(centroid) {
                    if (pos - centroid).squaredMag() < r2 && is_boundary_point(root, pos) {
                        marked.insert(*idx);
                    }
                }
            }
            Self::Composite(f, _) => {
                f.l.candidate_points(root, centroid, r2, marked);
                f.r.candidate_points(root, centroid, r2, marked);
            }
            _ => (),
        }
    }

    fn to_indirect_array(&self, out: &mut Vec<u32>) -> u32 {
        match self {
            Self::Circle(_, _, idx) | Self::HalfPlane(_, _, idx) | Self::Bezier(_, _, idx) => {
                assert_ne!(*idx, usize::MAX, "Attempt to push temporary value");
                out.push((*idx as u32) << 1);
                1
            }
            Self::Constant(v, idx) => {
                if *idx == usize::MAX {
                    //out.push(v.to_bits() | 1);
                    if *v > 0.0 {
                        out.push(u32::MAX);
                    } else if *v < 0.0 {
                        out.push(u32::MAX - 1);
                    } else {
                        out.push(u32::MAX);
                    }
                } else {
                    out.push((*idx as u32) << 1);
                }
                1
            }
            Self::Composite(f, idx) => {
                let l = f.l.to_indirect_array(out);
                let r = f.r.to_indirect_array(out);
                out.push((*idx as u32) << 1);
                l + r + 1
            }
        }
    }
}

impl CompositeField {
    fn recurse_array(&mut self, v: &mut Vec<f32>) -> usize {
        self.l.recurse_array(v);
        self.r.recurse_array(v);
        let idx = v.len();
        v.push(bytemuck::cast(self.op as u32));
        idx
    }
}

// Gets a vector of indices pointing at the location of all shape primitives in a shape array
pub fn get_indices(shapes: &[f32]) -> (Vec<u32>, u32) {
    let mut i = 0;
    let mut v = Vec::new();
    let mut lines = 0;
    let mut circles = 0;
    let mut beziers = 0;

    while i < shapes.len() {
        let b = shapes[i].to_bits() as u8;

        const UNION: u8 = ShapeOp::OpUnion as u8;
        const INTERSECT: u8 = ShapeOp::OpIntersect as u8;
        const BEZIER: u8 = ShapeOp::Bezier as u8;
        const CIRCLE: u8 = ShapeOp::Circle as u8;
        const HP: u8 = ShapeOp::HalfPlane as u8;

        match b & OP_MASK {
            UNION | INTERSECT => {
                i += 1;
            }
            BEZIER => {
                v.push(i as u32);
                i += 7;
                beziers += 1;
            }
            CIRCLE | HP => {
                v.push(i as u32);
                i += 4;
                if b & OP_MASK == CIRCLE {
                    circles += 1;
                } else {
                    lines += 1;
                }
            }
            _ => panic!("Unknown shape!"),
        }
    }

    (v, max_points(beziers, circles, lines))
}

struct App {
    state: Option<AppState>,
    composite: Shape,
    flatten: Vec<f32>,
    points: Vec<(Complex, (usize, usize))>,
    offset: [f32; 2],
    mousepos: [f32; 2],
    lastpos: PhysicalPosition<f64>,
    mousedown: bool,
    draw_stats: rolling_stats::Stats<f32>,
    compute_stats: rolling_stats::Stats<f32>,
    quad_stats: rolling_stats::Stats<f32>,
    quad_v: Vec<u32>,
}

impl ApplicationHandler for App {
    fn resumed(&mut self, event_loop: &ActiveEventLoop) {
        if self.state.is_some() {
            return;
        }

        #[cfg(not(target_family = "wasm"))]
        let window_attributes = WindowAttributes::default()
            .with_title(env!("CARGO_CRATE_NAME"))
            //.with_inner_size(PhysicalSize::new(403, 317))
            //.with_inner_size(PhysicalSize::new(392, 307))
            .with_inner_size(PhysicalSize::new(650, 650))
            .with_resizable(true);
        #[cfg(target_family = "wasm")]
        let window_attributes = WindowAttributes::default()
            .with_title(env!("CARGO_CRATE_NAME"))
            .with_resizable(true)
            .with_platform_attributes(Box::new(WindowAttributesWeb::default().with_append(true)));

        let window = match event_loop.create_window(window_attributes) {
            Ok(window) => Arc::new(window),
            Err(err) => {
                eprintln!("error creating window: {err}");
                event_loop.exit();
                return;
            }
        };

        let mut desc = InstanceDescriptor::new_with_display_handle(Box::new(
            event_loop.owned_display_handle(),
        ));

        #[cfg(debug_assertions)]
        {
            desc.flags = InstanceFlags::debugging();
        }

        #[cfg(not(debug_assertions))]
        {
            desc.flags = InstanceFlags::DISCARD_HAL_LABELS;
        }

        let instance = wgpu::Instance::new(desc);

        let surface: wgpu::Surface<'static> = instance
            .create_surface(window.clone())
            .expect("Failed to create surface");

        let driver = futures_lite::future::block_on(Driver::new(&instance, &surface))
            .expect("Failed to create driver");

        let size = window.inner_size();
        let mut config = surface
            .get_default_config(&driver.adapter, size.width, size.height)
            .expect("Failed to find a default configuration");
        let view_format = config.format.add_srgb_suffix();
        //let view_format = config.format.remove_srgb_suffix();
        config.format = view_format;
        config.view_formats.push(view_format);
        surface.configure(&driver.device, &config);

        use zerocopy::IntoBytes;
        let shapes = driver.device.create_buffer_init(&BufferInitDescriptor {
            label: Some("Shapes"),
            usage: BufferUsages::STORAGE | BufferUsages::COPY_DST,
            contents: self.flatten.as_bytes(),
            //contents: [f32::from_bits(0), 200.0, 100.0, 0.0, 20.0, 40.0, 60.0].as_bytes(),
            /*contents: [
                f32::from_bits(1),
                -200.0 + 0.5,
                -30.0 + 0.5,
                200.0 + 0.5,
                30.0 + 0.5,
                20.0,
            ]
            .as_bytes(),*/
        });

        let (shape_indexes, max_points) = get_indices(&self.flatten);
        let blank_points: Vec<f32> =
            Vec::from_iter((0..(std::cmp::max(1, max_points) * 2)).map(|_| f32::MAX));

        let points = driver.device.create_buffer_init(&BufferInitDescriptor {
            label: Some("Points"),
            usage: BufferUsages::STORAGE | BufferUsages::COPY_DST | BufferUsages::COPY_SRC,
            contents: blank_points.as_bytes(),
        });

        let atomic_offset = driver.device.create_buffer_init(&BufferInitDescriptor {
            label: Some("Atomic total offset"),
            usage: BufferUsages::STORAGE | BufferUsages::COPY_DST,
            contents: [0u32].as_bytes(),
        });

        use euclid::default::Box2D;
        use euclid::default::Point2D;
        use euclid::default::Size2D;

        self.quad_v.resize(1 << 15, 0);
        let timer = std::time::Instant::now();
        let mut hmap = HashMap::new();
        build_quadtree(
            Box2D::from_origin_and_size(
                Point2D::new(config.width as f32 * -0.5, config.height as f32 * -0.5),
                Size2D::new(config.width as f32, config.height as f32),
            ),
            &mut self.quad_v,
            &self.composite,
            &self.points,
            &mut hmap,
        );
        let diff = std::time::Instant::now() - timer;
        self.quad_stats.update(diff.as_secs_f32() * 1000.0);

        let quadtree = driver.device.create_buffer_init(&BufferInitDescriptor {
            label: Some("Quadtree"),
            usage: BufferUsages::STORAGE | BufferUsages::COPY_DST,
            contents: self.quad_v.as_bytes(),
        });

        /*let shapes = driver.device.create_buffer(&BufferDescriptor {
            label: None,
            size: 8,
            usage: BufferUsages::STORAGE | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let points = driver.device.create_buffer(&BufferDescriptor {
            label: None,
            size: 8,
            usage: BufferUsages::STORAGE | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });*/

        let bindings = [
            BindGroupEntry {
                binding: 0,
                resource: shapes.as_entire_binding(),
            },
            BindGroupEntry {
                binding: 1,
                resource: points.as_entire_binding(),
            },
            BindGroupEntry {
                binding: 2,
                resource: quadtree.as_entire_binding(),
            },
        ];

        let (pipeline, bind_group_layout) = driver.get_draw_pipeline(config.view_formats[0]);

        let group = driver.device.create_bind_group(&wgpu::BindGroupDescriptor {
            layout: &bind_group_layout,
            entries: &bindings,
            label: None,
        });

        let shape_idx = driver.device.create_buffer_init(&BufferInitDescriptor {
            label: Some("Shape indexes"),
            usage: BufferUsages::STORAGE | BufferUsages::COPY_DST,
            contents: shape_indexes.as_bytes(),
        });

        let compute_bindings = [
            BindGroupEntry {
                binding: 0,
                resource: shapes.as_entire_binding(),
            },
            BindGroupEntry {
                binding: 1,
                resource: points.as_entire_binding(),
            },
            BindGroupEntry {
                binding: 2,
                resource: shape_idx.as_entire_binding(),
            },
            BindGroupEntry {
                binding: 3,
                resource: atomic_offset.as_entire_binding(),
            },
        ];

        let (compute_pipeline, compute_bind_group_layout) = driver.get_compute_pipeline();

        let compute_group = driver.device.create_bind_group(&wgpu::BindGroupDescriptor {
            layout: &compute_bind_group_layout,
            entries: &compute_bindings,
            label: None,
        });

        driver.device.set_device_lost_callback(|r, e| {
            eprintln!("{e} {:?}", r);
        });

        self.state = Some(AppState {
            window,
            instance,
            surface,
            driver,
            config,
            draw_pipeline: pipeline,
            shapes,
            points,
            draw_group: group,
            compute_pipeline,
            compute_group,
            shape_idx,
            atomic_offset,
            quadtree,
        });
    }

    fn window_event(&mut self, event_loop: &ActiveEventLoop, _: WindowId, event: WindowEvent) {
        match event {
            WindowEvent::CloseRequested => {
                event_loop.exit();
            }
            WindowEvent::Resized(size) => {
                let state = self.state.as_mut().expect("resize event without a window");

                let config = &mut state.config;
                if config.width == size.width && config.height == size.height {
                    return; // nothing to do
                }
                config.width = size.width;
                config.height = size.height;
                state.surface.configure(&state.driver.device, config);

                {
                    use euclid::default::Box2D;
                    use euclid::default::Point2D;
                    use euclid::default::Size2D;
                    use zerocopy::IntoBytes;

                    self.quad_v.clear();

                    let timer = std::time::Instant::now();
                    let mut hmap = HashMap::new();
                    build_quadtree(
                        Box2D::from_origin_and_size(
                            Point2D::new(config.width as f32 * -0.5, config.height as f32 * -0.5),
                            Size2D::new(config.width as f32, config.height as f32),
                        ),
                        &mut self.quad_v,
                        &self.composite,
                        &self.points,
                        &mut hmap,
                    );
                    let diff = std::time::Instant::now() - timer;
                    self.quad_stats.update(diff.as_secs_f32() * 1000.0);

                    state
                        .driver
                        .queue
                        .write_buffer(&state.quadtree, 0, self.quad_v.as_bytes());
                }
                state.window.request_redraw();
            }
            WindowEvent::RedrawRequested => {
                let state = self
                    .state
                    .as_mut()
                    .expect("redraw request without a window");
                let window = state.window.clone();

                window.pre_present_notify();

                let encoder =
                    state
                        .driver
                        .device
                        .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                            label: Some("Root Encoder"),
                        });

                state
                    .draw(encoder, self.offset, self.mousepos)
                    .expect("draw failure");
                let timestamps = state.driver.get_timestamps();

                let diffms = ((timestamps[1] - timestamps[0]) as f32
                    * state.driver.queue.get_timestamp_period())
                    / 1000000.0;

                let computediff = ((timestamps[3] - timestamps[2]) as f32
                    * state.driver.queue.get_timestamp_period())
                    / 1000000.0;

                self.draw_stats.update(diffms);
                self.compute_stats.update(computediff);

                let name = env!("CARGO_CRATE_NAME");
                window.set_title(
                    format!(
                        "{:.3}\u{00b1}{:.2}ms (draw) - {:.3}\u{00b1}{:.2}ms (compute) - {:.3}\u{00b1}{:.2}ms (quad)",
                        self.draw_stats.mean,
                        self.draw_stats.std_dev,
                        self.compute_stats.mean,
                        self.compute_stats.std_dev,
                        self.quad_stats.mean,
                        self.quad_stats.std_dev
                    )
                    .as_str(),
                );

                state.window.request_redraw();

                /*let mut pointcheck = state.driver.get_gpu_points();
                pointcheck.resize(self.points.len() + 5, Complex::zeroed());
                pointcheck.sort();
                self.points.sort();
                println!("-----------------");
                println!("{pointcheck:?}");
                println!("{:?}", self.points);*/
            }
            WindowEvent::CursorMoved { position, .. } => {
                self.mousepos[0] = position.x as f32;
                self.mousepos[1] = position.y as f32;
                if self.mousedown {
                    if !self.lastpos.x.is_nan() && !self.lastpos.y.is_nan() {
                        self.offset[0] -= (position.x - self.lastpos.x) as f32;
                        self.offset[1] += (position.y - self.lastpos.y) as f32;
                    }
                    self.lastpos = position;

                    if let Some(state) = &self.state {
                        state.window.request_redraw();
                    }
                }
            }
            WindowEvent::MouseInput { state, button, .. } => {
                if button == MouseButton::Left {
                    self.mousedown = state.is_pressed();
                    self.lastpos.x = f64::NAN;
                    self.lastpos.y = f64::NAN;
                }
            }
            _ => (),
        }
    }
}

fn new_app<T>(any_thread: bool) -> eyre::Result<EventLoop<T>> {
    #[cfg(target_os = "windows")]
    let event_loop = EventLoop::with_user_event()
        .with_any_thread(any_thread)
        .with_dpi_aware(true)
        .build()?;
    #[cfg(not(target_os = "windows"))]
        let event_loop = EventLoop::with_user_event()
            .with_any_thread(any_thread)
            .build()
            .map_err(|e| {
                if e.to_string()
                    .eq_ignore_ascii_case("Could not find wayland compositor")
                {
                    eyre::eyre!(
                        "Wayland initialization failed! winit cannot automatically fall back to X11 (https://github.com/rust-windowing/winit/issues/4267). Try running the program with `WAYLAND_DISPLAY=\"\"`"
                    )
                } else {
                    e.into()
                }
            })?;

    Ok(event_loop)
}

impl From<&Shape> for sdf::HalfPlane {
    fn from(val: &Shape) -> Self {
        if let Shape::HalfPlane(hp, _, _) = val {
            *hp
        } else {
            panic!("invalid cast");
        }
    }
}

impl From<&Shape> for sdf::Circle {
    fn from(val: &Shape) -> Self {
        if let Shape::Circle(c, _, _) = val {
            *c
        } else {
            panic!("invalid cast");
        }
    }
}

impl From<&Shape> for sdf::Bezier2o2d {
    fn from(val: &Shape) -> Self {
        if let Shape::Bezier(b, _, _) = val {
            *b
        } else {
            panic!("invalid cast");
        }
    }
}

enum PointIter {
    Zero,
    One(core::array::IntoIter<Complex, 1>, (usize, usize)),
    Two(core::array::IntoIter<Complex, 2>, (usize, usize)),
    Four(core::array::IntoIter<Complex, 4>, (usize, usize)),
}

impl Iterator for PointIter {
    type Item = (Complex, (usize, usize));
    fn next(&mut self) -> Option<Self::Item> {
        match self {
            PointIter::Zero => None,
            PointIter::One(i, pair) => i.next().map(|x| (x, *pair)),
            PointIter::Two(i, pair) => i.next().map(|x| (x, *pair)),
            PointIter::Four(i, pair) => i.next().map(|x| (x, *pair)),
        }
    }
}

pub fn implied_points<'a>(
    shapes: impl Iterator<Item = &'a Shape> + Clone,
) -> impl Iterator<Item = (Complex, (usize, usize))> {
    use itertools::Itertools;
    shapes
        .clone()
        .tuple_combinations()
        .map(|(l, r)| match (l, r) {
            (Shape::Circle(c1, _, i), Shape::Circle(c2, _, j)) => {
                PointIter::Two(sdf::twoCirclesIntersect(*c1, *c2).into_iter(), (*i, *j))
            }
            (Shape::HalfPlane(hp, _, i), Shape::Circle(c, _, j))
            | (Shape::Circle(c, _, i), Shape::HalfPlane(hp, _, j)) => {
                PointIter::Two(sdf::circleLineIntersect(*c, *hp).into_iter(), (*i, *j))
            }
            (Shape::HalfPlane(hp1, _, i), Shape::HalfPlane(hp2, _, j)) => {
                PointIter::One([sdf::twoLinesIntersect(*hp1, *hp2)].into_iter(), (*i, *j))
            }
            (Shape::Bezier(b, _, i), Shape::Circle(c, _, j))
            | (Shape::Circle(c, _, i), Shape::Bezier(b, _, j)) => {
                PointIter::Four(b.bezier2o2dCircleIntersect(*c).into_iter(), (*i, *j))
            }
            (Shape::Bezier(b, _, i), Shape::HalfPlane(hp, _, j))
            | (Shape::HalfPlane(hp, _, i), Shape::Bezier(b, _, j)) => {
                PointIter::Two(b.bezier2o2dLineIntersect(*hp).into_iter(), (*i, *j))
            }
            (Shape::Bezier(b1, _, i), Shape::Bezier(b2, _, j)) => {
                PointIter::Four(b1.twoBezier2o2dsIntersect(*b2).into_iter(), (*i, *j))
            }
            _ => panic!("Composite in shape iter!"),
        })
        .chain(shapes.map(|x| match x {
            Shape::Bezier(b, _, i) => PointIter::Two([b.0, b.2].into_iter(), (*i, *i)),
            _ => PointIter::Zero,
        }))
        .flatten()
}

// Count all possible points from the types of intersections
#[allow(clippy::identity_op)]
const fn max_points(beziers: u32, circles: u32, lines: u32) -> u32 {
    beziers * 2 + // unconditional points
    triangle_count(lines as i32 - 1) as u32 * 1 + // line-line intersection
    triangle_count(circles as i32 - 1) as u32 * 2 + // circle-circle intersection
    triangle_count((lines + circles) as i32 - 1) as u32 * 2 + // line-circle intersection
    triangle_count((beziers + lines) as i32 - 1) as u32 * 2 + // bezier-line intersection
    triangle_count((beziers + circles) as i32 - 1) as u32 * 4 + // bezier-circle intersection
    triangle_count(beziers as i32 - 1) as u32 * 4 // bezier-bezier intersection
}

struct QuadTree {
    children: Option<Box<QuadTree>>,
    indirect: Vec<usize>,
}

fn is_large_shape(s: &Shape) -> bool {
    // Currently, we consider anything larger than 2 shapes to be a "large" shape
    match s {
        Shape::Composite(f, _) => {
            return matches!(f.l, Shape::Composite(_, _)) || matches!(f.r, Shape::Composite(_, _));
        }
        _ => false,
    }
}

pub const QUAD_CHILD: u32 = 1 << 31;

pub static DEBUG_COUNT: std::sync::atomic::AtomicIsize = std::sync::atomic::AtomicIsize::new(0);

impl PartialEq for Shape {
    fn eq(&self, other: &Self) -> bool {
        match (self, other) {
            (
                Self::Circle(_, _, i) | Self::HalfPlane(_, _, i) | Self::Bezier(_, _, i),
                Self::Circle(_, _, j) | Self::HalfPlane(_, _, j) | Self::Bezier(_, _, j),
            ) => *i == *j,
            (Self::Constant(l, i), Self::Constant(r, j)) => {
                if *i != *j {
                    return false;
                }
                if *i == usize::MAX {
                    (*l >= 0.0 && *r >= 0.0) || (*l < 0.0 && *r < 0.0)
                } else {
                    true
                }
            }
            (Self::Composite(l, i), Self::Composite(r, j)) => {
                i == j && l.l.eq(&r.l) && l.r.eq(&r.r)
            }
            _ => false,
        }
    }
}

impl Eq for Shape {}

impl std::hash::Hash for Shape {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        core::mem::discriminant(self).hash(state);

        match &self {
            Self::Circle(_, _, idx) | Self::HalfPlane(_, _, idx) | Self::Bezier(_, _, idx) => {
                state.write_usize(*idx);
            }
            Self::Constant(v, idx) => {
                state.write_usize(*idx);
                state.write_u32(v.to_bits());
            }
            Self::Composite(f, flag) => {
                state.write_usize(*flag);
                f.l.hash(state);
                f.r.hash(state);
            }
        }
    }
}

pub fn build_quadtree(
    area: euclid::default::Box2D<f32>,
    v: &mut Vec<u32>,
    sdf: &Shape,
    points: &[(sdf::Complex, (usize, usize))],
    hmap: &mut HashMap<Shape, u32>,
) -> u32 {
    use euclid::default::Box2D;
    use euclid::default::Point2D;
    const MIN_SIZE: f32 = 16.0;
    let centroid = area.center(); //+ Vector2D::new(5.0, 5.0);
    let mut nearest = Complex::new(f32::MAX, f32::MAX);

    //if centroid.distance_to(Point2D::new(439.0 - 325.0, -(379.0 - 325.0))) < 10.0 {
    //    // DEBUG
    //    nearest = nearest;
    //}

    sdf.nbp(&sdf, centroid.into(), &mut nearest, points);
    let mut trimmed = sdf.trim_shape(
        sdf::Complex::new(centroid.x, centroid.y),
        nearest,
        Complex::new(area.width(), area.height()).mag(),
        points,
    );

    let idx = v.len() as u32;

    if !matches!(trimmed, Shape::Constant(_, _))
        && ((area.height() > MIN_SIZE && area.width() > MIN_SIZE && is_large_shape(&trimmed))
            || v.is_empty())
    {
        let quad = area.size() * 0.5;
        let topleft = area.min;

        let q = [
            Box2D::from_origin_and_size(topleft, quad),
            Box2D::from_origin_and_size(Point2D::new(topleft.x + quad.width, topleft.y), quad),
            Box2D::from_origin_and_size(Point2D::new(topleft.x, topleft.y + quad.height), quad),
            Box2D::from_origin_and_size(
                Point2D::new(topleft.x + quad.width, topleft.y + quad.height),
                quad,
            ),
        ];

        v.extend_from_slice(&[0, 0, 0, 0]);

        let begin = idx as usize;
        for i in begin..begin + 4 {
            v[i] = build_quadtree(q[i - begin], v, &sdf, points, hmap);
        }

        idx
    } else {
        v.push(0);
        /*
        if DEBUG_COUNT.fetch_add(1, std::sync::atomic::Ordering::Relaxed) == 106 {
            if let Shape::Composite(f, _) = &mut trimmed {
                if let Shape::Composite(ff, _) = &mut f.l {
                    if let Shape::Constant(t, _) = &mut ff.r {
                        // println!("catch: {}", *t);
                        // *t = -*t;
                    }
                }
                let check = sdf.trim_shape(
                    sdf::Complex::new(centroid.x, centroid.y),
                    nearest,
                    Complex::new(area.width(), area.height()).mag(),
                    points,
                );
            }
            let count = trimmed.to_indirect_array(v);
            v[idx as usize] = count;
        } else { */
        if let Some(v) = hmap.get(&trimmed) {
            let c = DEBUG_COUNT.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
            v | QUAD_CHILD
        } else {
            let count = trimmed.to_indirect_array(v);
            v[idx as usize] = count;

            if hmap.len() % 100 == 0 {
                println!("total: {}", hmap.len());
                println!("sample: {:?}", &v[idx as usize..(idx + count) as usize]);
            }

            hmap.insert(trimmed, idx);
            idx | QUAD_CHILD
        }
        //}
    }
}

use rand::prelude::*;

fn gen_shape(pos: Complex, radius: f32, rng: &mut Xoshiro128PlusPlus) -> Shape {
    use rand::distr::Uniform;

    let range = Uniform::try_from(-radius..=radius).unwrap();

    match 0 {
        0 => Shape::circle(
            (range.sample(rng) + pos.x, range.sample(rng) + pos.y),
            rng.random_range(radius * 0.12..=radius * 0.25),
        ),
        1 => Shape::halfplane(
            (rng.random_range(0.0..=pos.y), 1.0),
            rng.random_range(pos.x..=pos.x + radius),
        ),
        _ => panic!("2 + 2 != 4??"),
    }
}
fn gen_composite_shape(
    n: usize,
    radius: f32,
    offset: Complex,
    rng: &mut Xoshiro128PlusPlus,
) -> Shape {
    let mut pos = offset;

    let mut composite = gen_shape(pos, radius, rng);

    for i in 0..n {
        let sub = if rng.random_bool(0.5) { 1.0 } else { -1.0 };
        if rng.random_bool(0.5) {
            pos.x += radius * sub;
        } else {
            pos.y += radius * sub;
        }
        if (i % 3) == 0 {
            composite = composite | gen_shape(pos, radius, rng);
        } else {
            composite = composite | gen_shape(pos, radius, rng);
        }
    }

    composite
}

fn gen_grid_shape(w: usize, h: usize, radius: f32, offset: Complex) -> Shape {
    let dist = radius * 4.0;
    let mut composite = Shape::circle((offset.x - dist, offset.y - dist), radius);
    for i in 0..w {
        for j in 0..h {
            composite = composite
                | Shape::circle(
                    (offset.x + dist * i as f32, offset.y + dist * j as f32),
                    radius,
                );
        }
    }
    composite
}
fn main() -> Result<(), Box<dyn std::error::Error>> {
    #[cfg(target_family = "wasm")]
    console_error_panic_hook::set_once();

    let hp1 = Shape::halfplane((1.0, 0.0), 240.0);
    let hp2 = Shape::halfplane((0.0, 1.0), -120.0);
    let c1 = Shape::circle((-180.0, 0.0), 240.0);
    let c2 = Shape::circle((180.0, 0.0), 240.0);
    let c3 = Shape::circle((0.0, -180.0), 240.0);
    let b1 = Shape::bezier([(0.0, 320.0), (0.0, -640.0), (120.0, 320.0)]);
    let b2 = Shape::bezier([(320.0, 0.0), (-640.0, 0.0), (320.0, 120.0)]);

    //u (i (n (u (n (hp exampleHalfPlanes[0])) (hp exampleHalfPlanes[1]))) (i (n (disk exampleDisks[0])) (disk exampleDisks[1]))) (bez exampleBezier2o2ds[0])
    //let mut composite = ((-((-hp1) | hp2)) & ((-c1) & c2)) | (b1 | b2);

    let mut rng = Xoshiro128PlusPlus::from_seed(2873493u128.to_le_bytes());
    let mut composite = gen_composite_shape(600, 50.0, Complex::new(200.0, 0.0), &mut rng);
    //let mut composite = ((-((-hp1) | hp2)) | (c2));
    //let mut composite = ((hp1) & -hp2) | (c2);

    //let mut composite = gen_grid_shape(7, 7, 20.0, Complex::new(-200.0, -200.0));
    let box1 = Shape::halfplane((30.0 - (-30.0), (-200.0) - 200.0), 0.5);
    let box2 = Shape::halfplane((30.0 - (-30.0), 200.0 - (-200.0)), -0.5);
    //let mut composite = box1 | box2;
    //let shapes = Vec::from_iter(composite.iter());
    //println!("shapes: {shapes:?}");
    composite = composite.demorgan(false);
    let flatten = composite.to_array();
    let points = Vec::from_iter(
        implied_points(composite.iter()).filter(|(x, _)| is_boundary_point(&composite, *x)),
    );

    //println!("points: {points:?}");
    let event_loop = new_app(true)?;
    let mut app = App {
        state: None,
        flatten,
        composite,
        points,
        lastpos: PhysicalPosition { x: 0.0, y: 0.0 },
        offset: [0.0, 0.0],
        mousepos: [0.0, 0.0],
        mousedown: false,
        draw_stats: Default::default(),
        compute_stats: Default::default(),
        quad_stats: Default::default(),
        quad_v: Vec::new(),
    };

    event_loop.run_app(&mut app)?;

    Ok(())
}

pub fn is_boundary_point(f: &Shape, pos: Complex) -> bool {
    f.eval(pos).abs() <= sdf::BOUNDARY_THRESHOLD
}
