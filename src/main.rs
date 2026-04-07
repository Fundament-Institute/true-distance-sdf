use std::num::NonZero;
use std::sync::Arc;

use wgpu::util::{BufferInitDescriptor, DeviceExt};
use winit::application::ApplicationHandler;
use winit::dpi::PhysicalPosition;
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

use crate::sdf::Complex;

pub mod sdf;

pub const fn triangle_count(n: u32) -> u32 {
    return (n * (1 + n)) / 2;
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

        let mut required_limits = wgpu::Limits::default();
        required_limits.max_immediate_size = size_of::<[f32; 4]>() as u32;

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

        Ok(Self {
            adapter,
            device,
            queue,
            draw_shader,
            compute_shader,
            qset,
            resolve_buffer,
            destination_buffer,
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
                .slice(
                    ..(size_of::<u64>() as wgpu::BufferAddress
                        * NUM_QUERIES as wgpu::BufferAddress),
                )
                .get_mapped_range();
            bytemuck::cast_slice(&timestamp_view).to_vec()
        };

        self.destination_buffer.unmap();

        timestamps
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

        return (pipeline, bind_group_layout);
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
                        /*BindGroupLayoutEntry {
                            binding: 2,
                            visibility: wgpu::ShaderStages::VERTEX,
                            ty: wgpu::BindingType::Buffer {
                                ty: wgpu::BufferBindingType::Uniform,
                                has_dynamic_offset: false,
                                min_binding_size: std::num::NonZero::<u64>::new(
                                    size_of::<[f32; 4]>() as u64,
                                ),
                            },
                            count: None,
                        },*/
                    ],
                });

        let layout = self
            .device
            .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("Draw Pipeline"),
                bind_group_layouts: &[Some(&bind_group_layout)],
                immediate_size: size_of::<[f32; 4]>() as u32,
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

        return (pipeline, bind_group_layout);
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
                (triangle_count((self.shape_idx.size() / size_of::<f32>() as u64) as u32) / 128)
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
        self.driver.queue.submit(Some(encoder.finish()));
        frame.present();

        Ok(())
    }
}

#[derive(Clone, Debug)]
pub enum Shape {
    Circle(sdf::Circle, u8),
    HalfPlane(sdf::HalfPlane, u8),
    Bezier(sdf::Bezier2o2d, u8),
    Composite(Box<CompositeField>),
}

impl From<sdf::Circle> for Shape {
    fn from(value: sdf::Circle) -> Self {
        Shape::Circle(value, 0)
    }
}
impl From<sdf::HalfPlane> for Shape {
    fn from(value: sdf::HalfPlane) -> Self {
        Shape::HalfPlane(value, 0)
    }
}
impl From<sdf::Bezier2o2d> for Shape {
    fn from(value: sdf::Bezier2o2d) -> Self {
        Shape::Bezier(value, 0)
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
                x @ Shape::Circle(_, _) | x @ Shape::HalfPlane(_, _) | x @ Shape::Bezier(_, _) => {
                    return Some(x);
                }
                Shape::Composite(f) => {
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
        Shape::Composite(Box::new(CompositeField {
            l: self,
            op: ShapeOp::OpUnion as u8,
            r: rhs.into(),
        }))
    }
}

impl<T: Into<Self>> std::ops::BitAnd<T> for Shape {
    type Output = Shape;

    fn bitand(self, rhs: T) -> Self::Output {
        Shape::Composite(Box::new(CompositeField {
            l: self,
            op: ShapeOp::OpIntersect as u8,
            r: rhs.into(),
        }))
    }
}

impl std::ops::Neg for Shape {
    type Output = Shape;

    fn neg(self) -> Self::Output {
        match self {
            Self::Circle(c, f) => Self::Circle(c, f ^ ShapeOp::OpNegate as u8),
            Self::HalfPlane(hp, f) => Self::HalfPlane(hp, f ^ ShapeOp::OpNegate as u8),
            Self::Bezier(b, f) => Self::Bezier(b, f ^ ShapeOp::OpNegate as u8),
            Self::Composite(mut f) => {
                f.op ^= ShapeOp::OpNegate as u8;
                Self::Composite(f)
            }
        }
    }
}

impl Shape {
    pub fn iter(&self) -> ShapeIter<'_> {
        ShapeIter::new(self)
    }

    pub fn abs(self) -> Self {
        match self {
            Self::Circle(c, f) => Self::Circle(c, f | ShapeOp::OpHollow as u8),
            Self::HalfPlane(hp, f) => Self::HalfPlane(hp, f | ShapeOp::OpHollow as u8),
            Self::Bezier(b, f) => Self::Bezier(b, f | ShapeOp::OpHollow as u8),
            Self::Composite(mut f) => {
                f.op |= ShapeOp::OpHollow as u8;
                Self::Composite(f)
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
        )
    }
    pub fn halfplane(normal: (f32, f32), shift: f32) -> Self {
        Self::HalfPlane(
            sdf::HalfPlane {
                normal: normal.into(),
                shift,
            },
            0,
        )
    }
    pub fn bezier(control: [(f32, f32); 3]) -> Self {
        Self::Bezier(
            sdf::Bezier2o2d(control[0].into(), control[1].into(), control[2].into()),
            0,
        )
    }
    fn recurse_array(&self, v: &mut Vec<f32>) {
        use sdf::AsArrayRef;

        match self {
            Self::Circle(c, f) => {
                v.push(bytemuck::cast((ShapeOp::Circle as u8 | f) as u32));
                v.extend_from_slice(c.as_array());
            }
            Self::HalfPlane(hp, f) => {
                v.push(bytemuck::cast((ShapeOp::HalfPlane as u8 | f) as u32));
                v.extend_from_slice(hp.as_array())
            }
            Self::Bezier(b, f) => {
                v.push(bytemuck::cast((ShapeOp::Bezier as u8 | f) as u32));
                v.extend_from_slice(b.as_array())
            }
            Self::Composite(f) => {
                f.recurse_array(v);
            }
        }
    }
    /// Converts an expression tree to reverse polish notation using postfix traversal
    pub fn to_array(&self) -> Vec<f32> {
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

    pub fn eval(&self, pos: Complex) -> f32 {
        match self {
            Self::Circle(circle, f) => Self::unary_op(*f, sdf::sdf_disk(*circle)(pos)),
            Self::HalfPlane(half_plane, f) => {
                Self::unary_op(*f, sdf::sdf_halfPlane(*half_plane)(pos))
            }
            Self::Bezier(b, f) => Self::unary_op(*f, b.sdf()(pos)),
            Self::Composite(f) => {
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
}

impl CompositeField {
    fn recurse_array(&self, v: &mut Vec<f32>) {
        self.l.recurse_array(v);
        self.r.recurse_array(v);
        v.push(bytemuck::cast(self.op as u32));
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
    points: Vec<Complex>,
    offset: [f32; 2],
    lastpos: PhysicalPosition<f64>,
    mousedown: bool,
}

impl ApplicationHandler for App {
    fn resumed(&mut self, event_loop: &ActiveEventLoop) {
        if self.state.is_some() {
            return;
        }

        #[cfg(not(target_family = "wasm"))]
        let window_attributes = WindowAttributes::default()
            .with_title(env!("CARGO_CRATE_NAME"))
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
        let flatten = self.composite.to_array();
        let shapes = driver.device.create_buffer_init(&BufferInitDescriptor {
            label: Some("Shapes"),
            usage: BufferUsages::STORAGE | BufferUsages::COPY_DST,
            contents: flatten.as_bytes(),
        });

        let (shape_indexes, max_points) = get_indices(&flatten);
        let blank_points: Vec<f32> = Vec::from_iter((0..(max_points * 2)).map(|_| f32::MAX));

        let points = driver.device.create_buffer_init(&BufferInitDescriptor {
            label: Some("Points"),
            usage: BufferUsages::STORAGE | BufferUsages::COPY_DST,
            contents: blank_points.as_bytes(),
        });

        let atomic_offset = driver.device.create_buffer_init(&BufferInitDescriptor {
            label: Some("Atomic total offset"),
            usage: BufferUsages::STORAGE | BufferUsages::COPY_DST,
            contents: [0u32].as_bytes(),
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
            /*BindGroupEntry {
                binding: 2,
                resource: extent.as_entire_binding(),
            },*/
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
                config.width = size.width;
                config.height = size.height;
                state.surface.configure(&state.driver.device, config);
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

                state.draw(encoder, self.offset).expect("draw failure");
                let timestamps = state.driver.get_timestamps();

                let diffns = ((timestamps[1] - timestamps[0]) as f32
                    * state.driver.queue.get_timestamp_period())
                    / 1000000.0;
                let name = env!("CARGO_CRATE_NAME");
                window.set_title(format!("{name} - {diffns}ms").as_str());
                window.request_redraw();
            }
            WindowEvent::CursorMoved { position, .. } => {
                if self.mousedown {
                    if !self.lastpos.x.is_nan() && !self.lastpos.y.is_nan() {
                        self.offset[0] -= (position.x - self.lastpos.x) as f32;
                        self.offset[1] += (position.y - self.lastpos.y) as f32;
                    }
                    self.lastpos = position;
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
        if let Shape::HalfPlane(hp, _) = val {
            *hp
        } else {
            panic!("invalid cast");
        }
    }
}

impl From<&Shape> for sdf::Circle {
    fn from(val: &Shape) -> Self {
        if let Shape::Circle(c, _) = val {
            *c
        } else {
            panic!("invalid cast");
        }
    }
}

impl From<&Shape> for sdf::Bezier2o2d {
    fn from(val: &Shape) -> Self {
        if let Shape::Bezier(b, _) = val {
            *b
        } else {
            panic!("invalid cast");
        }
    }
}

// Count all possible points from the types of intersections
const fn max_points(beziers: u32, circles: u32, lines: u32) -> u32 {
    beziers * 2 + // unconditional points
    triangle_count(lines - 1) * 1 + // line-line intersection
    triangle_count(circles - 1) * 2 + // circle-circle intersection
    triangle_count(lines + circles - 1) * 2 + // line-circle intersection
    triangle_count(beziers + lines - 1) * 2 + // bezier-line intersection
    triangle_count(beziers + circles - 1) * 4 + // bezier-circle intersection
    triangle_count(beziers - 1) * 4 // bezier-bezier intersection
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    #[cfg(target_family = "wasm")]
    console_error_panic_hook::set_once();

    let hp1 = Shape::halfplane((1.0, 0.0), 240.0);
    let hp2 = Shape::halfplane((0.0, 1.0), -120.0);
    let c1 = Shape::circle((-180.0, 0.0), 240.0);
    let c2 = Shape::circle((180.0, 0.0), 240.0);
    let b1 = Shape::bezier([(0.0, 320.0), (0.0, -640.0), (120.0, 320.0)]);
    let b2 = Shape::bezier([(320.0, 0.0), (-640.0, 0.0), (320.0, 120.0)]);

    //u (i (n (u (n (hp exampleHalfPlanes[0])) (hp exampleHalfPlanes[1]))) (i (n (disk exampleDisks[0])) (disk exampleDisks[1]))) (bez exampleBezier2o2ds[0])
    let composite = ((-((-hp1) | hp2)) & ((-c1) & c2)) | (b1 | b2);

    //let composite = b1;
    let composite2 = composite.clone();
    let psdf = move |pos: Complex| composite2.eval(pos);

    let shapes = Vec::from_iter(composite.iter());
    println!("shapes: {shapes:?}");

    let mut points = Vec::from_iter(
        implied_points(composite.iter()).filter(|x| sdf::isBoundaryPoint(&psdf, *x)),
    );

    if points.is_empty() {
        points.push(Complex::new(f32::MAX, f32::MAX));
    }
    let event_loop = new_app(true)?;
    let mut app = App {
        state: None,
        composite,
        points,
        lastpos: PhysicalPosition { x: 0.0, y: 0.0 },
        offset: [0.0, 0.0],
        mousedown: false,
    };

    event_loop.run_app(&mut app)?;

    Ok(())
}

enum PointIter {
    Zero,
    One(core::array::IntoIter<Complex, 1>),
    Two(core::array::IntoIter<Complex, 2>),
    Four(core::array::IntoIter<Complex, 4>),
}

impl Iterator for PointIter {
    type Item = Complex;
    fn next(&mut self) -> Option<Self::Item> {
        match self {
            PointIter::Zero => None,
            PointIter::One(i) => i.next(),
            PointIter::Two(i) => i.next(),
            PointIter::Four(i) => i.next(),
        }
    }
}

pub fn implied_points<'a>(
    shapes: impl Iterator<Item = &'a Shape> + Clone,
) -> impl Iterator<Item = Complex> {
    use itertools::Itertools;
    shapes
        .clone()
        .tuple_combinations()
        .map(|(l, r)| match (l, r) {
            (Shape::Circle(c1, _), Shape::Circle(c2, _)) => {
                PointIter::Two(sdf::twoCirclesIntersect(*c1, *c2).into_iter())
            }
            (Shape::HalfPlane(hp, _), Shape::Circle(c, _))
            | (Shape::Circle(c, _), Shape::HalfPlane(hp, _)) => {
                PointIter::Two(sdf::circleLineIntersect(*c, *hp).into_iter())
            }
            (Shape::HalfPlane(hp1, _), Shape::HalfPlane(hp2, _)) => {
                PointIter::One([sdf::twoLinesIntersect(*hp1, *hp2)].into_iter())
            }
            (Shape::Bezier(b, _), Shape::Circle(c, _))
            | (Shape::Circle(c, _), Shape::Bezier(b, _)) => {
                PointIter::Four(b.bezier2o2dCircleIntersect(*c).into_iter())
            }
            (Shape::Bezier(b, _), Shape::HalfPlane(hp, _))
            | (Shape::HalfPlane(hp, _), Shape::Bezier(b, _)) => {
                PointIter::Two(b.bezier2o2dLineIntersect(*hp).into_iter())
            }
            (Shape::Bezier(b1, _), Shape::Bezier(b2, _)) => {
                PointIter::Four(b1.twoBezier2o2dsIntersect(*b2).into_iter())
            }
            _ => panic!("Composite in shape iter!"),
        })
        .chain(shapes.map(|x| match x {
            Shape::Bezier(b, _) => PointIter::Two([b.0, b.2].into_iter()),
            _ => PointIter::Zero,
        }))
        .flatten()
}
