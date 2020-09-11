#[macro_use]
extern crate log;

#[cfg(not(any(feature = "vulkan", feature = "dx12", feature = "metal",)))]
extern crate gfx_backend_empty as backend;

#[cfg(feature = "dx12")]
extern crate gfx_backend_dx12 as backend;
#[cfg(feature = "metal")]
extern crate gfx_backend_metal as backend;
#[cfg(feature = "vulkan")]
extern crate gfx_backend_vulkan as backend;

extern crate shaderc;

use gfx_hal::{
    buffer, command, device, format, image, pass, pool, prelude::*, pso, queue, window, Instance,
};

use std::{io::Cursor, mem::ManuallyDrop};

const DIMS: window::Extent2D = window::Extent2D {
    width: 1024,
    height: 768,
};

#[derive(Debug, Clone, Copy)]
#[allow(non_snake_case)]
struct Vertex {
    a_Pos: [f32; 2],
    a_Uv: [f32; 2],
}

#[cfg_attr(rustfmt, rustfmt_skip)]
const QUAD: [Vertex; 6] = [
    Vertex { a_Pos: [ -0.5, 0.33 ], a_Uv: [0.0, 1.0] },
    Vertex { a_Pos: [  0.5, 0.33 ], a_Uv: [1.0, 1.0] },
    Vertex { a_Pos: [  0.5,-0.33 ], a_Uv: [1.0, 0.0] },

    Vertex { a_Pos: [ -0.5, 0.33 ], a_Uv: [0.0, 1.0] },
    Vertex { a_Pos: [  0.5,-0.33 ], a_Uv: [1.0, 0.0] },
    Vertex { a_Pos: [ -0.5,-0.33 ], a_Uv: [0.0, 0.0] },
];

fn main() {
    env_logger::init();

    let event_loop = winit::event_loop::EventLoop::new();

    let window = winit::window::WindowBuilder::new()
        .with_min_inner_size(winit::dpi::Size::Logical(winit::dpi::LogicalSize::new(
            64.0, 64.0,
        )))
        .with_inner_size(winit::dpi::Size::Physical(winit::dpi::PhysicalSize::new(
            DIMS.width,
            DIMS.height,
        )))
        .with_title("quad".to_string())
        .build(&event_loop)
        .unwrap();

   let mut renderer: Renderer<backend::Backend> = Renderer::new(&window);

    info!("Starting event loop");

    // TODO: Some sort of main loop struct instead of just windowing info??
    event_loop.run(move |event, _, control_flow| {
        *control_flow = winit::event_loop::ControlFlow::Poll;

        match event {
            winit::event::Event::WindowEvent { event, .. } => match event {
                winit::event::WindowEvent::CloseRequested => {
                    info!("Exiting...");
                    *control_flow = winit::event_loop::ControlFlow::Exit
                }
                winit::event::WindowEvent::KeyboardInput {
                    input:
                        winit::event::KeyboardInput {
                            virtual_keycode: Some(winit::event::VirtualKeyCode::Escape),
                            ..
                        },
                    ..
                } => {
                    info!("Exit requested via keypress");
                    *control_flow = winit::event_loop::ControlFlow::Exit;
                }
                winit::event::WindowEvent::Resized(dims) => {
                    // TODO: Recreate swapchain
                }
                _ => {}
            },
            winit::event::Event::MainEventsCleared => window.request_redraw(),
            winit::event::Event::RedrawRequested(_) => {
                renderer.render();
            }
            _ => {}
        }
    });
}

pub struct Renderer<B: gfx_hal::Backend> {
    instance: B::Instance,
    adapter: ManuallyDrop<gfx_hal::adapter::Adapter<B>>,
    device: B::Device,
    surface: ManuallyDrop<B::Surface>,
    render_passes: ManuallyDrop<Vec<B::RenderPass>>,
    pipeline_layouts: ManuallyDrop<Vec<B::PipelineLayout>>,
    pipelines: ManuallyDrop<Vec<B::GraphicsPipeline>>,
    command_pools: ManuallyDrop<Vec<B::CommandPool>>,
    submission_complete_fences: Vec<B::Fence>,
    rendering_complete_semaphores: Vec<B::Semaphore>,
}

impl<B> Renderer<B>
where
    B: gfx_hal::Backend,
{
    // FIXME: winit vs raw window handle
    pub fn new(window: &winit::window::Window) -> Renderer<B> {
        let (instance, mut adapters, surface) = {
            let instance = B::Instance::create("triangle", 1).expect("Could not create instance");
            let adapters = instance.enumerate_adapters();
            let surface = unsafe {
                instance
                    .create_surface(window)
                    .expect("Could not create surface")
            };

            (instance, adapters, surface)
        };

        let adapter = adapters.remove(0);

        // NOTE: ??
        let memory_types = adapter.physical_device.memory_properties().memory_types;
        let limits = adapter.physical_device.limits();

        let (device, mut queue_group) = {
            // Get graphics queue only
            let queue_family = adapter
                .queue_families
                .iter()
                .find(|family| {
                    surface.supports_queue_family(family) && family.queue_type().supports_graphics()
                })
                .unwrap();

            let mut gpu = unsafe {
                adapter
                    .physical_device
                    .open(&[(queue_family, &[1.0])], gfx_hal::Features::empty())
                    .expect("Could no open device")
            };

            (gpu.device, gpu.queue_groups.pop().unwrap())
        };

        let (command_pool, mut command_buffer) = unsafe {
            let mut command_pool = device
                .create_command_pool(queue_group.family, pool::CommandPoolCreateFlags::empty())
                .expect("Out of memory");

            let command_buffer = command_pool.allocate_one(command::Level::Primary);

            (command_pool, command_buffer)
        };

        let surface_color_format = {
            let supported_formats = surface
                .supported_formats(&adapter.physical_device)
                .unwrap_or(vec![]);
            let default_format = *supported_formats
                .get(0)
                .unwrap_or(&format::Format::Rgba8Srgb);

            supported_formats
                .into_iter()
                .find(|format| format.base_format().1 == format::ChannelType::Srgb)
                .unwrap_or(default_format)
        };

        let render_pass = {
            let color_attachment = pass::Attachment {
                format: Some(surface_color_format),
                samples: 1,
                ops: pass::AttachmentOps::new(
                    pass::AttachmentLoadOp::Clear,
                    pass::AttachmentStoreOp::Store,
                ),
                stencil_ops: pass::AttachmentOps::DONT_CARE,
                layouts: image::Layout::Undefined..image::Layout::Present,
            };

            let subpass = pass::SubpassDesc {
                colors: &[(0, image::Layout::ColorAttachmentOptimal)],
                depth_stencil: None,
                inputs: &[],
                resolves: &[],
                preserves: &[],
            };

                unsafe { device.create_render_pass(&[color_attachment], &[subpass], &[]).unwrap() }
        };

        let pipeline_layout = unsafe {
            device
                .create_pipeline_layout(&[], &[])
                .expect("Could not create pipeline layout")
        };

        // TODO: Look into gfx_auxil for shader compilaiton
        let (vs_spirv, fs_spirv) = {
            let vs_source = include_str!("assets/shaders/vertex.vert");
            let fs_source = include_str!("assets/shaders/fragment.frag");
            let mut compiler = shaderc::Compiler::new().unwrap();
            let vs_spirv = compiler
                .compile_into_spirv(
                    vs_source,
                    shaderc::ShaderKind::Vertex,
                    "vertex.vert",
                    "main",
                    None,
                )
                .unwrap();
            let fs_spirv = compiler
                .compile_into_spirv(
                    fs_source,
                    shaderc::ShaderKind::Fragment,
                    "fragment.frag",
                    "main",
                    None,
                )
                .unwrap();
            (vs_spirv, fs_spirv)
        };

        let pipeline = {
            let vs_module = {
                let spirv = gfx_auxil::read_spirv(Cursor::new(vs_spirv.as_binary_u8())).unwrap();
                unsafe { device.create_shader_module(&spirv).unwrap() }
            };
            let fs_module = {
                let spirv = gfx_auxil::read_spirv(Cursor::new(fs_spirv.as_binary_u8())).unwrap();
                unsafe { device.create_shader_module(&spirv).unwrap() }
            };

            let pipeline = {
                let (vs_entry, fs_entry): (pso::EntryPoint<B>, pso::EntryPoint<B>) = (
                    pso::EntryPoint {
                        entry: "main",
                        module: &vs_module,
                        specialization: pso::Specialization::default(),
                    },
                    pso::EntryPoint {
                        entry: "main",
                        module: &fs_module,
                        specialization: pso::Specialization::default(),
                    },
                );
                let mut pipline_desc = pso::GraphicsPipelineDesc::new(
                    pso::PrimitiveAssemblerDesc::Vertex {
                        buffers: &[],
                        attributes: &[],
                        input_assembler: pso::InputAssemblerDesc {
                            primitive: pso::Primitive::TriangleList,
                            with_adjacency: false,
                            restart_index: None,
                        },
                        vertex: vs_entry,
                        geometry: None,
                        tessellation: None,
                    },
                    pso::Rasterizer::FILL,
                    Some(fs_entry),
                    &pipeline_layout,
                    pass::Subpass {
                        index: 0,
                        main_pass: &render_pass,
                    },
                );

                pipline_desc.blender.targets.push(pso::ColorBlendDesc {
                    mask: pso::ColorMask::ALL,
                    blend: Some(pso::BlendState::ALPHA),
                });

                unsafe { device.create_graphics_pipeline(&pipline_desc, None) }
            };

            // TODO: Move shader stuff into its own module ??
            unsafe {
                device.destroy_shader_module(vs_module);
            }
            unsafe {
                device.destroy_shader_module(fs_module);
            }

            pipeline.unwrap()
        };

        let submission_complete_fence = device.create_fence(true).unwrap();
        let rendering_complete_semaphore = device.create_semaphore().unwrap();

        Renderer {
            instance,
            adapter: ManuallyDrop::new(adapter),
            device,
            surface: ManuallyDrop::new(surface),
            render_passes: ManuallyDrop::new(vec![render_pass]),
            pipeline_layouts: ManuallyDrop::new(vec![pipeline_layout]),
            pipelines: ManuallyDrop::new(vec![pipeline]),
            command_pools: ManuallyDrop::new(vec![command_pool]),
            submission_complete_fences: vec![submission_complete_fence],
            rendering_complete_semaphores: vec![rendering_complete_semaphore],
        }
    }

    pub fn render(&mut self) {}
}
