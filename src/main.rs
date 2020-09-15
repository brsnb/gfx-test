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
    buffer, command, device, format, image, memory, pass, pool, prelude::*, pso, queue, window,
    Instance,
};

use std::{borrow::Borrow, io::Cursor, iter, mem, mem::ManuallyDrop, ptr};

const DIMS: window::Extent2D = window::Extent2D {
    width: 1024,
    height: 768,
};

#[allow(non_snake_case)]
#[derive(Debug, Clone, Copy, serde::Deserialize)]
#[repr(C)]
struct Vertex {
    position: [f32; 3],
    normal: [f32; 3],
}

#[derive(Debug, Clone, Copy)]
#[repr(C)]
struct PushConstants {
    transform: [[f32; 4]; 4],
}

/// Create a matrix that positions, scales, and rotates.
fn make_transform(translate: [f32; 3], angle: f32, scale: f32) -> [[f32; 4]; 4] {
    let c = angle.cos() * scale;
    let s = angle.sin() * scale;
    let [dx, dy, dz] = translate;

    [
        [c, 0., s, 0.],
        [0., scale, 0., 0.],
        [-s, 0., c, 0.],
        [dx, dy, dz, 1.],
    ]
}

unsafe fn push_constant_bytes<T>(push_constants: &T) -> &[u32] {
    let size_in_bytes = std::mem::size_of::<T>();
    let size_in_u32s = size_in_bytes / std::mem::size_of::<u32>();
    let start_ptr = push_constants as *const T as *const u32;
    std::slice::from_raw_parts(start_ptr, size_in_u32s)
}


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
    renderer.recreate_swapchain();

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
                    renderer.recreate_swapchain();
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
    queue_group: queue::QueueGroup<B>,
    surface: ManuallyDrop<B::Surface>,
    render_passes: Vec<B::RenderPass>,
    pipeline_layouts: Vec<B::PipelineLayout>,
    pipelines: Vec<B::GraphicsPipeline>,
    command_pools: Vec<B::CommandPool>,
    command_buffers: Vec<B::CommandBuffer>,
    vertex_buffer: ManuallyDrop<B::Buffer>,
    buffer_memory: ManuallyDrop<B::Memory>,
    submission_complete_fences: Vec<B::Fence>,
    submission_complete_semaphores: Vec<B::Semaphore>,
    surface_color_format: format::Format,
    surface_extent: window::Extent2D,
    viewport: pso::Viewport,
    frames_in_flight: usize,
    frame: u64,
    mesh: Vec<Vertex>,
}

impl<B> Renderer<B>
where
    B: gfx_hal::Backend,
{
    // FIXME: winit vs raw window handle
    pub fn new(window: &winit::window::Window) -> Renderer<B> {
        let mesh_data = include_bytes!("../assets/teapot_mesh.bin");
        let mesh: Vec<Vertex> = bincode::deserialize(mesh_data).unwrap();
        println!("MESH LEN: {}", mesh.len());

        let (instance, mut adapters, mut surface) = {
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

        let (device, queue_group) = {
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

        let frames_in_flight = 3;
        let mut command_pools = Vec::with_capacity(frames_in_flight);
        let mut command_buffers = Vec::with_capacity(frames_in_flight);

        // TODO: Allocate command_buffer for each frame
        let command_pool = unsafe {
            let command_pool = device
                .create_command_pool(queue_group.family, pool::CommandPoolCreateFlags::empty())
                .expect("Out of memory");
            command_pool
        };

        command_pools.push(command_pool);
        for _ in 1..frames_in_flight {
            unsafe {
                command_pools.push(
                    device
                        .create_command_pool(
                            queue_group.family,
                            pool::CommandPoolCreateFlags::empty(),
                        )
                        .unwrap(),
                );
            }
        }

        let memory_types = adapter.physical_device.memory_properties().memory_types;
        let limits = adapter.physical_device.limits();

        // Buffer
        let non_coherent_alignment = limits.non_coherent_atom_size as u64;

        let buffer_stride = std::mem::size_of::<Vertex>() as u64;
        println!("BUFFER_STRIDE: {}", buffer_stride);
        let buffer_len = mesh.len() as u64 * buffer_stride;
        assert_ne!(buffer_len, 0);
        let padded_buffer_len = ((buffer_len + non_coherent_alignment - 1)
            / non_coherent_alignment)
            * non_coherent_alignment;

        let mut vertex_buffer = ManuallyDrop::new(
            unsafe { device.create_buffer(padded_buffer_len, buffer::Usage::VERTEX) }.unwrap(),
        );

        println!("BUFFER_LEN: {}", buffer_len);
        println!("PADDED_BUFFER_LEN: {}", padded_buffer_len);

        let buffer_req = unsafe { device.get_buffer_requirements(&vertex_buffer) };

        let upload_type = memory_types
            .iter()
            .enumerate()
            .position(|(id, mem_type)| {
                buffer_req.type_mask & (1 << id) != 0
                    && mem_type
                        .properties
                        .contains(memory::Properties::CPU_VISIBLE)
            })
            .unwrap()
            .into();

        let buffer_memory = unsafe {
            println!("BUFFER_REQ SIZE: {}", buffer_req.size);
            let m = device
                .allocate_memory(upload_type, buffer_req.size)
                .expect("Could not allocate buffer memory");
            println!("AFTERALLOCATE");
            device
                .bind_buffer_memory(&m, 0, &mut vertex_buffer)
                .unwrap();
            let mapping = device.map_memory(&m, memory::Segment::ALL).unwrap();
            ptr::copy_nonoverlapping(mesh.as_ptr() as *const u8, mapping, buffer_len as usize);
            device
                .flush_mapped_memory_ranges(iter::once((&m, memory::Segment::ALL)))
                .unwrap();
            device.unmap_memory(&m);
            ManuallyDrop::new(m)
        };

        let mut submission_complete_fences = Vec::with_capacity(frames_in_flight);
        let mut submission_complete_semaphores = Vec::with_capacity(frames_in_flight);

        for i in 0..frames_in_flight {
            submission_complete_semaphores.push(
                device
                    .create_semaphore()
                    .expect("Could not create semaphore"),
            );
            submission_complete_fences
                .push(device.create_fence(true).expect("Could not create fence"));
            command_buffers.push(unsafe { command_pools[i].allocate_one(command::Level::Primary) });
        }

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

            unsafe {
                device
                    .create_render_pass(&[color_attachment], &[subpass], &[])
                    .unwrap()
            }
        };

        let pipeline_layout = unsafe {
            device
                .create_pipeline_layout(&[], &[(pso::ShaderStageFlags::VERTEX, 0..8)])
                .expect("Could not create pipeline layout")
        };

        // TODO: Look into gfx_auxil for shader compilaiton
        let (vs_spirv, fs_spirv) = {
            let vs_source = include_str!("../assets/shaders/vertex.vert");
            let fs_source = include_str!("../assets/shaders/fragment.frag");
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

                let vertex_buffers = vec![pso::VertexBufferDesc {
                    binding: 0,
                    stride: mem::size_of::<Vertex>() as u32,
                    rate: pso::VertexInputRate::Vertex,
                }];

                let attributes = vec![
                    pso::AttributeDesc {
                        location: 0,
                        binding: 0,
                        element: pso::Element {
                            format: format::Format::Rgb32Sfloat,
                            offset: 0,
                        },
                    },
                    pso::AttributeDesc {
                        location: 1,
                        binding: 0,
                        element: pso::Element {
                            format: format::Format::Rgb32Sfloat,
                            offset: 12,
                        },
                    },
                ];

                let mut pipeline_desc = pso::GraphicsPipelineDesc::new(
                    pso::PrimitiveAssemblerDesc::Vertex {
                        buffers: &vertex_buffers,
                        attributes: &attributes,
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

                pipeline_desc.blender.targets.push(pso::ColorBlendDesc {
                    mask: pso::ColorMask::ALL,
                    blend: Some(pso::BlendState::ALPHA),
                });

                unsafe { device.create_graphics_pipeline(&pipeline_desc, None) }
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

        // FIXME: Duplication b/t this and recreate_swapchain
        let caps = surface.capabilities(&adapter.physical_device);
        let formats = surface.supported_formats(&adapter.physical_device);
        info!("formats: {:?}", formats);
        let format = formats.map_or(format::Format::Rgba8Srgb, |formats| {
            formats
                .iter()
                .find(|format| format.base_format().1 == format::ChannelType::Srgb)
                .map(|format| *format)
                .unwrap_or(formats[0])
        });

        let swap_config = window::SwapchainConfig::from_caps(&caps, format, DIMS);
        info!("{:?}", swap_config);
        let surface_extent = swap_config.extent;
        unsafe {
            surface
                .configure_swapchain(&device, swap_config)
                .expect("Can't configure swapchain");
        };

        let viewport = pso::Viewport {
            rect: pso::Rect {
                x: 0,
                y: 0,
                w: surface_extent.width as _,
                h: surface_extent.height as _,
            },
            depth: 0.0..1.0,
        };

        Renderer {
            instance,
            adapter: ManuallyDrop::new(adapter),
            device,
            queue_group,
            surface: ManuallyDrop::new(surface),
            render_passes: vec![render_pass],
            pipeline_layouts: vec![pipeline_layout],
            pipelines: vec![pipeline],
            command_pools,
            command_buffers,
            vertex_buffer,
            buffer_memory,
            submission_complete_fences,
            submission_complete_semaphores,
            surface_color_format,
            surface_extent,
            viewport,
            frames_in_flight,
            frame: 0,
            mesh,
        }
    }

    pub fn recreate_swapchain(&mut self) {
        let caps = self.surface.capabilities(&self.adapter.physical_device);
        let mut swapchain_config = window::SwapchainConfig::from_caps(
            &caps,
            self.surface_color_format,
            self.surface_extent,
        );
        info!("{:?}", swapchain_config);
        if caps.image_count.contains(&3) {
            swapchain_config.image_count = 3;
        }

        self.surface_extent = swapchain_config.extent;
        self.viewport.rect.w = self.surface_extent.width as _;
        self.viewport.rect.h = self.surface_extent.height as _;

        unsafe {
            self.surface
                .configure_swapchain(&self.device, swapchain_config)
                .unwrap();
        }
    }

    pub fn render(&mut self) {
        let surface_image = unsafe {
            let acquire_timeout = 1_000_000_000;
            match self.surface.acquire_image(acquire_timeout) {
                Ok((image, _)) => image,
                Err(_) => {
                    self.recreate_swapchain();
                    return;
                }
            }
        };

        // TODO: Check that w/h work
        let framebuffer = unsafe {
            self.device
                .create_framebuffer(
                    &self.render_passes[0],
                    iter::once(surface_image.borrow()),
                    image::Extent {
                        width: self.viewport.rect.w as u32,
                        height: self.viewport.rect.h as u32,
                        depth: 1,
                    },
                )
                .unwrap()
        };

        // Get current frame
        // FIXME: This seems unsound
        let frame_idx = self.frame as usize % self.frames_in_flight;

        unsafe {
            let render_timeout = 1_000_000_000;
            let fence = &self.submission_complete_fences[frame_idx];
            self.device.wait_for_fence(fence, render_timeout).unwrap();
            self.device.reset_fence(fence).unwrap();
            self.command_pools[frame_idx].reset(false);
        }

        let command_buffer = &mut self.command_buffers[frame_idx];
        unsafe {
            command_buffer.begin_primary(command::CommandBufferFlags::ONE_TIME_SUBMIT);

            command_buffer.set_viewports(0, &[self.viewport.clone()]);
            command_buffer.set_scissors(0, &[self.viewport.rect]);

            command_buffer.bind_vertex_buffers(
                0,
                iter::once((&*self.vertex_buffer, buffer::SubRange::WHOLE)),
            );

            // TODO: Handle multiple render passes?
            command_buffer.begin_render_pass(
                &self.render_passes[0],
                &framebuffer,
                self.viewport.rect,
                &[command::ClearValue {
                    color: command::ClearColor {
                        float32: [0.0, 0.0, 0.0, 1.0],
                    },
                }],
                command::SubpassContents::Inline,
            );

            // TODO: Handle multiple pipelines?
            command_buffer.bind_graphics_pipeline(&self.pipelines[0]);

            let teapot = PushConstants {
                transform: make_transform([0.0, 0.0, 0.5], 0.0, 1.0),
            };

            command_buffer.push_graphics_constants(
                &self.pipeline_layouts[0],
                pso::ShaderStageFlags::VERTEX,
                0,
                push_constant_bytes(&teapot),
            );

            command_buffer.draw(0..self.mesh.len() as _, 0..1);
            command_buffer.end_render_pass();
            command_buffer.finish();

            let submission = queue::Submission {
                command_buffers: iter::once(&*command_buffer),
                wait_semaphores: None,
                signal_semaphores: iter::once(&self.submission_complete_semaphores[frame_idx]),
            };

            self.queue_group.queues[0].submit(
                submission,
                Some(&self.submission_complete_fences[frame_idx]),
            );

            let result = self.queue_group.queues[0].present(
                &mut self.surface,
                surface_image,
                Some(&self.submission_complete_semaphores[frame_idx]),
            );

            self.device.destroy_framebuffer(framebuffer);

            if result.is_err() {
                self.recreate_swapchain();
            }
        }

        self.frame += 1;
    }
}

impl<B> Drop for Renderer<B>
where
    B: gfx_hal::Backend,
{
    fn drop(&mut self) {
        self.device.wait_idle().unwrap();
        unsafe {
            self.device
                .destroy_buffer(ManuallyDrop::take(&mut self.vertex_buffer));
            self.device
                .free_memory(ManuallyDrop::take(&mut self.buffer_memory));

            for s in self.submission_complete_semaphores.drain(..) {
                self.device.destroy_semaphore(s);
            }

            for f in self.submission_complete_fences.drain(..) {
                self.device.destroy_fence(f);
            }

            for pool in self.command_pools.drain(..) {
                self.device.destroy_command_pool(pool);
            }

            for p in self.pipelines.drain(..) {
                self.device.destroy_graphics_pipeline(p);
            }

            for rp in self.render_passes.drain(..) {
                self.device.destroy_render_pass(rp);
            }

            self.surface.unconfigure_swapchain(&self.device);
            self.instance
                .destroy_surface(ManuallyDrop::take(&mut self.surface));
        }
    }
}
