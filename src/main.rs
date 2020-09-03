#[macro_use]
extern crate log;

use gfx_hal::{device, window, Instance};

const DIMS: window::Extent2D = window::Extent2D {
    width: 1024,
    height: 768,
};

fn main() {
    env_logger::init();

    info!("Starting");

    let window = Window::new().unwrap();

}

pub struct Window {
    event_loop: winit::event_loop::EventLoop<()>,
    window: winit::window::Window,
}

impl Window {
    pub fn new() -> Result<Self, &'static str> {
        let event_loop = winit::event_loop::EventLoop::new();

        let window = winit::window::WindowBuilder::new()
            .with_min_inner_size(winit::dpi::Size::Logical(winit::dpi::LogicalSize::new(
                64.0, 64.0,
            )))
            .with_inner_size(winit::dpi::Size::Physical(winit::dpi::PhysicalSize::new(
                DIMS.width,
                DIMS.height,
            )))
            .with_title("triangle".to_string())
            .build(&event_loop)
            .map_err(|_| "Could not create window")?;

        Ok(
            Window {
                event_loop,
                window,
            }
        )
    }
}

pub struct Renderer<B: gfx_hal::Backend> {
    instance: B::Instance,
    adapter: gfx_hal::adapter::Adapter<B>,
    device: B::Device,
}

impl<B> Renderer<B>
where
    B: gfx_hal::Backend
{
    pub fn new(window: &winit::window::Window) -> Self {

        let (instance, mut adapters, surface) = {
            let instance =
                backend::Instance::create("triangle", 1).expect("Could not create instance");
            let adapters = instance.enumerate_adapters();
            let surface = unsafe {
                instance
                    .create_surface(window)
                    .expect("Could not create surface")
            };

            (instance, adapters, surface)
        };

        let adapter = adapters.remove(0);
        let memory_types = adapter.physical_device.memory_properties().memory_types;
        let limits = adapter.physical_device.limits();
    }

}
