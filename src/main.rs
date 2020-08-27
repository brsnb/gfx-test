use gfx_hal::{device, window, Instance};

const DIMS: window::Extent2D = window::Extent2D {
    width: 1024,
    height: 768,
};

fn main() {
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
        .expect("Could not create window");

    let (instance, mut adapters, surface) = {
        let instance =
            backend::Instance::create("triangle", 1).expect("Could not create instance");
        let adapters = instance.enumerate_adapters();
        let surface = unsafe {
            instance
                .create_surface(&window)
                .expect("Could not create surface")
        };

        (instance, adapters, surface)
    };

    let adapter = adapters.remove(0);
}

struct Context<B: gfx_hal::Backend> {
    instance: B::Instance,
}

impl<B> Context<B>
where
    B: gfx_hal::Backend
{

}
