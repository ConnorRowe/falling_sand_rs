use std::collections::HashSet;

use coffee::{Game, input, Result, Timer};
use coffee::graphics::{Batch, Color, Font, Frame, Image, Point, Rectangle, Sprite, Text, Window, WindowSettings};
use coffee::input::{Input, keyboard, mouse};
use coffee::load::{Join, loading_screen::ProgressBar, Task};
use nalgebra::Vector2;
use rand::*;
use rayon::prelude::*;

fn main() -> Result<()> {
    FallingSand::run(WindowSettings {
        title: String::from("Falling Sand - Coffee"),
        size: (512, 512),
        resizable: false,
        fullscreen: false,
        maximized: false,
    })
}

struct Inputs {
    cursor_position: Point,
    mouse_wheel: Point,
    keys_pressed: HashSet<keyboard::KeyCode>,
    mouse_buttons_pressed: HashSet<mouse::Button>,
    text_buffer: String,
}

impl Input for Inputs {
    fn new() -> Inputs {
        Inputs {
            cursor_position: Point::new(0.0, 0.0),
            mouse_wheel: Point::new(0.0, 0.0),
            keys_pressed: HashSet::new(),
            mouse_buttons_pressed: HashSet::new(),
            text_buffer: String::new(),
        }
    }

    fn update(&mut self, event: input::Event) {
        match event {
            input::Event::Mouse(mouse_event) => match mouse_event {
                mouse::Event::CursorMoved { x, y } => {
                    self.cursor_position = Point::new(x, y);
                }
                mouse::Event::Input { state, button } => match state {
                    input::ButtonState::Pressed => {
                        self.mouse_buttons_pressed.insert(button);
                    }
                    input::ButtonState::Released => {
                        self.mouse_buttons_pressed.remove(&button);
                    }
                },
                mouse::Event::WheelScrolled { delta_x, delta_y } => {
                    self.mouse_wheel = Point::new(delta_x, delta_y);
                }
                _ => {}
            },
            input::Event::Keyboard(keyboard_event) => match keyboard_event {
                keyboard::Event::TextEntered { character } => {
                    self.text_buffer.push(character);
                }
                keyboard::Event::Input { key_code, state } => match state {
                    input::ButtonState::Pressed => {
                        self.keys_pressed.insert(key_code);
                    }
                    input::ButtonState::Released => {
                        self.keys_pressed.remove(&key_code);
                    }
                },
            },
            _ => {}
        }
    }

    fn clear(&mut self) {
        self.text_buffer.clear();
    }
}

#[repr(C)]
#[derive(Clone, Copy)]
pub struct Particle {
    strain: Strain,
    update: bool,
}

impl Default for Particle {
    fn default() -> Self {
        Particle {
            strain: Strain::Empty,
            update: false,
        }
    }
}

#[repr(u8)]
#[derive(Clone, Copy, Eq, PartialEq)]
pub enum Strain {
    Empty = 0,
    Sand = 1,
    Water = 2,
}

impl Strain {
    fn to_colour_id(&self) -> u16 {
        match self {
            Strain::Sand => 0,
            Strain::Water => 1,
            _ => 2
        }
    }

    fn density(&self) -> u16 {
        match self {
            Strain::Sand => 1600,
            Strain::Water => 1000,
            _ => 1000
        }
    }

    fn to_str(&self) -> &'static str {
        match self {
            Strain::Empty => "Empty",
            Strain::Sand => "Sand",
            Strain::Water => "Water",
            _ => ""
        }
    }
}

struct FallingSand {
    font: Font,
    grid: Vec<Particle>,
    grid_width: usize,
    grid_height: usize,
    batch: Batch,
    update: bool,
    cursor_position: Point,
    mouse_wheel: Point,
    keys_pressed: HashSet<keyboard::KeyCode>,
    mouse_buttons_pressed: HashSet<mouse::Button>,
    text_buffer: String,
    particles_updated: u64,
    active_strain: Strain,
}

impl FallingSand {
    const MAX_TEXTSIZE: usize = 40;

    fn new(batch: Batch, font: Font, x: usize, y: usize) -> FallingSand {
        FallingSand {
            font,
            grid: vec![Particle::default(); x * y],
            grid_width: x,
            grid_height: y,
            batch,
            update: false,
            cursor_position: Point::new(0.0, 0.0),
            mouse_wheel: Point::new(0.0, 0.0),
            keys_pressed: HashSet::new(),
            mouse_buttons_pressed: HashSet::new(),
            text_buffer: String::with_capacity(Self::MAX_TEXTSIZE),
            particles_updated: 0,
            active_strain: Strain::Sand,
        }
    }

    fn load() -> Task<FallingSand> {
        (
            Task::using_gpu(|gpu| Image::from_colors(gpu, &COLORS)),
            Font::load_from_bytes(include_bytes!(
                "../resources/Gamepixies-8MO6n.ttf"
            )),
        )
            .join()
            .map(|(palette, font)| FallingSand::new(Batch::new(palette), font, 128, 128))
    }

    fn set_row(&mut self, strain: Strain, row: usize) {
        for x in 0..128 {
            self.set_strain(x, row, strain);
        }
    }

    fn index(&mut self, x: usize, y: usize) -> usize {
        x + self.grid_width * y
    }

    fn get(&mut self, x: usize, y: usize) -> Particle {
        let i: usize = self.index(x, y);

        self.grid[i]
    }

    fn set(&mut self, x: usize, y: usize, p: Particle) {
        let i: usize = self.index(x, y);

        self.grid[i] = p;
    }

    fn set_strain(&mut self, x: usize, y: usize, s: Strain) {
        let i: usize = self.index(x, y);

        self.grid[i].strain = s;
    }

    fn is_particle_empty(&mut self, x: usize, y: usize) -> bool {
        self.get(x, y).strain == Strain::Empty
    }

    fn apply_tumble(&mut self, x: usize, y: usize) -> bool {
        let mut translate: Vector2<isize> = nalgebra::Vector2::new(0, 0);

        let &grid_height: &usize = &self.grid_height;
        let &grid_width: &usize = &self.grid_width;

        if y + 1 < grid_height && self.is_particle_empty(x, y + 1)
        {
            // Move down
            translate.y = 1;
        } else if y + 1 < grid_height && x > 0 && self.is_particle_empty(x - 1, y + 1) {

            // Move down left
            translate.x = -1;
            translate.y = 1;
        } else if y + 1 < grid_height && x + 1 < grid_width && self.is_particle_empty(x + 1, y + 1) {
            // Move down right
            translate.x = 1;
            translate.y = 1;
        }

        if translate.x != 0 || translate.y != 0 {
            // Move particle
            let p = self.get(x, y);

            self.set((x as isize + translate.x) as usize, (y as isize + translate.y) as usize, p);

            // Clear old cell
            self.set(x, y, Particle::default());

            return true;
        }

        false
    }

    fn apply_spread(&mut self, x: usize, y: usize) -> bool {
        let mut trans_x: isize = 0;

        let &grid_width: &usize = &self.grid_width;

        let dir: isize;
        if random() { dir = -1; } else { dir = 1; }

        if x > 0 && (x as isize + dir < grid_width as isize) && self.is_particle_empty((x as isize + dir) as usize, y)
        {
            trans_x = dir;
        }

        if trans_x != 0 {
            // Move particle
            let p = self.get(x, y);
            self.set((x as isize + trans_x) as usize, y, p);

            // Clear old cell
            self.set(x, y, Particle::default());

            return true;
        }

        false
    }

    fn apply_density(&mut self, x: usize, y: usize) -> bool {
        if y + 1 < self.grid_height {
            let cur = self.get(x, y);
            let bel = self.get(x, y + 1);

            // If the current particle's density is greater than the particle below's
            if cur.strain.density() > bel.strain.density() {

                // Swap
                self.set(x, y, bel);
                self.set(x, y + 1, cur);

                return true;
            }
        }

        false
    }

    fn spawn_particle(&mut self, x: usize, y: usize, p: Particle) {
        if x < self.grid_width && y < self.grid_height {
            self.set(x, y, p);
        }
    }
}

impl Game for FallingSand {
    type Input = Inputs;
    type LoadingScreen = ProgressBar;

    const TICKS_PER_SECOND: u16 = 60;

    fn load(_window: &Window) -> Task<Self> {
        Task::stage("Loading...", FallingSand::load())
    }

    fn draw(&mut self, frame: &mut Frame, _timer: &Timer) {
        frame.clear(Color::BLACK);

        let target = &mut frame.as_target();

        // multiplies the size of particles
        let scale: f32 = 4.;

        self.batch.clear();

        // collects all particles into a batch of rectangle sprites
        for x in 0..self.grid_width {
            for y in 0..self.grid_height {
                let p = self.get(x as usize, y as usize);
                if p.strain != Strain::Empty
                {
                    self.batch.add(
                        Sprite {
                            source: Rectangle {
                                x: p.strain.to_colour_id(),
                                y: 0,
                                width: 1,
                                height: 1,
                            },
                            position: Point::new(x as f32 * scale, y as f32 * scale),
                            scale: (1.0 * scale, 1.0 * scale),
                        });
                }
            }
        }

        // draw particle batch
        self.batch.draw(target);

        // add and then draw text
        self.font.add(Text {
            content: &*format!("particles_updated={}", self.particles_updated),
            position: Point::new(20.0, 20.0),
            size: 16.0,
            color: COLORS[0],
            ..Text::default()
        });

        self.font.add(Text {
            content: &*format!("active: {}", self.active_strain.to_str()),
            position: Point::new(20., 36.),
            size: 16.0,
            color: COLORS[self.active_strain.to_colour_id() as usize],
            ..Text::default()
        });

        self.font.draw(target);
    }

    //noinspection RsBorrowChecker
    fn interact(&mut self, input: &mut Inputs, _window: &mut Window) {
        self.cursor_position = input.cursor_position;
        self.mouse_wheel = input.mouse_wheel;
        self.keys_pressed = input.keys_pressed.clone();
        self.mouse_buttons_pressed = input.mouse_buttons_pressed.clone();

        if !input.text_buffer.is_empty() {
            for c in input.text_buffer.chars() {
                match c {
                    // Match ASCII backspace and delete from the text buffer
                    '\u{0008}' => {
                        self.text_buffer.pop();
                    }
                    _ => {
                        if self.text_buffer.chars().count() < Self::MAX_TEXTSIZE
                        {
                            self.text_buffer.push_str(&input.text_buffer);
                        }
                    }
                }
            }
        }
    }

    fn update(&mut self, _window: &Window)
    {
        // Update current strain for mouse click
        let x: Option<&keyboard::KeyCode> = self.keys_pressed.par_iter().find_first(|&&x| x == keyboard::KeyCode::E || x == keyboard::KeyCode::Key1 || x == keyboard::KeyCode::Key2);

        if x != None
        {
            self.active_strain =
                match x.unwrap() {
                    keyboard::KeyCode::E => Strain::Empty,
                    keyboard::KeyCode::Key1 => Strain::Sand,
                    keyboard::KeyCode::Key2 => Strain::Water,
                    _ => Strain::Sand
                }
        }

        // Spawn particle at mouse
        let left_down = self.mouse_buttons_pressed.contains(&mouse::Button::Left);
        let right_down = self.mouse_buttons_pressed.contains(&mouse::Button::Right);

        if left_down {
            let x: usize = (self.cursor_position.x / 4.) as usize;
            let y: usize = (self.cursor_position.y / 4.) as usize;

            self.spawn_particle(x, y, Particle {
                strain: self.active_strain,
                update: false,
            })
        }

        // Reset updated particles stat
        self.particles_updated = 0;

        // Update particle grid - bottom to top; left to right
        for y in (0..self.grid_height).rev() {
            for x in 0..self.grid_width {
                let mut p = self.get(x, y);

                // check the particle has not been updated this frame & ensure it isn't empty
                if p.update == self.update && p.strain != Strain::Empty {
                    p.update = !p.update;

                    // save update state to grid
                    self.set(x, y, p);

                    // select particle update behaviour depending on its Strain
                    match p.strain {
                        Strain::Sand => {
                            self.apply_tumble(x, y);
                            self.apply_density(x, y);
                        }
                        Strain::Water => {
                            if !self.apply_tumble(x, y) {
                                self.apply_spread(x, y);
                            }

                            self.apply_density(x, y);
                        }
                        _ => {}
                    }
                    self.particles_updated += 1;
                }
            }
        }

        self.update = !self.update;
    }
}

const COLORS: [Color; 7] = [
    // Sand
    Color {
        r: 1.0,
        g: 0.87,
        b: 0.67,
        a: 1.0,
    },
    // Water
    Color {
        r: 0.117,
        g: 0.564,
        b: 1.0,
        a: 1.0,
    },
    // White
    Color {
        r: 1.0,
        g: 1.0,
        b: 1.0,
        a: 1.0,
    },
    Color {
        r: 0.7,
        g: 0.7,
        b: 0.7,
        a: 1.0,
    },
    Color {
        r: 0.8,
        g: 0.8,
        b: 0.8,
        a: 1.0,
    },
    Color {
        r: 0.9,
        g: 0.9,
        b: 0.9,
        a: 1.0,
    },
    Color {
        r: 0.8,
        g: 0.8,
        b: 1.0,
        a: 1.0,
    },
];