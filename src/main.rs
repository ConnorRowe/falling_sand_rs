use std::collections::HashSet;

use coffee::graphics::{
    Batch, Color, Font, Frame, Image, Point, Rectangle, Sprite, Text, Window, WindowSettings,
};
use coffee::input::{keyboard, mouse, Input};
use coffee::load::{loading_screen::ProgressBar, Join, Task};
use coffee::{input, Game, Result, Timer};
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
    lifetime: i16,
}

impl Default for Particle {
    fn default() -> Self {
        Particle {
            strain: Strain::Empty,
            update: false,
            lifetime: -1,
        }
    }
}

#[repr(u8)]
#[derive(Clone, Copy, Eq, PartialEq)]
pub enum Strain {
    Empty = 0,
    Sand = 1,
    Water = 2,
    Wood = 3,
    Fire = 4,
    Glass = 5,
    MoltenGlass = 6,
}

impl Strain {
    fn to_colour_id(&self) -> u16 {
        match self {
            Strain::Sand => 1,
            Strain::Water => 2,
            Strain::Wood => 3,
            Strain::Fire => 4,
            Strain::Glass => 5,
            Strain::MoltenGlass => 6,
            _ => 0,
        }
    }

    fn density(&self) -> u16 {
        match self {
            Strain::Sand => 1600,
            Strain::Water => 1000,
            Strain::Wood => 9999,
            Strain::Fire => 600,
            Strain::Glass => 9999,
            Strain::MoltenGlass => 1600,
            _ => 1000,
        }
    }

    fn to_str(&self) -> &'static str {
        match self {
            Strain::Empty => "Empty",
            Strain::Sand => "Sand",
            Strain::Water => "Water",
            Strain::Wood => "Wood",
            Strain::Fire => "Fire",
            Strain::Glass => "Glass",
            Strain::MoltenGlass => "Molten Glass",
            _ => "",
        }
    }

    // how long it survives in ticks
    fn base_lifetime(&self) -> i16 {
        let mut rng = rand::thread_rng();

        match self {
            Strain::Fire => rng.gen_range(60, 100),
            Strain::MoltenGlass => rng.gen_range(240, 480),
            _ => -1,
        }
    }

    // what it turns into when it dies
    fn death_strain(&self) -> Strain {
        match self {
            Strain::MoltenGlass => Strain::Glass,
            _ => Strain::Empty,
        }
    }

    // out of 100
    fn ignite_chance(&self) -> u8 {
        match self {
            Strain::Wood => 5,
            _ => 0,
        }
    }

    fn reactable_strains(&self) -> Vec<(Strain, i8, Strain)> {
        match self {
            Strain::Sand => vec![(Strain::Fire, 1, Strain::MoltenGlass)],
            Strain::MoltenGlass => vec![(Strain::Water, 50, Strain::Glass)],
            Strain::Glass => vec![(Strain::Fire, 10, Strain::MoltenGlass)],

            _ => vec![],
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
    four_adj_particles: [Vector2<isize>; 4],
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
            four_adj_particles: [
                Vector2::new(-1, 0),
                Vector2::new(1, 0),
                Vector2::new(0, -1),
                Vector2::new(0, 1),
            ],
        }
    }

    fn load() -> Task<FallingSand> {
        (
            Task::using_gpu(|gpu| Image::from_colors(gpu, &COLORS)),
            Font::load_from_bytes(include_bytes!("../resources/Gamepixies-8MO6n.ttf")),
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

    fn swap(&mut self, x1: usize, y1: usize, x2: usize, y2: usize) {
        let cur: &Particle = &self.get(x1, y1);
        let other: &Particle = &self.get(x2, y2);

        self.set(x1, y1, *other);
        self.set(x2, y2, *cur);
    }

    fn apply_gravity(&mut self, x: usize, y: usize, val: isize) -> bool {
        if ((y as isize + val) <= 0) || (y as isize + val > self.grid_height as isize - 1) {
            return false;
        } else {
            // Compare densities
            let dx = self.get(x, y).strain.density();
            let dy = self.get(x, (y as isize + val) as usize).strain.density();

            if dx > dy || self.is_particle_empty(x, (y as isize + val) as usize) {
                self.swap(x, y, x, (y as isize + val) as usize);

                return true;
            }
        }

        return false;
    }

    fn apply_tumble(&mut self, x: usize, y: usize) -> bool {
        let mut translate: Vector2<isize> = nalgebra::Vector2::new(0, 0);

        let &grid_height: &usize = &self.grid_height;
        let &grid_width: &usize = &self.grid_width;

        if y + 1 < grid_height && x > 0 && self.is_particle_empty(x - 1, y + 1) {
            // Move down left
            translate.x = -1;
            translate.y = 1;
        } else if y + 1 < grid_height && x + 1 < grid_width && self.is_particle_empty(x + 1, y + 1)
        {
            // Move down right
            translate.x = 1;
            translate.y = 1;
        }

        if translate.x != 0 || translate.y != 0 {
            self.swap(
                x,
                y,
                (x as isize + translate.x) as usize,
                (y as isize + translate.y) as usize,
            );

            return true;
        }

        false
    }

    fn apply_spread(&mut self, x: usize, y: usize) -> bool {
        let mut trans_x: isize = 0;

        let &grid_width: &usize = &self.grid_width;

        let dir: isize;
        if random() {
            dir = -1;
        } else {
            dir = 1;
        }

        if x > 0
            && (x as isize + dir < grid_width as isize)
            && self.is_particle_empty((x as isize + dir) as usize, y)
        {
            trans_x = dir;
        }

        if trans_x != 0 {
            self.swap(x, y, (x as isize + trans_x) as usize, y);

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
                self.swap(x, y, x, y + 1);

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
                if p.strain != Strain::Empty {
                    self.batch.add(Sprite {
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
            position: Point::new(8.0, 2.0),
            size: 16.0,
            color: COLORS[0],
            ..Text::default()
        });

        let cur = self.cursor_position;
        let cur_x = (cur.x / 4.) as usize;
        let cur_y = (cur.y / 4.) as usize;

        let under_cur = if cur_x < self.grid_width && cur_y < self.grid_height {
            self.get((cur.x / 4.) as usize, (cur.y / 4.) as usize)
                .strain
                .to_str()
        } else {
            "Empty"
        };

        self.font.add(Text {
            content: &*format!("under cursor: {}", under_cur),
            position: Point::new(8., 16.),
            size: 16.0,
            color: COLORS[0],
            ..Text::default()
        });

        self.font.add(Text {
            content: &*format!("active: {}", self.active_strain.to_str()),
            position: Point::new(8., 30.),
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
                        if self.text_buffer.chars().count() < Self::MAX_TEXTSIZE {
                            self.text_buffer.push_str(&input.text_buffer);
                        }
                    }
                }
            }
        }
    }

    fn update(&mut self, _window: &Window) {
        // Update current strain for mouse click
        let x: Option<&keyboard::KeyCode> = self.keys_pressed.par_iter().find_first(|&&x| {
            x == keyboard::KeyCode::E
                || x == keyboard::KeyCode::Key1
                || x == keyboard::KeyCode::Key2
                || x == keyboard::KeyCode::Key3
                || x == keyboard::KeyCode::Key4
        });

        if x != None {
            self.active_strain = match x.unwrap() {
                keyboard::KeyCode::E => Strain::Empty,
                keyboard::KeyCode::Key1 => Strain::Sand,
                keyboard::KeyCode::Key2 => Strain::Water,
                keyboard::KeyCode::Key3 => Strain::Wood,
                keyboard::KeyCode::Key4 => Strain::Fire,
                _ => Strain::Sand,
            }
        }

        // Spawn particle at mouse
        let left_down = self.mouse_buttons_pressed.contains(&mouse::Button::Left);
        let right_down = self.mouse_buttons_pressed.contains(&mouse::Button::Right);

        if left_down {
            let x: usize = (self.cursor_position.x / 4.) as usize;
            let y: usize = (self.cursor_position.y / 4.) as usize;

            let points: Vec<Vector2<isize>> = vec![
                Vector2::new(0, 0),
                Vector2::new(-1, 0),
                Vector2::new(1, 0),
                Vector2::new(0, -1),
                Vector2::new(0, 1),
            ];

            for v in points {
                let xp = (x as isize + v.x) as usize;
                let yp = (y as isize + v.y) as usize;

                if xp < self.grid_width
                    && yp < self.grid_height
                    && (self.is_particle_empty(xp, yp) || self.active_strain == Strain::Empty)
                {
                    self.spawn_particle(
                        xp,
                        yp,
                        Particle {
                            strain: self.active_strain,
                            lifetime: self.active_strain.base_lifetime(),
                            ..Default::default()
                        },
                    );
                }
            }
        }

        // Reset updated particles stat
        self.particles_updated = 0;

        // Update particle grid - bottom to top; left to right
        for y in (0..self.grid_height).rev() {
            for x in 0..self.grid_width {
                let mut p = self.get(x, y);

                // check if dead
                if p.lifetime == 0 {
                    p.strain = p.strain.death_strain();
                    p.lifetime = p.strain.base_lifetime();

                    // save
                    self.set(x, y, p);
                }
                // check the particle has not been updated this frame & ensure it isn't empty
                else if p.update == self.update && p.strain != Strain::Empty {
                    p.update = !p.update;

                    // decrease lifetime if needed
                    if p.lifetime > 0 {
                        p.lifetime -= 1;
                    }

                    // Attempt reaction
                    for r in p.strain.reactable_strains().iter() {
                        let itr = self.four_adj_particles;
                        for v in itr.iter() {
                            if ((x as isize + v.x) as usize) < self.grid_width
                                && ((y as isize + v.y) as usize) < self.grid_height
                            {
                                let other = self
                                    .get((x as isize + v.x) as usize, (y as isize + v.y) as usize);

                                if other.strain == r.0 && thread_rng().gen_range(0, 100) <= r.1 {
                                    // Reaction successful
                                    p.strain = r.2;
                                    p.lifetime = p.strain.base_lifetime();
                                    break;
                                }
                            }
                        }
                    }

                    // save state to grid
                    self.set(x, y, p);

                    // select particle update behaviour depending on its Strain
                    match p.strain {
                        Strain::Sand => {
                            if !self.apply_gravity(x, y, 1) {
                                self.apply_tumble(x, y);
                            }
                        }
                        Strain::Water => {
                            if !self.apply_gravity(x, y, 1) {
                                if !self.apply_tumble(x, y) {
                                    self.apply_spread(x, y);
                                }
                            }
                        }
                        Strain::Fire => {
                            if random() {
                                self.apply_gravity(x, y, -1);
                            }
                            if random() {
                                self.apply_spread(x, y);
                            }

                            // Attempt to ignite nearby particles
                            let itr = self.four_adj_particles;
                            for v in itr.iter() {
                                if ((x as isize + v.x) as usize) < self.grid_width
                                    && ((y as isize + v.y) as usize) < self.grid_height
                                {
                                    let mut p = self.get(
                                        (x as isize + v.x) as usize,
                                        (y as isize + v.y) as usize,
                                    );

                                    if p.strain != Strain::Empty && p.strain.ignite_chance() > 0 {
                                        let mut rng = thread_rng();
                                        if rng.gen_range(0, 100) <= p.strain.ignite_chance() {
                                            p.strain = Strain::Fire;
                                            p.lifetime = p.strain.base_lifetime();

                                            self.set(
                                                (x as isize + v.x) as usize,
                                                (y as isize + v.y) as usize,
                                                p,
                                            );
                                            break;
                                        }
                                    }
                                }
                            }
                        }
                        Strain::MoltenGlass => {
                            if !self.apply_gravity(x, y, 1) {
                                // Randomly dont move to appear thicker
                                if random() {
                                    if !self.apply_tumble(x, y) {
                                        self.apply_spread(x, y);
                                    }
                                }
                            }
                        }
                        _ => {}
                    }

                    self.particles_updated += 1;
                }
            }
        }

        self.update = !self.update;
    }

    const DEBUG_KEY: Option<keyboard::KeyCode> = Some(keyboard::KeyCode::F12);
}

const COLORS: [Color; 7] = [
    // White
    Color {
        r: 1.0,
        g: 1.0,
        b: 1.0,
        a: 1.0,
    },
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
    // Wood
    Color {
        r: 0.6274,
        g: 0.3215,
        b: 0.1647,
        a: 1.0,
    },
    // Fire
    Color {
        r: 1.0,
        g: 0.2705,
        b: 0.0,
        a: 1.0,
    },
    // Glass
    Color {
        r: 0.85,
        g: 0.85,
        b: 0.85,
        a: 1.0,
    },
    // Molten glass
    Color {
        r: 1.0,
        g: 0.498,
        b: 0.3137,
        a: 1.0,
    },
];
