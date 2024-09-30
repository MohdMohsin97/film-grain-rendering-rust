use film_grain_rendering::{film_grain_rendering_pixel_wise, FilmGrainOptions};

mod film_grain_rendering;

fn main() {
    let img_in = image::open("bateau.png").unwrap().into_luma8();

    let m_in = img_in.width();
    let n_in = img_in.height();

    println!("image input size : {m_in} x {n_in}");

    // Film grain rendering options
    let options = FilmGrainOptions {
        mu_r: 0.1,
        sigma_r: 0.0,
        sigma_filter: 0.8,
        n_monte_carlo: 800,
        x_a: 0.0,
        y_a: 0.0,
        x_b: m_in as f32,
        y_b: n_in as f32,
        m_out: m_in,
        n_out: n_in,
        grain_seed: 42,
    };

    // Apply the film grain effect
    let img_out = film_grain_rendering_pixel_wise(&img_in, &options);

    // Save the output image
    img_out.save("output_image.png").unwrap();

    println!("Film grain effect applied and image saved.");
}
