use std::f32::consts::PI;

use image::{GrayImage, ImageBuffer, Luma};
use rand::{rngs::StdRng, Rng, SeedableRng};
use rand_distr::{Distribution, Normal};
use rayon::iter::{ParallelBridge, ParallelIterator};

const MAX_GRAY_LEVEL: u8 = 255;
const EPSION_GRAY_LEVEL: f32 = 0.1;
const NORMAL_QUANTILE: f32 = 3.0902; // standard normal quantile for alpha=0.999

#[derive(Debug)]
pub struct FilmGrainOptions {
    pub mu_r: f32,          // representing the average size of the grains.
    pub sigma_r: f32,       // Standard deviation of the grain radius
    pub sigma_filter: f32,  // Standard deviation of the filter
    pub n_monte_carlo: u32, // Number of Monte Carlo simulations
    pub x_a: f32,           // x-coordinate of the top-left boundary of the image
    pub y_a: f32,           // y-coordinate of the top-left boundary of the image
    pub x_b: f32,           // x-coordinate of the bottom-right boundary of the image
    pub y_b: f32,           // y-coordinate of the bottom-right boundary of the image
    pub m_out: u32,         // Number of rows in the output image
    pub n_out: u32,         // Number of columns in the output image
    pub grain_seed: u32, // Random seed used to initialize the pseudo-random number generator for consistent grain rendering
}

pub fn film_grain_rendering_pixel_wise(
    img_in: &GrayImage,
    film_grain_options: &FilmGrainOptions,
) -> GrayImage {
    let grain_radius = film_grain_options.mu_r;
    let grain_std = film_grain_options.sigma_r;
    let sigma_filter = film_grain_options.sigma_filter;

    let n_monte_carlo = film_grain_options.n_monte_carlo;

    let mut rng = StdRng::from_entropy();
    let normal_dist = Normal::new(0.0, sigma_filter).unwrap();

    //draw the random (gaussian) translation vectors
    let x_gaussian_list: Vec<f32> = (0..n_monte_carlo)
        .map(|_| normal_dist.sample(&mut rng))
        .collect();

    let y_gaussian_list: Vec<f32> = (0..n_monte_carlo)
        .map(|_| normal_dist.sample(&mut rng))
        .collect();

    //pre-calculate lambda and exp(-lambda) for each possible grey-level
    let lambda_list: Vec<f32> = (0..=MAX_GRAY_LEVEL)
        .map(|i| {
            let u = (i as f32) / (MAX_GRAY_LEVEL as f32 + EPSION_GRAY_LEVEL);
            let ag = 1.0 / (1.0 / grain_radius).ceil();
            let lambda_temp = -((ag * ag)
                / (PI * (grain_radius * grain_radius + grain_std * grain_std)))
                * (1.0 - u).ln();
            lambda_temp
        })
        .collect();

    let exp_lambda_list: Vec<f32> = lambda_list.iter().map(|&l| (-l).exp()).collect();

    let mut img_out = ImageBuffer::new(film_grain_options.m_out, film_grain_options.n_out);

    let m_out = img_out.width();
    let n_out = img_out.height();

    img_out
        .enumerate_pixels_mut()
        .par_bridge()
        .for_each(|(x, y, pixel)| {
            let new_value = render_pixel(
                img_in,
                x,
                y,
                img_in.width(),
                img_in.height(),
                m_out,
                n_out,
                film_grain_options.grain_seed,
                n_monte_carlo,
                grain_radius,
                grain_std,
                sigma_filter,
                film_grain_options.x_a,
                film_grain_options.y_a,
                film_grain_options.x_b,
                film_grain_options.y_b,
                &lambda_list,
                &exp_lambda_list,
                &x_gaussian_list,
                &y_gaussian_list,
            );

            *pixel = Luma([(new_value * 255 as f32).clamp(0.0, 255.0) as u8]);
        });

    img_out
}

/// Render one pixel in a Monte Carlo simulation-based pixel-wise algorithm.
///
/// # Parameters
/// - `img_in`: Input image as a 1D array of floats.
/// - `y_out, x_out`: Coordinates of the output pixel.
/// - `m_in, n_in`: Input image size.
/// - `m_out, n_out`: Output image size.
/// - `offset`: Offset to put into the pseudo-random number generator.
/// - `n_monte_carlo`: Number of iterations in the Monte Carlo simulation.
/// - `grain_radius`: Average grain radius.
/// - `sigma_r`: Standard deviation of the grain radius.
/// - `sigma_filter`: Standard deviation of the blur kernel.
/// - `(x_a, y_a), (x_b, y_b)`: Limits of the image to render.
/// - `lambda_list, exp_lambda_list`: Precomputed values for Poisson distribution.
/// - `x_gaussian_list, y_gaussian_list`: Precomputed random Gaussian values.
///
/// # Returns
/// The output value of the pixel after the Monte Carlo simulation.

fn render_pixel(
    img_in: &GrayImage, //Input image
    x: u32, y: u32, // Coordinates of the output pixel.
    m_in: u32, n_in: u32, // Input image size.
    m_out: u32, n_out: u32, // Output image size.
    offest: u32, // Offset to put into the pseudo-random number generator.
    n_monte_carlo: u32, // Number of iterations in the Monte Carlo simulation.
    grain_radius: f32, // Average grain radius.
    sigma_r: f32, // Standard deviation of the grain radius.
    sigma_filter: f32, // Standard deviation of the blur kernel.
    x_a: f32, y_a: f32, x_b: f32, y_b: f32, // Limits of the image to render.
    lambda_list: &[f32], exp_lambda_list: &[f32], // Precomputed values for Poisson distribution.
    x_gaussian_list: &[f32], y_gaussian_list: &[f32], // Precomputed random Gaussian values.
) -> f32 {
    let grain_radius_sq = grain_radius * grain_radius;
    let mut max_radius = grain_radius;
    let mut mu: f32 = 0.0;
    let mut sigma: f32 = 0.0;
    let sigma_sq: f32;
    let log_normal_quantile: f32;

    let ag = 1.0 / (1.0 / grain_radius).ceil();
    let scale_x = (m_out as f32 - 1.0) / (x_b - x_a);
    let scale_y = (n_out as f32 - 1.0) / (y_b - y_a);

    let mut pixel_out = 0.0;

    //conversion from output grid (x_out,y_out) to input grid (x_in,y_in)
    //we inspect the middle of the output pixel (1/2)
    //the size of a pixel is (x_b-x_a)/n_out
    let x_in = x_a + (x as f32 + 0.5) * ((x_b - x_a) / (m_out as f32));
    let y_in = y_a + (y as f32 + 0.5) * ((y_b - y_a) / (n_out as f32));

    //calculate the mu and sigma for the lognormal distribution
    if sigma_r > 0.0 {
        sigma = (((sigma_r / grain_radius).powi(2) + 1.0).ln()).sqrt();
        sigma_sq = sigma * sigma;
        mu = grain_radius.ln() - sigma_sq / 2.0;
        log_normal_quantile = (mu + sigma * NORMAL_QUANTILE).exp();
        max_radius = log_normal_quantile;
    }

    //loop over the number of Monte Carlo simulations
    for i in 0..n_monte_carlo as usize {
        let x_gaussian = x_in + sigma_filter * (x_gaussian_list[i]) / scale_x;
        let y_gaussian = y_in + sigma_filter * (y_gaussian_list[i]) / scale_y;

        //determine the bounding boxes around the current shifted pixel
        let min_x = ((x_gaussian - max_radius) / ag).floor() as u32;
        let max_x = ((x_gaussian + max_radius) / ag).floor() as u32;
        let min_y = ((y_gaussian - max_radius) / ag).floor() as u32;
        let max_y = ((y_gaussian + max_radius) / ag).floor() as u32;

        let mut pt_covered = false;

        for ncx in min_x..=max_x {
            if pt_covered == true {
                break;
            }
            for ncy in min_y..=max_y {
                if pt_covered == true {
                    break;
                }
                let cell_corner = Vec2d::new(ag * ncx as f32, ag * ncy as f32);

                let seed = cell_seed(ncx, ncy, offest);

                let mut rng = StdRng::seed_from_u64(seed as u64);

                let img_y = f32::min(f32::max(cell_corner.y.floor(), 0.0), (n_in - 1) as f32);
                let img_x = f32::min(f32::max(cell_corner.x.floor(), 0.0), (m_in - 1) as f32);

                let pixel = img_in.get_pixel(img_x as u32, img_y as u32)[0] as usize;

                let curr_lambda = lambda_list[pixel];
                let curr_exp_lambda = exp_lambda_list[pixel];

                let n_cell = my_rand_poisson(&mut rng, curr_lambda, curr_exp_lambda);

                for _k in 0..n_cell {
                    let x_centre_grain = cell_corner.x + ag * rng.gen_range(0.0..=1.0);
                    let y_centre_grain = cell_corner.y + ag * rng.gen_range(0.0..=1.0);

                    let curr_grain_radius_sq: f32;

                    if sigma_r > 0.0 {
                        let normal_dist = Normal::new(0.0, sigma_r).unwrap();
                        let rand_gaussian = normal_dist.sample(&mut rng) as f32;
                        let curr_radius = (mu + sigma + rand_gaussian).exp().min(max_radius);
                        curr_grain_radius_sq = curr_radius * curr_radius;
                    } else if sigma_r == 0.0 {
                        curr_grain_radius_sq = grain_radius_sq;
                    } else {
                        panic!("Error, the standard deviation of the grain should be positive.");
                    }

                    if sq_distance(x_centre_grain, y_centre_grain, x_gaussian, y_gaussian)
                        < curr_grain_radius_sq
                    {
                        pixel_out = pixel_out + 1.0;
                        pt_covered = true;
                        break;
                    }
                }
            }
        }
    }
    pixel_out / n_monte_carlo as f32
}

struct Vec2d {
    x: f32,
    y: f32,
}

impl Vec2d {
    fn new(x: f32, y: f32) -> Self {
        Vec2d { x, y }
    }
}

fn cell_seed(x: u32, y: u32, offest: u32) -> u32 {
    let period = 65536u32;
    let mut s = ((y % period) * period + (x % period)) + offest;
    if s == 0 {
        s = 1;
    }
    s
}

fn my_rand_poisson<R: Rng>(rng: &mut R, lambda: f32, prod_in: f32) -> u32 {
    let u = rng.gen::<f32>();
    let mut x: u32 = 0;
    let mut prod: f32;

    if prod_in <= 0.0 {
        prod = (-lambda).exp();
    } else {
        prod = prod_in;
    }

    let mut sum = prod;

    while u > sum && x < (10000.0 * lambda).floor() as u32 {
        x += 1;
        prod *= lambda / (x as f32);
        sum += prod;
    }

    x
}

fn sq_distance(x1: f32, y1: f32, x2: f32, y2: f32) -> f32 {
    (x1 - x2).powi(2) + (y1 - y2).powi(2)
}
