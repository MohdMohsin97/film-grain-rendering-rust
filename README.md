# Film Grain Rendering in Rust

This project applies a film grain effect to an input image using a pixel-wise algorithm in Rust. The effect is customizable via parameters such as grain radius, filter strength, and Monte Carlo simulation iterations.

## Requirements

Before running the project, ensure that you have the following installed:

- [Rust](https://www.rust-lang.org/tools/install)
- [Cargo](https://doc.rust-lang.org/cargo/getting-started/installation.html)

## How to Build and Run

1. **Clone the repository or download the source code**:
   ```bash
   git clone https://github.com/MohdMohsin97/film-grain-rendering-rust.git
   cd film-grain-rendering-rust
   ```

2. **Build the project**:
    ```bash
    cargo build --release
    ```

3. **Run the project**:
    ```bash
    cargo run --release
    ```