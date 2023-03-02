#![forbid(unsafe_code)]

use image::flat::SampleLayout;
use image::{imageops, imageops::FilterType, GenericImage, GenericImageView, ImageBuffer, Pixel};
use ndarray::{stack, Array3, Axis, ShapeBuilder};
#[allow(unused_imports)]
use onnxruntime::{
    ndarray::Array, CUDAProviderOptions, Environment, GraphOptimizationLevel, LoggingLevel,
    TensorrtProviderOptions, Value,
};
use std::collections::HashMap;
use std::fs::File;
use std::io::BufRead;
use std::io::BufReader;
use std::iter;
use std::time::Instant;
use tracing::Level;
use tracing_subscriber::FmtSubscriber;

type Error = Box<dyn std::error::Error>;

static N: usize = 100;
static BATCH_SIZE: usize = 1;
static WARM_UP: usize = 10;
static MODEL_WIDTH: usize = 640;
static MODEL_HEIGHT: usize = 640;

fn main() {
    if let Err(e) = run() {
        eprintln!("Error: {e}");
        std::process::exit(1);
    }
}

fn run() -> Result<(), Error> {
    // Setup the example's log level.
    // NOTE: ONNX Runtime's log level is controlled separately when building the environment.
    let subscriber = FmtSubscriber::builder()
        .with_max_level(Level::INFO)
        .finish();

    tracing::subscriber::set_global_default(subscriber).expect("setting default subscriber failed");

    println!(
        "available providers {:?}",
        onnxruntime::get_available_providers()?
    );

    let file = File::open("coco.names")?;
    let buf = BufReader::new(file);
    let names = buf
        .lines()
        .map(|l| l.expect("Could not parse line"))
        .collect::<Vec<_>>();

    let environment = Environment::builder()
        .with_name("test")
        .with_log_level(LoggingLevel::Warning)
        .build()?;

    let session = environment
        .new_session_builder()?
        .with_tensorrt(
            TensorrtProviderOptions::default()
                .with_fp16_enable(true)
                .with_engine_cache_enable(true)
                .with_engine_cache_path(Some("./")),
        )?
        // .with_cuda(CUDAProviderOptions::default())?
        .with_optimization_level(GraphOptimizationLevel::All)?
        .with_inter_op_num_threads(16)?
        .with_model_from_file("./yolov7_onms_bN_fp32.onnx")?;

    let img_raw = image::io::Reader::open("./vehicle.jpg")?
        .decode()?
        .to_rgb8();
    let (img, scale_x, scale_y) = resize_proportional(
        &img_raw,
        MODEL_WIDTH as u32,
        MODEL_HEIGHT as u32,
        image::imageops::Nearest,
    );

    let batch = stack(
        Axis(0),
        &iter::repeat(img.into_ndarray3().view())
            .take(BATCH_SIZE)
            .collect::<Vec<_>>(),
    )?
    .mapv(|v| v as f32 / 255.0)
    .into_dyn();

    let mut start = Instant::now();

    for i in 1..N + WARM_UP {
        println!("run: {i}");
        let outputs = session.run(HashMap::from([(
            "images",
            &OrtValue::try_from_array(&batch)?,
        )]))?;

        if i == WARM_UP {
            start = Instant::now()
        }

        if let Some(output) = outputs.get("output") {
            let output = output.array_view::<f32>()?;

            output.outer_iter().for_each(|output| {
                let batch_index = output[0] as usize;
                let class_index = output[5] as usize;

                println!(
                    "{} {} {} {:.2} {} {} {} {}",
                    batch_index,
                    class_index,
                    names.get(class_index).unwrap(),
                    output[6],
                    ((output[1] / MODEL_WIDTH as f32) * img_raw.width() as f32 * scale_x) as usize,
                    ((output[2] / MODEL_HEIGHT as f32) * img_raw.height() as f32 * scale_y)
                        as usize,
                    ((output[3] / MODEL_WIDTH as f32) * img_raw.width() as f32 * scale_x) as usize,
                    ((output[4] / MODEL_HEIGHT as f32) * img_raw.height() as f32 * scale_y)
                        as usize,
                )
            });
        }
    }

    let duration = start.elapsed();

    println!(
        "batch {} duration {:.3} ms {:.2} fps",
        BATCH_SIZE,
        duration.as_nanos() as f64 / N as f64 / 1000000_f64,
        1000f64 / (duration.as_nanos() as f64 / N as f64 / 1000000_f64) * BATCH_SIZE as f64,
    );

    Ok(())
}

/// Resize the supplied image to the specified dimensions maintaining input image proportions.
/// ```target_width``` and ```target_height``` are the new dimensions.
/// ```filter``` is the sampling filter to use.
#[allow(dead_code, clippy::type_complexity)]
pub fn resize_proportional<I: GenericImageView>(
    image: &I,
    target_width: u32,
    target_height: u32,
    filter: FilterType,
) -> (
    ImageBuffer<I::Pixel, Vec<<I::Pixel as Pixel>::Subpixel>>,
    f32,
    f32,
)
where
    I::Pixel: 'static,
    <I::Pixel as Pixel>::Subpixel: 'static,
{
    let image_ratio = image.width() as f32 / image.height() as f32;
    let target_ratio = target_width as f32 / target_height as f32;

    // create an empty black image as a square with of the longest-edge
    // and calculate ratios for scaling
    let (mut pad_image, scale_x, scale_y) = match image_ratio.partial_cmp(&target_ratio) {
        Some(std::cmp::Ordering::Less) => (
            ImageBuffer::new(
                (image.height() as f32 * target_ratio) as u32,
                image.height(),
            ),
            image_ratio / target_ratio,
            1.0,
        ),
        Some(std::cmp::Ordering::Greater) => (
            ImageBuffer::new(image.width(), (image.width() as f32 / target_ratio) as u32),
            1.0,
            image_ratio / target_ratio,
        ),
        _ => (ImageBuffer::new(image.width(), image.height()), 1.0, 1.0),
    };

    // copy the original image to the top-left corner of the black image
    pad_image.copy_from(image, 0, 0).unwrap();

    // return the resized image
    (
        imageops::resize(&pad_image, target_width, target_height, filter),
        scale_x,
        scale_y,
    )
}

pub trait ToNdarray3 {
    type Out;

    fn into_ndarray3(self) -> Self::Out;
}

impl<P> ToNdarray3 for ImageBuffer<P, Vec<P::Subpixel>>
where
    P: Pixel + 'static,
{
    type Out = Array3<P::Subpixel>;

    fn into_ndarray3(self) -> Self::Out {
        let SampleLayout {
            channels,
            channel_stride,
            height,
            height_stride,
            width,
            width_stride,
        } = self.sample_layout();
        let shape = (channels as usize, height as usize, width as usize);
        let strides = (channel_stride, height_stride, width_stride);
        let stride_shape = shape.strides(strides);
        Array3::from_shape_vec(stride_shape, self.into_raw()).unwrap()
    }
}
