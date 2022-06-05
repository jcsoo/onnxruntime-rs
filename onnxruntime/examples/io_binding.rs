#![forbid(unsafe_code)]
use ndarray::{stack, Axis};
use nshare::ToNdarray3;

use onnxruntime::{
    environment::Environment, ndarray::Array, AllocatorType, GraphOptimizationLevel, LoggingLevel,
    MemType, MemoryInfo, TypedArray, TypedOrtOwnedTensor,
};
use tracing::Level;
use tracing_subscriber::FmtSubscriber;

type Error = Box<dyn std::error::Error>;

use std::fs::File;
use std::io::BufRead;
use std::io::BufReader;
use std::time::Instant;

static N: usize = 1000;
static BATCH_SIZE: usize = 6;

fn main() {
    if let Err(e) = run() {
        eprintln!("Error: {}", e);
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
        // The ONNX Runtime's log level can be different than the one of the wrapper crate or the application.
        .with_log_level(LoggingLevel::Info)
        .build()?;

    let session = environment
        .new_session_builder()?
        .use_tensorrt(0, true)?
        .with_optimization_level(GraphOptimizationLevel::All)?
        .with_number_threads(8)?
        .with_model_from_file(format!("yolov4_b{}_c3_h320_w320.onnx", BATCH_SIZE))?;

    let img_raw = image::io::Reader::open("person.jpg")?.decode()?.to_rgb8();
    let img = image::imageops::resize(&img_raw, 320, 320, image::imageops::Nearest)
        .into_ndarray3()
        .mapv(|v| v as f32);

    let batch = stack![Axis(0), img, img, img, img, img, img]
        .into_shape(vec![BATCH_SIZE, 3, 320, 320])
        .unwrap();

    let max_output_boxes_per_class = Array::from_elem(1, 200).into_shape(vec![1]).unwrap();
    let iou_threshold = Array::from_elem(1, 0.5).into_shape(vec![1]).unwrap();
    let score_threshold = Array::from_elem(1, 0.5).into_shape(vec![1]).unwrap();

    let mut io_binding = session.io_binding()?;

    let start = Instant::now();

    io_binding.bind_input("input", TypedArray::F32(batch))?;
    io_binding.bind_input(
        "max_output_boxes_per_class",
        TypedArray::I64(max_output_boxes_per_class),
    )?;
    io_binding.bind_input("iou_threshold", TypedArray::F32(iou_threshold))?;
    io_binding.bind_input("score_threshold", TypedArray::F32(score_threshold))?;

    io_binding.bind_output(
        "boxes",
        MemoryInfo::new(
            onnxruntime::DeviceName::Cuda,
            0,
            AllocatorType::Device,
            MemType::Default,
        )?,
    )?;
    io_binding.bind_output(
        "confidences",
        MemoryInfo::new(
            onnxruntime::DeviceName::Cuda,
            0,
            AllocatorType::Device,
            MemType::Default,
        )?,
    )?;
    io_binding.bind_output(
        "selected_indices",
        MemoryInfo::new(
            onnxruntime::DeviceName::Cuda,
            0,
            AllocatorType::Device,
            MemType::Default,
        )?,
    )?;

    for i in 1..N {
        println!("run_with_iobinding: {:?}", i);
        session.run_with_iobinding(&io_binding)?;
    }
    let duration = start.elapsed();

    let outputs = io_binding.copy_outputs_to_cpu()?;
    if let (
        Some(TypedOrtOwnedTensor::I64(selected_indicies)),
        Some(TypedOrtOwnedTensor::F32(boxes)),
        Some(TypedOrtOwnedTensor::F32(confidences)),
    ) = (
        outputs.get("selected_indices"),
        outputs.get("boxes"),
        outputs.get("confidences"),
    ) {
        selected_indicies.outer_iter().for_each(|selected_index| {
            let batch_index = selected_index[0] as usize;
            let class_index = selected_index[1] as usize;
            let box_index = selected_index[2] as usize;

            println!(
                "{} {} {} {} {} {}",
                names.get(class_index).unwrap(),
                confidences[[batch_index, box_index, class_index]],
                boxes[[batch_index, box_index, 0]],
                boxes[[batch_index, box_index, 1]],
                boxes[[batch_index, box_index, 2]],
                boxes[[batch_index, box_index, 3]],
            )
        });
    }

    println!(
        "duration {:.3} ms {:.2} fps",
        duration.as_nanos() as f64 / N as f64 / 1000000_f64,
        1000f64 / (duration.as_nanos() as f64 / N as f64 / 1000000_f64) * BATCH_SIZE as f64,
    );

    Ok(())
}
