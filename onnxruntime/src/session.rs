//! Module containing session types

use crate::{
    char_p_to_string,
    environment::Environment,
    error::{
        assert_not_null_pointer, assert_null_pointer, status_to_result, NonMatchingDimensionsError,
        OrtApiError, OrtError, Result,
    },
    g_ort,
    io_binding::IoBinding,
    memory::MemoryInfo,
    tensor::{ort_owned_tensor::OrtOwnedTensorExtractor, OrtTensor},
    AllocatorType, DeviceName, GraphOptimizationLevel, MemType, TensorElementDataType,
    TypeToTensorElementDataType, TypedArray, TypedOrtOwnedTensor, TypedOrtTensor,
};
use ndarray::Array;
use onnxruntime_sys as sys;
use std::collections::HashMap;
use std::os::raw::c_char;
use std::os::unix::ffi::OsStrExt;
use std::{ffi::CString, fmt::Debug, path::Path};
use tracing::{error, trace};

/// Type used to create a session using the _builder pattern_
///
/// A `SessionBuilder` is created by calling the
/// [`Environment::new_session_builder()`](../env/struct.Environment.html#method.new_session_builder)
/// method on the environment.
///
/// Once created, use the different methods to configure the session.
///
/// Once configured, use the [`SessionBuilder::with_model_from_file()`](../session/struct.SessionBuilder.html#method.with_model_from_file)
/// method to "commit" the builder configuration into a [`Session`](../session/struct.Session.html).
///
/// # Example
///
/// ```no_run
/// # use std::error::Error;
/// # use onnxruntime::{environment::Environment, LoggingLevel, GraphOptimizationLevel};
/// # fn main() -> Result<(), Box<dyn Error>> {
/// let environment = Environment::builder()
///     .with_name("test")
///     .with_log_level(LoggingLevel::Verbose)
///     .build()?;
/// let mut session = environment
///     .new_session_builder()?
///     .with_optimization_level(GraphOptimizationLevel::Basic)?
///     .with_number_threads(1)?
///     .with_model_from_file("squeezenet.onnx")?;
/// # Ok(())
/// # }
/// ```
#[derive(Debug)]
pub struct SessionBuilder<'a> {
    env: &'a Environment,
    session_options_ptr: *mut sys::OrtSessionOptions,

    allocator: AllocatorType,
    memory_type: MemType,
}

impl<'a> Drop for SessionBuilder<'a> {
    #[tracing::instrument]
    fn drop(&mut self) {
        if self.session_options_ptr.is_null() {
            error!("SessionBuilder pointer is null, not dropping");
        } else {
            trace!("Dropping SessionBuilder.");
            unsafe { g_ort().ReleaseSessionOptions.unwrap()(self.session_options_ptr) };
        }

        self.session_options_ptr = std::ptr::null_mut();
    }
}

impl<'a> SessionBuilder<'a> {
    pub(crate) fn new(env: &'a Environment) -> Result<SessionBuilder<'a>> {
        let mut session_options_ptr: *mut sys::OrtSessionOptions = std::ptr::null_mut();
        let status = unsafe { g_ort().CreateSessionOptions.unwrap()(&mut session_options_ptr) };

        status_to_result(status).map_err(OrtError::SessionOptions)?;
        assert_null_pointer(status, "SessionStatus")?;
        assert_not_null_pointer(session_options_ptr, "SessionOptions")?;

        Ok(SessionBuilder {
            env,
            session_options_ptr,
            allocator: AllocatorType::Arena,
            memory_type: MemType::Default,
        })
    }

    /// Configure the session to use a number of threads
    pub fn with_number_threads(self, num_threads: i16) -> Result<SessionBuilder<'a>> {
        // FIXME: Pre-built binaries use OpenMP, set env variable instead

        // We use a u16 in the builder to cover the 16-bits positive values of a i32.
        let num_threads = num_threads as i32;
        let status =
            unsafe { g_ort().SetIntraOpNumThreads.unwrap()(self.session_options_ptr, num_threads) };
        status_to_result(status).map_err(OrtError::SessionOptions)?;
        assert_null_pointer(status, "SessionStatus")?;
        Ok(self)
    }

    /// Set the session's optimization level
    pub fn with_optimization_level(
        self,
        opt_level: GraphOptimizationLevel,
    ) -> Result<SessionBuilder<'a>> {
        // Sets graph optimization level
        unsafe {
            g_ort().SetSessionGraphOptimizationLevel.unwrap()(
                self.session_options_ptr,
                opt_level.into(),
            )
        };
        Ok(self)
    }

    /// Set the session to use cpu
    #[cfg(feature = "cuda")]
    pub fn with_cpu(self, use_arena: i32) -> Result<SessionBuilder<'a>> {
        unsafe {
            sys::OrtSessionOptionsAppendExecutionProvider_CPU(self.session_options_ptr, use_arena);
        }
        Ok(self)
    }

    /// Set the session to use cuda
    #[cfg(feature = "cuda")]
    pub fn with_cuda(self, options: CUDAProviderOptions) -> Result<SessionBuilder<'a>> {
        unsafe {
            let mut cuda_options_ptr: *mut sys::OrtCUDAProviderOptionsV2 = std::ptr::null_mut();
            let status = g_ort().CreateCUDAProviderOptions.unwrap()(&mut cuda_options_ptr);
            status_to_result(status).map_err(OrtError::Allocator)?;
            assert_not_null_pointer(cuda_options_ptr, "OrtCUDAProviderOptionsV2")?;

            let (keys, values) = options.get_keys_values();

            let status = g_ort().UpdateCUDAProviderOptions.unwrap()(
                cuda_options_ptr,
                keys.iter().map(|k| k.as_ptr()).collect::<Vec<_>>().as_ptr(),
                values
                    .iter()
                    .map(|v| v.as_ptr())
                    .collect::<Vec<_>>()
                    .as_ptr(),
                keys.len(),
            );
            status_to_result(status).map_err(OrtError::Allocator)?;

            let status = g_ort()
                .SessionOptionsAppendExecutionProvider_CUDA_V2
                .unwrap()(self.session_options_ptr, cuda_options_ptr);

            status_to_result(status).map_err(OrtError::Allocator)?;

            g_ort().ReleaseCUDAProviderOptions.unwrap()(cuda_options_ptr);
        }
        Ok(self)
    }

    /// Set the session to use cuda
    #[cfg(feature = "cuda")]
    pub fn with_tensorrt(self, options: TensorrtProviderOptions) -> Result<SessionBuilder<'a>> {
        unsafe {
            let mut trt_options_ptr: *mut sys::OrtTensorRTProviderOptionsV2 = std::ptr::null_mut();
            let status = g_ort().CreateTensorRTProviderOptions.unwrap()(&mut trt_options_ptr);
            status_to_result(status).map_err(OrtError::Allocator)?;
            assert_not_null_pointer(trt_options_ptr, "OrtTensorRTProviderOptionsV2")?;

            let (keys, values) = options.get_keys_values();

            let status = g_ort().UpdateTensorRTProviderOptions.unwrap()(
                trt_options_ptr,
                keys.iter().map(|k| k.as_ptr()).collect::<Vec<_>>().as_ptr(),
                values
                    .iter()
                    .map(|v| v.as_ptr())
                    .collect::<Vec<_>>()
                    .as_ptr(),
                keys.len(),
            );
            status_to_result(status).map_err(OrtError::Allocator)?;

            let status = g_ort()
                .SessionOptionsAppendExecutionProvider_TensorRT_V2
                .unwrap()(self.session_options_ptr, trt_options_ptr);

            status_to_result(status).map_err(OrtError::Allocator)?;

            g_ort().ReleaseTensorRTProviderOptions.unwrap()(trt_options_ptr);
        }
        Ok(self)
    }

    /// Set the session's allocator
    ///
    /// Defaults to [`AllocatorType::Arena`](../enum.AllocatorType.html#variant.Arena)
    pub fn with_allocator(mut self, allocator: AllocatorType) -> Result<SessionBuilder<'a>> {
        self.allocator = allocator;
        Ok(self)
    }

    /// Set the session's memory type
    ///
    /// Defaults to [`MemType::Default`](../enum.MemType.html#variant.Default)
    pub fn with_memory_type(mut self, memory_type: MemType) -> Result<SessionBuilder<'a>> {
        self.memory_type = memory_type;
        Ok(self)
    }

    // TODO: Add all functions changing the options.
    //       See all OrtApi methods taking a `options: *mut OrtSessionOptions`.

    /// Load an ONNX graph from a file and commit the session
    #[tracing::instrument]
    pub fn with_model_from_file<P>(self, model_filepath_ref: P) -> Result<Session<'a>>
    where
        P: AsRef<Path> + Debug + 'a,
    {
        let model_filepath = model_filepath_ref.as_ref();
        let mut session_ptr: *mut sys::OrtSession = std::ptr::null_mut();

        if !model_filepath.exists() {
            return Err(OrtError::FileDoesNotExists {
                filename: model_filepath.to_path_buf(),
            });
        }

        // Build an OsString than a vector of bytes to pass to C
        let model_path = std::ffi::OsString::from(model_filepath);
        let model_path: Vec<std::os::raw::c_char> = model_path
            .as_bytes()
            .iter()
            .chain(std::iter::once(&b'\0')) // Make sure we have a null terminated string
            .map(|b| *b as std::os::raw::c_char)
            .collect();

        let env_ptr: *const sys::OrtEnv = self.env.env_ptr();

        let status = unsafe {
            g_ort().CreateSession.unwrap()(
                env_ptr,
                model_path.as_ptr(),
                self.session_options_ptr,
                &mut session_ptr,
            )
        };
        status_to_result(status).map_err(OrtError::Session)?;
        assert_null_pointer(status, "SessionStatus")?;
        assert_not_null_pointer(session_ptr, "Session")?;

        let mut allocator_ptr: *mut sys::OrtAllocator = std::ptr::null_mut();
        let status = unsafe { g_ort().GetAllocatorWithDefaultOptions.unwrap()(&mut allocator_ptr) };
        status_to_result(status).map_err(OrtError::Allocator)?;
        assert_null_pointer(status, "SessionStatus")?;
        assert_not_null_pointer(allocator_ptr, "Allocator")?;

        let memory_info =
            MemoryInfo::new(DeviceName::Cpu, 0, AllocatorType::Arena, MemType::Default)?;

        // Extract input and output properties
        let num_input_nodes = dangerous::extract_inputs_count(session_ptr)?;
        let num_output_nodes = dangerous::extract_outputs_count(session_ptr)?;
        let inputs = (0..num_input_nodes)
            .map(|i| {
                let input = dangerous::extract_input(session_ptr, allocator_ptr, i)?;
                Ok((input.name.clone(), input))
            })
            .collect::<Result<HashMap<String, Input>>>()?;
        let outputs = (0..num_output_nodes)
            .map(|i| {
                let output = dangerous::extract_output(session_ptr, allocator_ptr, i)?;
                Ok((output.name.clone(), output))
            })
            .collect::<Result<HashMap<String, Output>>>()?;

        trace!("Creating Session.");
        Ok(Session {
            env: self.env,
            ptr: session_ptr,
            allocator_ptr,
            memory_info,
            inputs,
            outputs,
        })
    }

    /// Load an ONNX graph from memory and commit the session
    pub fn with_model_from_memory<B>(self, model_bytes: B) -> Result<Session<'a>>
    where
        B: AsRef<[u8]>,
    {
        self.with_model_from_memory_monomorphized(model_bytes.as_ref())
    }

    #[tracing::instrument]
    fn with_model_from_memory_monomorphized(self, model_bytes: &[u8]) -> Result<Session<'a>> {
        let mut session_ptr: *mut sys::OrtSession = std::ptr::null_mut();

        let env_ptr: *const sys::OrtEnv = self.env.env_ptr();

        let status = unsafe {
            let model_data = model_bytes.as_ptr() as *const std::ffi::c_void;
            let model_data_length = model_bytes.len();
            g_ort().CreateSessionFromArray.unwrap()(
                env_ptr,
                model_data,
                model_data_length,
                self.session_options_ptr,
                &mut session_ptr,
            )
        };
        status_to_result(status).map_err(OrtError::Session)?;
        assert_null_pointer(status, "SessionStatus")?;
        assert_not_null_pointer(session_ptr, "Session")?;

        let mut allocator_ptr: *mut sys::OrtAllocator = std::ptr::null_mut();
        let status = unsafe { g_ort().GetAllocatorWithDefaultOptions.unwrap()(&mut allocator_ptr) };
        status_to_result(status).map_err(OrtError::Allocator)?;
        assert_null_pointer(status, "SessionStatus")?;
        assert_not_null_pointer(allocator_ptr, "Allocator")?;

        let memory_info =
            MemoryInfo::new(DeviceName::Cpu, 0, AllocatorType::Arena, MemType::Default)?;

        // Extract input and output properties
        let num_input_nodes = dangerous::extract_inputs_count(session_ptr)?;
        let num_output_nodes = dangerous::extract_outputs_count(session_ptr)?;
        let inputs = (0..num_input_nodes)
            .map(|i| {
                let input = dangerous::extract_input(session_ptr, allocator_ptr, i)?;
                Ok((input.name.clone(), input))
            })
            .collect::<Result<HashMap<String, Input>>>()?;
        let outputs = (0..num_output_nodes)
            .map(|i| {
                let output = dangerous::extract_output(session_ptr, allocator_ptr, i)?;
                Ok((output.name.clone(), output))
            })
            .collect::<Result<HashMap<String, Output>>>()?;

        trace!("Creating Session.");
        Ok(Session {
            env: self.env,
            ptr: session_ptr,
            allocator_ptr,
            memory_info,
            inputs,
            outputs,
        })
    }
}

#[cfg(feature = "cuda")]
#[derive(Debug, Clone)]
/// Configuration options for the CUDA Execution Provider.
pub struct CUDAProviderOptions {
    /// The device ID.
    device_id: usize,
    /// The size limit of the device memory arena in bytes.
    gpu_mem_limit: usize,
    /// The strategy for extending the device memory arena.
    arena_extend_strategy: ArenaExtendStrategy,
    /// The type of search done for cuDNN convolution algorithms.
    cudnn_conv_algo_search: CuDNNConvAlgoSearch,
    /// Whether to do copies in the default stream or use separate streams.
    do_copy_in_default_stream: bool,
    /// Allow ORT to allocate the maximum possible workspace as determined by CuDNN.
    cudnn_conv_use_max_workspace: bool,
    /// Convolution Input Padding in the CUDA EP.
    cudnn_conv1d_pad_to_nc1d: bool,
    /// Enable the usage of CUDA Graphs.
    enable_cuda_graph: bool,
}

#[cfg(feature = "cuda")]
impl Default for CUDAProviderOptions {
    fn default() -> Self {
        Self {
            device_id: 0,
            gpu_mem_limit: 18446744073709551615,
            arena_extend_strategy: ArenaExtendStrategy::NextPowerOfTwo,
            cudnn_conv_algo_search: CuDNNConvAlgoSearch::Exhaustive,
            do_copy_in_default_stream: true,
            cudnn_conv_use_max_workspace: false,
            cudnn_conv1d_pad_to_nc1d: false,
            enable_cuda_graph: false,
        }
    }
}

#[cfg(feature = "cuda")]
impl CUDAProviderOptions {
    fn get_keys_values(&self) -> (Vec<CString>, Vec<CString>) {
        let keys = vec![
            "device_id",
            "gpu_mem_limit",
            "arena_extend_strategy",
            "cudnn_conv_algo_search",
            "do_copy_in_default_stream",
            "cudnn_conv_use_max_workspace",
            "cudnn_conv1d_pad_to_nc1d",
            "enable_cuda_graph",
        ]
        .into_iter()
        .map(|k| CString::new(k).unwrap())
        .collect::<Vec<_>>();

        let values = vec![
            self.device_id.to_string(),
            self.gpu_mem_limit.to_string(),
            match self.arena_extend_strategy {
                ArenaExtendStrategy::NextPowerOfTwo => "kNextPowerOfTwo",
                ArenaExtendStrategy::SameAsRequested => "kSameAsRequested",
            }
            .to_string(),
            match self.cudnn_conv_algo_search {
                CuDNNConvAlgoSearch::Exhaustive => "EXHAUSTIVE",
                CuDNNConvAlgoSearch::Heuristic => "HEURISTIC",
                CuDNNConvAlgoSearch::Default => "DEFAULT",
            }
            .to_string(),
            (if self.do_copy_in_default_stream { 1 } else { 0 }).to_string(),
            (if self.cudnn_conv_use_max_workspace {
                1
            } else {
                0
            })
            .to_string(),
            (if self.cudnn_conv1d_pad_to_nc1d { 1 } else { 0 }).to_string(),
            (if self.enable_cuda_graph { 1 } else { 0 }).to_string(),
        ]
        .into_iter()
        .map(|k| CString::new(k).unwrap())
        .collect::<Vec<_>>();

        (keys, values)
    }

    /// Set device_id
    pub fn with_device_id(mut self, device_id: usize) -> Self {
        self.device_id = device_id;
        self
    }

    /// Set gpu_mem_limit
    pub fn with_gpu_mem_limit(mut self, gpu_mem_limit: usize) -> Self {
        self.gpu_mem_limit = gpu_mem_limit;
        self
    }

    /// Set arena_extend_strategy
    pub fn with_arena_extend_strategy(
        mut self,
        arena_extend_strategy: ArenaExtendStrategy,
    ) -> Self {
        self.arena_extend_strategy = arena_extend_strategy;
        self
    }

    /// Set cudnn_conv_algo_search
    pub fn with_cudnn_conv_algo_search(
        mut self,
        cudnn_conv_algo_search: CuDNNConvAlgoSearch,
    ) -> Self {
        self.cudnn_conv_algo_search = cudnn_conv_algo_search;
        self
    }

    /// Set do_copy_in_default_stream
    pub fn with_do_copy_in_default_stream(mut self, do_copy_in_default_stream: bool) -> Self {
        self.do_copy_in_default_stream = do_copy_in_default_stream;
        self
    }

    /// Set cudnn_conv_use_max_workspace
    pub fn with_cudnn_conv_use_max_workspace(mut self, cudnn_conv_use_max_workspace: bool) -> Self {
        self.cudnn_conv_use_max_workspace = cudnn_conv_use_max_workspace;
        self
    }

    /// Set cudnn_conv1d_pad_to_nc1d
    pub fn with_cudnn_conv1d_pad_to_nc1d(mut self, cudnn_conv1d_pad_to_nc1d: bool) -> Self {
        self.cudnn_conv1d_pad_to_nc1d = cudnn_conv1d_pad_to_nc1d;
        self
    }

    /// Set enable_cuda_graph
    pub fn with_enable_cuda_graph(mut self, enable_cuda_graph: bool) -> Self {
        self.enable_cuda_graph = enable_cuda_graph;
        self
    }
}

#[derive(Debug, Clone)]
/// The strategy for extending the device memory arena.
pub enum ArenaExtendStrategy {
    /// subsequent extensions extend by larger amounts (multiplied by powers of two)
    NextPowerOfTwo = 0,
    /// extend by the requested amount
    SameAsRequested = 1,
}

#[derive(Debug, Clone)]
/// The type of search done for cuDNN convolution algorithms.
pub enum CuDNNConvAlgoSearch {
    /// expensive exhaustive benchmarking using cudnnFindConvolutionForwardAlgorithmEx
    Exhaustive,
    /// lightweight heuristic based search using cudnnGetConvolutionForwardAlgorithm_v7
    Heuristic,
    /// default algorithm using CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_PRECOMP_GEMM
    Default,
}

#[cfg(feature = "cuda")]
#[derive(Debug, Clone)]
/// Configuration options for the TensorRT Execution Provider.
pub struct TensorrtProviderOptions {
    /// The device ID.
    device_id: usize,
    /// The maximum workspace size for TensorRT engine.
    trt_max_workspace_size: usize,
    /// The maximum number of iterations allowed in model partitioning for TensorRT.
    trt_max_partition_iterations: usize,
    /// The minimum node size in a subgraph after partitioning.
    trt_min_subgraph_size: usize,
    /// Enable FP16 mode in TensorRT.
    trt_fp16_enable: bool,
    /// Enable FP16 mode in TensorRT.
    trt_int8_enable: bool,
    /// Specify INT8 calibration table file for non-QDQ models in INT8 mode.
    trt_int8_calibration_table_name: Option<String>,
    ///  Select what calibration table is used for non-QDQ models in INT8 mode.
    /// If 1, native TensorRT generated calibration table is used; if 0, ONNXRUNTIME tool generated calibration table is used.
    trt_int8_use_native_calibration_table: bool,
    /// Enable Deep Learning Accelerator (DLA).
    trt_dla_enable: bool,
    /// Specify DLA core to execute on.
    trt_dla_core: usize,
    /// Enable TensorRT engine caching.
    trt_engine_cache_enable: bool,
    /// Specify path for TensorRT engine and profile files.
    trt_engine_cache_path: Option<String>,
    /// Dumps the subgraphs that are transformed into TRT engines in onnx format to the filesystem.
    trt_dump_subgraphs: bool,
    /// Sequentially build TensorRT engines across provider instances in multi-GPU environment.
    trt_force_sequential_engine_build: bool,
}

#[cfg(feature = "cuda")]
impl Default for TensorrtProviderOptions {
    fn default() -> Self {
        Self {
            device_id: 0,
            trt_max_workspace_size: 1073741824,
            trt_max_partition_iterations: 1000,
            trt_min_subgraph_size: 1,
            trt_fp16_enable: false,
            trt_int8_enable: false,
            trt_int8_calibration_table_name: None,
            trt_int8_use_native_calibration_table: false,
            trt_dla_enable: false,
            trt_dla_core: 0,
            trt_engine_cache_enable: false,
            trt_engine_cache_path: None,
            trt_dump_subgraphs: false,
            trt_force_sequential_engine_build: false,
        }
    }
}

#[cfg(feature = "cuda")]
impl TensorrtProviderOptions {
    fn get_keys_values(&self) -> (Vec<CString>, Vec<CString>) {
        let mut keys = vec![
            "device_id",
            "trt_max_workspace_size",
            "trt_max_partition_iterations",
            "trt_min_subgraph_size",
            "trt_fp16_enable",
            "trt_int8_enable",
            "trt_int8_use_native_calibration_table",
            "trt_dla_enable",
            "trt_dla_core",
            "trt_engine_cache_enable",
            "trt_dump_subgraphs",
            "trt_force_sequential_engine_build",
        ]
        .into_iter()
        .map(|k| CString::new(k).unwrap())
        .collect::<Vec<_>>();

        let mut values = vec![
            self.device_id.to_string(),
            self.trt_max_workspace_size.to_string(),
            self.trt_max_partition_iterations.to_string(),
            self.trt_min_subgraph_size.to_string(),
            (if self.trt_fp16_enable { 1 } else { 0 }).to_string(),
            (if self.trt_int8_enable { 1 } else { 0 }).to_string(),
            (if self.trt_int8_use_native_calibration_table {
                1
            } else {
                0
            })
            .to_string(),
            (if self.trt_dla_enable { 1 } else { 0 }).to_string(),
            self.trt_dla_core.to_string(),
            (if self.trt_engine_cache_enable { 1 } else { 0 }).to_string(),
            (if self.trt_dump_subgraphs { 1 } else { 0 }).to_string(),
            (if self.trt_force_sequential_engine_build {
                1
            } else {
                0
            })
            .to_string(),
        ]
        .into_iter()
        .map(|k| CString::new(k).unwrap())
        .collect::<Vec<_>>();

        if let Some(trt_engine_cache_path) = &self.trt_engine_cache_path {
            keys.push(CString::new("trt_engine_cache_path").unwrap());
            values.push(CString::new(trt_engine_cache_path.clone()).unwrap());
        };

        if let Some(trt_int8_calibration_table_name) = &self.trt_int8_calibration_table_name {
            keys.push(CString::new("trt_int8_calibration_table_name").unwrap());
            values.push(CString::new(trt_int8_calibration_table_name.clone()).unwrap());
        };

        (keys, values)
    }

    /// Set device_id
    pub fn with_device_id(mut self, device_id: usize) -> Self {
        self.device_id = device_id;
        self
    }

    /// Set trt_max_workspace_size
    pub fn with_trt_max_workspace_size(mut self, trt_max_workspace_size: usize) -> Self {
        self.trt_max_workspace_size = trt_max_workspace_size;
        self
    }

    /// Set trt_max_partition_iterations
    pub fn with_trt_max_partition_iterations(
        mut self,
        trt_max_partition_iterations: usize,
    ) -> Self {
        self.trt_max_partition_iterations = trt_max_partition_iterations;
        self
    }

    /// Set trt_min_subgraph_size
    pub fn with_trt_min_subgraph_size(mut self, trt_min_subgraph_size: usize) -> Self {
        self.trt_min_subgraph_size = trt_min_subgraph_size;
        self
    }

    /// Set trt_fp16_enable
    pub fn with_trt_fp16_enable(mut self, trt_fp16_enable: bool) -> Self {
        self.trt_fp16_enable = trt_fp16_enable;
        self
    }

    /// Set trt_int8_enable
    pub fn with_trt_int8_enable(mut self, trt_int8_enable: bool) -> Self {
        self.trt_int8_enable = trt_int8_enable;
        self
    }

    /// Set trt_int8_calibration_table_name
    pub fn with_trt_int8_calibration_table_name(
        mut self,
        trt_int8_calibration_table_name: Option<&str>,
    ) -> Self {
        self.trt_int8_calibration_table_name =
            trt_int8_calibration_table_name.map(|v| v.to_string());
        self
    }

    /// Set trt_int8_use_native_calibration_table
    pub fn with_trt_int8_use_native_calibration_table(
        mut self,
        trt_int8_use_native_calibration_table: bool,
    ) -> Self {
        self.trt_int8_use_native_calibration_table = trt_int8_use_native_calibration_table;
        self
    }

    /// Set trt_dla_enable
    pub fn with_trt_dla_enable(mut self, trt_dla_enable: bool) -> Self {
        self.trt_dla_enable = trt_dla_enable;
        self
    }

    /// Set trt_dla_core
    pub fn with_trt_dla_core(mut self, trt_dla_core: usize) -> Self {
        self.trt_dla_core = trt_dla_core;
        self
    }

    /// Set trt_engine_cache_enable
    pub fn with_trt_engine_cache_enable(mut self, trt_engine_cache_enable: bool) -> Self {
        self.trt_engine_cache_enable = trt_engine_cache_enable;
        self
    }

    /// Set trt_engine_cache_path
    pub fn with_trt_engine_cache_path(mut self, trt_engine_cache_path: Option<&str>) -> Self {
        self.trt_engine_cache_path = trt_engine_cache_path.map(|v| v.to_string());
        self
    }

    /// Set trt_dump_subgraphs
    pub fn with_trt_dump_subgraphs(mut self, trt_dump_subgraphs: bool) -> Self {
        self.trt_dump_subgraphs = trt_dump_subgraphs;
        self
    }

    /// Set trt_force_sequential_engine_build
    pub fn with_trt_force_sequential_engine_build(
        mut self,
        trt_force_sequential_engine_build: bool,
    ) -> Self {
        self.trt_force_sequential_engine_build = trt_force_sequential_engine_build;
        self
    }
}

/// Type storing the session information, built from an [`Environment`](environment/struct.Environment.html)
#[derive(Debug)]
#[allow(dead_code)]
pub struct Session<'a> {
    env: &'a Environment,
    pub(crate) ptr: *mut sys::OrtSession,
    pub(crate) allocator_ptr: *mut sys::OrtAllocator,
    pub(crate) memory_info: MemoryInfo,
    /// Information about the ONNX's inputs as stored in loaded file
    pub inputs: HashMap<String, Input>,
    /// Information about the ONNX's outputs as stored in loaded file
    pub outputs: HashMap<String, Output>,
}

unsafe impl<'a> Send for Session<'a> {}
unsafe impl<'a> Sync for Session<'a> {}

/// Information about an ONNX's input as stored in loaded file
#[derive(Clone, Debug)]
pub struct Input {
    /// Name of the input layer
    pub name: String,
    /// Type of the input layer's elements
    pub element_type: TensorElementDataType,
    /// Shape of the input layer
    ///
    /// C API uses a i64 for the dimensions. We use an unsigned of the same range of the positive values.
    pub dimensions: Vec<Option<u32>>,
}

/// Information about an ONNX's output as stored in loaded file
#[derive(Clone, Debug)]
pub struct Output {
    /// Name of the output layer
    pub name: String,
    /// Type of the output layer's elements
    pub element_type: TensorElementDataType,
    /// Shape of the output layer
    ///
    /// C API uses a i64 for the dimensions. We use an unsigned of the same range of the positive values.
    pub dimensions: Vec<Option<u32>>,
}

impl Input {
    /// Return an iterator over the shape elements of the input layer
    ///
    /// Note: The member [`Input::dimensions`](struct.Input.html#structfield.dimensions)
    /// stores `u32` (since ONNX uses `i64` but which cannot be negative) so the
    /// iterator converts to `usize`.
    pub fn dimensions(&self) -> impl Iterator<Item = Option<usize>> + '_ {
        self.dimensions.iter().map(|d| d.map(|d2| d2 as usize))
    }
}

impl Output {
    /// Return an iterator over the shape elements of the output layer
    ///
    /// Note: The member [`Output::dimensions`](struct.Output.html#structfield.dimensions)
    /// stores `u32` (since ONNX uses `i64` but which cannot be negative) so the
    /// iterator converts to `usize`.
    pub fn dimensions(&self) -> impl Iterator<Item = Option<usize>> + '_ {
        self.dimensions.iter().map(|d| d.map(|d2| d2 as usize))
    }
}

impl<'a> Drop for Session<'a> {
    #[tracing::instrument]
    fn drop(&mut self) {
        if self.ptr.is_null() {
            error!("Session pointer is null, not dropping");
        } else {
            trace!("Dropping Session.");
            unsafe { g_ort().ReleaseSession.unwrap()(self.ptr) };
        }

        self.ptr = std::ptr::null_mut();
        self.allocator_ptr = std::ptr::null_mut();
    }
}

impl<'a> Session<'a> {
    /// Run the input data through the ONNX graph, performing inference.
    ///
    /// Note that ONNX models can have multiple inputs; a `Vec<>` is thus
    /// used for the input data here.
    #[tracing::instrument]
    pub fn run<'s, 't, 'm, D>(
        &'s self,
        inputs: HashMap<String, TypedArray<D>>,
    ) -> Result<HashMap<String, TypedOrtOwnedTensor<ndarray::Dim<ndarray::IxDynImpl>>>>
    where
        D: ndarray::Dimension,
        'm: 't, // 'm outlives 't (memory info outlives tensor)
        's: 'm, // 's outlives 'm (session outlives memory info)
    {
        // self.validate_untyped_input_shapes(&inputs)?;

        // Build arguments to Run()
        let input_names_ptr: Vec<*const c_char> = inputs
            .iter()
            .map(|(name, _)| name.clone())
            .map(|n| CString::new(n).unwrap())
            .map(|n| n.into_raw() as *const c_char)
            .collect();

        let output_names_cstring: Vec<CString> = self
            .outputs
            .iter()
            .map(|(name, _)| name.clone())
            .map(|n| CString::new(n).unwrap())
            .collect();
        let output_names_ptr: Vec<*const c_char> = output_names_cstring
            .iter()
            .map(|n| n.as_ptr() as *const c_char)
            .collect();

        // The C API expects pointers for the arrays (pointers to C-arrays)
        let input_ort_tensors: Vec<TypedOrtTensor<D>> = inputs
            .into_iter()
            .map(|(_, input_array)| match input_array {
                TypedArray::F32(input_array) => {
                    OrtTensor::from_array(&self.memory_info, self.allocator_ptr, input_array)
                        .map(|t| TypedOrtTensor::F32(t))
                }
                TypedArray::U8(input_array) => {
                    OrtTensor::from_array(&self.memory_info, self.allocator_ptr, input_array)
                        .map(|t| TypedOrtTensor::U8(t))
                }
                TypedArray::I8(input_array) => {
                    OrtTensor::from_array(&self.memory_info, self.allocator_ptr, input_array)
                        .map(|t| TypedOrtTensor::I8(t))
                }
                TypedArray::U16(input_array) => {
                    OrtTensor::from_array(&self.memory_info, self.allocator_ptr, input_array)
                        .map(|t| TypedOrtTensor::U16(t))
                }
                TypedArray::I16(input_array) => {
                    OrtTensor::from_array(&self.memory_info, self.allocator_ptr, input_array)
                        .map(|t| TypedOrtTensor::I16(t))
                }
                TypedArray::I32(input_array) => {
                    OrtTensor::from_array(&self.memory_info, self.allocator_ptr, input_array)
                        .map(|t| TypedOrtTensor::I32(t))
                }
                TypedArray::I64(input_array) => {
                    OrtTensor::from_array(&self.memory_info, self.allocator_ptr, input_array)
                        .map(|t| TypedOrtTensor::I64(t))
                }
                TypedArray::F64(input_array) => {
                    OrtTensor::from_array(&self.memory_info, self.allocator_ptr, input_array)
                        .map(|t| TypedOrtTensor::F64(t))
                }
                TypedArray::U32(input_array) => {
                    OrtTensor::from_array(&self.memory_info, self.allocator_ptr, input_array)
                        .map(|t| TypedOrtTensor::U32(t))
                }
                TypedArray::U64(input_array) => {
                    OrtTensor::from_array(&self.memory_info, self.allocator_ptr, input_array)
                        .map(|t| TypedOrtTensor::U64(t))
                }
            })
            .collect::<Result<Vec<TypedOrtTensor<D>>>>()?;
        let input_ort_values: Vec<*const sys::OrtValue> = input_ort_tensors
            .iter()
            .map(|input_array_ort| match input_array_ort {
                TypedOrtTensor::F32(input_array_ort) => input_array_ort.ptr as *const sys::OrtValue,
                TypedOrtTensor::U8(input_array_ort) => input_array_ort.ptr as *const sys::OrtValue,
                TypedOrtTensor::I8(input_array_ort) => input_array_ort.ptr as *const sys::OrtValue,
                TypedOrtTensor::U16(input_array_ort) => input_array_ort.ptr as *const sys::OrtValue,
                TypedOrtTensor::I16(input_array_ort) => input_array_ort.ptr as *const sys::OrtValue,
                TypedOrtTensor::I32(input_array_ort) => input_array_ort.ptr as *const sys::OrtValue,
                TypedOrtTensor::I64(input_array_ort) => input_array_ort.ptr as *const sys::OrtValue,
                TypedOrtTensor::F64(input_array_ort) => input_array_ort.ptr as *const sys::OrtValue,
                TypedOrtTensor::U32(input_array_ort) => input_array_ort.ptr as *const sys::OrtValue,
                TypedOrtTensor::U64(input_array_ort) => input_array_ort.ptr as *const sys::OrtValue,
            })
            .collect();

        let mut output_tensor_extractors_ptrs: Vec<*mut sys::OrtValue> =
            vec![std::ptr::null_mut(); output_names_cstring.len()];

        let run_options_ptr: *const sys::OrtRunOptions = std::ptr::null();

        let status = unsafe {
            g_ort().Run.unwrap()(
                self.ptr,
                run_options_ptr,
                input_names_ptr.as_ptr(),
                input_ort_values.as_ptr(),
                input_ort_values.len(),
                output_names_ptr.as_ptr(),
                output_names_ptr.len(),
                output_tensor_extractors_ptrs.as_mut_ptr(),
            )
        };
        status_to_result(status).map_err(OrtError::Run)?;

        let memory_info_ref = &self.memory_info;
        let outputs = output_tensor_extractors_ptrs
            .into_iter()
            .zip(&self.outputs)
            .filter_map(|(ptr, (output_name, output))| {
                let mut tensor_info_ptr: *mut sys::OrtTensorTypeAndShapeInfo = std::ptr::null_mut();
                let status = unsafe {
                    g_ort().GetTensorTypeAndShape.unwrap()(ptr, &mut tensor_info_ptr as _)
                };
                status_to_result(status)
                    .map_err(OrtError::GetTensorTypeAndShape)
                    .unwrap();

                let dims = get_tensor_dimensions(tensor_info_ptr).unwrap();
                unsafe { g_ort().ReleaseTensorTypeAndShapeInfo.unwrap()(tensor_info_ptr) };
                let dims: Vec<_> = dims.iter().map(|&n| n as usize).collect();

                (!dims.contains(&0)).then(|| {
                    let mut output_tensor_extractor =
                        OrtOwnedTensorExtractor::new(memory_info_ref, ndarray::IxDyn(&dims));
                    output_tensor_extractor.ptr = ptr;
                    let ort_owned_tensor = output_tensor_extractor.extract(output)?;

                    Ok((output_name.clone(), ort_owned_tensor))
                })
            })
            .collect::<Result<HashMap<String, TypedOrtOwnedTensor<_>>>>();

        // Reconvert to CString so drop impl is called and memory is freed
        let cstrings: Result<Vec<CString>> = input_names_ptr
            .into_iter()
            .map(|p| {
                assert_not_null_pointer(p, "i8 for CString")?;
                unsafe { Ok(CString::from_raw(p as *mut c_char)) }
            })
            .collect();
        cstrings?;

        outputs
    }

    /// Run the input data through the ONNX graph, performing inference.
    pub fn run_with_iobinding<D>(&self, io_binding: &IoBinding<D>) -> Result<()>
    where
        D: ndarray::Dimension,
    {
        let run_options_ptr: *const sys::OrtRunOptions = std::ptr::null();
        let status =
            unsafe { g_ort().RunWithBinding.unwrap()(self.ptr, run_options_ptr, io_binding.ptr) };
        status_to_result(status).map_err(OrtError::Run)?;
        Ok(())
    }

    // pub fn tensor_from_array<'a, 'b, T, D>(&'a self, array: Array<T, D>) -> Tensor<'b, T, D>
    // where
    //     'a: 'b, // 'a outlives 'b
    // {
    //     Tensor::from_array(self, array)
    // }

    fn validate_input_shape<TIn, D, DEB>(
        &self,
        input_array: &Array<TIn, D>,
        input: &Input,
        input_arrays_dimensions: &[Vec<usize>],
        input_arrays: &[DEB],
    ) -> Result<()>
    where
        TIn: TypeToTensorElementDataType + Debug + Clone,
        D: ndarray::Dimension,
        DEB: Debug,
    {
        // Verify length
        if input_array.shape().len() != input.dimensions.len() {
            error!(
                "Different input lengths: {:?} vs {:?}",
                self.inputs, input_arrays
            );
            return Err(OrtError::NonMatchingDimensions(
                NonMatchingDimensionsError::InputsLength {
                    inference_input: input_arrays_dimensions.to_vec(),
                    model_input: self
                        .inputs
                        .iter()
                        .map(|(_, input)| input.dimensions.clone())
                        .collect(),
                },
            ));
        }

        // Verify shape
        let inputs_different_shape =
            input_array
                .shape()
                .iter()
                .zip(input.dimensions.iter())
                .any(|(l2, r2)| match r2 {
                    Some(r3) => *r3 as usize != *l2,
                    None => false, // None means dynamic size; in that case shape always match
                });
        if inputs_different_shape {
            error!(
                "Different input lengths: {:?} vs {:?}",
                self.inputs, input_arrays
            );
            return Err(OrtError::NonMatchingDimensions(
                NonMatchingDimensionsError::InputsLength {
                    inference_input: input_arrays_dimensions.to_vec(),
                    model_input: self
                        .inputs
                        .iter()
                        .map(|(_, input)| input.dimensions.clone())
                        .collect(),
                },
            ));
        }

        Ok(())
    }

    #[allow(dead_code)]
    fn validate_untyped_input_shapes<D: ndarray::Dimension>(
        &mut self,
        input_arrays: &[TypedArray<D>],
    ) -> Result<()> {
        // ******************************************************************
        // FIXME: Properly handle errors here
        // Make sure all dimensions match (except dynamic ones)

        // Verify length of inputs
        if input_arrays.len() != self.inputs.len() {
            error!(
                "Non-matching number of inputs: {} (inference) vs {} (model)",
                input_arrays.len(),
                self.inputs.len()
            );
            return Err(OrtError::NonMatchingDimensions(
                NonMatchingDimensionsError::InputsCount {
                    inference_input_count: 0,
                    model_input_count: 0,
                    inference_input: input_arrays
                        .iter()
                        .map(|input_array| match input_array {
                            TypedArray::F32(array) => array.shape().to_vec(),
                            TypedArray::U8(array) => array.shape().to_vec(),
                            TypedArray::I8(array) => array.shape().to_vec(),
                            TypedArray::U16(array) => array.shape().to_vec(),
                            TypedArray::I16(array) => array.shape().to_vec(),
                            TypedArray::I32(array) => array.shape().to_vec(),
                            TypedArray::I64(array) => array.shape().to_vec(),
                            TypedArray::F64(array) => array.shape().to_vec(),
                            TypedArray::U32(array) => array.shape().to_vec(),
                            TypedArray::U64(array) => array.shape().to_vec(),
                        })
                        .collect(),
                    model_input: self
                        .inputs
                        .iter()
                        .map(|(_, input)| input.dimensions.clone())
                        .collect(),
                },
            ));
        }

        // Verify length of each individual inputs
        let inputs_different_length = input_arrays
            .iter()
            .zip(self.inputs.iter())
            .any(|(l, (_, r))| match l {
                TypedArray::F32(array) => array.shape().len(),
                TypedArray::U8(array) => array.shape().len(),
                TypedArray::I8(array) => array.shape().len(),
                TypedArray::U16(array) => array.shape().len(),
                TypedArray::I16(array) => array.shape().len(),
                TypedArray::I32(array) => array.shape().len(),
                TypedArray::I64(array) => array.shape().len(),
                TypedArray::F64(array) => array.shape().len(),
                TypedArray::U32(array) => array.shape().len(),
                TypedArray::U64(array) => array.shape().len(),
            } != r.dimensions.len());

        if inputs_different_length {
            error!(
                "Different input lengths: {:?} vs {:?}",
                self.inputs, input_arrays
            );
            return Err(OrtError::NonMatchingDimensions(
                NonMatchingDimensionsError::InputsLength {
                    inference_input: input_arrays
                        .iter()
                        .map(|input_array| match input_array {
                            TypedArray::F32(array) => array.shape().to_vec(),
                            TypedArray::U8(array) => array.shape().to_vec(),
                            TypedArray::I8(array) => array.shape().to_vec(),
                            TypedArray::U16(array) => array.shape().to_vec(),
                            TypedArray::I16(array) => array.shape().to_vec(),
                            TypedArray::I32(array) => array.shape().to_vec(),
                            TypedArray::I64(array) => array.shape().to_vec(),
                            TypedArray::F64(array) => array.shape().to_vec(),
                            TypedArray::U32(array) => array.shape().to_vec(),
                            TypedArray::U64(array) => array.shape().to_vec(),
                        })
                        .collect(),
                    model_input: self
                        .inputs
                        .iter()
                        .map(|(_, input)| input.dimensions.clone())
                        .collect(),
                },
            ));
        }

        let input_arrays_dimensions = input_arrays
            .iter()
            .map(|input_array| match input_array {
                TypedArray::F32(input_array) => input_array.shape().to_vec(),
                TypedArray::U8(input_array) => input_array.shape().to_vec(),
                TypedArray::I8(input_array) => input_array.shape().to_vec(),
                TypedArray::U16(input_array) => input_array.shape().to_vec(),
                TypedArray::I16(input_array) => input_array.shape().to_vec(),
                TypedArray::I32(input_array) => input_array.shape().to_vec(),
                TypedArray::I64(input_array) => input_array.shape().to_vec(),
                TypedArray::F64(input_array) => input_array.shape().to_vec(),
                TypedArray::U32(input_array) => input_array.shape().to_vec(),
                TypedArray::U64(input_array) => input_array.shape().to_vec(),
            })
            .collect::<Vec<_>>();

        for (input_array, (_, input)) in input_arrays.iter().zip(self.inputs.iter()) {
            match input_array {
                TypedArray::F32(input_array) => self.validate_input_shape(
                    input_array,
                    input,
                    &input_arrays_dimensions,
                    input_arrays,
                ),
                TypedArray::U8(input_array) => self.validate_input_shape(
                    input_array,
                    input,
                    &input_arrays_dimensions,
                    input_arrays,
                ),
                TypedArray::I8(input_array) => self.validate_input_shape(
                    input_array,
                    input,
                    &input_arrays_dimensions,
                    input_arrays,
                ),
                TypedArray::U16(input_array) => self.validate_input_shape(
                    input_array,
                    input,
                    &input_arrays_dimensions,
                    input_arrays,
                ),
                TypedArray::I16(input_array) => self.validate_input_shape(
                    input_array,
                    input,
                    &input_arrays_dimensions,
                    input_arrays,
                ),
                TypedArray::I32(input_array) => self.validate_input_shape(
                    input_array,
                    input,
                    &input_arrays_dimensions,
                    input_arrays,
                ),
                TypedArray::I64(input_array) => self.validate_input_shape(
                    input_array,
                    input,
                    &input_arrays_dimensions,
                    input_arrays,
                ),
                TypedArray::F64(input_array) => self.validate_input_shape(
                    input_array,
                    input,
                    &input_arrays_dimensions,
                    input_arrays,
                ),
                TypedArray::U32(input_array) => self.validate_input_shape(
                    input_array,
                    input,
                    &input_arrays_dimensions,
                    input_arrays,
                ),
                TypedArray::U64(input_array) => self.validate_input_shape(
                    input_array,
                    input,
                    &input_arrays_dimensions,
                    input_arrays,
                ),
            }?;
        }

        Ok(())
    }

    #[allow(dead_code)]
    fn validate_input_shapes<TIn, D>(&mut self, input_arrays: &[Array<TIn, D>]) -> Result<()>
    where
        TIn: TypeToTensorElementDataType + Debug + Clone,
        D: ndarray::Dimension,
    {
        // ******************************************************************
        // FIXME: Properly handle errors here
        // Make sure all dimensions match (except dynamic ones)

        // Verify length of inputs
        if input_arrays.len() != self.inputs.len() {
            error!(
                "Non-matching number of inputs: {} (inference) vs {} (model)",
                input_arrays.len(),
                self.inputs.len()
            );
            return Err(OrtError::NonMatchingDimensions(
                NonMatchingDimensionsError::InputsCount {
                    inference_input_count: 0,
                    model_input_count: 0,
                    inference_input: input_arrays
                        .iter()
                        .map(|input_array| input_array.shape().to_vec())
                        .collect(),
                    model_input: self
                        .inputs
                        .iter()
                        .map(|(_, input)| input.dimensions.clone())
                        .collect(),
                },
            ));
        }

        // Verify length of each individual inputs
        let inputs_different_length = input_arrays
            .iter()
            .zip(self.inputs.iter())
            .any(|(l, (_, r))| l.shape().len() != r.dimensions.len());
        if inputs_different_length {
            error!(
                "Different input lengths: {:?} vs {:?}",
                self.inputs, input_arrays
            );
            return Err(OrtError::NonMatchingDimensions(
                NonMatchingDimensionsError::InputsLength {
                    inference_input: input_arrays
                        .iter()
                        .map(|input_array| input_array.shape().to_vec())
                        .collect(),
                    model_input: self
                        .inputs
                        .iter()
                        .map(|(_, input)| input.dimensions.clone())
                        .collect(),
                },
            ));
        }

        let input_arrays_dimensions = input_arrays
            .iter()
            .map(|input_array| input_array.shape().to_vec())
            .collect::<Vec<_>>();

        for (input_array, (_, input)) in input_arrays.iter().zip(self.inputs.iter()) {
            self.validate_input_shape(input_array, input, &input_arrays_dimensions, input_arrays)?;
        }

        Ok(())
    }

    /// Create or return the session [`IoBinding`](../io_binding/struct.IoBinding.html)
    pub fn io_binding<D>(&self) -> Result<IoBinding<D>>
    where
        D: ndarray::Dimension,
    {
        unsafe { IoBinding::new(self) }
    }
}

pub(crate) fn get_tensor_dimensions(
    tensor_info_ptr: *const sys::OrtTensorTypeAndShapeInfo,
) -> Result<Vec<i64>> {
    let mut num_dims = 0;
    let status = unsafe { g_ort().GetDimensionsCount.unwrap()(tensor_info_ptr, &mut num_dims) };
    status_to_result(status).map_err(OrtError::GetDimensionsCount)?;
    (num_dims != 0)
        .then(|| ())
        .ok_or(OrtError::InvalidDimensions)?;

    let mut node_dims: Vec<i64> = vec![0; num_dims as usize];
    let status = unsafe {
        g_ort().GetDimensions.unwrap()(tensor_info_ptr, node_dims.as_mut_ptr(), num_dims)
    };
    status_to_result(status).map_err(OrtError::GetDimensions)?;

    Ok(node_dims)
}

pub(crate) fn get_tensor_element_type(
    tensor_info_ptr: *const sys::OrtTensorTypeAndShapeInfo,
) -> Result<TensorElementDataType> {
    let mut type_sys = sys::ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_UNDEFINED;
    let status = unsafe { g_ort().GetTensorElementType.unwrap()(tensor_info_ptr, &mut type_sys) };
    status_to_result(status).map_err(OrtError::GetTensorElementType)?;
    (type_sys != sys::ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_UNDEFINED)
        .then(|| ())
        .ok_or(OrtError::UndefinedTensorElementType)?;

    Ok(unsafe { std::mem::transmute(type_sys) })
}

/// This module contains dangerous functions working on raw pointers.
/// Those functions are only to be used from inside the
/// `SessionBuilder::with_model_from_file()` method.
mod dangerous {
    use super::*;

    pub(super) fn extract_inputs_count(session_ptr: *mut sys::OrtSession) -> Result<usize> {
        let f = g_ort().SessionGetInputCount.unwrap();
        extract_io_count(f, session_ptr)
    }

    pub(super) fn extract_outputs_count(session_ptr: *mut sys::OrtSession) -> Result<usize> {
        let f = g_ort().SessionGetOutputCount.unwrap();
        extract_io_count(f, session_ptr)
    }

    fn extract_io_count(
        f: extern_system_fn! { unsafe fn(*const sys::OrtSession, *mut usize) -> *mut sys::OrtStatus },
        session_ptr: *mut sys::OrtSession,
    ) -> Result<usize> {
        let mut num_nodes: usize = 0;
        let status = unsafe { f(session_ptr, &mut num_nodes) };
        status_to_result(status).map_err(OrtError::InOutCount)?;
        assert_null_pointer(status, "SessionStatus")?;
        (num_nodes != 0).then(|| ()).ok_or_else(|| {
            OrtError::InOutCount(OrtApiError::Msg("No nodes in model".to_owned()))
        })?;
        Ok(num_nodes)
    }

    pub(super) fn extract_input(
        session_ptr: *mut sys::OrtSession,
        allocator_ptr: *mut sys::OrtAllocator,
        i: usize,
    ) -> Result<Input> {
        let input_name = extract_input_name(session_ptr, allocator_ptr, i)?;
        let f = g_ort().SessionGetInputTypeInfo.unwrap();
        let (tensor_element_type, tensor_dimensions) = extract_io_type_info(f, session_ptr, i)?;
        Ok(Input {
            name: input_name,
            element_type: tensor_element_type,
            dimensions: tensor_dimensions,
        })
    }

    fn extract_input_name(
        session_ptr: *mut sys::OrtSession,
        allocator_ptr: *mut sys::OrtAllocator,
        i: usize,
    ) -> Result<String> {
        let mut name_bytes: *mut c_char = std::ptr::null_mut();
        let status = unsafe {
            g_ort().SessionGetInputName.unwrap()(session_ptr, i, allocator_ptr, &mut name_bytes)
        };
        status_to_result(status).map_err(OrtError::SessionGetInputName)?;
        assert_not_null_pointer(name_bytes, "SessionGetInputName")?;
        char_p_to_string(name_bytes)
    }

    fn extract_output_name(
        session_ptr: *mut sys::OrtSession,
        allocator_ptr: *mut sys::OrtAllocator,
        i: usize,
    ) -> Result<String> {
        let mut name_bytes: *mut c_char = std::ptr::null_mut();
        let status = unsafe {
            g_ort().SessionGetOutputName.unwrap()(session_ptr, i, allocator_ptr, &mut name_bytes)
        };
        status_to_result(status).map_err(OrtError::SessionGetOutputName)?;
        assert_not_null_pointer(name_bytes, "SessionGetOutputName")?;
        char_p_to_string(name_bytes)
    }

    pub(super) fn extract_output(
        session_ptr: *mut sys::OrtSession,
        allocator_ptr: *mut sys::OrtAllocator,
        i: usize,
    ) -> Result<Output> {
        let output_name = extract_output_name(session_ptr, allocator_ptr, i)?;
        let f = g_ort().SessionGetOutputTypeInfo.unwrap();
        let (tensor_element_type, tensor_dimensions) = extract_io_type_info(f, session_ptr, i)?;
        Ok(Output {
            name: output_name,
            element_type: tensor_element_type,
            dimensions: tensor_dimensions,
        })
    }

    fn extract_io_type_info(
        f: extern_system_fn! { unsafe fn(
            *const sys::OrtSession,
            usize,
            *mut *mut sys::OrtTypeInfo,
        ) -> *mut sys::OrtStatus },
        session_ptr: *mut sys::OrtSession,
        i: usize,
    ) -> Result<(TensorElementDataType, Vec<Option<u32>>)> {
        let mut typeinfo_ptr: *mut sys::OrtTypeInfo = std::ptr::null_mut();

        let status = unsafe { f(session_ptr, i, &mut typeinfo_ptr) };
        status_to_result(status).map_err(OrtError::GetTypeInfo)?;
        assert_not_null_pointer(typeinfo_ptr, "TypeInfo")?;

        let mut tensor_info_ptr: *const sys::OrtTensorTypeAndShapeInfo = std::ptr::null_mut();
        let status = unsafe {
            g_ort().CastTypeInfoToTensorInfo.unwrap()(typeinfo_ptr, &mut tensor_info_ptr)
        };
        status_to_result(status).map_err(OrtError::CastTypeInfoToTensorInfo)?;
        assert_not_null_pointer(tensor_info_ptr, "TensorInfo")?;

        let tensor_dimensions = get_tensor_dimensions(tensor_info_ptr)?;
        let tensor_element_type = get_tensor_element_type(tensor_info_ptr)?;

        unsafe { g_ort().ReleaseTypeInfo.unwrap()(typeinfo_ptr) };

        Ok((
            tensor_element_type,
            tensor_dimensions
                .into_iter()
                .map(|d| if d == -1 { None } else { Some(d as u32) })
                .collect(),
        ))
    }
}
