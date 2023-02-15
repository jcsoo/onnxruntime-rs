//! Module containing session types

use crate::{
    char_p_to_string,
    environment::Environment,
    error::{
        assert_not_null_pointer, assert_null_pointer, status_to_result, OrtApiError, OrtError,
        Result,
    },
    g_ort,
    io_binding::IoBinding,
    memory_info::MemoryInfo,
    AllocatorType, DeviceName, GraphOptimizationLevel, MemType, OrtValue, TensorElementDataType,
};
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
pub struct SessionBuilder<'e> {
    env: &'e Environment,
    session_options_ptr: *mut sys::OrtSessionOptions,

    allocator: AllocatorType,
    memory_type: MemType,
}

impl<'e> Drop for SessionBuilder<'e> {
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

impl<'e> SessionBuilder<'e> {
    pub(crate) fn new(env: &'e Environment) -> Result<SessionBuilder<'e>> {
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
    pub fn with_inter_op_num_threads(self, num_threads: u16) -> Result<SessionBuilder<'e>> {
        // We use a u16 in the builder to cover the 16-bits positive values of a i32.
        let status = unsafe {
            g_ort().SetIntraOpNumThreads.unwrap()(self.session_options_ptr, num_threads as i32)
        };
        status_to_result(status).map_err(OrtError::SessionOptions)?;
        Ok(self)
    }

    /// Set the session's optimization level
    pub fn with_optimization_level(
        self,
        opt_level: GraphOptimizationLevel,
    ) -> Result<SessionBuilder<'e>> {
        // Sets graph optimization level
        let status = unsafe {
            g_ort().SetSessionGraphOptimizationLevel.unwrap()(
                self.session_options_ptr,
                opt_level.into(),
            )
        };
        status_to_result(status).map_err(OrtError::SessionOptions)?;
        Ok(self)
    }

    /// Set the session's disable per session threads
    pub fn with_disable_per_session_threads(self) -> Result<SessionBuilder<'e>> {
        let status = unsafe { g_ort().DisablePerSessionThreads.unwrap()(self.session_options_ptr) };
        status_to_result(status).map_err(OrtError::SessionOptions)?;
        Ok(self)
    }

    /// Enable profiling for a session.
    pub fn with_profiling(
        self,
        profile_file_prefix: Option<impl Into<String>>,
    ) -> Result<SessionBuilder<'e>> {
        let status = unsafe {
            if let Some(profile_file_prefix) = profile_file_prefix {
                let profile_file_prefix = CString::new(profile_file_prefix.into()).unwrap();
                g_ort().EnableProfiling.unwrap()(
                    self.session_options_ptr,
                    profile_file_prefix.as_ptr(),
                )
            } else {
                g_ort().DisableProfiling.unwrap()(self.session_options_ptr)
            }
        };
        status_to_result(status).map_err(OrtError::SessionOptions)?;

        Ok(self)
    }

    /// Set ExecutionMode for a session.
    pub fn with_execution_mode(self, exection_mode: ExecutionMode) -> Result<SessionBuilder<'e>> {
        let status = unsafe {
            g_ort().SetSessionExecutionMode.unwrap()(self.session_options_ptr, exection_mode.into())
        };
        status_to_result(status).map_err(OrtError::SessionOptions)?;

        Ok(self)
    }

    /// Set MemPattern for a session.
    pub fn with_mem_pattern(self, mem_pattern: bool) -> Result<SessionBuilder<'e>> {
        let status = unsafe {
            if mem_pattern {
                g_ort().EnableMemPattern.unwrap()(self.session_options_ptr)
            } else {
                g_ort().DisableMemPattern.unwrap()(self.session_options_ptr)
            }
        };
        status_to_result(status).map_err(OrtError::SessionOptions)?;

        Ok(self)
    }

    /// Set CpuMemArena for a session.
    pub fn with_cpu_mem_arena(self, cpu_mem_arena: bool) -> Result<SessionBuilder<'e>> {
        let status = unsafe {
            if cpu_mem_arena {
                g_ort().EnableCpuMemArena.unwrap()(self.session_options_ptr)
            } else {
                g_ort().DisableCpuMemArena.unwrap()(self.session_options_ptr)
            }
        };
        status_to_result(status).map_err(OrtError::SessionOptions)?;

        Ok(self)
    }

    /// Set the session to use cpu
    pub fn with_cpu(self, use_arena: bool) -> Result<SessionBuilder<'e>> {
        unsafe {
            sys::OrtSessionOptionsAppendExecutionProvider_CPU(
                self.session_options_ptr,
                i32::from(use_arena),
            );
        };

        Ok(self)
    }

    /// Set the session to use cuda
    #[cfg(feature = "cuda")]
    pub fn with_cuda(self, options: CUDAProviderOptions) -> Result<SessionBuilder<'e>> {
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
    pub fn with_tensorrt(self, options: TensorrtProviderOptions) -> Result<SessionBuilder<'e>> {
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
    pub fn with_allocator(mut self, allocator: AllocatorType) -> Result<SessionBuilder<'e>> {
        self.allocator = allocator;
        Ok(self)
    }

    /// Set the session's memory type
    ///
    /// Defaults to [`MemType::Default`](../enum.MemType.html#variant.Default)
    pub fn with_memory_type(mut self, memory_type: MemType) -> Result<SessionBuilder<'e>> {
        self.memory_type = memory_type;
        Ok(self)
    }

    // TODO: Add all functions changing the options.
    //       See all OrtApi methods taking a `options: *mut OrtSessionOptions`.

    /// Load an ONNX graph from a file and commit the session
    #[tracing::instrument]
    pub fn with_model_from_file<P>(self, model_filepath_ref: P) -> Result<Session<'e>>
    where
        P: AsRef<Path> + Debug + 'e,
    {
        let model_filepath = model_filepath_ref.as_ref();
        let mut session_ptr: *mut sys::OrtSession = std::ptr::null_mut();

        if !model_filepath.exists() {
            return Err(OrtError::FileDoesNotExist {
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
    #[tracing::instrument(skip(model_bytes))]
    pub fn with_model_from_memory<B>(self, model_bytes: B) -> Result<Session<'e>>
    where
        B: AsRef<[u8]> + Debug,
    {
        self.with_model_from_memory_monomorphized(model_bytes.as_ref())
    }

    #[tracing::instrument(skip(model_bytes))]
    fn with_model_from_memory_monomorphized(self, model_bytes: &[u8]) -> Result<Session<'e>> {
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

/// Execution mode
#[derive(Debug, Clone)]
pub enum ExecutionMode {
    /// Sequential
    Sequential,
    /// Parallel
    Parallel,
}

impl From<ExecutionMode> for sys::ExecutionMode {
    fn from(val: ExecutionMode) -> Self {
        match val {
            ExecutionMode::Sequential => sys::ExecutionMode::ORT_SEQUENTIAL,
            ExecutionMode::Parallel => sys::ExecutionMode::ORT_PARALLEL,
        }
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
            i32::from(self.do_copy_in_default_stream).to_string(),
            i32::from(self.cudnn_conv_use_max_workspace).to_string(),
            i32::from(self.cudnn_conv1d_pad_to_nc1d).to_string(),
            i32::from(self.enable_cuda_graph).to_string(),
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
    max_workspace_size: usize,
    /// The maximum number of iterations allowed in model partitioning for TensorRT.
    max_partition_iterations: usize,
    /// The minimum node size in a subgraph after partitioning.
    min_subgraph_size: usize,
    /// Enable FP16 mode in TensorRT.
    fp16_enable: bool,
    /// Enable FP16 mode in TensorRT.
    int8_enable: bool,
    /// Specify INT8 calibration table file for non-QDQ models in INT8 mode.
    int8_calibration_table_name: Option<String>,
    ///  Select what calibration table is used for non-QDQ models in INT8 mode.
    /// If true, native TensorRT generated calibration table is used
    /// If false, ONNXRUNTIME tool generated calibration table is used.
    int8_use_native_calibration_table: bool,
    /// Enable Deep Learning Accelerator (DLA).
    dla_enable: bool,
    /// Specify DLA core to execute on.
    dla_core: usize,
    /// Enable TensorRT engine caching.
    engine_cache_enable: bool,
    /// Specify path for TensorRT engine and profile files.
    engine_cache_path: Option<String>,
    /// Dumps the subgraphs that are transformed into TRT engines in onnx format to the filesystem.
    dump_subgraphs: bool,
    /// Sequentially build TensorRT engines across provider instances in multi-GPU environment.
    force_sequential_engine_build: bool,
    /// Enable context memory sharing between subgraphs.
    #[cfg(feature = "ort_1_14_0")]
    context_memory_sharing_enable: bool,
    /// Force Pow + Reduce ops in layer norm to FP32.
    #[cfg(feature = "ort_1_14_0")]
    layer_norm_fp32_fallback: bool,
}

#[cfg(feature = "cuda")]
impl Default for TensorrtProviderOptions {
    fn default() -> Self {
        Self {
            device_id: 0,
            max_workspace_size: 1073741824,
            max_partition_iterations: 1000,
            min_subgraph_size: 1,
            fp16_enable: false,
            int8_enable: false,
            int8_calibration_table_name: None,
            int8_use_native_calibration_table: false,
            dla_enable: false,
            dla_core: 0,
            engine_cache_enable: false,
            engine_cache_path: None,
            dump_subgraphs: false,
            force_sequential_engine_build: false,
            #[cfg(feature = "ort_1_14_0")]
            context_memory_sharing_enable: false,
            #[cfg(feature = "ort_1_14_0")]
            layer_norm_fp32_fallback: false,
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

        #[cfg(feature = "ort_1_14_0")]
        keys.append(
            &mut vec![
                "trt_context_memory_sharing_enable",
                "trt_layer_norm_fp32_fallback",
            ]
            .into_iter()
            .map(|k| CString::new(k).unwrap())
            .collect::<Vec<_>>(),
        );

        let mut values = vec![
            self.device_id.to_string(),
            self.max_workspace_size.to_string(),
            self.max_partition_iterations.to_string(),
            self.min_subgraph_size.to_string(),
            i32::from(self.fp16_enable).to_string(),
            i32::from(self.int8_enable).to_string(),
            i32::from(self.int8_use_native_calibration_table).to_string(),
            i32::from(self.dla_enable).to_string(),
            self.dla_core.to_string(),
            i32::from(self.engine_cache_enable).to_string(),
            i32::from(self.dump_subgraphs).to_string(),
            i32::from(self.force_sequential_engine_build).to_string(),
        ]
        .into_iter()
        .map(|k| CString::new(k).unwrap())
        .collect::<Vec<_>>();

        #[cfg(feature = "ort_1_14_0")]
        values.append(
            &mut vec![
                i32::from(self.context_memory_sharing_enable).to_string(),
                i32::from(self.layer_norm_fp32_fallback).to_string(),
            ]
            .into_iter()
            .map(|k| CString::new(k).unwrap())
            .collect::<Vec<_>>(),
        );

        if let Some(engine_cache_path) = &self.engine_cache_path {
            keys.push(CString::new("trt_engine_cache_path").unwrap());
            values.push(CString::new(engine_cache_path.clone()).unwrap());
        };

        if let Some(int8_calibration_table_name) = &self.int8_calibration_table_name {
            keys.push(CString::new("trt_int8_calibration_table_name").unwrap());
            values.push(CString::new(int8_calibration_table_name.clone()).unwrap());
        };

        (keys, values)
    }

    /// Set device_id
    pub fn with_device_id(mut self, device_id: usize) -> Self {
        self.device_id = device_id;
        self
    }

    /// Set trt_max_workspace_size
    pub fn with_max_workspace_size(mut self, max_workspace_size: usize) -> Self {
        self.max_workspace_size = max_workspace_size;
        self
    }

    /// Set trt_max_partition_iterations
    pub fn with_max_partition_iterations(mut self, max_partition_iterations: usize) -> Self {
        self.max_partition_iterations = max_partition_iterations;
        self
    }

    /// Set min_subgraph_size
    pub fn with_min_subgraph_size(mut self, min_subgraph_size: usize) -> Self {
        self.min_subgraph_size = min_subgraph_size;
        self
    }

    /// Set fp16_enable
    pub fn with_fp16_enable(mut self, fp16_enable: bool) -> Self {
        self.fp16_enable = fp16_enable;
        self
    }

    /// Set int8_enable
    pub fn with_int8_enable(mut self, int8_enable: bool) -> Self {
        self.int8_enable = int8_enable;
        self
    }

    /// Set int8_calibration_table_name
    pub fn with_int8_calibration_table_name(
        mut self,
        int8_calibration_table_name: Option<&str>,
    ) -> Self {
        self.int8_calibration_table_name = int8_calibration_table_name.map(|v| v.to_string());
        self
    }

    /// Set int8_use_native_calibration_table
    pub fn with_int8_use_native_calibration_table(
        mut self,
        int8_use_native_calibration_table: bool,
    ) -> Self {
        self.int8_use_native_calibration_table = int8_use_native_calibration_table;
        self
    }

    /// Set dla_enable
    pub fn with_dla_enable(mut self, dla_enable: bool) -> Self {
        self.dla_enable = dla_enable;
        self
    }

    /// Set dla_core
    pub fn with_dla_core(mut self, dla_core: usize) -> Self {
        self.dla_core = dla_core;
        self
    }

    /// Set engine_cache_enable
    pub fn with_engine_cache_enable(mut self, engine_cache_enable: bool) -> Self {
        self.engine_cache_enable = engine_cache_enable;
        self
    }

    /// Set engine_cache_path
    pub fn with_engine_cache_path(mut self, engine_cache_path: Option<&str>) -> Self {
        self.engine_cache_path = engine_cache_path.map(|v| v.to_string());
        self
    }

    /// Set dump_subgraphs
    pub fn with_dump_subgraphs(mut self, dump_subgraphs: bool) -> Self {
        self.dump_subgraphs = dump_subgraphs;
        self
    }

    /// Set force_sequential_engine_build
    pub fn with_force_sequential_engine_build(
        mut self,
        force_sequential_engine_build: bool,
    ) -> Self {
        self.force_sequential_engine_build = force_sequential_engine_build;
        self
    }

    /// Set context_memory_sharing_enable
    #[cfg(feature = "ort_1_14_0")]
    pub fn with_context_memory_sharing_enable(
        mut self,
        context_memory_sharing_enable: bool,
    ) -> Self {
        self.context_memory_sharing_enable = context_memory_sharing_enable;
        self
    }

    /// Set layer_norm_fp32_fallback
    #[cfg(feature = "ort_1_14_0")]
    pub fn with_layer_norm_fp32_fallback(mut self, layer_norm_fp32_fallback: bool) -> Self {
        self.layer_norm_fp32_fallback = layer_norm_fp32_fallback;
        self
    }
}

/// Type storing the session information, built from an [`Environment`](environment/struct.Environment.html)
#[derive(Debug)]
#[allow(dead_code)]
pub struct Session<'e> {
    env: &'e Environment,
    pub(crate) ptr: *mut sys::OrtSession,
    pub(crate) allocator_ptr: *mut sys::OrtAllocator,
    pub(crate) memory_info: MemoryInfo,
    /// Information about the ONNX's inputs as stored in loaded file
    pub inputs: HashMap<String, Input>,
    /// Information about the ONNX's outputs as stored in loaded file
    pub outputs: HashMap<String, Output>,
}

unsafe impl<'e> Send for Session<'e> {}
unsafe impl<'e> Sync for Session<'e> {}

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

impl<'e> Drop for Session<'e> {
    #[tracing::instrument]
    fn drop(&mut self) {
        if self.ptr.is_null() {
            error!("Session pointer is null, not dropping");
        } else {
            trace!("Dropping Session: {:?}.", self.ptr);
            unsafe { g_ort().ReleaseSession.unwrap()(self.ptr) };
        }

        self.ptr = std::ptr::null_mut();
        self.allocator_ptr = std::ptr::null_mut();
    }
}

impl<'e> Session<'e> {
    /// Run the input data through the ONNX graph, performing inference.
    ///
    /// Note that ONNX models can have multiple inputs; a `Vec<>` is thus
    /// used for the input data here.
    #[tracing::instrument]
    pub fn run<'s, 't, 'm, S>(
        &'s self,
        inputs: HashMap<S, &OrtValue>,
    ) -> Result<HashMap<String, OrtValue>>
    where
        'm: 't, // 'm outlives 't (memory info outlives tensor)
        's: 'm, // 's outlives 'm (session outlives memory info)
        S: Into<String> + Clone + Debug,
    {
        // self.validate_untyped_input_shapes(&inputs)?;

        // Build arguments to Run()
        let input_names_ptr: Vec<*const c_char> = inputs
            .keys()
            .cloned()
            .map(|n| CString::new(n.into()).unwrap())
            .map(|n| n.into_raw() as *const c_char)
            .collect();

        let output_names_ptr: Vec<*const c_char> = self
            .outputs
            .keys()
            .cloned()
            .map(|n| CString::new(n).unwrap())
            .map(|n| n.into_raw() as *const c_char)
            .collect();

        let input_ort_values = inputs
            .values()
            .map(|input| input.ptr as *const sys::OrtValue)
            .collect::<Vec<_>>();

        let mut output_ort_values: Vec<*mut sys::OrtValue> =
            vec![std::ptr::null_mut(); output_names_ptr.len()];

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
                output_ort_values.as_mut_ptr(),
            )
        };
        status_to_result(status).map_err(OrtError::Run)?;

        Ok(self
            .outputs
            .keys()
            .zip(output_ort_values.into_iter())
            .map(|(name, value)| (name.to_owned(), value.into()))
            .collect::<HashMap<_, _>>())
    }

    /// Run the input data through the ONNX graph, performing inference.
    pub fn run_with_iobinding(&self, io_binding: &IoBinding) -> Result<()> {
        let run_options_ptr: *const sys::OrtRunOptions = std::ptr::null();
        let status =
            unsafe { g_ort().RunWithBinding.unwrap()(self.ptr, run_options_ptr, io_binding.ptr) };
        status_to_result(status).map_err(OrtError::Run)?;
        Ok(())
    }

    /// Create or return the session [`IoBinding`](../io_binding/struct.IoBinding.html)
    pub fn io_binding(&self) -> Result<IoBinding> {
        unsafe { IoBinding::new(self) }
    }
}

/// This module contains dangerous functions working on raw pointers.
/// Those functions are only to be used from inside the
/// `SessionBuilder::with_model_from_file()` method.
mod dangerous {
    use crate::ort_tensor_type_and_shape_info::OrtTensorTypeAndShapeInfo;

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
        (num_nodes != 0).then_some(()).ok_or_else(|| {
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

        let type_and_shape_info: OrtTensorTypeAndShapeInfo =
            (tensor_info_ptr as *mut sys::OrtTensorTypeAndShapeInfo).try_into()?;

        Ok((
            type_and_shape_info.element_data_type.clone(),
            type_and_shape_info
                .dimensions
                .clone()
                .into_iter()
                .map(|d| if d == -1 { None } else { Some(d as u32) })
                .collect(),
        ))
    }
}
