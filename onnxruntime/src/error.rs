//! Module containing error definitions.

use crate::{char_p_to_string, g_ort, TensorElementDataType};
use onnxruntime_sys as sys;
use std::os::raw::c_char;
use std::{io, path::PathBuf};
use thiserror::Error;
use tracing::error;

/// Type alias for the `Result`
pub type Result<T> = std::result::Result<T, OrtError>;

/// Error type centralizing all possible errors
#[non_exhaustive]
#[derive(Error, Debug)]
pub enum OrtError {
    /// The C API can message to the caller using a C `char *` which needs to be converted
    /// to Rust's `String`. This operation can fail.
    #[error("Failed to construct String")]
    StringConversion(OrtApiError),
    /// Error occurred when getting available providers
    #[error("Failed to get available providers: {0}")]
    GetAvailableProviders(OrtApiError),
    /// Error occurred when releasing available providers
    #[error("Failed to release available providers: {0}")]
    ReleaseAvailableProviders(OrtApiError),
    /// An error occurred when creating an ONNX environment
    #[error("Failed to create environment: {0}")]
    Environment(OrtApiError),
    /// An error occurred when creating an ONNX environment
    #[error("Failed to create environment: {0}")]
    ThreadingOptions(OrtApiError),
    /// Error occurred when creating an ONNX session options
    #[error("Failed to create session options: {0}")]
    SessionOptions(OrtApiError),
    /// Error occurred when creating an ONNX session
    #[error("Failed to create session: {0}")]
    Session(OrtApiError),
    /// Error occurred when creating an ONNX allocator
    #[error("Failed to get allocator: {0}")]
    Allocator(OrtApiError),
    /// Error occurred when counting ONNX input or output count
    #[error("Failed to get input or output count: {0}")]
    InOutCount(OrtApiError),
    /// Error occurred when getting ONNX input name
    #[error("Failed to get input name: {0}")]
    SessionGetInputName(OrtApiError),
    /// Error occurred when getting ONNX output name
    #[error("Failed to get output name: {0}")]
    SessionGetOutputName(OrtApiError),
    /// Error occurred when getting ONNX type information
    #[error("Failed to get type info: {0}")]
    GetTypeInfo(OrtApiError),
    /// Error occurred when casting ONNX type information to tensor information
    #[error("Failed to cast type info to tensor info: {0}")]
    CastTypeInfoToTensorInfo(OrtApiError),
    /// Error occurred when getting tensor elements type
    #[error("Failed to get tensor element type: {0}")]
    GetTensorElementType(OrtApiError),
    /// Error occurred when getting ONNX dimensions count
    #[error("Failed to get dimensions count: {0}")]
    GetDimensionsCount(OrtApiError),
    /// Error occurred when getting ONNX dimensions
    #[error("Failed to get dimensions: {0}")]
    GetDimensions(OrtApiError),
    /// Error occurred when getting memory information
    #[error("Failed to get get memory information: {0}")]
    OrtMemoryInfo(OrtApiError),
    /// Error occurred when creating memory information
    #[error("Failed to get create memory information: {0}")]
    CreateMemoryInfo(OrtApiError),
    /// Error occurred when getting memory information
    #[error("Failed to get memory information: {0}")]
    GetTensorMemoryInfo(OrtApiError),
    /// Error occurred when geting name from memory information
    #[error("Failed to get memory information name: {0}")]
    MemoryInfoGetName(OrtApiError),
    /// Error occurred when geting id from memory information
    #[error("Failed to get memory information id: {0}")]
    MemoryInfoGetId(OrtApiError),
    /// Error occurred when geting name from memory information
    #[error("Failed to get memory information type: {0}")]
    MemoryInfoGetMemType(OrtApiError),
    /// Error occurred when geting allocator type from memory information
    #[error("Failed to get memory information allocator type: {0}")]
    MemoryInfoGetType(OrtApiError),
    /// Error occurred when creating CPU memory information
    #[error("Failed to get create io_binding: {0}")]
    CreateIoBinding(OrtApiError),
    /// Error occurred when trying to bind an input
    #[error("Failed to bind input: {0}")]
    BindInput(OrtApiError),
    /// Error occurred when trying to bind an output
    #[error("Failed to bind output to device: {0}")]
    BindOutputToDevice(OrtApiError),
    /// Error occurred when trying to get bound output names
    #[error("Failed to get bound output names: {0}")]
    GetBoundOutputNames(OrtApiError),
    /// Error occurred when trying to get bound output values
    #[error("Failed to get bound output values: {0}")]
    GetBoundOutputValues(OrtApiError),
    /// Error occurred when trying to copy bound output values to CPU
    #[error("Failed to copy bound output values: {0}")]
    CopyOutputsAcrossDevices(OrtApiError),
    /// Error occurred when creating ONNX tensor
    #[error("Failed to create tensor: {0}")]
    CreateTensor(OrtApiError),
    /// Error occurred when creating ONNX tensor with specific data
    #[error("Failed to create tensor with data: {0}")]
    CreateTensorWithData(OrtApiError),
    /// Error occurred when filling a tensor with string data
    #[error("Failed to fill string tensor: {0}")]
    FillStringTensor(OrtApiError),
    /// Error occurred when checking if ONNX tensor was properly initialized
    #[error("Failed to check if tensor: {0}")]
    IsTensor(OrtApiError),
    /// Error occurred when getting tensor type and shape
    #[error("Failed to get tensor type and shape: {0}")]
    GetTensorTypeAndShape(OrtApiError),
    /// Error occurred when ONNX inference operation was called
    #[error("Failed to run: {0}")]
    Run(OrtApiError),
    /// Error occurred when extracting data from an ONNX tensor into an C array to be used as an `ndarray::ArrayView`
    #[error("Failed to get tensor data: {0}")]
    GetTensorMutableData(OrtApiError),
    /// Data type of input data and ONNX model loaded from file do not match
    #[error("Data type does not match: {0}")]
    NonMachingTypes(NonMatchingDataTypes),
    /// File does not exists
    #[error("File {filename:?} does not exists")]
    FileDoesNotExist {
        /// Path which does not exists
        filename: PathBuf,
    },
    /// Path is an invalid UTF-8
    #[error("Path {path:?} cannot be converted to UTF-8")]
    NonUtf8Path {
        /// Path with invalid UTF-8
        path: PathBuf,
    },
    /// Attempt to build a Rust `CString` from a null pointer
    #[error("Failed to build CString when original contains null: {0}")]
    CStringNulError(#[from] std::ffi::NulError),
    #[error("{0} pointer should be null")]
    /// Ort Pointer should have been null
    PointerShouldBeNull(String),
    /// Ort pointer should not have been null
    #[error("{0} pointer should not be null")]
    PointerShouldNotBeNull(String),
    /// ONNX Model has invalid dimensions
    #[error("Invalid dimensions")]
    InvalidDimensions,
    /// The runtime type was undefined
    #[error("Undefined Tensor Element Type")]
    UndefinedTensorElementType,
    /// The OrtValue is not of type Tensor
    #[error("OrtValue is not Tensor")]
    NotTensor,
}

/// Error used when dimensions of input (from model and from inference call)
/// do not match (as they should).
#[derive(Error, Debug)]
pub enum NonMatchingDataTypes {
    /// Requested data type for input does not match requested data type
    #[error("Non-matching data types: {input:?} for input vs {requested:?}")]
    DataType {
        /// Number of input dimensions used by inference call
        requested: TensorElementDataType,
        /// Number of input dimensions defined in model
        input: TensorElementDataType,
    },
}

/// Error details when ONNX C API fail
#[non_exhaustive]
#[derive(Error, Debug)]
pub enum OrtApiError {
    /// Details as reported by the ONNX C API in case of error
    #[error("Error calling ONNX Runtime C function: {0}")]
    Msg(String),
    /// Details as reported by the ONNX C API in case of error cannot be converted to UTF-8
    #[error("Error calling ONNX Runtime C function and failed to convert error message to UTF-8")]
    IntoStringError(std::ffi::IntoStringError),
}

/// Error from downloading pre-trained model from the [ONNX Model Zoo](https://github.com/onnx/models).
#[non_exhaustive]
#[derive(Error, Debug)]
pub enum OrtDownloadError {
    /// Generic input/output error
    #[error("Error downloading data to file: {0}")]
    IoError(#[from] io::Error),
    #[cfg(feature = "model-fetching")]
    /// Download error by ureq
    #[error("Error downloading data to file: {0}")]
    UreqError(#[from] Box<ureq::Error>),
    /// Error getting content-length from an HTTP GET request
    #[error("Error getting content-length")]
    ContentLengthError,
    /// Mismatch between amount of downloaded and expected bytes
    #[error("Error copying data to file: expected {expected} length, received {io}")]
    CopyError {
        /// Expected amount of bytes to download
        expected: u64,
        /// Number of bytes read from network and written to file
        io: u64,
    },
}

/// Wrapper type around a ONNX C API's `OrtStatus` pointer
///
/// This wrapper exists to facilitate conversion from C raw pointers to Rust error types
#[derive(Debug)]
pub struct OrtStatusWrapper {
    ptr: *const sys::OrtStatus,
}

impl Drop for OrtStatusWrapper {
    fn drop(&mut self) {
        unsafe { g_ort().ReleaseStatus.unwrap()(self.ptr as *mut sys::OrtStatus) };

        self.ptr = std::ptr::null_mut();
    }
}

impl From<*const sys::OrtStatus> for OrtStatusWrapper {
    fn from(status: *const sys::OrtStatus) -> Self {
        OrtStatusWrapper { ptr: status }
    }
}

pub(crate) fn assert_null_pointer<T>(ptr: *const T, name: &str) -> Result<()> {
    ptr.is_null()
        .then_some(())
        .ok_or_else(|| OrtError::PointerShouldBeNull(name.to_owned()))
}

pub(crate) fn assert_not_null_pointer<T>(ptr: *const T, name: &str) -> Result<()> {
    (!ptr.is_null())
        .then_some(())
        .ok_or_else(|| OrtError::PointerShouldBeNull(name.to_owned()))
}

impl From<OrtStatusWrapper> for std::result::Result<(), OrtApiError> {
    fn from(status: OrtStatusWrapper) -> Self {
        if status.ptr.is_null() {
            Ok(())
        } else {
            let raw: *const c_char = unsafe { g_ort().GetErrorMessage.unwrap()(status.ptr) };
            match char_p_to_string(raw) {
                Ok(msg) => Err(OrtApiError::Msg(msg)),
                Err(err) => match err {
                    OrtError::StringConversion(OrtApiError::IntoStringError(e)) => {
                        Err(OrtApiError::IntoStringError(e))
                    }
                    _ => unreachable!(),
                },
            }
        }
    }
}

pub(crate) fn status_to_result(
    status: *const sys::OrtStatus,
) -> std::result::Result<(), OrtApiError> {
    let status_wrapper: OrtStatusWrapper = status.into();
    status_wrapper.into()
}
