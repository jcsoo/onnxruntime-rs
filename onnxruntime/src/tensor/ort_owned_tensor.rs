//! Module containing tensor with memory owned by the ONNX Runtime

use std::{fmt::Debug, ops::Deref};

use ndarray::ArrayView;
use tracing::{error, trace};

use onnxruntime_sys as sys;

use crate::{
    error::status_to_result, g_ort, memory::MemoryInfo, session::Output, OrtError, Result,
    TensorElementDataType, TypeToTensorElementDataType, TypedOrtOwnedTensor,
};

/// Tensor containing data owned by the ONNX Runtime C library, used to return values from inference.
///
/// This tensor type is returned by the [`Session::run()`](../session/struct.Session.html#method.run) method.
/// It is not meant to be created directly.
///
/// The tensor hosts an [`ndarray::ArrayView`](https://docs.rs/ndarray/latest/ndarray/type.ArrayView.html)
/// of the data on the C side. This allows manipulation on the Rust side using `ndarray` without copying the data.
///
/// `OrtOwnedTensor` implements the [`std::deref::Deref`](#impl-Deref) trait for ergonomic access to
/// the underlying [`ndarray::ArrayView`](https://docs.rs/ndarray/latest/ndarray/type.ArrayView.html).
#[derive(Debug)]
#[allow(dead_code)]
pub struct OrtOwnedTensor<'t, 'm, T, D>
where
    T: TypeToTensorElementDataType + Debug + Clone,
    D: ndarray::Dimension,
    'm: 't, // 'm outlives 't
{
    pub(crate) ptr: *mut sys::OrtValue,
    array_view: ArrayView<'t, T, D>,
    memory_info: &'m MemoryInfo,
}

unsafe impl<'t, 'm, T, D> Send for OrtOwnedTensor<'_, '_, T, D>
where
    T: TypeToTensorElementDataType + Debug + Clone,
    D: ndarray::Dimension,
    'm: 't, // 'm outlives 't
{
}

impl<'t, 'm, T, D> Deref for OrtOwnedTensor<'t, 'm, T, D>
where
    T: TypeToTensorElementDataType + Debug + Clone,
    D: ndarray::Dimension,
{
    type Target = ArrayView<'t, T, D>;

    fn deref(&self) -> &Self::Target {
        &self.array_view
    }
}

impl<'t, 'm, T, D> OrtOwnedTensor<'t, 'm, T, D>
where
    T: TypeToTensorElementDataType + Debug + Clone,
    D: ndarray::Dimension,
    'm: 't, // 'm outlives 't
{
    /// # Safety
    ///
    /// create a new OrtOwnedTensor.
    pub unsafe fn new(
        ptr: *mut sys::OrtValue,
        array_view: ArrayView<'t, T, D>,
        memory_info: &'m MemoryInfo,
    ) -> Self {
        Self {
            ptr,
            array_view,
            memory_info,
        }
    }
}

#[derive(Debug)]
pub(crate) struct OrtOwnedTensorExtractor<'m, D>
where
    D: ndarray::Dimension,
{
    pub(crate) ptr: *mut sys::OrtValue,
    memory_info: &'m MemoryInfo,
    shape: D,
}

impl<'m, D> OrtOwnedTensorExtractor<'m, D>
where
    D: ndarray::Dimension,
{
    pub(crate) fn new(memory_info: &'m MemoryInfo, shape: D) -> OrtOwnedTensorExtractor<'m, D> {
        OrtOwnedTensorExtractor {
            ptr: std::ptr::null_mut(),
            memory_info,
            shape,
        }
    }

    pub(crate) fn extract<'t>(self, output: &Output) -> Result<TypedOrtOwnedTensor<'t, 'm, D>> {
        Ok(match output.element_type {
            TensorElementDataType::Float => TypedOrtOwnedTensor::F32(self.extract_impl::<f32>()?),
            TensorElementDataType::Double => TypedOrtOwnedTensor::F64(self.extract_impl::<f64>()?),
            TensorElementDataType::Int8 => TypedOrtOwnedTensor::I8(self.extract_impl::<i8>()?),
            TensorElementDataType::Int16 => TypedOrtOwnedTensor::I16(self.extract_impl::<i16>()?),
            TensorElementDataType::Int32 => TypedOrtOwnedTensor::I32(self.extract_impl::<i32>()?),
            TensorElementDataType::Int64 => TypedOrtOwnedTensor::I64(self.extract_impl::<i64>()?),
            TensorElementDataType::Uint8 => TypedOrtOwnedTensor::U8(self.extract_impl::<u8>()?),
            TensorElementDataType::Uint16 => TypedOrtOwnedTensor::U16(self.extract_impl::<u16>()?),
            TensorElementDataType::Uint32 => TypedOrtOwnedTensor::U32(self.extract_impl::<u32>()?),
            TensorElementDataType::Uint64 => TypedOrtOwnedTensor::U64(self.extract_impl::<u64>()?),
            TensorElementDataType::String => todo!(),
        })
    }

    #[tracing::instrument]
    fn extract_impl<'t, T>(self) -> Result<OrtOwnedTensor<'t, 'm, T, D>>
    where
        T: TypeToTensorElementDataType + Debug + Clone,
    {
        trace!("Creating OrtOwnedTensor.");

        // Note: Both tensor and array will point to the same data, nothing is copied.
        // As such, there is no need too free the pointer used to create the ArrayView.
        assert_ne!(self.ptr, std::ptr::null_mut());

        let mut is_tensor = 0;
        let status = unsafe { g_ort().IsTensor.unwrap()(self.ptr, &mut is_tensor) };
        status_to_result(status).map_err(OrtError::IsTensor)?;
        (is_tensor == 1)
            .then(|| ())
            .ok_or(OrtError::IsTensorCheck)?;

        // Get pointer to output tensor float values
        let mut output_array_ptr: *mut T = std::ptr::null_mut();
        let output_array_ptr_ptr: *mut *mut T = &mut output_array_ptr;
        let output_array_ptr_ptr_void: *mut *mut std::ffi::c_void =
            output_array_ptr_ptr as *mut *mut std::ffi::c_void;
        let status =
            unsafe { g_ort().GetTensorMutableData.unwrap()(self.ptr, output_array_ptr_ptr_void) };
        status_to_result(status).map_err(OrtError::IsTensor)?;
        assert_ne!(output_array_ptr, std::ptr::null_mut());

        let array_view = unsafe { ArrayView::<T, _>::from_shape_ptr(self.shape, output_array_ptr) };

        let ort_owned_tensor = OrtOwnedTensor {
            ptr: self.ptr,
            array_view,
            memory_info: self.memory_info,
        };
        Ok(ort_owned_tensor)
    }
}

impl<'t, 'm, T, D> Drop for OrtOwnedTensor<'t, 'm, T, D>
where
    T: TypeToTensorElementDataType + Debug + Clone,
    D: ndarray::Dimension,
    'm: 't, // 'm outlives 't
{
    #[tracing::instrument]
    fn drop(&mut self) {
        if self.ptr.is_null() {
            error!("OrtOwnedTensor pointer is null, not dropping.");
        } else {
            trace!("Dropping OrtOwnedTensor.");
            unsafe { g_ort().ReleaseValue.unwrap()(self.ptr) };
        }

        self.ptr = std::ptr::null_mut();
    }
}
