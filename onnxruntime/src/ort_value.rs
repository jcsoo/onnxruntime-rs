//! Module containing tensor with memory owned by the ONNX Runtime

use std::fmt::Debug;
use std::ops::Deref;

use ndarray::{Array, ArrayView, Dim, IxDynImpl};
use tracing::{error, trace};

use onnxruntime_sys as sys;

use crate::error::{status_to_result, NonMatchingDataTypes, OrtError, Result};
use crate::ort_tensor_type_and_shape_info::OrtTensorTypeAndShapeInfo;
use crate::{
    g_ort, AllocatorType, DeviceName, MemType, MemoryInfo, TensorElementDataType,
    TypeToTensorElementDataType,
};

#[derive(Debug)]
/// An ::OrtValue
pub struct OrtValue {
    pub(crate) ptr: *mut sys::OrtValue,
}

unsafe impl Send for OrtValue {}
unsafe impl Sync for OrtValue {}

impl From<*mut sys::OrtValue> for OrtValue {
    fn from(val: *mut sys::OrtValue) -> Self {
        Self { ptr: val }
    }
}

impl Default for OrtValue {
    fn default() -> Self {
        Self {
            ptr: std::ptr::null_mut(),
        }
    }
}

impl OrtValue {
    fn new(ptr: *mut sys::OrtValue) -> Self {
        trace!("Creating OrtValue.");
        Self { ptr }
    }

    /// try to allocate a new ortvalue from an ndarray
    pub fn try_from_array<T, D>(array: &Array<T, D>) -> Result<Self>
    where
        T: TypeToTensorElementDataType + Debug + Clone,
        D: ndarray::Dimension,
    {
        let memory_info =
            MemoryInfo::new(DeviceName::Cpu, 0, AllocatorType::Arena, MemType::Default)?;

        // where onnxruntime will write the tensor data to
        let mut tensor_ptr: *mut sys::OrtValue = std::ptr::null_mut();
        let tensor_ptr_ptr: *mut *mut sys::OrtValue = &mut tensor_ptr;

        let shape: Vec<i64> = array.shape().iter().map(|d: &usize| *d as i64).collect();
        let shape_ptr: *const i64 = shape.as_ptr();
        let shape_len = array.shape().len();

        match T::tensor_element_data_type() {
            TensorElementDataType::Float
            | TensorElementDataType::Uint8
            | TensorElementDataType::Int8
            | TensorElementDataType::Uint16
            | TensorElementDataType::Int16
            | TensorElementDataType::Int32
            | TensorElementDataType::Int64
            | TensorElementDataType::Double
            | TensorElementDataType::Uint32
            | TensorElementDataType::Uint64 => {
                // primitive data is already suitably laid out in memory; provide it to
                // onnxruntime as is
                let tensor_values_ptr: *mut std::ffi::c_void =
                    array.as_ptr() as *mut std::ffi::c_void;

                let status = unsafe {
                    g_ort().CreateTensorWithDataAsOrtValue.unwrap()(
                        memory_info.ptr,
                        tensor_values_ptr,
                        array.len() * std::mem::size_of::<T>(),
                        shape_ptr,
                        shape_len,
                        T::tensor_element_data_type().into(),
                        tensor_ptr_ptr,
                    )
                };
                status_to_result(status).map_err(OrtError::IsTensor)?;
            }
            _ => todo!(),
        }

        Ok(Self::new(tensor_ptr))
    }

    /// Return if an OrtValue is a tensor type.
    pub fn is_tensor(&self) -> Result<bool> {
        // Note: Both tensor and array will point to the same data, nothing is copied.
        // As such, there is no need too free the pointer used to create the ArrayView.
        assert_ne!(self.ptr, std::ptr::null_mut());

        let mut is_tensor = 0;
        let status = unsafe { g_ort().IsTensor.unwrap()(self.ptr, &mut is_tensor) };
        status_to_result(status).map_err(OrtError::IsTensor)?;

        Ok(is_tensor == 1)
    }

    /// Return OrtTensorTypeAndShapeInfo if OrtValue is a tensor type.
    pub fn type_and_shape_info(&self) -> Result<OrtTensorTypeAndShapeInfo> {
        OrtTensorTypeAndShapeInfo::try_new(self)
    }

    /// Return MemoryInfo if OrtValue is a tensor type.
    pub fn memory_info(&self) -> Result<MemoryInfo> {
        let mut memory_info_ptr: *const sys::OrtMemoryInfo = std::ptr::null_mut();
        let status =
            unsafe { g_ort().GetTensorMemoryInfo.unwrap()(self.ptr, &mut memory_info_ptr) };
        status_to_result(status).map_err(OrtError::GetTensorMemoryInfo)?;

        MemoryInfo::try_from(memory_info_ptr)
    }

    /// # Safety
    pub unsafe fn try_from_ptr_unchecked(
        memory_info: &MemoryInfo,
        ptr_addr: *mut std::ffi::c_void,
        data_size: usize,
        shape: Vec<i64>,
    ) -> Result<Self> {
        let shape_ptr: *const i64 = shape.as_ptr();
        let shape_len = shape.len();

        let mut tensor_ptr: *mut sys::OrtValue = std::ptr::null_mut();
        let tensor_ptr_ptr: *mut *mut sys::OrtValue = &mut tensor_ptr;

        let status = g_ort().CreateTensorWithDataAsOrtValue.unwrap()(
            memory_info.ptr,
            ptr_addr,
            data_size,
            shape_ptr,
            shape_len,
            sys::ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT8,
            tensor_ptr_ptr,
        );

        status_to_result(status).map_err(OrtError::CreateTensorWithData)?;

        Ok(Self { ptr: tensor_ptr })
    }

    /// extract the array_view from the OrtValue
    pub fn array_view<T>(&self) -> Result<ArrayView<T, Dim<IxDynImpl>>>
    where
        T: TypeToTensorElementDataType,
    {
        let type_and_shape_info = self.type_and_shape_info()?;

        if T::tensor_element_data_type() != type_and_shape_info.element_data_type {
            return Err(OrtError::NonMachingTypes(NonMatchingDataTypes::DataType {
                input: type_and_shape_info.element_data_type.clone(),
                requested: T::tensor_element_data_type(),
            }));
        };

        if type_and_shape_info.dimensions.iter().any(|dim| dim == &0) {
            Ok(ArrayView::from_shape(
                type_and_shape_info
                    .dimensions
                    .iter()
                    .map(|dim| *dim as usize)
                    .collect::<Vec<_>>(),
                &[],
            )
            .unwrap())
        } else {
            self.array_view_unchecked::<T>()
        }
    }

    /// # Safety
    pub fn array_view_unchecked<T>(&self) -> Result<ArrayView<T, Dim<IxDynImpl>>> {
        let type_and_shape_info = self.type_and_shape_info()?;

        // Get pointer to output tensor values
        let mut output_array_ptr: *mut T = std::ptr::null_mut();
        let output_array_ptr_ptr: *mut *mut T = &mut output_array_ptr;
        let output_array_ptr_ptr_void: *mut *mut std::ffi::c_void =
            output_array_ptr_ptr as *mut *mut std::ffi::c_void;
        let status =
            unsafe { g_ort().GetTensorMutableData.unwrap()(self.ptr, output_array_ptr_ptr_void) };
        status_to_result(status).map_err(OrtError::GetTensorMutableData)?;

        Ok(unsafe {
            ArrayView::<T, _>::from_shape_ptr(
                type_and_shape_info
                    .dimensions
                    .iter()
                    .map(|dim| *dim as usize)
                    .collect::<Vec<_>>(),
                output_array_ptr,
            )
        })
    }
}

impl Deref for OrtValue {
    type Target = *mut sys::OrtValue;

    fn deref(&self) -> &Self::Target {
        &self.ptr
    }
}

impl Drop for OrtValue {
    #[tracing::instrument]
    fn drop(&mut self) {
        if self.ptr.is_null() {
            error!("OrtValue pointer is null, not dropping.");
        } else {
            trace!("Dropping OrtValue: {:?}.", self.ptr);
            unsafe { g_ort().ReleaseValue.unwrap()(self.ptr) };
        }

        self.ptr = std::ptr::null_mut();
    }
}
