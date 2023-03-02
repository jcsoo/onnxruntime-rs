//! Module abstracting OrtValue.

use std::fmt::Debug;
use std::ops::Deref;

use ndarray::{Array, ArrayView, Dim, IxDynImpl};
use tracing::{error, trace};

use onnxruntime_sys as sys;

use crate::{
    error::{NonMatchingDataTypes, NonMatchingDeviceName, OrtError, Result},
    g_ort, status_to_result, AllocatorType, DeviceName, MemType, MemoryInfo, TensorElementDataType,
    TensorTypeAndShapeInfo, TypeToTensorElementDataType,
};

#[derive(Debug)]
/// An ::OrtValue
pub struct Value {
    pub(crate) ptr: *mut sys::OrtValue,
}

unsafe impl Send for Value {}
unsafe impl Sync for Value {}

impl From<*mut sys::OrtValue> for Value {
    fn from(ptr: *mut sys::OrtValue) -> Self {
        trace!("Created Value: {ptr:?}.");
        Self { ptr }
    }
}

impl Default for Value {
    fn default() -> Self {
        Self {
            ptr: std::ptr::null_mut(),
        }
    }
}

impl Value {
    fn new(ptr: *mut sys::OrtValue) -> Self {
        trace!("Created Value {ptr:?}.");
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

        // shapes
        let shape: Vec<i64> = array.shape().iter().map(|d: &usize| *d as i64).collect();
        let shape_ptr = shape.as_ptr();
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
                let tensor_values_ptr = array.as_ptr() as *mut std::ffi::c_void;

                let status = unsafe {
                    g_ort().CreateTensorWithDataAsOrtValue.unwrap()(
                        memory_info.ptr,
                        tensor_values_ptr,
                        array.len() * std::mem::size_of::<T>(),
                        shape_ptr,
                        shape_len,
                        T::tensor_element_data_type().into(),
                        &mut tensor_ptr,
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
        let mut is_tensor = 0;
        let status = unsafe { g_ort().IsTensor.unwrap()(self.ptr, &mut is_tensor) };
        status_to_result(status).map_err(OrtError::IsTensor)?;

        Ok(is_tensor == 1)
    }

    /// Return OrtTensorTypeAndShapeInfo if OrtValue is a tensor type.
    pub fn type_and_shape_info(&self) -> Result<TensorTypeAndShapeInfo> {
        TensorTypeAndShapeInfo::try_new(self)
    }

    /// Return MemoryInfo of OrtValue
    pub fn memory_info(&self) -> Result<MemoryInfo> {
        let mut memory_info_ptr: *const sys::OrtMemoryInfo = std::ptr::null_mut();
        let status =
            unsafe { g_ort().GetTensorMemoryInfo.unwrap()(self.ptr, &mut memory_info_ptr) };
        status_to_result(status).map_err(OrtError::GetTensorMemoryInfo)?;

        MemoryInfo::try_from(memory_info_ptr)
    }

    /// Provide an array_view over the data contained in the OrtValue (CPU Only)
    pub fn array_view<T>(&self) -> Result<ArrayView<T, Dim<IxDynImpl>>>
    where
        T: TypeToTensorElementDataType,
    {
        let memory_info = self.memory_info()?;
        if !matches!(memory_info.name(), DeviceName::Cpu) {
            return Err(OrtError::GetTensorMutableDataNonMatchingDeviceName(
                NonMatchingDeviceName::DeviceName {
                    tensor: memory_info.name().clone(),
                    requested: DeviceName::Cpu,
                },
            ));
        }

        let type_and_shape_info = self.type_and_shape_info()?;
        if T::tensor_element_data_type() != type_and_shape_info.element_data_type {
            return Err(OrtError::NonMachingTypes(NonMatchingDataTypes::DataType {
                input: type_and_shape_info.element_data_type.clone(),
                requested: T::tensor_element_data_type(),
            }));
        };

        // return empty array if any dimension is 0
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
            self.array_view_unchecked::<T>(Some(type_and_shape_info))
        }
    }

    /// # Safety
    pub fn array_view_unchecked<T>(
        &self,
        type_and_shape_info: Option<TensorTypeAndShapeInfo>,
    ) -> Result<ArrayView<T, Dim<IxDynImpl>>> {
        let type_and_shape_info = type_and_shape_info
            .map(Result::Ok)
            .unwrap_or_else(|| self.type_and_shape_info())?;

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

impl Deref for Value {
    type Target = *mut sys::OrtValue;

    fn deref(&self) -> &Self::Target {
        &self.ptr
    }
}

impl Drop for Value {
    #[tracing::instrument]
    fn drop(&mut self) {
        if self.ptr.is_null() {
            error!("Value pointer is null, not dropping.");
        } else {
            trace!("Dropping Value: {:?}.", self.ptr);
            unsafe { g_ort().ReleaseValue.unwrap()(self.ptr) };
        }

        self.ptr = std::ptr::null_mut();
    }
}
