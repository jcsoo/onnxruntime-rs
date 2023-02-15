use crate::error::{status_to_result, OrtError, Result};
use crate::{g_ort, OrtValue, TensorElementDataType};
use onnxruntime_sys as sys;
use std::fmt::Debug;
use tracing::{error, trace};

#[derive(Debug)]
/// A tensor’s type and shape information.
pub struct OrtTensorTypeAndShapeInfo {
    ptr: *mut sys::OrtTensorTypeAndShapeInfo,
    /// The Tensory Data Type
    pub element_data_type: TensorElementDataType,
    /// The shape of the Tensor
    pub dimensions: Vec<i64>,
}

impl TryFrom<*mut sys::OrtTensorTypeAndShapeInfo> for OrtTensorTypeAndShapeInfo {
    type Error = OrtError;

    fn try_from(ptr: *mut sys::OrtTensorTypeAndShapeInfo) -> Result<Self> {
        let element_data_type = OrtTensorTypeAndShapeInfo::try_get_data_type(ptr)?;
        let dimensions = OrtTensorTypeAndShapeInfo::try_get_dimensions(ptr)?;

        Ok(Self {
            ptr,
            element_data_type,
            dimensions,
        })
    }
}

impl OrtTensorTypeAndShapeInfo {
    pub(crate) fn try_new(ort_value: &OrtValue) -> Result<Self> {
        trace!("Creating OrtTensorTypeAndShapeInfo.");

        // ensure tensor
        ort_value
            .is_tensor()?
            .then_some(())
            .ok_or(OrtError::NotTensor)?;

        // get info
        let mut ptr: *mut sys::OrtTensorTypeAndShapeInfo = std::ptr::null_mut();
        let status = unsafe { g_ort().GetTensorTypeAndShape.unwrap()(**ort_value, &mut ptr) };
        status_to_result(status)
            .map_err(OrtError::GetTensorTypeAndShape)
            .unwrap();

        ptr.try_into()
    }

    fn try_get_data_type(
        type_and_shape_info: *mut sys::OrtTensorTypeAndShapeInfo,
    ) -> Result<TensorElementDataType> {
        let mut onnx_data_type =
            sys::ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_UNDEFINED;
        let status = unsafe {
            g_ort().GetTensorElementType.unwrap()(type_and_shape_info, &mut onnx_data_type)
        };
        status_to_result(status).map_err(OrtError::GetTensorElementType)?;
        (onnx_data_type != sys::ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_UNDEFINED)
            .then_some(())
            .ok_or(OrtError::UndefinedTensorElementType)?;

        Ok(onnx_data_type.into())
    }

    fn try_get_dimensions(
        type_and_shape_info: *mut sys::OrtTensorTypeAndShapeInfo,
    ) -> Result<Vec<i64>> {
        let mut num_dims = 0;
        let status =
            unsafe { g_ort().GetDimensionsCount.unwrap()(type_and_shape_info, &mut num_dims) };
        status_to_result(status).map_err(OrtError::GetDimensionsCount)?;
        (num_dims != 0)
            .then_some(())
            .ok_or(OrtError::InvalidDimensions)?;

        let mut dimensions: Vec<i64> = vec![0; num_dims];
        let status = unsafe {
            g_ort().GetDimensions.unwrap()(type_and_shape_info, dimensions.as_mut_ptr(), num_dims)
        };
        status_to_result(status).map_err(OrtError::GetDimensions)?;

        Ok(dimensions)
    }
}

impl Drop for OrtTensorTypeAndShapeInfo {
    #[tracing::instrument]
    fn drop(&mut self) {
        if self.ptr.is_null() {
            error!("OrtTensorTypeAndShapeInfo pointer is null, not dropping.");
        } else {
            trace!("Dropping OrtTensorTypeAndShapeInfo: {:?}.", self.ptr);
            unsafe { g_ort().ReleaseTensorTypeAndShapeInfo.unwrap()(self.ptr) };
        }

        self.ptr = std::ptr::null_mut();
    }
}
