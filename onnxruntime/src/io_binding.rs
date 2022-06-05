//! Module containing IoBinding.

use crate::{
    error::{assert_not_null_pointer, status_to_result, OrtError, Result},
    g_ort,
    memory::MemoryInfo,
    session::Session,
    tensor::ort_owned_tensor::OrtOwnedTensorExtractor,
    OrtTensor, TypedArray, TypedOrtOwnedTensor, TypedOrtTensor,
};
use onnxruntime_sys as sys;
use std::collections::HashMap;
use std::ffi::{CStr, CString};
use std::fmt::Debug;
use tracing::{error, trace};

/// IoBinding used to declare memory device of input and output tensors
#[derive(Debug)]
pub struct IoBinding<'a, 'b, D>
where
    D: ndarray::Dimension,
    'a: 'b,
{
    pub(crate) ptr: *mut sys::OrtIoBinding,
    session: &'b Session<'a>,
    inputs: HashMap<String, TypedOrtTensor<'b, D>>,
}

impl<'a, 'b, D> IoBinding<'a, 'b, D>
where
    D: ndarray::Dimension,
{
    /// Create a new io_binding instance
    #[tracing::instrument]
    pub(crate) unsafe fn new(session: &'b Session<'a>) -> Result<Self>
    where
        D: ndarray::Dimension,
    {
        trace!("Creating new io_binding.");
        let mut io_binding_ptr: *mut sys::OrtIoBinding = std::ptr::null_mut();
        let status = g_ort().CreateIoBinding.unwrap()(session.ptr, &mut io_binding_ptr);
        status_to_result(status).map_err(OrtError::CreateIoBinding)?;
        assert_not_null_pointer(io_binding_ptr, "IoBinding")?;

        Ok(Self {
            ptr: io_binding_ptr,
            session,
            inputs: HashMap::new(),
        })
    }

    /// Bind an ::OrtValue to an ::OrtIoBinding input
    #[tracing::instrument]
    pub fn bind_input<S>(&mut self, name: S, input_tensor: TypedArray<D>) -> Result<()>
    where
        S: Into<String> + Debug,
        D: ndarray::Dimension,
    {
        trace!("Binding input.");
        let name = name.into();
        let cname = CString::new(name.clone()).unwrap();

        // The C API expects pointers for the arrays (pointers to C-arrays)
        let input_ort_tensor: TypedOrtTensor<D> = match input_tensor {
            TypedArray::F32(input_array) => OrtTensor::from_array(
                &self.session.memory_info,
                self.session.allocator_ptr,
                input_array,
            )
            .map(|t| TypedOrtTensor::F32(t)),
            TypedArray::U8(input_array) => OrtTensor::from_array(
                &self.session.memory_info,
                self.session.allocator_ptr,
                input_array,
            )
            .map(|t| TypedOrtTensor::U8(t)),
            TypedArray::I8(input_array) => OrtTensor::from_array(
                &self.session.memory_info,
                self.session.allocator_ptr,
                input_array,
            )
            .map(|t| TypedOrtTensor::I8(t)),
            TypedArray::U16(input_array) => OrtTensor::from_array(
                &self.session.memory_info,
                self.session.allocator_ptr,
                input_array,
            )
            .map(|t| TypedOrtTensor::U16(t)),
            TypedArray::I16(input_array) => OrtTensor::from_array(
                &self.session.memory_info,
                self.session.allocator_ptr,
                input_array,
            )
            .map(|t| TypedOrtTensor::I16(t)),
            TypedArray::I32(input_array) => OrtTensor::from_array(
                &self.session.memory_info,
                self.session.allocator_ptr,
                input_array,
            )
            .map(|t| TypedOrtTensor::I32(t)),
            TypedArray::I64(input_array) => OrtTensor::from_array(
                &self.session.memory_info,
                self.session.allocator_ptr,
                input_array,
            )
            .map(|t| TypedOrtTensor::I64(t)),
            TypedArray::F64(input_array) => OrtTensor::from_array(
                &self.session.memory_info,
                self.session.allocator_ptr,
                input_array,
            )
            .map(|t| TypedOrtTensor::F64(t)),
            TypedArray::U32(input_array) => OrtTensor::from_array(
                &self.session.memory_info,
                self.session.allocator_ptr,
                input_array,
            )
            .map(|t| TypedOrtTensor::U32(t)),
            TypedArray::U64(input_array) => OrtTensor::from_array(
                &self.session.memory_info,
                self.session.allocator_ptr,
                input_array,
            )
            .map(|t| TypedOrtTensor::U64(t)),
        }?;

        self.inputs.insert(name.clone(), input_ort_tensor);
        let input_ort_tensor = self.inputs.get(&name).unwrap();

        let input_ort_value: *const sys::OrtValue = match input_ort_tensor {
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
        };

        let status =
            unsafe { g_ort().BindInput.unwrap()(self.ptr, cname.as_ptr(), input_ort_value) };
        status_to_result(status).map_err(OrtError::BindInput)?;

        Ok(())
    }

    /// Bind an ::OrtIoBinding output to a device
    #[tracing::instrument]
    pub fn bind_output<S>(&mut self, name: S, mem_info: MemoryInfo) -> Result<()>
    where
        S: Into<String> + Debug,
        D: ndarray::Dimension,
    {
        trace!("Binding output.");
        let cname = CString::new(name.into()).unwrap();

        let status =
            unsafe { g_ort().BindOutputToDevice.unwrap()(self.ptr, cname.as_ptr(), mem_info.ptr) };
        status_to_result(status).map_err(OrtError::BindOutputToDevice)?;

        Ok(())
    }

    /// Bind an ::OrtIoBinding output to a device
    #[tracing::instrument]
    pub fn copy_outputs_to_cpu(
        &self,
    ) -> Result<HashMap<String, TypedOrtOwnedTensor<ndarray::Dim<ndarray::IxDynImpl>>>> {
        trace!("Copying outputs to CPU.");

        let mut output_names_ptr = Vec::new().as_mut_ptr();
        let mut lengths: Vec<usize> = Vec::new();
        let mut lengths_ptr = lengths.as_mut_ptr();
        let mut count = 0;

        let status = unsafe {
            g_ort().GetBoundOutputNames.unwrap()(
                self.ptr,
                self.session.allocator_ptr,
                &mut output_names_ptr,
                &mut lengths_ptr,
                &mut count,
            )
        };
        status_to_result(status).map_err(OrtError::GetBoundOutputNames)?;
        assert_not_null_pointer(output_names_ptr, "GetBoundOutputNames")?;

        let lengths = unsafe { std::slice::from_raw_parts(lengths_ptr, count).to_vec() };
        let output_names_cstr = unsafe { CStr::from_ptr(output_names_ptr) }.to_string_lossy();
        let mut output_names_chars = output_names_cstr.chars();
        let output_names = lengths
            .iter()
            .map(|length| {
                output_names_chars
                    .by_ref()
                    .take(*length)
                    .collect::<String>()
            })
            .collect::<Vec<_>>();

        let mut output_values_ptr: *mut *mut sys::OrtValue =
            vec![std::ptr::null_mut(); count].as_mut_ptr();

        let status = unsafe {
            g_ort().GetBoundOutputValues.unwrap()(
                self.ptr,
                self.session.allocator_ptr,
                &mut output_values_ptr,
                &mut count,
            )
        };
        status_to_result(status).map_err(OrtError::GetBoundOutputValues)?;
        assert_not_null_pointer(output_values_ptr, "GetBoundOutputValues")?;

        let mut output_values_ptr: Vec<*mut sys::OrtValue> = vec![std::ptr::null_mut(); count];
        let status = unsafe {
            g_ort().CopyOutputsAcrossDevices.unwrap()(
                self.ptr,
                count,
                output_values_ptr.as_mut_ptr(),
            )
        };
        status_to_result(status).map_err(OrtError::CopyOutputsAcrossDevices)?;
        assert_not_null_pointer(output_names_ptr, "CopyOutputsAcrossDevices")?;

        let outputs = output_values_ptr
            .iter()
            .zip(output_names)
            .map(|(ptr, output_name)| {
                let mut tensor_info_ptr: *mut sys::OrtTensorTypeAndShapeInfo = std::ptr::null_mut();
                let status = unsafe {
                    g_ort().GetTensorTypeAndShape.unwrap()(*ptr, &mut tensor_info_ptr as _)
                };
                status_to_result(status).map_err(OrtError::GetTensorTypeAndShape)?;

                let dims = crate::session::get_tensor_dimensions(tensor_info_ptr)?;
                unsafe { g_ort().ReleaseTensorTypeAndShapeInfo.unwrap()(tensor_info_ptr) };

                let dims: Vec<_> = dims.iter().map(|&n| n as usize).collect();

                let output = self.session.outputs.get(&output_name).unwrap();

                let mut output_tensor_extractor =
                    OrtOwnedTensorExtractor::new(&self.session.memory_info, ndarray::IxDyn(&dims));
                output_tensor_extractor.ptr = *ptr;
                Ok((output_name, output_tensor_extractor.extract(output)?))
            })
            .collect::<Result<HashMap<String, TypedOrtOwnedTensor<_>>>>()?;

        Ok(outputs)
    }
}

impl<'a, 'b, D> Drop for IoBinding<'a, 'b, D>
where
    D: ndarray::Dimension,
{
    #[tracing::instrument]
    fn drop(&mut self) {
        if self.ptr.is_null() {
            error!("IoBinding pointer is null, not dropping.");
        } else {
            trace!("Dropping IoBinding.");
            unsafe { g_ort().ReleaseIoBinding.unwrap()(self.ptr) };
        }

        self.ptr = std::ptr::null_mut();
    }
}
