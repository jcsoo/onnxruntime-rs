//! Module abstracting OrtIoBinding.

use crate::{
    assert_not_null_pointer,
    error::{OrtError, Result},
    g_ort,
    memory_info::MemoryInfo,
    status_to_result, Session, Value,
};
use onnxruntime_sys as sys;
use std::ffi::CString;
use std::fmt::Debug;
use std::mem::ManuallyDrop;
use std::{collections::HashMap, os::raw::c_char};
use tracing::{error, trace};

/// IoBinding used to declare memory device of input and output tensors
#[derive(Debug)]
pub struct IoBinding<'s, 'i>
where
    's: 'i, // 's outlives 'i (session outlives io_binding)
{
    pub(crate) ptr: *mut sys::OrtIoBinding,
    session: &'i Session<'s>,
}

impl<'s, 'i> IoBinding<'s, 'i> {
    /// Create a new io_binding instance
    #[tracing::instrument]
    pub(crate) unsafe fn new(session: &'i Session<'s>) -> Result<Self> {
        let mut ptr: *mut sys::OrtIoBinding = std::ptr::null_mut();
        let status = g_ort().CreateIoBinding.unwrap()(session.ptr, &mut ptr);
        status_to_result(status).map_err(OrtError::CreateIoBinding)?;
        assert_not_null_pointer(ptr, "IoBinding")?;

        trace!("Created IoBinding: {ptr:?}.");
        Ok(Self { ptr, session })
    }

    /// Bind an ::OrtValue to an ::OrtIoBinding input
    #[tracing::instrument]
    #[allow(clippy::not_unsafe_ptr_arg_deref)]
    pub fn bind_input<S>(&mut self, name: S, ort_value: &Value) -> Result<()>
    where
        S: Into<String> + Clone + Debug,
    {
        let name = name.into();
        trace!("Binding Input '{name}': {ort_value:?}.");
        let cname = CString::new(name).unwrap();

        let status = unsafe { g_ort().BindInput.unwrap()(self.ptr, cname.as_ptr(), **ort_value) };
        status_to_result(status).map_err(OrtError::BindInput)?;

        Ok(())
    }

    /// Bind an ::OrtIoBinding output to a device
    #[tracing::instrument]
    pub fn bind_output<S>(&mut self, name: S, mem_info: MemoryInfo) -> Result<()>
    where
        S: Into<String> + Clone + Debug,
    {
        let name = name.into();
        trace!("Binding Output '{name}': {mem_info:?}.");
        let cname = CString::new(name).unwrap();

        let status =
            unsafe { g_ort().BindOutputToDevice.unwrap()(self.ptr, cname.as_ptr(), mem_info.ptr) };
        status_to_result(status).map_err(OrtError::BindOutputToDevice)?;

        Ok(())
    }

    /// Retrieve the outputs of the ::OrtIoBinding as OrtValue
    #[tracing::instrument]
    pub fn outputs(&self) -> Result<HashMap<String, Value>> {
        // get keys
        let mut output_names_ptr: *mut c_char = std::ptr::null_mut();
        let mut lengths: Vec<usize> = Vec::new();
        let mut lengths_ptr = lengths.as_mut_ptr();
        let mut count = 0;

        let status = unsafe {
            g_ort().GetBoundOutputNames.unwrap()(
                self.ptr,
                self.session.allocator.ptr,
                &mut output_names_ptr,
                &mut lengths_ptr,
                &mut count,
            )
        };
        status_to_result(status).map_err(OrtError::GetBoundOutputNames)?;
        assert_not_null_pointer(output_names_ptr, "GetBoundOutputNames")?;

        if count == 0 {
            return Ok(HashMap::new());
        }

        let lengths = unsafe { std::slice::from_raw_parts(lengths_ptr, count).to_vec() };
        let output_names = unsafe {
            ManuallyDrop::new(String::from_raw_parts(
                output_names_ptr as *mut u8,
                lengths.iter().sum(),
                lengths.iter().sum(),
            ))
        };
        let mut output_names_chars = output_names.chars();

        let output_names = lengths
            .into_iter()
            .map(|length| output_names_chars.by_ref().take(length).collect::<String>())
            .collect::<Vec<_>>();

        let status = unsafe {
            g_ort().AllocatorFree.unwrap()(
                self.session.allocator.ptr,
                output_names_ptr as *mut std::ffi::c_void,
            )
        };
        status_to_result(status).map_err(OrtError::AllocatorFree)?;

        // get values
        let mut output_values_ptr: *mut *mut sys::OrtValue =
            vec![std::ptr::null_mut(); count].as_mut_ptr();

        let status = unsafe {
            g_ort().GetBoundOutputValues.unwrap()(
                self.ptr,
                self.session.allocator.ptr,
                &mut output_values_ptr,
                &mut count,
            )
        };

        status_to_result(status).map_err(OrtError::GetBoundOutputValues)?;
        assert_not_null_pointer(output_values_ptr, "GetBoundOutputValues")?;

        let output_values_ptr =
            unsafe { std::slice::from_raw_parts(output_values_ptr, count).to_vec() }
                .into_iter()
                .map(Value::from);

        Ok(output_names
            .into_iter()
            .zip(output_values_ptr)
            .collect::<HashMap<_, _>>())
    }
}

impl<'s, 'i> Drop for IoBinding<'s, 'i> {
    #[tracing::instrument]
    fn drop(&mut self) {
        if self.ptr.is_null() {
            error!("IoBinding pointer is null, not dropping.");
        } else {
            trace!("Dropping IoBinding: {:?}.", self.ptr);
            unsafe { g_ort().ReleaseIoBinding.unwrap()(self.ptr) };
        }

        self.ptr = std::ptr::null_mut();
    }
}
