//! Module containing MemoryInfo.

use crate::{
    error::{assert_not_null_pointer, status_to_result, OrtError, Result},
    g_ort, AllocatorType, DeviceName, MemType,
};
use onnxruntime_sys as sys;
use std::ffi::{CStr, CString};
use std::fmt::Debug;
use std::os::raw::c_char;
use tracing::{error, trace};

#[derive(Debug)]
/// MemoryInfo
pub struct MemoryInfo {
    pub(crate) ptr: *mut sys::OrtMemoryInfo,
    id: i32,
    name: DeviceName,
    allocator_type: AllocatorType,
    mem_type: MemType,
}

impl MemoryInfo {
    #[tracing::instrument]
    /// Create new MemoryInfo
    pub fn new(
        name: DeviceName,
        id: i32,
        allocator_type: AllocatorType,
        mem_type: MemType,
    ) -> Result<Self> {
        trace!("Creating MemoryInfo.");

        let mut memory_info_ptr: *mut sys::OrtMemoryInfo = std::ptr::null_mut();

        let status = unsafe {
            g_ort().CreateMemoryInfo.unwrap()(
                CString::from(name.clone()).as_ptr(),
                allocator_type.clone().into(),
                id,
                mem_type.clone().into(),
                &mut memory_info_ptr,
            )
        };
        status_to_result(status).map_err(OrtError::CreateMemoryInfo)?;
        assert_not_null_pointer(memory_info_ptr, "MemoryInfo")?;

        Ok(Self {
            ptr: memory_info_ptr,
            id,
            name,
            allocator_type,
            mem_type,
        })
    }

    /// gets the AllocatorType from MemoryInfo
    pub fn allocator_type(&self) -> &AllocatorType {
        &self.allocator_type
    }

    /// gets the id from MemoryInfo
    pub fn id(&self) -> i32 {
        self.id
    }

    /// gets the MemType from MemoryInfo
    pub fn mem_type(&self) -> &MemType {
        &self.mem_type
    }

    /// gets the name from MemoryInfo
    pub fn name(&self) -> &DeviceName {
        &self.name
    }
}

#[allow(dead_code)]
unsafe fn get_allocator_type(memory_info_ptr: *mut sys::OrtMemoryInfo) -> Result<AllocatorType> {
    let mut allocator_type = sys::OrtAllocatorType::OrtInvalidAllocator;
    let status = g_ort().MemoryInfoGetType.unwrap()(memory_info_ptr, &mut allocator_type);
    status_to_result(status).map_err(OrtError::MemoryInfoGetType)?;
    Ok(allocator_type.into())
}

#[allow(dead_code)]
unsafe fn get_id(memory_info_ptr: *mut sys::OrtMemoryInfo) -> Result<i32> {
    let mut id = 0;
    let status = g_ort().MemoryInfoGetId.unwrap()(memory_info_ptr, &mut id);
    status_to_result(status).map_err(OrtError::MemoryInfoGetId)?;
    Ok(id)
}

#[allow(dead_code)]
unsafe fn get_mem_type(memory_info_ptr: *mut sys::OrtMemoryInfo) -> Result<MemType> {
    let mut mem_type = sys::OrtMemType::OrtMemTypeDefault;
    let status = g_ort().MemoryInfoGetMemType.unwrap()(memory_info_ptr, &mut mem_type);
    status_to_result(status).map_err(OrtError::MemoryInfoGetMemType)?;
    Ok(mem_type.into())
}

#[allow(dead_code)]
unsafe fn get_name(memory_info_ptr: *mut sys::OrtMemoryInfo) -> Result<String> {
    let mut name_ptr: *const c_char = std::ptr::null_mut();
    let status = g_ort().MemoryInfoGetName.unwrap()(memory_info_ptr, &mut name_ptr);
    status_to_result(status).map_err(OrtError::MemoryInfoGetName)?;
    assert_not_null_pointer(name_ptr, "Name")?;
    Ok(CStr::from_ptr(name_ptr).to_string_lossy().into_owned())
}

impl Drop for MemoryInfo {
    #[tracing::instrument]
    fn drop(&mut self) {
        if self.ptr.is_null() {
            error!("MemoryInfo pointer is null, not dropping.");
        } else {
            trace!("Dropping MemoryInfo.");
            unsafe { g_ort().ReleaseMemoryInfo.unwrap()(self.ptr) };
        }

        self.ptr = std::ptr::null_mut();
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use test_log::test;

    #[test]
    fn memory_info_constructor_destructor() {
        let memory_info =
            MemoryInfo::new(DeviceName::Cpu, 0, AllocatorType::Arena, MemType::Default).unwrap();
        std::mem::drop(memory_info);
    }
}
