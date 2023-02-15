//! Module containing environment types

use std::{
    ffi::CString,
    sync::{atomic::AtomicPtr, Arc, Mutex},
};

use lazy_static::lazy_static;
use tracing::{debug, error, info, trace, warn};

use onnxruntime_sys as sys;

use crate::{
    error::{status_to_result, OrtError, Result},
    g_ort,
    onnxruntime::custom_logger,
    session::SessionBuilder,
    LoggingLevel,
};

lazy_static! {
    static ref G_ENV: Arc<Mutex<EnvironmentSingleton>> =
        Arc::new(Mutex::new(EnvironmentSingleton {
            name: String::from("uninitialized"),
            env_ptr: AtomicPtr::new(std::ptr::null_mut()),
        }));
}

#[derive(Debug)]
struct EnvironmentSingleton {
    name: String,
    env_ptr: AtomicPtr<sys::OrtEnv>,
}

/// An [`Environment`](session/struct.Environment.html) is the main entry point of the ONNX Runtime.
///
/// Only one ONNX environment can be created per process. The `onnxruntime` crate
/// uses a singleton (through `lazy_static!()`) to enforce this.
///
/// Once an environment is created, a [`Session`](../session/struct.Session.html)
/// can be obtained from it.
///
/// **NOTE**: While the [`Environment`](environment/struct.Environment.html) constructor takes a `name` parameter
/// to name the environment, only the first name will be considered if many environments
/// are created.
///
/// # Example
///
/// ```no_run
/// # use std::error::Error;
/// # use onnxruntime::{environment::Environment, LoggingLevel};
/// # fn main() -> Result<(), Box<dyn Error>> {
/// let environment = Environment::builder()
///     .with_name("test")
///     .with_log_level(LoggingLevel::Verbose)
///     .build()?;
/// # Ok(())
/// # }
/// ```
#[derive(Debug, Clone)]
pub struct Environment {
    env: Arc<Mutex<EnvironmentSingleton>>,
}

impl Environment {
    /// Create a new environment builder using default values
    /// (name: `default`, log level: [LoggingLevel::Warning](../enum.LoggingLevel.html#variant.Warning))
    pub fn builder() -> EnvBuilder {
        EnvBuilder {
            name: "default".into(),
            log_level: LoggingLevel::Warning,
            global_inter_op_num_threads: None,
            global_intra_op_num_threads: None,
            global_spin_control: None,
        }
    }

    /// Return the name of the current environment
    pub fn name(&self) -> String {
        self.env.lock().unwrap().name.to_string()
    }

    pub(crate) fn env_ptr(&self) -> *const sys::OrtEnv {
        *self.env.lock().unwrap().env_ptr.get_mut()
    }

    #[tracing::instrument]
    fn new(builder: &EnvBuilder) -> Result<Environment> {
        // NOTE: Because 'G_ENV' is a lazy_static, locking it will, initially, create
        //      a new Arc<Mutex<EnvironmentSingleton>> with a strong count of 1.
        //      Cloning it to embed it inside the 'Environment' to return
        //      will thus increase the strong count to 2.
        let mut environment_guard = G_ENV
            .lock()
            .expect("Failed to acquire lock: another thread panicked?");
        let g_env_ptr = environment_guard.env_ptr.get_mut();
        if g_env_ptr.is_null() {
            debug!("Environment not yet initialized, creating a new one.");

            let mut threading_options_ptr: *mut sys::OrtThreadingOptions = std::ptr::null_mut();
            let status =
                { unsafe { g_ort().CreateThreadingOptions.unwrap()(&mut threading_options_ptr) } };
            status_to_result(status).map_err(OrtError::ThreadingOptions)?;

            if let Some(inter_op_num_threads) = builder.global_inter_op_num_threads {
                let status = {
                    unsafe {
                        g_ort().SetGlobalInterOpNumThreads.unwrap()(
                            threading_options_ptr,
                            inter_op_num_threads as i32,
                        )
                    }
                };
                status_to_result(status).map_err(OrtError::ThreadingOptions)?;
            }
            if let Some(intra_op_num_threads) = builder.global_intra_op_num_threads {
                let status = {
                    unsafe {
                        g_ort().SetGlobalIntraOpNumThreads.unwrap()(
                            threading_options_ptr,
                            intra_op_num_threads as i32,
                        )
                    }
                };
                status_to_result(status).map_err(OrtError::ThreadingOptions)?;
            }
            if let Some(spin_control) = builder.global_spin_control {
                let status = {
                    unsafe {
                        g_ort().SetGlobalSpinControl.unwrap()(
                            threading_options_ptr,
                            i32::from(spin_control),
                        )
                    }
                };
                status_to_result(status).map_err(OrtError::ThreadingOptions)?;
            }

            let mut env_ptr: *mut sys::OrtEnv = std::ptr::null_mut();
            let logging_function: sys::OrtLoggingFunction = Some(custom_logger);
            let logger_param: *mut std::ffi::c_void = std::ptr::null_mut();
            let cname = CString::new(builder.name.clone()).unwrap();
            let status = {
                unsafe {
                    g_ort()
                        .CreateEnvWithCustomLoggerAndGlobalThreadPools
                        .unwrap()(
                        logging_function,
                        logger_param,
                        builder.log_level.clone().into(),
                        cname.as_ptr(),
                        threading_options_ptr,
                        &mut env_ptr,
                    )
                }
            };
            status_to_result(status).map_err(OrtError::Environment)?;

            // disable telemetry
            let status = { unsafe { g_ort().DisableTelemetryEvents.unwrap()(env_ptr) } };
            status_to_result(status).map_err(OrtError::Environment)?;

            debug!(
                env_ptr = format!("{env_ptr:?}").as_str(),
                "Environment created."
            );

            *g_env_ptr = env_ptr;
            environment_guard.name = builder.name.clone();

            // NOTE: Cloning the lazy_static 'G_ENV' will increase its strong count by one.
            //       If this 'Environment' is the only one in the process, the strong count
            //       will be 2:
            //          * one lazy_static 'G_ENV'
            //          * one inside the 'Environment' returned
            Ok(Environment { env: G_ENV.clone() })
        } else {
            info!(
                name = environment_guard.name.as_str(),
                env_ptr = format!("{:?}", environment_guard.env_ptr).as_str(),
                "Environment already initialized, reusing it.",
            );

            // NOTE: Cloning the lazy_static 'G_ENV' will increase its strong count by one.
            //       If this 'Environment' is the only one in the process, the strong count
            //       will be 2:
            //          * one lazy_static 'G_ENV'
            //          * one inside the 'Environment' returned
            Ok(Environment { env: G_ENV.clone() })
        }
    }

    /// Create a new [`SessionBuilder`](../session/struct.SessionBuilder.html)
    /// used to create a new ONNX session.
    pub fn new_session_builder(&self) -> Result<SessionBuilder> {
        SessionBuilder::new(self)
    }
}

impl Drop for Environment {
    #[tracing::instrument]
    fn drop(&mut self) {
        let mut environment_guard = self
            .env
            .lock()
            .expect("Failed to acquire lock: another thread panicked?");

        // NOTE: If we drop an 'Environment' we (obviously) have _at least_
        //       one 'G_ENV' strong count (the one in the 'env' member).
        //       There is also the "original" 'G_ENV' which is a the lazy_static global.
        //       If there is no other environment, the strong count should be two and we
        //       can properly free the sys::OrtEnv pointer.
        if Arc::strong_count(&G_ENV) == 2 {
            let ptr: *mut sys::OrtEnv = *environment_guard.env_ptr.get_mut();
            if ptr.is_null() {
                error!("Environment pointer is null, not dropping");
            } else {
                trace!("Dropping Environment: {:?}.", ptr);
                unsafe { g_ort().ReleaseEnv.unwrap()(ptr) };
            }

            environment_guard.env_ptr = AtomicPtr::new(std::ptr::null_mut());
            environment_guard.name = String::from("uninitialized");
        }
    }
}

/// Struct used to build an environment [`Environment`](environment/struct.Environment.html)
///
/// This is the crate's main entry point. An environment _must_ be created
/// as the first step. An [`Environment`](environment/struct.Environment.html) can only be built
/// using `EnvBuilder` to configure it.
///
/// **NOTE**: If the same configuration method (for example [`with_name()`](struct.EnvBuilder.html#method.with_name))
/// is called multiple times, the last value will have precedence.
#[derive(Debug)]
pub struct EnvBuilder {
    name: String,
    log_level: LoggingLevel,
    global_inter_op_num_threads: Option<u16>,
    global_intra_op_num_threads: Option<u16>,
    global_spin_control: Option<bool>,
}

impl EnvBuilder {
    /// Configure the environment with a given name
    ///
    /// **NOTE**: Since ONNX can only define one environment per process,
    /// creating multiple environments using multiple `EnvBuilder` will
    /// end up re-using the same environment internally; a new one will _not_
    /// be created. New parameters will be ignored.
    pub fn with_name<S>(mut self, name: S) -> EnvBuilder
    where
        S: Into<String>,
    {
        self.name = name.into();
        self
    }

    /// Configure the environment with a given log level
    ///
    /// **NOTE**: Since ONNX can only define one environment per process,
    /// creating multiple environments using multiple `EnvBuilder` will
    /// end up re-using the same environment internally; a new one will _not_
    /// be created. New parameters will be ignored.
    pub fn with_log_level(mut self, log_level: LoggingLevel) -> EnvBuilder {
        self.log_level = log_level;
        self
    }

    /// Set global inter-op thread count.
    pub fn with_global_inter_op_num_threads(mut self, inter_op_num_threads: u16) -> EnvBuilder {
        self.global_inter_op_num_threads = Some(inter_op_num_threads);
        self
    }

    /// Set global intra-op thread count.
    pub fn with_global_intra_op_num_threads(mut self, intra_op_num_threads: u16) -> EnvBuilder {
        self.global_intra_op_num_threads = Some(intra_op_num_threads);
        self
    }

    /// Set global spin control options.
    pub fn with_global_spin_control(mut self, spin_control: bool) -> EnvBuilder {
        self.global_spin_control = Some(spin_control);
        self
    }

    /// Commit the configuration to a new [`Environment`](environment/struct.Environment.html)
    pub fn build(self) -> Result<Environment> {
        Environment::new(&self)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::{RwLock, RwLockWriteGuard};

    impl G_ENV {
        fn is_initialized(&self) -> bool {
            Arc::strong_count(self) >= 2
        }

        fn env_ptr(&self) -> *const sys::OrtEnv {
            *self.lock().unwrap().env_ptr.get_mut()
        }
    }

    struct ConcurrentTestRun {
        lock: Arc<RwLock<()>>,
    }

    lazy_static! {
        static ref CONCURRENT_TEST_RUN: ConcurrentTestRun = ConcurrentTestRun {
            lock: Arc::new(RwLock::new(()))
        };
    }

    impl CONCURRENT_TEST_RUN {
        fn single_test_run(&self) -> RwLockWriteGuard<()> {
            self.lock.write().unwrap()
        }
    }

    #[test]
    fn env_is_initialized() {
        let _run_lock = CONCURRENT_TEST_RUN.single_test_run();

        assert!(!G_ENV.is_initialized());
        assert_eq!(G_ENV.env_ptr(), std::ptr::null_mut());

        let env = Environment::builder()
            .with_name("env_is_initialized")
            .with_log_level(LoggingLevel::Warning)
            .build()
            .unwrap();
        assert!(G_ENV.is_initialized());
        assert_ne!(G_ENV.env_ptr(), std::ptr::null_mut());

        std::mem::drop(env);
        assert!(!G_ENV.is_initialized());
        assert_eq!(G_ENV.env_ptr(), std::ptr::null_mut());
    }

    #[ignore]
    #[test]
    fn sequential_environment_creation() {
        let _concurrent_run_lock_guard = CONCURRENT_TEST_RUN.single_test_run();

        let mut prev_env_ptr = G_ENV.env_ptr();

        for i in 0..10 {
            let name = format!("sequential_environment_creation: {i}");
            let env = Environment::builder()
                .with_name(name.clone())
                .with_log_level(LoggingLevel::Warning)
                .build()
                .unwrap();
            let next_env_ptr = G_ENV.env_ptr();
            assert_ne!(next_env_ptr, prev_env_ptr);
            prev_env_ptr = next_env_ptr;

            assert_eq!(env.name(), name);
        }
    }

    #[test]
    fn concurrent_environment_creations() {
        let _concurrent_run_lock_guard = CONCURRENT_TEST_RUN.single_test_run();

        let initial_name = String::from("concurrent_environment_creation");
        let main_env = Environment::builder()
            .with_name(initial_name.clone())
            .with_log_level(LoggingLevel::Warning)
            .build()
            .unwrap();
        let main_env_ptr = main_env.env_ptr() as usize;

        let children = (0..10).map(|t| {
            let initial_name_cloned = initial_name.clone();
            std::thread::spawn(move || {
                let name = format!("concurrent_environment_creation: {t}");
                let env = Environment::builder()
                    .with_name(name)
                    .with_log_level(LoggingLevel::Warning)
                    .build()
                    .unwrap();

                assert_eq!(env.name(), initial_name_cloned);
                assert_eq!(env.env_ptr() as usize, main_env_ptr);
            })
        });

        assert_eq!(main_env.name(), initial_name);
        assert_eq!(main_env.env_ptr() as usize, main_env_ptr);

        let res = children.into_iter().map(|child| child.join());
        assert!(res.into_iter().all(|r| std::result::Result::is_ok(&r)));
    }
}
