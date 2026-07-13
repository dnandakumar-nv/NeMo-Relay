// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::fmt;
use std::io::{self, Write};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, Ordering};

#[derive(Debug)]
pub(super) struct TokenFileError {
    code: &'static str,
    message: String,
}

impl TokenFileError {
    fn new(code: &'static str, message: impl Into<String>) -> Self {
        Self {
            code,
            message: message.into(),
        }
    }

    pub(super) const fn code(&self) -> &'static str {
        self.code
    }
}

impl fmt::Display for TokenFileError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(&self.message)
    }
}

pub(super) struct TransientTokenFile {
    path: PathBuf,
    present: AtomicBool,
}

impl TransientTokenFile {
    pub(super) fn preflight(path: &Path) -> Result<(), TokenFileError> {
        if path.as_os_str().is_empty() {
            return Err(TokenFileError::new(
                "invalid_token_file",
                "token-file path must not be empty",
            ));
        }
        match std::fs::symlink_metadata(path) {
            Ok(metadata) => {
                let kind = if metadata.file_type().is_symlink() {
                    "a symbolic link"
                } else if metadata.is_dir() {
                    "a directory"
                } else if metadata.is_file() {
                    "an existing file"
                } else {
                    "a non-regular filesystem object"
                };
                Err(TokenFileError::new(
                    "token_file_exists",
                    format!("token-file destination {} is {kind}", path.display()),
                ))
            }
            Err(error) if error.kind() == io::ErrorKind::NotFound => {
                let parent = path
                    .parent()
                    .filter(|parent| !parent.as_os_str().is_empty());
                if parent.is_some_and(|parent| !parent.is_dir()) {
                    return Err(TokenFileError::new(
                        "invalid_token_file",
                        format!(
                            "token-file parent {} is not an existing directory",
                            parent.unwrap().display()
                        ),
                    ));
                }
                Ok(())
            }
            Err(error) => Err(TokenFileError::new(
                "invalid_token_file",
                format!("cannot inspect token-file destination: {error}"),
            )),
        }
    }

    pub(super) fn create(path: PathBuf, launch_url: &str) -> Result<Self, TokenFileError> {
        Self::preflight(&path)?;
        let mut file = platform::create_new(&path).map_err(|error| {
            let code = if error.kind() == io::ErrorKind::AlreadyExists {
                "token_file_exists"
            } else {
                "token_file_create_failed"
            };
            TokenFileError::new(code, format!("cannot create token file: {error}"))
        })?;
        let write_result = (|| {
            file.write_all(launch_url.as_bytes())?;
            file.write_all(b"\n")?;
            file.flush()?;
            file.sync_all()?;
            platform::verify(&path, &file)
        })();
        if let Err(error) = write_result {
            drop(file);
            let _ = std::fs::remove_file(&path);
            return Err(TokenFileError::new(
                "token_file_write_failed",
                format!("cannot write owner-only token file: {error}"),
            ));
        }
        drop(file);
        Ok(Self {
            path,
            present: AtomicBool::new(true),
        })
    }

    pub(super) fn cleanup(&self) -> Result<(), TokenFileError> {
        if !self.present.swap(false, Ordering::AcqRel) {
            return Ok(());
        }
        match std::fs::remove_file(&self.path) {
            Ok(()) => Ok(()),
            Err(error) if error.kind() == io::ErrorKind::NotFound => Ok(()),
            Err(error) => {
                self.present.store(true, Ordering::Release);
                Err(TokenFileError::new(
                    "token_file_cleanup_failed",
                    format!("cannot remove token file: {error}"),
                ))
            }
        }
    }

    #[cfg(test)]
    pub(super) fn path(&self) -> &Path {
        &self.path
    }
}

impl fmt::Debug for TransientTokenFile {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("TransientTokenFile")
            .field("path", &self.path)
            .field("present", &self.present.load(Ordering::Acquire))
            .finish()
    }
}

impl Drop for TransientTokenFile {
    fn drop(&mut self) {
        let _ = self.cleanup();
    }
}

#[cfg(unix)]
mod platform {
    use std::fs::{File, OpenOptions, Permissions};
    use std::io;
    use std::os::unix::fs::{OpenOptionsExt, PermissionsExt};
    use std::path::Path;

    pub(super) fn create_new(path: &Path) -> io::Result<File> {
        OpenOptions::new()
            .write(true)
            .create_new(true)
            .mode(0o600)
            .open(path)
    }

    pub(super) fn verify(path: &Path, file: &File) -> io::Result<()> {
        file.set_permissions(Permissions::from_mode(0o600))?;
        let metadata = std::fs::symlink_metadata(path)?;
        if metadata.file_type().is_symlink()
            || !metadata.is_file()
            || metadata.permissions().mode() & 0o077 != 0
        {
            return Err(io::Error::new(
                io::ErrorKind::PermissionDenied,
                "token file is not an owner-only regular file",
            ));
        }
        Ok(())
    }
}

#[cfg(windows)]
mod platform {
    use std::fs::{File, OpenOptions};
    use std::io;
    use std::mem;
    use std::os::windows::io::AsRawHandle;
    use std::path::Path;
    use std::ptr;

    use windows_sys::Win32::Foundation::{CloseHandle, ERROR_SUCCESS, HANDLE};
    use windows_sys::Win32::Security::Authorization::{SE_FILE_OBJECT, SetSecurityInfo};
    use windows_sys::Win32::Security::{
        ACCESS_ALLOWED_ACE, ACL, ACL_REVISION, AddAccessAllowedAceEx, CopySid,
        DACL_SECURITY_INFORMATION, GetLengthSid, GetTokenInformation, InitializeAcl,
        PROTECTED_DACL_SECURITY_INFORMATION, PSID, TOKEN_QUERY, TOKEN_USER, TokenUser,
    };
    use windows_sys::Win32::Storage::FileSystem::FILE_ALL_ACCESS;
    use windows_sys::Win32::System::Threading::{GetCurrentProcess, OpenProcessToken};

    struct OwnedHandle(HANDLE);

    impl Drop for OwnedHandle {
        fn drop(&mut self) {
            unsafe {
                CloseHandle(self.0);
            }
        }
    }

    struct SidBuffer(Vec<usize>);

    impl SidBuffer {
        fn as_ptr(&self) -> PSID {
            self.0.as_ptr().cast_mut().cast()
        }
    }

    pub(super) fn create_new(path: &Path) -> io::Result<File> {
        let file = OpenOptions::new().write(true).create_new(true).open(path)?;
        apply_owner_only_acl(&file)?;
        Ok(file)
    }

    pub(super) fn verify(path: &Path, _file: &File) -> io::Result<()> {
        let metadata = std::fs::symlink_metadata(path)?;
        if metadata.file_type().is_symlink() || !metadata.is_file() {
            return Err(io::Error::new(
                io::ErrorKind::PermissionDenied,
                "token file is not a regular file",
            ));
        }
        Ok(())
    }

    fn apply_owner_only_acl(file: &File) -> io::Result<()> {
        let sid = current_process_sid()?;
        let sid_length = unsafe { GetLengthSid(sid.as_ptr()) } as usize;
        if sid_length == 0 {
            return Err(io::Error::last_os_error());
        }
        let acl_bytes = mem::size_of::<ACL>() + mem::size_of::<ACCESS_ALLOWED_ACE>()
            - mem::size_of::<u32>()
            + sid_length;
        let mut acl_words = zeroed_words(acl_bytes);
        let acl = acl_words.as_mut_ptr().cast::<ACL>();
        if unsafe { InitializeAcl(acl, acl_bytes as u32, ACL_REVISION) } == 0
            || unsafe { AddAccessAllowedAceEx(acl, ACL_REVISION, 0, FILE_ALL_ACCESS, sid.as_ptr()) }
                == 0
        {
            return Err(io::Error::last_os_error());
        }
        let status = unsafe {
            SetSecurityInfo(
                file.as_raw_handle().cast(),
                SE_FILE_OBJECT,
                DACL_SECURITY_INFORMATION | PROTECTED_DACL_SECURITY_INFORMATION,
                ptr::null_mut(),
                ptr::null_mut(),
                acl,
                ptr::null(),
            )
        };
        if status == ERROR_SUCCESS {
            Ok(())
        } else {
            Err(io::Error::from_raw_os_error(status as i32))
        }
    }

    fn current_process_sid() -> io::Result<SidBuffer> {
        let mut token = ptr::null_mut();
        if unsafe { OpenProcessToken(GetCurrentProcess(), TOKEN_QUERY, &mut token) } == 0 {
            return Err(io::Error::last_os_error());
        }
        let token = OwnedHandle(token);
        let mut bytes = 0;
        unsafe {
            GetTokenInformation(token.0, TokenUser, ptr::null_mut(), 0, &mut bytes);
        }
        if bytes == 0 {
            return Err(io::Error::last_os_error());
        }
        let mut token_words = zeroed_words(bytes as usize);
        if unsafe {
            GetTokenInformation(
                token.0,
                TokenUser,
                token_words.as_mut_ptr().cast(),
                bytes,
                &mut bytes,
            )
        } == 0
        {
            return Err(io::Error::last_os_error());
        }
        let source = unsafe { &*token_words.as_ptr().cast::<TOKEN_USER>() }
            .User
            .Sid;
        let sid_bytes = unsafe { GetLengthSid(source) };
        if sid_bytes == 0 {
            return Err(io::Error::last_os_error());
        }
        let mut words = zeroed_words(sid_bytes as usize);
        if unsafe { CopySid(sid_bytes, words.as_mut_ptr().cast(), source) } == 0 {
            return Err(io::Error::last_os_error());
        }
        Ok(SidBuffer(words))
    }

    fn zeroed_words(bytes: usize) -> Vec<usize> {
        vec![0; bytes.div_ceil(mem::size_of::<usize>())]
    }
}

#[cfg(not(any(unix, windows)))]
mod platform {
    use std::fs::{File, OpenOptions};
    use std::io;
    use std::path::Path;

    pub(super) fn create_new(path: &Path) -> io::Result<File> {
        OpenOptions::new().write(true).create_new(true).open(path)
    }

    pub(super) fn verify(path: &Path, _file: &File) -> io::Result<()> {
        let metadata = std::fs::symlink_metadata(path)?;
        if metadata.file_type().is_symlink() || !metadata.is_file() {
            return Err(io::Error::new(
                io::ErrorKind::PermissionDenied,
                "token file is not a regular file",
            ));
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn token_file_is_create_new_owner_only_and_idempotently_removed() {
        let temporary = tempfile::tempdir().unwrap();
        let path = temporary.path().join("dashboard.token");
        let url = "http://127.0.0.1:4040/#bootstrap=not-a-real-credential";
        let token = TransientTokenFile::create(path.clone(), url).unwrap();
        assert_eq!(token.path(), path);
        assert_eq!(std::fs::read_to_string(&path).unwrap(), format!("{url}\n"));
        assert!(!format!("{token:?}").contains("not-a-real-credential"));
        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt as _;
            assert_eq!(
                std::fs::metadata(&path).unwrap().permissions().mode() & 0o777,
                0o600
            );
        }
        token.cleanup().unwrap();
        token.cleanup().unwrap();
        assert!(!path.exists());
    }

    #[test]
    fn existing_and_nonregular_destinations_are_refused_without_changes() {
        let temporary = tempfile::tempdir().unwrap();
        let existing = temporary.path().join("existing");
        std::fs::write(&existing, "keep").unwrap();
        assert_eq!(
            TransientTokenFile::create(existing.clone(), "secret")
                .unwrap_err()
                .code(),
            "token_file_exists"
        );
        assert_eq!(std::fs::read_to_string(existing).unwrap(), "keep");
        assert_eq!(
            TransientTokenFile::create(temporary.path().to_path_buf(), "secret")
                .unwrap_err()
                .code(),
            "token_file_exists"
        );
    }

    #[cfg(unix)]
    #[test]
    fn symbolic_link_destination_is_refused_without_touching_its_target() {
        use std::os::unix::fs::symlink;

        let temporary = tempfile::tempdir().unwrap();
        let target = temporary.path().join("target");
        let link = temporary.path().join("link");
        std::fs::write(&target, "keep").unwrap();
        symlink(&target, &link).unwrap();
        assert_eq!(
            TransientTokenFile::create(link, "secret")
                .unwrap_err()
                .code(),
            "token_file_exists"
        );
        assert_eq!(std::fs::read_to_string(target).unwrap(), "keep");
    }
}
