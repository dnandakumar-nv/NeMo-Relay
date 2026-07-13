// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Secure filesystem boundary for the Router evidence ledger.

use std::error::Error;
use std::fmt;
use std::path::Path;

use rusqlite::{Connection, OpenFlags};

const SIDECAR_SUFFIXES: [&str; 3] = ["-wal", "-shm", "-journal"];

/// Stable class for a ledger filesystem failure.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum LedgerFsErrorKind {
    /// The path, directory, database, or sidecar failed structural validation.
    InvalidFilesystem,
    /// Owner-only access could not be established or verified.
    InvalidPermissions,
    /// SQLite could not safely open the secured database path.
    OpenFailed,
    /// The host cannot enforce the required filesystem security contract.
    #[cfg_attr(
        all(unix, not(target_os = "redox")),
        allow(dead_code, reason = "constructed by the fail-closed platform branches")
    )]
    UnsupportedSecurity,
}

/// Sanitized ledger filesystem error.
#[derive(Debug)]
pub(crate) struct LedgerFsError {
    kind: LedgerFsErrorKind,
}

impl LedgerFsError {
    const fn new(kind: LedgerFsErrorKind) -> Self {
        Self { kind }
    }

    fn io(kind: LedgerFsErrorKind, _error: std::io::Error) -> Self {
        Self::new(kind)
    }

    fn sqlite(_error: rusqlite::Error) -> Self {
        Self::new(LedgerFsErrorKind::OpenFailed)
    }

    /// Returns the stable failure class without path or operating-system text.
    pub(crate) const fn kind(&self) -> LedgerFsErrorKind {
        self.kind
    }
}

impl fmt::Display for LedgerFsError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(match self.kind {
            LedgerFsErrorKind::InvalidFilesystem => "ledger filesystem validation failed",
            LedgerFsErrorKind::InvalidPermissions => {
                "ledger filesystem permissions are not owner-only"
            }
            LedgerFsErrorKind::OpenFailed => "ledger database could not be opened",
            LedgerFsErrorKind::UnsupportedSecurity => {
                "ledger filesystem security is unsupported on this platform"
            }
        })
    }
}

impl Error for LedgerFsError {}

/// Opens a database only after establishing the owner-only filesystem boundary.
pub(crate) fn open_secure_connection(path: &Path) -> Result<Connection, LedgerFsError> {
    platform::open_secure_connection(path)
}

/// Opens an existing database read-only after validating the same filesystem boundary.
pub(crate) fn open_secure_read_connection(path: &Path) -> Result<Connection, LedgerFsError> {
    platform::open_secure_read_connection(path)
}

/// Revalidates and enforces owner-only access on existing SQLite sidecars.
///
/// Call this after enabling WAL and after migrations have materialized the WAL
/// and shared-memory files.
pub(crate) fn enforce_sidecar_permissions(path: &Path) -> Result<(), LedgerFsError> {
    platform::enforce_sidecar_permissions(path)
}

fn sqlite_open_flags() -> OpenFlags {
    OpenFlags::SQLITE_OPEN_READ_WRITE
        | OpenFlags::SQLITE_OPEN_CREATE
        | OpenFlags::SQLITE_OPEN_NO_MUTEX
        | OpenFlags::SQLITE_OPEN_NOFOLLOW
}

fn sqlite_read_open_flags() -> OpenFlags {
    OpenFlags::SQLITE_OPEN_READ_ONLY
        | OpenFlags::SQLITE_OPEN_NO_MUTEX
        | OpenFlags::SQLITE_OPEN_NOFOLLOW
}

#[cfg(all(unix, not(target_os = "redox")))]
mod platform {
    use std::ffi::{OsStr, OsString};
    use std::os::fd::OwnedFd;
    use std::path::{Component, Path, PathBuf};

    use rusqlite::Connection;
    use rustix::fs::{self, FileType, Mode, OFlags, Stat};
    use rustix::io::Errno;
    use rustix::process::geteuid;

    use super::{
        LedgerFsError, LedgerFsErrorKind, SIDECAR_SUFFIXES, sqlite_open_flags,
        sqlite_read_open_flags,
    };

    const DIRECTORY_MODE: Mode = Mode::RWXU;
    const FILE_MODE: Mode = Mode::RUSR.union(Mode::WUSR);

    struct SecureDirectory {
        _ancestors: Vec<OwnedFd>,
        parent: OwnedFd,
        database_name: OsString,
        absolute_path: PathBuf,
    }

    impl SecureDirectory {
        fn prepare(path: &Path) -> Result<Self, LedgerFsError> {
            let absolute_path = absolute_lexical_path(path)?;
            let parent_path = absolute_path
                .parent()
                .ok_or_else(|| LedgerFsError::new(LedgerFsErrorKind::InvalidFilesystem))?;
            if !parent_path
                .components()
                .any(|component| matches!(component, Component::Normal(_)))
            {
                return Err(LedgerFsError::new(LedgerFsErrorKind::InvalidFilesystem));
            }
            let database_name = absolute_path
                .file_name()
                .filter(|name| !name.is_empty())
                .ok_or_else(|| LedgerFsError::new(LedgerFsErrorKind::InvalidFilesystem))?
                .to_os_string();

            let root = fs::open(
                "/",
                OFlags::RDONLY | OFlags::DIRECTORY | OFlags::NOFOLLOW | OFlags::CLOEXEC,
                Mode::empty(),
            )
            .map_err(|error| errno_error(LedgerFsErrorKind::InvalidPermissions, error))?;
            let mut ancestors = vec![root];

            for component in parent_path.components() {
                match component {
                    Component::RootDir => {}
                    Component::Normal(name) => {
                        let next = open_or_create_directory(
                            ancestors
                                .last()
                                .expect("secure path always retains its root"),
                            name,
                        )?;
                        ancestors.push(next);
                    }
                    Component::CurDir | Component::ParentDir | Component::Prefix(_) => {
                        return Err(LedgerFsError::new(LedgerFsErrorKind::InvalidFilesystem));
                    }
                }
            }

            let parent = ancestors
                .pop()
                .expect("absolute parent traversal always retains one directory");
            validate_ancestors(&ancestors)?;
            secure_parent(&parent)?;

            Ok(Self {
                _ancestors: ancestors,
                parent,
                database_name,
                absolute_path,
            })
        }

        fn create_or_open_database(&self) -> Result<(OwnedFd, Stat), LedgerFsError> {
            let create_flags =
                OFlags::RDWR | OFlags::CREATE | OFlags::EXCL | OFlags::NOFOLLOW | OFlags::CLOEXEC;
            let descriptor = match fs::openat(
                &self.parent,
                Path::new(&self.database_name),
                create_flags,
                FILE_MODE,
            ) {
                Ok(descriptor) => descriptor,
                Err(Errno::EXIST) => self
                    .open_existing_file(&self.database_name, false)?
                    .ok_or_else(|| LedgerFsError::new(LedgerFsErrorKind::InvalidFilesystem))?,
                Err(error) => return Err(classify_file_open(error)),
            };
            let stat = secure_regular_file(&descriptor)?;
            Ok((descriptor, stat))
        }

        fn open_database_for_read(&self) -> Result<(OwnedFd, Stat), LedgerFsError> {
            let descriptor = self
                .open_existing_file(&self.database_name, true)?
                .ok_or_else(|| LedgerFsError::new(LedgerFsErrorKind::InvalidFilesystem))?;
            let stat = fs::fstat(&descriptor)
                .map_err(|error| errno_error(LedgerFsErrorKind::InvalidFilesystem, error))?;
            Ok((descriptor, stat))
        }

        fn open_existing_file(
            &self,
            name: &OsStr,
            allow_missing: bool,
        ) -> Result<Option<OwnedFd>, LedgerFsError> {
            let path = Path::new(name);
            let flags = OFlags::RDWR | OFlags::NOFOLLOW | OFlags::CLOEXEC;
            match fs::openat(&self.parent, path, flags, Mode::empty()) {
                Ok(descriptor) => {
                    secure_regular_file(&descriptor)?;
                    Ok(Some(descriptor))
                }
                Err(Errno::NOENT) if allow_missing => Ok(None),
                Err(Errno::ACCESS | Errno::PERM) => {
                    let descriptor = fs::openat(
                        &self.parent,
                        path,
                        OFlags::RDONLY | OFlags::NOFOLLOW | OFlags::CLOEXEC,
                        Mode::empty(),
                    )
                    .map_err(classify_file_open)?;
                    secure_regular_file(&descriptor)?;
                    Ok(Some(descriptor))
                }
                Err(error) => Err(classify_file_open(error)),
            }
        }

        fn secure_existing_sidecars(&self) -> Result<(), LedgerFsError> {
            for suffix in SIDECAR_SUFFIXES {
                let name = suffixed_name(&self.database_name, suffix);
                let _ = self.open_existing_file(&name, true)?;
            }
            Ok(())
        }

        fn verify_database_identity(&self, before: &Stat) -> Result<(), LedgerFsError> {
            let descriptor = self
                .open_existing_file(&self.database_name, true)?
                .ok_or_else(|| LedgerFsError::new(LedgerFsErrorKind::InvalidFilesystem))?;
            let after = fs::fstat(&descriptor)
                .map_err(|error| errno_error(LedgerFsErrorKind::InvalidFilesystem, error))?;
            if before.st_dev != after.st_dev || before.st_ino != after.st_ino {
                return Err(LedgerFsError::new(LedgerFsErrorKind::InvalidFilesystem));
            }
            Ok(())
        }
    }

    pub(super) fn open_secure_connection(path: &Path) -> Result<Connection, LedgerFsError> {
        let secure = SecureDirectory::prepare(path)?;
        let (_database, identity) = secure.create_or_open_database()?;
        secure.secure_existing_sidecars()?;

        let connection = Connection::open_with_flags(&secure.absolute_path, sqlite_open_flags())
            .map_err(LedgerFsError::sqlite)?;
        secure.verify_database_identity(&identity)?;
        secure.secure_existing_sidecars()?;
        Ok(connection)
    }

    pub(super) fn open_secure_read_connection(path: &Path) -> Result<Connection, LedgerFsError> {
        let secure = SecureDirectory::prepare(path)?;
        let (_database, identity) = secure.open_database_for_read()?;
        secure.secure_existing_sidecars()?;

        let connection =
            Connection::open_with_flags(&secure.absolute_path, sqlite_read_open_flags())
                .map_err(LedgerFsError::sqlite)?;
        secure.verify_database_identity(&identity)?;
        secure.secure_existing_sidecars()?;
        Ok(connection)
    }

    pub(super) fn enforce_sidecar_permissions(path: &Path) -> Result<(), LedgerFsError> {
        let secure = SecureDirectory::prepare(path)?;
        let database = secure
            .open_existing_file(&secure.database_name, true)?
            .ok_or_else(|| LedgerFsError::new(LedgerFsErrorKind::InvalidFilesystem))?;
        let identity = fs::fstat(&database)
            .map_err(|error| errno_error(LedgerFsErrorKind::InvalidFilesystem, error))?;
        secure.verify_database_identity(&identity)?;
        secure.secure_existing_sidecars()
    }

    fn absolute_lexical_path(path: &Path) -> Result<PathBuf, LedgerFsError> {
        if path.as_os_str().is_empty() || path.file_name().is_none() {
            return Err(LedgerFsError::new(LedgerFsErrorKind::InvalidFilesystem));
        }
        let absolute = if path.is_absolute() {
            path.to_path_buf()
        } else {
            std::env::current_dir()
                .map_err(|error| LedgerFsError::io(LedgerFsErrorKind::InvalidFilesystem, error))?
                .join(path)
        };
        #[cfg(target_os = "macos")]
        let absolute = normalize_macos_system_alias(absolute);
        if absolute.components().any(|component| {
            matches!(
                component,
                Component::CurDir | Component::ParentDir | Component::Prefix(_)
            )
        }) {
            return Err(LedgerFsError::new(LedgerFsErrorKind::InvalidFilesystem));
        }
        Ok(absolute)
    }

    #[cfg(target_os = "macos")]
    fn normalize_macos_system_alias(absolute: PathBuf) -> PathBuf {
        for (alias, target) in [("/var", "/private/var"), ("/tmp", "/private/tmp")] {
            if let Ok(suffix) = absolute.strip_prefix(alias) {
                return Path::new(target).join(suffix);
            }
        }
        absolute
    }

    fn open_or_create_directory(parent: &OwnedFd, name: &OsStr) -> Result<OwnedFd, LedgerFsError> {
        let path = Path::new(name);
        let flags = OFlags::RDONLY | OFlags::DIRECTORY | OFlags::NOFOLLOW | OFlags::CLOEXEC;
        let descriptor = match fs::openat(parent, path, flags, Mode::empty()) {
            Ok(descriptor) => descriptor,
            Err(Errno::NOENT) => {
                match fs::mkdirat(parent, path, DIRECTORY_MODE) {
                    Ok(()) | Err(Errno::EXIST) => {}
                    Err(error) => {
                        return Err(errno_error(LedgerFsErrorKind::InvalidPermissions, error));
                    }
                }
                fs::openat(parent, path, flags, Mode::empty()).map_err(classify_directory_open)?
            }
            Err(error) => return Err(classify_directory_open(error)),
        };
        let stat = fs::fstat(&descriptor)
            .map_err(|error| errno_error(LedgerFsErrorKind::InvalidPermissions, error))?;
        if !FileType::from_raw_mode(stat.st_mode).is_dir() {
            return Err(LedgerFsError::new(LedgerFsErrorKind::InvalidFilesystem));
        }
        Ok(descriptor)
    }

    fn validate_ancestors(ancestors: &[OwnedFd]) -> Result<(), LedgerFsError> {
        let effective_uid = geteuid().as_raw();
        for descriptor in ancestors {
            let stat = fs::fstat(descriptor)
                .map_err(|error| errno_error(LedgerFsErrorKind::InvalidPermissions, error))?;
            if !FileType::from_raw_mode(stat.st_mode).is_dir() {
                return Err(LedgerFsError::new(LedgerFsErrorKind::InvalidFilesystem));
            }
            if stat.st_uid != effective_uid && stat.st_uid != 0 {
                return Err(LedgerFsError::new(LedgerFsErrorKind::InvalidPermissions));
            }
            let mode = Mode::from_raw_mode(stat.st_mode);
            let broadly_writable = mode.intersects(Mode::WGRP | Mode::WOTH);
            if broadly_writable && !mode.contains(Mode::SVTX) {
                return Err(LedgerFsError::new(LedgerFsErrorKind::InvalidPermissions));
            }
        }
        Ok(())
    }

    fn secure_parent(parent: &OwnedFd) -> Result<(), LedgerFsError> {
        let before = fs::fstat(parent)
            .map_err(|error| errno_error(LedgerFsErrorKind::InvalidPermissions, error))?;
        if !FileType::from_raw_mode(before.st_mode).is_dir() {
            return Err(LedgerFsError::new(LedgerFsErrorKind::InvalidFilesystem));
        }
        if before.st_uid != geteuid().as_raw() {
            return Err(LedgerFsError::new(LedgerFsErrorKind::InvalidPermissions));
        }
        fs::fchmod(parent, DIRECTORY_MODE)
            .map_err(|error| errno_error(LedgerFsErrorKind::InvalidPermissions, error))?;
        let after = fs::fstat(parent)
            .map_err(|error| errno_error(LedgerFsErrorKind::InvalidPermissions, error))?;
        if Mode::from_raw_mode(after.st_mode) != DIRECTORY_MODE {
            return Err(LedgerFsError::new(LedgerFsErrorKind::InvalidPermissions));
        }
        Ok(())
    }

    fn secure_regular_file(descriptor: &OwnedFd) -> Result<Stat, LedgerFsError> {
        let before = fs::fstat(descriptor)
            .map_err(|error| errno_error(LedgerFsErrorKind::InvalidFilesystem, error))?;
        if !FileType::from_raw_mode(before.st_mode).is_file() {
            return Err(LedgerFsError::new(LedgerFsErrorKind::InvalidFilesystem));
        }
        if before.st_uid != geteuid().as_raw() {
            return Err(LedgerFsError::new(LedgerFsErrorKind::InvalidPermissions));
        }
        fs::fchmod(descriptor, FILE_MODE)
            .map_err(|error| errno_error(LedgerFsErrorKind::InvalidPermissions, error))?;
        let after = fs::fstat(descriptor)
            .map_err(|error| errno_error(LedgerFsErrorKind::InvalidPermissions, error))?;
        if !FileType::from_raw_mode(after.st_mode).is_file()
            || Mode::from_raw_mode(after.st_mode) != FILE_MODE
        {
            return Err(LedgerFsError::new(LedgerFsErrorKind::InvalidPermissions));
        }
        Ok(after)
    }

    fn suffixed_name(database_name: &OsStr, suffix: &str) -> OsString {
        let mut name = database_name.to_os_string();
        name.push(suffix);
        name
    }

    fn classify_directory_open(error: Errno) -> LedgerFsError {
        match error {
            Errno::LOOP => errno_error(LedgerFsErrorKind::InvalidFilesystem, error),
            Errno::NOTDIR => errno_error(LedgerFsErrorKind::InvalidFilesystem, error),
            _ => errno_error(LedgerFsErrorKind::InvalidPermissions, error),
        }
    }

    fn classify_file_open(error: Errno) -> LedgerFsError {
        match error {
            Errno::LOOP => errno_error(LedgerFsErrorKind::InvalidFilesystem, error),
            Errno::ISDIR | Errno::NOTDIR => {
                errno_error(LedgerFsErrorKind::InvalidFilesystem, error)
            }
            _ => errno_error(LedgerFsErrorKind::InvalidPermissions, error),
        }
    }

    fn errno_error(kind: LedgerFsErrorKind, error: Errno) -> LedgerFsError {
        LedgerFsError::io(
            kind,
            std::io::Error::from_raw_os_error(error.raw_os_error()),
        )
    }
}

#[cfg(windows)]
mod platform {
    use std::ffi::{OsStr, OsString};
    use std::iter;
    use std::mem::{self, MaybeUninit};
    use std::os::windows::ffi::OsStrExt;
    use std::path::{Component, Path, PathBuf, Prefix};
    use std::ptr;

    use rusqlite::Connection;
    use windows_sys::Win32::Foundation::{
        CloseHandle, ERROR_ALREADY_EXISTS, ERROR_FILE_EXISTS, ERROR_FILE_NOT_FOUND,
        ERROR_PATH_NOT_FOUND, ERROR_SUCCESS, GetLastError, HANDLE, INVALID_HANDLE_VALUE, LocalFree,
    };
    use windows_sys::Win32::Security::Authorization::{
        GetSecurityInfo, SE_FILE_OBJECT, SetSecurityInfo,
    };
    use windows_sys::Win32::Security::{
        ACCESS_ALLOWED_ACE, ACL, ACL_REVISION, ACL_SIZE_INFORMATION, AclSizeInformation,
        AddAccessAllowedAceEx, CONTAINER_INHERIT_ACE, CopySid, DACL_SECURITY_INFORMATION, EqualSid,
        GetAce, GetAclInformation, GetLengthSid, GetSecurityDescriptorControl, GetTokenInformation,
        InitializeAcl, InitializeSecurityDescriptor, OBJECT_INHERIT_ACE,
        OWNER_SECURITY_INFORMATION, PROTECTED_DACL_SECURITY_INFORMATION, PSID, SE_DACL_PROTECTED,
        SECURITY_ATTRIBUTES, SECURITY_DESCRIPTOR, SetSecurityDescriptorControl,
        SetSecurityDescriptorDacl, SetSecurityDescriptorOwner, TOKEN_QUERY, TOKEN_USER, TokenUser,
    };
    use windows_sys::Win32::Storage::FileSystem::{
        BY_HANDLE_FILE_INFORMATION, CREATE_NEW, CreateDirectoryW, CreateFileW, FILE_ALL_ACCESS,
        FILE_ATTRIBUTE_DIRECTORY, FILE_ATTRIBUTE_NORMAL, FILE_ATTRIBUTE_REPARSE_POINT,
        FILE_ATTRIBUTE_TAG_INFO, FILE_FLAG_BACKUP_SEMANTICS, FILE_FLAG_OPEN_REPARSE_POINT,
        FILE_LIST_DIRECTORY, FILE_READ_ATTRIBUTES, FILE_SHARE_READ, FILE_SHARE_WRITE,
        FILE_TRAVERSE, FILE_TYPE_DISK, FileAttributeTagInfo, GetFileInformationByHandle,
        GetFileInformationByHandleEx, GetFileType, GetVolumeInformationByHandleW, OPEN_EXISTING,
        READ_CONTROL, WRITE_DAC,
    };
    use windows_sys::Win32::System::Threading::{GetCurrentProcess, OpenProcessToken};

    use super::{
        LedgerFsError, LedgerFsErrorKind, SIDECAR_SUFFIXES, sqlite_open_flags,
        sqlite_read_open_flags,
    };

    const ACCESS_ALLOWED_ACE_TYPE: u8 = 0;
    const FILE_PERSISTENT_ACLS: u32 = 0x0000_0008;
    const SECURE_FILE_ACCESS: u32 = FILE_READ_ATTRIBUTES | READ_CONTROL | WRITE_DAC;
    const SECURITY_DESCRIPTOR_REVISION: u32 = 1;

    struct OwnedHandle(HANDLE);

    impl Drop for OwnedHandle {
        fn drop(&mut self) {
            // SAFETY: `OwnedHandle` is constructed only from a successful Win32 open.
            unsafe {
                CloseHandle(self.0);
            }
        }
    }

    struct SidBuffer {
        words: Vec<usize>,
    }

    impl SidBuffer {
        fn as_ptr(&self) -> PSID {
            self.words.as_ptr().cast_mut().cast()
        }
    }

    struct OwnerOnlyDescriptor {
        acl_words: Vec<usize>,
        descriptor: SECURITY_DESCRIPTOR,
    }

    impl OwnerOnlyDescriptor {
        fn new(sid: &SidBuffer, directory: bool) -> Result<Self, LedgerFsError> {
            // SAFETY: `sid` was copied from a validated process token SID.
            let sid_length = unsafe { GetLengthSid(sid.as_ptr()) } as usize;
            if sid_length == 0 {
                return Err(last_io_error(LedgerFsErrorKind::InvalidPermissions));
            }
            let acl_bytes = mem::size_of::<ACL>() + mem::size_of::<ACCESS_ALLOWED_ACE>()
                - mem::size_of::<u32>()
                + sid_length;
            let mut acl_words = zeroed_words(acl_bytes);
            let acl = acl_words.as_mut_ptr().cast::<ACL>();
            // SAFETY: the aligned backing allocation is at least `acl_bytes` long.
            if unsafe { InitializeAcl(acl, acl_bytes as u32, ACL_REVISION) } == 0 {
                return Err(last_io_error(LedgerFsErrorKind::InvalidPermissions));
            }
            let flags = if directory {
                OBJECT_INHERIT_ACE | CONTAINER_INHERIT_ACE
            } else {
                0
            };
            // SAFETY: `acl` and `sid` remain valid for the lifetime of this descriptor.
            if unsafe {
                AddAccessAllowedAceEx(acl, ACL_REVISION, flags, FILE_ALL_ACCESS, sid.as_ptr())
            } == 0
            {
                return Err(last_io_error(LedgerFsErrorKind::InvalidPermissions));
            }

            let mut descriptor = SECURITY_DESCRIPTOR::default();
            let descriptor_ptr = (&mut descriptor as *mut SECURITY_DESCRIPTOR).cast();
            // SAFETY: all pointers reference initialized, suitably aligned storage.
            if unsafe { InitializeSecurityDescriptor(descriptor_ptr, SECURITY_DESCRIPTOR_REVISION) }
                == 0
                || unsafe { SetSecurityDescriptorOwner(descriptor_ptr, sid.as_ptr(), 0) } == 0
                || unsafe { SetSecurityDescriptorDacl(descriptor_ptr, 1, acl, 0) } == 0
                || unsafe {
                    SetSecurityDescriptorControl(
                        descriptor_ptr,
                        SE_DACL_PROTECTED,
                        SE_DACL_PROTECTED,
                    )
                } == 0
            {
                return Err(last_io_error(LedgerFsErrorKind::InvalidPermissions));
            }
            Ok(Self {
                acl_words,
                descriptor,
            })
        }

        fn security_attributes(&mut self) -> SECURITY_ATTRIBUTES {
            SECURITY_ATTRIBUTES {
                nLength: mem::size_of::<SECURITY_ATTRIBUTES>() as u32,
                lpSecurityDescriptor: (&mut self.descriptor as *mut SECURITY_DESCRIPTOR).cast(),
                bInheritHandle: 0,
            }
        }

        fn acl(&self) -> *const ACL {
            self.acl_words.as_ptr().cast()
        }
    }

    struct SecureDirectory {
        _handles: Vec<OwnedHandle>,
        _parent: OwnedHandle,
        database_name: OsString,
        absolute_path: PathBuf,
        sid: SidBuffer,
    }

    impl SecureDirectory {
        fn prepare(path: &Path) -> Result<Self, LedgerFsError> {
            let absolute_path = absolute_lexical_path(path)?;
            validate_windows_path(&absolute_path)?;
            let parent_path = absolute_path
                .parent()
                .ok_or_else(|| LedgerFsError::new(LedgerFsErrorKind::InvalidFilesystem))?;
            if !parent_path
                .components()
                .any(|component| matches!(component, Component::Normal(_)))
            {
                return Err(LedgerFsError::new(LedgerFsErrorKind::InvalidFilesystem));
            }
            let database_name = absolute_path
                .file_name()
                .filter(|name| !name.is_empty())
                .ok_or_else(|| LedgerFsError::new(LedgerFsErrorKind::InvalidFilesystem))?
                .to_os_string();
            let sid = current_process_sid()?;
            let mut handles = Vec::new();
            let mut current = PathBuf::new();

            for component in parent_path.components() {
                match component {
                    Component::Prefix(prefix) => current.push(prefix.as_os_str()),
                    Component::RootDir => {
                        current.push(component.as_os_str());
                        let root = open_directory(&current, false, false)?.ok_or_else(|| {
                            LedgerFsError::new(LedgerFsErrorKind::InvalidFilesystem)
                        })?;
                        ensure_persistent_acls(&root)?;
                        handles.push(root);
                    }
                    Component::Normal(name) => {
                        current.push(name);
                        let handle = match open_directory(&current, false, true)? {
                            Some(handle) => handle,
                            None => {
                                create_directory(&current, &sid)?;
                                let handle =
                                    open_directory(&current, false, false)?.ok_or_else(|| {
                                        LedgerFsError::new(LedgerFsErrorKind::InvalidFilesystem)
                                    })?;
                                verify_owner(&handle, &sid)?;
                                handle
                            }
                        };
                        handles.push(handle);
                    }
                    Component::CurDir | Component::ParentDir => {
                        return Err(LedgerFsError::new(LedgerFsErrorKind::InvalidFilesystem));
                    }
                }
            }

            let traversed_parent = handles
                .pop()
                .ok_or_else(|| LedgerFsError::new(LedgerFsErrorKind::InvalidFilesystem))?;
            let parent = reopen_directory_for_acl(parent_path)?;
            apply_and_verify_acl(&parent, &sid, true)?;
            handles.push(traversed_parent);
            Ok(Self {
                _handles: handles,
                _parent: parent,
                database_name,
                absolute_path,
                sid,
            })
        }

        fn create_or_open_database(
            &self,
        ) -> Result<(OwnedHandle, BY_HANDLE_FILE_INFORMATION), LedgerFsError> {
            let path = self.absolute_path.as_path();
            let mut descriptor = OwnerOnlyDescriptor::new(&self.sid, false)?;
            let attributes = descriptor.security_attributes();
            let wide = to_wide(path)?;
            // SAFETY: the path and security descriptor buffers remain live for the call.
            let raw = unsafe {
                CreateFileW(
                    wide.as_ptr(),
                    SECURE_FILE_ACCESS,
                    FILE_SHARE_READ | FILE_SHARE_WRITE,
                    &attributes,
                    CREATE_NEW,
                    FILE_ATTRIBUTE_NORMAL | FILE_FLAG_OPEN_REPARSE_POINT,
                    ptr::null_mut(),
                )
            };
            let handle = if raw == INVALID_HANDLE_VALUE {
                let error = unsafe { GetLastError() };
                if !matches!(error, ERROR_FILE_EXISTS | ERROR_ALREADY_EXISTS) {
                    return Err(win32_error(LedgerFsErrorKind::InvalidPermissions, error));
                }
                open_file(path, false)?
                    .ok_or_else(|| LedgerFsError::new(LedgerFsErrorKind::InvalidFilesystem))?
            } else {
                OwnedHandle(raw)
            };
            validate_regular_file(&handle)?;
            apply_and_verify_acl(&handle, &self.sid, false)?;
            let identity = file_identity(&handle)?;
            Ok((handle, identity))
        }

        fn open_database_for_read(
            &self,
        ) -> Result<(OwnedHandle, BY_HANDLE_FILE_INFORMATION), LedgerFsError> {
            let handle = open_file(&self.absolute_path, true)?
                .ok_or_else(|| LedgerFsError::new(LedgerFsErrorKind::InvalidFilesystem))?;
            validate_regular_file(&handle)?;
            apply_and_verify_acl(&handle, &self.sid, false)?;
            let identity = file_identity(&handle)?;
            Ok((handle, identity))
        }

        fn secure_existing_sidecars(&self) -> Result<(), LedgerFsError> {
            for suffix in SIDECAR_SUFFIXES {
                let name = suffixed_name(&self.database_name, suffix);
                let path = self
                    .absolute_path
                    .parent()
                    .expect("validated database path has a parent")
                    .join(name);
                match open_file(&path, true)? {
                    Some(handle) => {
                        validate_regular_file(&handle)?;
                        apply_and_verify_acl(&handle, &self.sid, false)?;
                    }
                    None => continue,
                }
            }
            Ok(())
        }

        fn verify_database_identity(
            &self,
            expected: &BY_HANDLE_FILE_INFORMATION,
        ) -> Result<(), LedgerFsError> {
            let handle = open_file(&self.absolute_path, false)?
                .ok_or_else(|| LedgerFsError::new(LedgerFsErrorKind::InvalidFilesystem))?;
            validate_regular_file(&handle)?;
            let actual = file_identity(&handle)?;
            if expected.dwVolumeSerialNumber != actual.dwVolumeSerialNumber
                || expected.nFileIndexHigh != actual.nFileIndexHigh
                || expected.nFileIndexLow != actual.nFileIndexLow
            {
                return Err(LedgerFsError::new(LedgerFsErrorKind::InvalidFilesystem));
            }
            Ok(())
        }
    }

    pub(super) fn open_secure_connection(path: &Path) -> Result<Connection, LedgerFsError> {
        let secure = SecureDirectory::prepare(path)?;
        let (_database, identity) = secure.create_or_open_database()?;
        secure.secure_existing_sidecars()?;
        let connection = Connection::open_with_flags(&secure.absolute_path, sqlite_open_flags())
            .map_err(LedgerFsError::sqlite)?;
        secure.verify_database_identity(&identity)?;
        secure.secure_existing_sidecars()?;
        Ok(connection)
    }

    pub(super) fn open_secure_read_connection(path: &Path) -> Result<Connection, LedgerFsError> {
        let secure = SecureDirectory::prepare(path)?;
        let (_database, identity) = secure.open_database_for_read()?;
        secure.secure_existing_sidecars()?;
        let connection =
            Connection::open_with_flags(&secure.absolute_path, sqlite_read_open_flags())
                .map_err(LedgerFsError::sqlite)?;
        secure.verify_database_identity(&identity)?;
        secure.secure_existing_sidecars()?;
        Ok(connection)
    }

    pub(super) fn enforce_sidecar_permissions(path: &Path) -> Result<(), LedgerFsError> {
        let secure = SecureDirectory::prepare(path)?;
        let database = open_file(&secure.absolute_path, true)?
            .ok_or_else(|| LedgerFsError::new(LedgerFsErrorKind::InvalidFilesystem))?;
        validate_regular_file(&database)?;
        apply_and_verify_acl(&database, &secure.sid, false)?;
        secure.secure_existing_sidecars()
    }

    fn absolute_lexical_path(path: &Path) -> Result<PathBuf, LedgerFsError> {
        if path.as_os_str().is_empty() || path.file_name().is_none() {
            return Err(LedgerFsError::new(LedgerFsErrorKind::InvalidFilesystem));
        }
        if !path.is_absolute() && matches!(path.components().next(), Some(Component::Prefix(_))) {
            return Err(LedgerFsError::new(LedgerFsErrorKind::InvalidFilesystem));
        }
        let absolute = if path.is_absolute() {
            path.to_path_buf()
        } else {
            std::env::current_dir()
                .map(|current| current.join(path))
                .map_err(|error| LedgerFsError::io(LedgerFsErrorKind::InvalidFilesystem, error))?
        };
        if !absolute.is_absolute() {
            return Err(LedgerFsError::new(LedgerFsErrorKind::InvalidFilesystem));
        }
        Ok(absolute)
    }

    fn validate_windows_path(path: &Path) -> Result<(), LedgerFsError> {
        let mut components = path.components();
        match components.next() {
            Some(Component::Prefix(prefix)) if matches!(prefix.kind(), Prefix::Disk(_)) => {}
            _ => {
                return Err(LedgerFsError::new(LedgerFsErrorKind::InvalidFilesystem));
            }
        }
        if !matches!(components.next(), Some(Component::RootDir)) {
            return Err(LedgerFsError::new(LedgerFsErrorKind::InvalidFilesystem));
        }
        for component in components {
            match component {
                Component::Normal(name) => validate_windows_component(name)?,
                Component::Prefix(_)
                | Component::RootDir
                | Component::CurDir
                | Component::ParentDir => {
                    return Err(LedgerFsError::new(LedgerFsErrorKind::InvalidFilesystem));
                }
            }
        }
        Ok(())
    }

    fn validate_windows_component(component: &OsStr) -> Result<(), LedgerFsError> {
        let units = component.encode_wide().collect::<Vec<_>>();
        let invalid = units.is_empty()
            || matches!(units.last(), Some(value) if *value == b'.' as u16 || *value == b' ' as u16)
            || units.iter().any(|value| {
                *value == 0
                    || matches!(
                        *value,
                        value if value == b'<' as u16
                            || value == b'>' as u16
                            || value == b':' as u16
                            || value == b'"' as u16
                            || value == b'|' as u16
                            || value == b'?' as u16
                            || value == b'*' as u16
                    )
            });
        if invalid {
            return Err(LedgerFsError::new(LedgerFsErrorKind::InvalidFilesystem));
        }
        Ok(())
    }

    fn current_process_sid() -> Result<SidBuffer, LedgerFsError> {
        let mut token = ptr::null_mut();
        // SAFETY: `token` is a valid out pointer and the pseudo process handle is always valid.
        if unsafe { OpenProcessToken(GetCurrentProcess(), TOKEN_QUERY, &mut token) } == 0 {
            return Err(last_io_error(LedgerFsErrorKind::InvalidPermissions));
        }
        let token = OwnedHandle(token);
        let mut bytes = 0;
        // SAFETY: the null-buffer size query writes only `bytes`.
        unsafe {
            GetTokenInformation(token.0, TokenUser, ptr::null_mut(), 0, &mut bytes);
        }
        if bytes == 0 {
            return Err(last_io_error(LedgerFsErrorKind::InvalidPermissions));
        }
        let mut token_words = zeroed_words(bytes as usize);
        // SAFETY: the aligned buffer has the size returned by the preceding query.
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
            return Err(last_io_error(LedgerFsErrorKind::InvalidPermissions));
        }
        // SAFETY: a successful TokenUser query initializes a TOKEN_USER at the buffer start.
        let source = unsafe { &*token_words.as_ptr().cast::<TOKEN_USER>() }
            .User
            .Sid;
        // SAFETY: the token returned a validated SID pointer.
        let sid_bytes = unsafe { GetLengthSid(source) };
        if sid_bytes == 0 {
            return Err(last_io_error(LedgerFsErrorKind::InvalidPermissions));
        }
        let mut words = zeroed_words(sid_bytes as usize);
        // SAFETY: the destination buffer is at least `sid_bytes` long.
        if unsafe { CopySid(sid_bytes, words.as_mut_ptr().cast(), source) } == 0 {
            return Err(last_io_error(LedgerFsErrorKind::InvalidPermissions));
        }
        Ok(SidBuffer { words })
    }

    fn open_directory(
        path: &Path,
        write_acl: bool,
        allow_missing: bool,
    ) -> Result<Option<OwnedHandle>, LedgerFsError> {
        let wide = to_wide(path)?;
        let mut access = FILE_LIST_DIRECTORY | FILE_TRAVERSE | READ_CONTROL;
        if write_acl {
            access |= WRITE_DAC;
        }
        // SAFETY: `wide` is a terminated path buffer.
        let raw = unsafe {
            CreateFileW(
                wide.as_ptr(),
                access,
                FILE_SHARE_READ | FILE_SHARE_WRITE,
                ptr::null(),
                OPEN_EXISTING,
                FILE_FLAG_BACKUP_SEMANTICS | FILE_FLAG_OPEN_REPARSE_POINT,
                ptr::null_mut(),
            )
        };
        if raw == INVALID_HANDLE_VALUE {
            let error = unsafe { GetLastError() };
            if allow_missing && matches!(error, ERROR_FILE_NOT_FOUND | ERROR_PATH_NOT_FOUND) {
                return Ok(None);
            }
            return Err(win32_error(LedgerFsErrorKind::InvalidPermissions, error));
        }
        let handle = OwnedHandle(raw);
        validate_handle_kind(&handle, true)?;
        Ok(Some(handle))
    }

    fn reopen_directory_for_acl(path: &Path) -> Result<OwnedHandle, LedgerFsError> {
        open_directory(path, true, false)?
            .ok_or_else(|| LedgerFsError::new(LedgerFsErrorKind::InvalidFilesystem))
    }

    fn create_directory(path: &Path, sid: &SidBuffer) -> Result<(), LedgerFsError> {
        let wide = to_wide(path)?;
        let mut descriptor = OwnerOnlyDescriptor::new(sid, true)?;
        let attributes = descriptor.security_attributes();
        // SAFETY: the path and security descriptor remain valid for the call.
        if unsafe { CreateDirectoryW(wide.as_ptr(), &attributes) } == 0 {
            let error = unsafe { GetLastError() };
            if !matches!(error, ERROR_FILE_EXISTS | ERROR_ALREADY_EXISTS) {
                return Err(win32_error(LedgerFsErrorKind::InvalidPermissions, error));
            }
        }
        Ok(())
    }

    fn open_file(path: &Path, allow_missing: bool) -> Result<Option<OwnedHandle>, LedgerFsError> {
        let wide = to_wide(path)?;
        // SAFETY: `wide` is a terminated path buffer.
        let raw = unsafe {
            CreateFileW(
                wide.as_ptr(),
                SECURE_FILE_ACCESS,
                FILE_SHARE_READ | FILE_SHARE_WRITE,
                ptr::null(),
                OPEN_EXISTING,
                FILE_ATTRIBUTE_NORMAL | FILE_FLAG_OPEN_REPARSE_POINT,
                ptr::null_mut(),
            )
        };
        if raw == INVALID_HANDLE_VALUE {
            let error = unsafe { GetLastError() };
            if allow_missing && matches!(error, ERROR_FILE_NOT_FOUND | ERROR_PATH_NOT_FOUND) {
                return Ok(None);
            }
            return Err(win32_error(LedgerFsErrorKind::InvalidPermissions, error));
        }
        Ok(Some(OwnedHandle(raw)))
    }

    fn validate_regular_file(handle: &OwnedHandle) -> Result<(), LedgerFsError> {
        validate_handle_kind(handle, false)
    }

    fn validate_handle_kind(
        handle: &OwnedHandle,
        expect_directory: bool,
    ) -> Result<(), LedgerFsError> {
        let mut info = MaybeUninit::<FILE_ATTRIBUTE_TAG_INFO>::zeroed();
        // SAFETY: `info` points to a correctly sized output structure.
        if unsafe {
            GetFileInformationByHandleEx(
                handle.0,
                FileAttributeTagInfo,
                info.as_mut_ptr().cast(),
                mem::size_of::<FILE_ATTRIBUTE_TAG_INFO>() as u32,
            )
        } == 0
        {
            return Err(last_io_error(LedgerFsErrorKind::InvalidFilesystem));
        }
        // SAFETY: the Win32 call initialized the output structure.
        let info = unsafe { info.assume_init() };
        if info.FileAttributes & FILE_ATTRIBUTE_REPARSE_POINT != 0 {
            return Err(LedgerFsError::new(LedgerFsErrorKind::InvalidFilesystem));
        }
        let is_directory = info.FileAttributes & FILE_ATTRIBUTE_DIRECTORY != 0;
        // SAFETY: the handle remains valid for this query.
        if unsafe { GetFileType(handle.0) } != FILE_TYPE_DISK || is_directory != expect_directory {
            return Err(LedgerFsError::new(LedgerFsErrorKind::InvalidFilesystem));
        }
        Ok(())
    }

    fn apply_and_verify_acl(
        handle: &OwnedHandle,
        sid: &SidBuffer,
        directory: bool,
    ) -> Result<(), LedgerFsError> {
        verify_owner(handle, sid)?;
        let descriptor = OwnerOnlyDescriptor::new(sid, directory)?;
        // SAFETY: the handle and ACL pointer remain valid for the call.
        let status = unsafe {
            SetSecurityInfo(
                handle.0,
                SE_FILE_OBJECT,
                DACL_SECURITY_INFORMATION | PROTECTED_DACL_SECURITY_INFORMATION,
                ptr::null_mut(),
                ptr::null_mut(),
                descriptor.acl(),
                ptr::null(),
            )
        };
        if status != ERROR_SUCCESS {
            return Err(win32_error(LedgerFsErrorKind::InvalidPermissions, status));
        }
        verify_owner_only_acl(handle, sid, directory)
    }

    fn verify_owner(handle: &OwnedHandle, sid: &SidBuffer) -> Result<(), LedgerFsError> {
        with_security_info(handle, |owner, _, _| {
            if owner.is_null() || unsafe { EqualSid(owner, sid.as_ptr()) } == 0 {
                return Err(LedgerFsError::new(LedgerFsErrorKind::InvalidPermissions));
            }
            Ok(())
        })
    }

    fn verify_owner_only_acl(
        handle: &OwnedHandle,
        sid: &SidBuffer,
        directory: bool,
    ) -> Result<(), LedgerFsError> {
        with_security_info(handle, |owner, acl, security_descriptor| {
            if owner.is_null() || unsafe { EqualSid(owner, sid.as_ptr()) } == 0 {
                return Err(LedgerFsError::new(LedgerFsErrorKind::InvalidPermissions));
            }
            if acl.is_null() {
                return Err(LedgerFsError::new(LedgerFsErrorKind::InvalidPermissions));
            }
            let mut control = 0;
            let mut revision = 0;
            // SAFETY: the descriptor is owned by `with_security_info` for this call.
            if unsafe {
                GetSecurityDescriptorControl(security_descriptor, &mut control, &mut revision)
            } == 0
                || control & SE_DACL_PROTECTED == 0
            {
                return Err(LedgerFsError::new(LedgerFsErrorKind::InvalidPermissions));
            }
            let mut size = ACL_SIZE_INFORMATION::default();
            // SAFETY: `acl` is part of the live security descriptor.
            if unsafe {
                GetAclInformation(
                    acl,
                    (&mut size as *mut ACL_SIZE_INFORMATION).cast(),
                    mem::size_of::<ACL_SIZE_INFORMATION>() as u32,
                    AclSizeInformation,
                )
            } == 0
                || size.AceCount != 1
            {
                return Err(LedgerFsError::new(LedgerFsErrorKind::InvalidPermissions));
            }
            let mut raw_ace = ptr::null_mut();
            // SAFETY: the ACL contains exactly one ACE.
            if unsafe { GetAce(acl, 0, &mut raw_ace) } == 0 || raw_ace.is_null() {
                return Err(LedgerFsError::new(LedgerFsErrorKind::InvalidPermissions));
            }
            // SAFETY: the ACE type is checked before reading the allowed-ACE layout.
            let ace = unsafe { &*raw_ace.cast::<ACCESS_ALLOWED_ACE>() };
            let expected_flags = if directory {
                (OBJECT_INHERIT_ACE | CONTAINER_INHERIT_ACE) as u8
            } else {
                0
            };
            let ace_sid = (&ace.SidStart as *const u32).cast_mut().cast();
            if ace.Header.AceType != ACCESS_ALLOWED_ACE_TYPE
                || ace.Header.AceFlags != expected_flags
                || ace.Mask != FILE_ALL_ACCESS
                || unsafe { EqualSid(ace_sid, sid.as_ptr()) } == 0
            {
                return Err(LedgerFsError::new(LedgerFsErrorKind::InvalidPermissions));
            }
            Ok(())
        })
    }

    fn with_security_info<T>(
        handle: &OwnedHandle,
        inspect: impl FnOnce(PSID, *mut ACL, *mut core::ffi::c_void) -> Result<T, LedgerFsError>,
    ) -> Result<T, LedgerFsError> {
        let mut owner = ptr::null_mut();
        let mut acl = ptr::null_mut();
        let mut descriptor = ptr::null_mut();
        // SAFETY: all output pointers are valid and the result is freed below.
        let status = unsafe {
            GetSecurityInfo(
                handle.0,
                SE_FILE_OBJECT,
                OWNER_SECURITY_INFORMATION | DACL_SECURITY_INFORMATION,
                &mut owner,
                ptr::null_mut(),
                &mut acl,
                ptr::null_mut(),
                &mut descriptor,
            )
        };
        if status != ERROR_SUCCESS {
            return Err(win32_error(LedgerFsErrorKind::InvalidPermissions, status));
        }
        let result = inspect(owner, acl, descriptor);
        // SAFETY: GetSecurityInfo allocated this descriptor with LocalAlloc.
        unsafe {
            LocalFree(descriptor);
        }
        result
    }

    fn ensure_persistent_acls(handle: &OwnedHandle) -> Result<(), LedgerFsError> {
        let mut flags = 0;
        // SAFETY: optional output buffers are null and `flags` is a valid out pointer.
        if unsafe {
            GetVolumeInformationByHandleW(
                handle.0,
                ptr::null_mut(),
                0,
                ptr::null_mut(),
                ptr::null_mut(),
                &mut flags,
                ptr::null_mut(),
                0,
            )
        } == 0
        {
            return Err(last_io_error(LedgerFsErrorKind::UnsupportedSecurity));
        }
        if flags & FILE_PERSISTENT_ACLS == 0 {
            return Err(LedgerFsError::new(LedgerFsErrorKind::UnsupportedSecurity));
        }
        Ok(())
    }

    fn file_identity(handle: &OwnedHandle) -> Result<BY_HANDLE_FILE_INFORMATION, LedgerFsError> {
        let mut info = MaybeUninit::<BY_HANDLE_FILE_INFORMATION>::zeroed();
        // SAFETY: `info` points to a correctly sized output structure.
        if unsafe { GetFileInformationByHandle(handle.0, info.as_mut_ptr()) } == 0 {
            return Err(last_io_error(LedgerFsErrorKind::InvalidFilesystem));
        }
        // SAFETY: the Win32 call initialized the output structure.
        Ok(unsafe { info.assume_init() })
    }

    fn to_wide(path: &Path) -> Result<Vec<u16>, LedgerFsError> {
        let wide = path
            .as_os_str()
            .encode_wide()
            .chain(iter::once(0))
            .collect::<Vec<_>>();
        if wide[..wide.len() - 1].contains(&0) {
            return Err(LedgerFsError::new(LedgerFsErrorKind::InvalidFilesystem));
        }
        Ok(wide)
    }

    fn suffixed_name(database_name: &OsStr, suffix: &str) -> OsString {
        let mut name = database_name.to_os_string();
        name.push(suffix);
        name
    }

    fn zeroed_words(bytes: usize) -> Vec<usize> {
        vec![0; bytes.div_ceil(mem::size_of::<usize>())]
    }

    fn last_io_error(kind: LedgerFsErrorKind) -> LedgerFsError {
        LedgerFsError::io(kind, std::io::Error::last_os_error())
    }

    fn win32_error(kind: LedgerFsErrorKind, error: u32) -> LedgerFsError {
        LedgerFsError::io(kind, std::io::Error::from_raw_os_error(error as i32))
    }
}

#[cfg(any(target_os = "redox", not(any(unix, windows))))]
mod platform {
    use std::path::Path;

    use rusqlite::Connection;

    use super::{LedgerFsError, LedgerFsErrorKind};

    pub(super) fn open_secure_connection(_path: &Path) -> Result<Connection, LedgerFsError> {
        Err(LedgerFsError::new(LedgerFsErrorKind::UnsupportedSecurity))
    }

    pub(super) fn open_secure_read_connection(_path: &Path) -> Result<Connection, LedgerFsError> {
        Err(LedgerFsError::new(LedgerFsErrorKind::UnsupportedSecurity))
    }

    pub(super) fn enforce_sidecar_permissions(_path: &Path) -> Result<(), LedgerFsError> {
        Err(LedgerFsError::new(LedgerFsErrorKind::UnsupportedSecurity))
    }
}

#[cfg(all(test, unix, not(target_os = "redox")))]
mod tests {
    use std::fs;
    use std::os::unix::fs::{PermissionsExt, symlink};

    use tempfile::tempdir;

    use super::{
        LedgerFsErrorKind, enforce_sidecar_permissions, open_secure_connection,
        open_secure_read_connection,
    };

    #[test]
    fn secure_open_sets_exact_parent_database_and_sidecar_modes() {
        let temporary = tempdir().expect("tempdir should be created");
        secure_tempdir_root(&temporary);
        let parent = temporary.path().join("ledger");
        fs::create_dir(&parent).expect("ledger parent should be created");
        fs::set_permissions(&parent, fs::Permissions::from_mode(0o777))
            .expect("test should widen parent permissions");
        let database = parent.join("router.db");

        let connection = open_secure_connection(&database).expect("secure open should succeed");
        connection
            .pragma_update(None, "journal_mode", "WAL")
            .expect("WAL should be enabled");
        connection
            .execute_batch("CREATE TABLE secure_test (value INTEGER NOT NULL); INSERT INTO secure_test VALUES (1);")
            .expect("WAL-backed write should succeed");
        enforce_sidecar_permissions(&database).expect("sidecars should be secured");

        assert_mode(&parent, 0o700);
        assert_mode(&database, 0o600);
        assert_mode(&database.with_file_name("router.db-wal"), 0o600);
        assert_mode(&database.with_file_name("router.db-shm"), 0o600);
    }

    #[test]
    fn secure_open_rejects_database_symlink() {
        let temporary = tempdir().expect("tempdir should be created");
        secure_tempdir_root(&temporary);
        let parent = temporary.path().join("ledger");
        fs::create_dir(&parent).expect("ledger parent should be created");
        let outside = temporary.path().join("outside.db");
        fs::write(&outside, b"sentinel").expect("outside sentinel should be written");
        let database = parent.join("router.db");
        symlink(&outside, &database).expect("database symlink should be created");

        let error = open_secure_connection(&database).expect_err("symlink must be rejected");
        assert_eq!(error.kind(), LedgerFsErrorKind::InvalidFilesystem);
        assert_eq!(
            fs::read(&outside).expect("outside sentinel should remain readable"),
            b"sentinel"
        );
    }

    #[test]
    fn secure_open_rejects_symlinked_parent_component() {
        let temporary = tempdir().expect("tempdir should be created");
        secure_tempdir_root(&temporary);
        let actual_parent = temporary.path().join("actual-ledger");
        fs::create_dir(&actual_parent).expect("actual ledger parent should be created");
        let linked_parent = temporary.path().join("linked-ledger");
        symlink(&actual_parent, &linked_parent).expect("parent symlink should be created");

        let error = open_secure_connection(&linked_parent.join("router.db"))
            .expect_err("symlinked parent must be rejected");
        assert_eq!(error.kind(), LedgerFsErrorKind::InvalidFilesystem);
        assert!(!actual_parent.join("router.db").exists());
    }

    #[test]
    fn secure_open_rejects_non_regular_database() {
        let temporary = tempdir().expect("tempdir should be created");
        secure_tempdir_root(&temporary);
        let parent = temporary.path().join("ledger");
        fs::create_dir(&parent).expect("ledger parent should be created");
        let database = parent.join("router.db");
        fs::create_dir(&database).expect("directory should occupy database path");

        let error = open_secure_connection(&database).expect_err("directory must be rejected");
        assert_eq!(error.kind(), LedgerFsErrorKind::InvalidFilesystem);
    }

    #[test]
    fn secure_open_rejects_sidecar_symlink_before_sqlite() {
        let temporary = tempdir().expect("tempdir should be created");
        secure_tempdir_root(&temporary);
        let parent = temporary.path().join("ledger");
        fs::create_dir(&parent).expect("ledger parent should be created");
        let database = parent.join("router.db");
        fs::write(&database, b"").expect("database placeholder should be created");
        let outside = temporary.path().join("outside-wal");
        fs::write(&outside, b"sentinel").expect("outside sentinel should be written");
        symlink(&outside, database.with_file_name("router.db-wal"))
            .expect("WAL symlink should be created");

        let error = open_secure_connection(&database).expect_err("sidecar link must be rejected");
        assert_eq!(error.kind(), LedgerFsErrorKind::InvalidFilesystem);
        assert_eq!(
            fs::read(&outside).expect("outside sentinel should remain readable"),
            b"sentinel"
        );
    }

    #[test]
    fn secure_open_rejects_non_sticky_writable_ancestor() {
        let temporary = tempdir().expect("tempdir should be created");
        fs::set_permissions(temporary.path(), fs::Permissions::from_mode(0o770))
            .expect("test should widen ancestor permissions");
        let parent = temporary.path().join("ledger");
        fs::create_dir(&parent).expect("ledger parent should be created");

        let error = open_secure_connection(&parent.join("router.db"))
            .expect_err("writable ancestor must be rejected");
        assert_eq!(error.kind(), LedgerFsErrorKind::InvalidPermissions);
    }

    #[test]
    fn secure_read_open_requires_an_existing_database_and_is_read_only() {
        let temporary = tempdir().expect("tempdir should be created");
        secure_tempdir_root(&temporary);
        let database = temporary.path().join("ledger/router.db");

        let missing = open_secure_read_connection(&database)
            .expect_err("a read-only open must not create the database");
        assert_eq!(missing.kind(), LedgerFsErrorKind::InvalidFilesystem);

        let writer = open_secure_connection(&database).expect("secure writer should open");
        writer
            .execute_batch(
                "CREATE TABLE read_test (value INTEGER NOT NULL); INSERT INTO read_test VALUES (7);",
            )
            .expect("test data should be written");
        drop(writer);

        let reader =
            open_secure_read_connection(&database).expect("secure reader should open existing DB");
        let value: i64 = reader
            .query_row("SELECT value FROM read_test", [], |row| row.get(0))
            .expect("reader should query existing data");
        assert_eq!(value, 7);
        assert!(
            reader
                .execute("INSERT INTO read_test VALUES (8)", [])
                .is_err(),
            "read-only SQLite flags must reject writes"
        );
    }

    #[test]
    fn filesystem_errors_expose_only_stable_sanitized_classes() {
        let error = open_secure_connection(std::path::Path::new("/router.db"))
            .expect_err("filesystem root must not become a ledger directory");

        assert_eq!(error.kind(), LedgerFsErrorKind::InvalidFilesystem);
        assert_eq!(error.to_string(), "ledger filesystem validation failed");
        assert!(std::error::Error::source(&error).is_none());
        assert_eq!(
            format!("{error:?}"),
            "LedgerFsError { kind: InvalidFilesystem }"
        );
    }

    fn assert_mode(path: &std::path::Path, expected: u32) {
        let mode = fs::symlink_metadata(path)
            .expect("secured path should exist")
            .permissions()
            .mode()
            & 0o7777;
        assert_eq!(mode, expected, "unexpected mode for {}", path.display());
    }

    fn secure_tempdir_root(directory: &tempfile::TempDir) {
        fs::set_permissions(directory.path(), fs::Permissions::from_mode(0o700))
            .expect("temporary root should be owner-only");
    }
}

#[cfg(all(test, windows))]
mod windows_tests {
    use std::path::Path;
    use std::sync::{Arc, Barrier};
    use std::thread;

    use tempfile::tempdir;

    use super::{
        LedgerFsErrorKind, enforce_sidecar_permissions, open_secure_connection,
        open_secure_read_connection,
    };

    #[test]
    fn secure_open_allows_sqlite_and_live_sidecar_handles() {
        let temporary = tempdir().expect("tempdir should be created");
        let database = temporary.path().join("ledger/router.db");

        let connection = open_secure_connection(&database).expect("secure open should succeed");
        connection
            .pragma_update(None, "journal_mode", "WAL")
            .expect("WAL should be enabled");
        connection
            .execute_batch("CREATE TABLE secure_test (value INTEGER NOT NULL); INSERT INTO secure_test VALUES (1);")
            .expect("WAL-backed write should succeed");
        enforce_sidecar_permissions(&database).expect("live sidecars should be secured");
    }

    #[test]
    fn secure_open_rejects_drive_relative_paths() {
        let error = open_secure_connection(Path::new(r"C:relative\router.db"))
            .expect_err("drive-relative paths must be rejected");

        assert_eq!(error.kind(), LedgerFsErrorKind::InvalidFilesystem);
    }

    #[test]
    fn secure_read_open_requires_existing_database_and_rejects_writes() {
        let temporary = tempdir().expect("tempdir should be created");
        let database = temporary.path().join("ledger/router.db");
        let writer = open_secure_connection(&database).expect("secure writer should open");
        writer
            .execute_batch(
                "CREATE TABLE read_test (value INTEGER NOT NULL); INSERT INTO read_test VALUES (7);",
            )
            .expect("test data should be written");
        drop(writer);

        let reader =
            open_secure_read_connection(&database).expect("secure reader should open existing DB");
        let value: i64 = reader
            .query_row("SELECT value FROM read_test", [], |row| row.get(0))
            .expect("reader should query existing data");
        assert_eq!(value, 7);
        assert!(
            reader
                .execute("INSERT INTO read_test VALUES (8)", [])
                .is_err()
        );
    }

    #[test]
    fn concurrent_secure_openers_share_directory_creation() {
        let temporary = tempdir().expect("tempdir should be created");
        let database = Arc::new(temporary.path().join("nested/ledger/router.db"));
        let barrier = Arc::new(Barrier::new(2));
        let openers = (0..2)
            .map(|_| {
                let database = Arc::clone(&database);
                let barrier = Arc::clone(&barrier);
                thread::spawn(move || {
                    barrier.wait();
                    open_secure_connection(database.as_ref()).map(drop)
                })
            })
            .collect::<Vec<_>>();

        for opener in openers {
            opener
                .join()
                .expect("secure opener should not panic")
                .expect("concurrent secure open should succeed");
        }
    }
}
