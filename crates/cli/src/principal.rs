// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Authenticated local process principal for Router mutation audit records.

use std::io;

#[cfg(unix)]
pub(crate) fn authenticated_principal() -> Result<String, io::Error> {
    Ok(format!("uid:{}", rustix::process::geteuid().as_raw()))
}

#[cfg(windows)]
pub(crate) fn authenticated_principal() -> Result<String, io::Error> {
    use std::ptr::{null_mut, slice_from_raw_parts};

    use windows_sys::Win32::Foundation::{CloseHandle, ERROR_INSUFFICIENT_BUFFER, GetLastError};
    use windows_sys::Win32::Security::Authorization::ConvertSidToStringSidW;
    use windows_sys::Win32::Security::{GetTokenInformation, TOKEN_QUERY, TOKEN_USER, TokenUser};
    use windows_sys::Win32::System::Threading::{GetCurrentProcess, OpenProcessToken};

    let mut token = null_mut();
    // SAFETY: The process pseudo-handle is valid for the current process and `token` is writable.
    if unsafe { OpenProcessToken(GetCurrentProcess(), TOKEN_QUERY, &mut token) } == 0 {
        return Err(io::Error::last_os_error());
    }
    let result = (|| {
        let mut bytes = 0_u32;
        // SAFETY: A null buffer with length zero is the documented size-query call.
        let queried = unsafe { GetTokenInformation(token, TokenUser, null_mut(), 0, &mut bytes) };
        if queried != 0 || bytes == 0 || unsafe { GetLastError() } != ERROR_INSUFFICIENT_BUFFER {
            return Err(io::Error::last_os_error());
        }
        let mut buffer = vec![
            0_u8;
            usize::try_from(bytes).map_err(|_| {
                io::Error::new(io::ErrorKind::InvalidData, "token identity is too large")
            })?
        ];
        // SAFETY: `buffer` has the exact size requested by the preceding Windows API call.
        if unsafe {
            GetTokenInformation(
                token,
                TokenUser,
                buffer.as_mut_ptr().cast(),
                bytes,
                &mut bytes,
            )
        } == 0
        {
            return Err(io::Error::last_os_error());
        }
        // SAFETY: A successful TokenUser query initializes a TOKEN_USER at the buffer start.
        let user = unsafe { &*buffer.as_ptr().cast::<TOKEN_USER>() };
        let mut sid = null_mut();
        // SAFETY: The queried token owns a valid SID for the duration of this conversion.
        if unsafe { ConvertSidToStringSidW(user.User.Sid, &mut sid) } == 0 {
            return Err(io::Error::last_os_error());
        }
        let converted = (|| {
            let mut length = 0_usize;
            // SAFETY: ConvertSidToStringSidW returns a null-terminated LocalAlloc buffer.
            while unsafe { *sid.add(length) } != 0 {
                length = length.checked_add(1).ok_or_else(|| {
                    io::Error::new(io::ErrorKind::InvalidData, "SID string is too large")
                })?;
            }
            // SAFETY: The loop established the initialized UTF-16 length before the terminator.
            let wide = unsafe { &*slice_from_raw_parts(sid, length) };
            String::from_utf16(wide)
                .map(|value| format!("sid:{value}"))
                .map_err(|_| io::Error::new(io::ErrorKind::InvalidData, "SID is not valid UTF-16"))
        })();
        // SAFETY: The SID string was allocated by ConvertSidToStringSidW with LocalAlloc.
        unsafe {
            windows_sys::Win32::Foundation::LocalFree(sid.cast());
        }
        converted
    })();
    // SAFETY: OpenProcessToken returned this owned token handle.
    unsafe {
        CloseHandle(token);
    }
    result
}

#[cfg(not(any(unix, windows)))]
pub(crate) fn authenticated_principal() -> Result<String, io::Error> {
    Err(io::Error::new(
        io::ErrorKind::Unsupported,
        "authenticated OS principal is unavailable on this platform",
    ))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn principal_uses_the_authenticated_platform_identity() {
        let principal = authenticated_principal().unwrap();
        #[cfg(unix)]
        assert_eq!(
            principal,
            format!("uid:{}", rustix::process::geteuid().as_raw())
        );
        #[cfg(windows)]
        assert!(principal.starts_with("sid:S-1-"));
    }
}
