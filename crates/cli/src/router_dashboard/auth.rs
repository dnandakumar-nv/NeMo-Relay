// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::fmt;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};

use axum::http::header::{AUTHORIZATION, COOKIE, HOST, ORIGIN};
use axum::http::{HeaderMap, HeaderValue, Method};
use cookie::{Cookie, SameSite};
use nemo_relay_router::inspection::{
    InspectionError, InspectionHttpAuthContext, InspectionHttpAuthGuard,
};
use ring::digest::{SHA256, digest};
use ring::rand::{SecureRandom, SystemRandom};
use subtle::ConstantTimeEq as _;
use zeroize::Zeroizing;

use super::ActivityTracker;

pub(super) const SESSION_COOKIE_NAME: &str = "nemo_relay_dashboard_session";
const TOKEN_BYTES: usize = 32;
const TOKEN_TEXT_BYTES: usize = 43;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum AuthError {
    Unauthorized,
    Forbidden,
    Invalid,
    Entropy,
}

impl AuthError {
    fn inspection(self) -> InspectionError {
        match self {
            Self::Unauthorized => InspectionError::Unauthorized,
            Self::Forbidden => InspectionError::Forbidden,
            Self::Invalid | Self::Entropy => InspectionError::IntegrityError,
        }
    }
}

pub(super) struct BootstrapSecret(Zeroizing<String>);

impl BootstrapSecret {
    pub(super) fn launch_url(&self, origin: &str) -> Zeroizing<String> {
        Zeroizing::new(format!("{origin}/#bootstrap={}", self.0.as_str()))
    }
}

#[derive(Clone)]
pub(super) struct DashboardAuth {
    inner: Arc<AuthInner>,
}

struct AuthInner {
    origin: String,
    host: String,
    secure_cookie: bool,
    bootstrap: Mutex<BootstrapState>,
    session_digest: [u8; 32],
    valid: AtomicBool,
    activity: ActivityTracker,
}

struct BootstrapState {
    nonce_digest: [u8; 32],
    pending_session: Option<Zeroizing<String>>,
}

impl DashboardAuth {
    pub(super) fn new(
        origin: String,
        host: String,
        secure_cookie: bool,
        activity: ActivityTracker,
    ) -> Result<(Self, BootstrapSecret), AuthError> {
        let nonce = random_token()?;
        let session = random_token()?;
        let nonce_digest = token_digest(nonce.as_bytes());
        let session_digest = token_digest(session.as_bytes());
        Ok((
            Self {
                inner: Arc::new(AuthInner {
                    origin,
                    host,
                    secure_cookie,
                    bootstrap: Mutex::new(BootstrapState {
                        nonce_digest,
                        pending_session: Some(session),
                    }),
                    session_digest,
                    valid: AtomicBool::new(true),
                    activity,
                }),
            },
            BootstrapSecret(nonce),
        ))
    }

    pub(super) fn bootstrap(&self, headers: &HeaderMap) -> Result<HeaderValue, AuthError> {
        require_exact_header(headers, HOST, &self.inner.host, AuthError::Forbidden)?;
        require_exact_header(headers, ORIGIN, &self.inner.origin, AuthError::Forbidden)?;
        let authorization = one_header(headers, AUTHORIZATION, AuthError::Unauthorized)?;
        let nonce = authorization
            .strip_prefix("Bootstrap ")
            .filter(|value| valid_token_text(value))
            .ok_or(AuthError::Unauthorized)?;
        let actual = token_digest(nonce.as_bytes());
        let mut bootstrap = self
            .inner
            .bootstrap
            .lock()
            .map_err(|_| AuthError::Invalid)?;
        if !bool::from(bootstrap.nonce_digest.ct_eq(&actual)) {
            return Err(AuthError::Unauthorized);
        }
        let session = bootstrap
            .pending_session
            .take()
            .ok_or(AuthError::Unauthorized)?;
        let cookie = Cookie::build((SESSION_COOKIE_NAME, session.as_str().to_owned()))
            .http_only(true)
            .same_site(SameSite::Strict)
            .path("/")
            .secure(self.inner.secure_cookie)
            .build()
            .to_string();
        drop(session);
        self.inner.activity.touch();
        HeaderValue::from_str(&cookie).map_err(|_| AuthError::Invalid)
    }

    pub(super) fn invalidate(&self) {
        self.inner.valid.store(false, Ordering::Release);
        if let Ok(mut bootstrap) = self.inner.bootstrap.lock() {
            bootstrap.pending_session.take();
        }
    }

    #[cfg(test)]
    pub(super) fn is_bootstrapped(&self) -> bool {
        self.inner
            .bootstrap
            .lock()
            .map(|bootstrap| bootstrap.pending_session.is_none())
            .unwrap_or(false)
    }

    fn authorize_session(&self, context: &InspectionHttpAuthContext<'_>) -> Result<(), AuthError> {
        if context.operations || !self.inner.valid.load(Ordering::Acquire) {
            return Err(AuthError::Forbidden);
        }
        require_exact_header(
            context.headers,
            HOST,
            &self.inner.host,
            AuthError::Forbidden,
        )?;
        let origin = optional_one_header(context.headers, ORIGIN, AuthError::Forbidden)?;
        if matches!(*context.method, Method::GET | Method::HEAD) {
            if origin.is_some_and(|origin| origin != self.inner.origin) {
                return Err(AuthError::Forbidden);
            }
        } else if origin != Some(self.inner.origin.as_str()) {
            return Err(AuthError::Forbidden);
        }
        let token = session_cookie(context.headers)?;
        let actual = token_digest(token.as_bytes());
        if !bool::from(self.inner.session_digest.ct_eq(&actual)) {
            return Err(AuthError::Unauthorized);
        }
        self.inner.activity.touch();
        Ok(())
    }
}

impl InspectionHttpAuthGuard for DashboardAuth {
    fn authorize(&self, context: &InspectionHttpAuthContext<'_>) -> Result<(), InspectionError> {
        self.authorize_session(context)
            .map_err(AuthError::inspection)
    }
}

impl fmt::Debug for DashboardAuth {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("DashboardAuth")
            .field("origin", &self.inner.origin)
            .field("host", &self.inner.host)
            .field("secure_cookie", &self.inner.secure_cookie)
            .finish_non_exhaustive()
    }
}

fn random_token() -> Result<Zeroizing<String>, AuthError> {
    use base64::Engine as _;

    let mut bytes = Zeroizing::new([0_u8; TOKEN_BYTES]);
    SystemRandom::new()
        .fill(bytes.as_mut())
        .map_err(|_| AuthError::Entropy)?;
    Ok(Zeroizing::new(
        base64::engine::general_purpose::URL_SAFE_NO_PAD.encode(bytes.as_ref()),
    ))
}

fn token_digest(token: &[u8]) -> [u8; 32] {
    digest(&SHA256, token)
        .as_ref()
        .try_into()
        .expect("SHA-256 always returns 32 bytes")
}

fn valid_token_text(value: &str) -> bool {
    value.len() == TOKEN_TEXT_BYTES
        && value
            .bytes()
            .all(|byte| byte.is_ascii_alphanumeric() || matches!(byte, b'-' | b'_'))
}

fn session_cookie(headers: &HeaderMap) -> Result<Zeroizing<String>, AuthError> {
    let mut found = None;
    for raw in headers.get_all(COOKIE).iter() {
        let raw = raw.to_str().map_err(|_| AuthError::Unauthorized)?;
        for parsed in Cookie::split_parse(raw) {
            let cookie = parsed.map_err(|_| AuthError::Unauthorized)?;
            if cookie.name() == SESSION_COOKIE_NAME {
                if found.is_some() || !valid_token_text(cookie.value()) {
                    return Err(AuthError::Unauthorized);
                }
                found = Some(Zeroizing::new(cookie.value().to_owned()));
            }
        }
    }
    found.ok_or(AuthError::Unauthorized)
}

fn require_exact_header(
    headers: &HeaderMap,
    name: axum::http::header::HeaderName,
    expected: &str,
    error: AuthError,
) -> Result<(), AuthError> {
    if one_header(headers, name, error)? == expected {
        Ok(())
    } else {
        Err(error)
    }
}

fn one_header(
    headers: &HeaderMap,
    name: axum::http::header::HeaderName,
    error: AuthError,
) -> Result<&str, AuthError> {
    optional_one_header(headers, name, error)?.ok_or(error)
}

fn optional_one_header(
    headers: &HeaderMap,
    name: axum::http::header::HeaderName,
    error: AuthError,
) -> Result<Option<&str>, AuthError> {
    let mut values = headers.get_all(name).iter();
    let Some(value) = values.next() else {
        return Ok(None);
    };
    if values.next().is_some() {
        return Err(error);
    }
    value.to_str().map(Some).map_err(|_| error)
}

#[cfg(test)]
mod tests {
    use std::sync::{Arc, Barrier};
    use std::thread;

    use super::*;

    fn authority() -> (DashboardAuth, BootstrapSecret, ActivityTracker) {
        let activity = ActivityTracker::new();
        let (auth, secret) = DashboardAuth::new(
            "http://127.0.0.1:4040".into(),
            "127.0.0.1:4040".into(),
            false,
            activity.clone(),
        )
        .unwrap();
        (auth, secret, activity)
    }

    fn bootstrap_headers(secret: &BootstrapSecret) -> HeaderMap {
        let url = secret.launch_url("http://127.0.0.1:4040");
        let nonce = url.split("#bootstrap=").nth(1).unwrap();
        let mut headers = HeaderMap::new();
        headers.insert(HOST, HeaderValue::from_static("127.0.0.1:4040"));
        headers.insert(ORIGIN, HeaderValue::from_static("http://127.0.0.1:4040"));
        headers.insert(
            AUTHORIZATION,
            HeaderValue::from_str(&format!("Bootstrap {nonce}")).unwrap(),
        );
        headers
    }

    #[test]
    fn bootstrap_is_atomic_and_cookie_is_process_scoped() {
        let (auth, secret, _) = authority();
        let headers = bootstrap_headers(&secret);
        let auth = Arc::new(auth);
        let barrier = Arc::new(Barrier::new(8));
        let mut workers = Vec::new();
        for _ in 0..8 {
            let auth = Arc::clone(&auth);
            let headers = headers.clone();
            let barrier = Arc::clone(&barrier);
            workers.push(thread::spawn(move || {
                barrier.wait();
                auth.bootstrap(&headers)
            }));
        }
        let results = workers
            .into_iter()
            .map(|worker| worker.join().unwrap())
            .collect::<Vec<_>>();
        assert_eq!(results.iter().filter(|result| result.is_ok()).count(), 1);
        let cookie = results.into_iter().find_map(Result::ok).unwrap();
        let cookie = cookie.to_str().unwrap();
        assert!(cookie.contains("HttpOnly"));
        assert!(cookie.contains("SameSite=Strict"));
        assert!(cookie.contains("Path=/"));
        assert!(!cookie.contains("Secure"));
        assert!(!cookie.contains("Max-Age"));
        assert!(!cookie.contains("Expires"));
        assert!(auth.is_bootstrapped());
        assert_eq!(
            auth.bootstrap(&bootstrap_headers(&secret)),
            Err(AuthError::Unauthorized)
        );
    }

    #[test]
    fn session_requires_exact_host_cookie_and_unsafe_origin_then_invalidates() {
        let (auth, secret, activity) = authority();
        let set_cookie = auth.bootstrap(&bootstrap_headers(&secret)).unwrap();
        let parsed = Cookie::parse(set_cookie.to_str().unwrap()).unwrap();
        let request_cookie = format!("{}={}", parsed.name(), parsed.value());
        let mut headers = HeaderMap::new();
        headers.insert(HOST, HeaderValue::from_static("127.0.0.1:4040"));
        headers.insert(COOKIE, HeaderValue::from_str(&request_cookie).unwrap());
        let get = InspectionHttpAuthContext {
            method: &Method::GET,
            path: "/api/router/v1/status",
            headers: &headers,
            operations: false,
        };
        assert_eq!(auth.authorize(&get), Ok(()));
        assert!(activity.elapsed() < std::time::Duration::from_secs(1));

        let post = InspectionHttpAuthContext {
            method: &Method::POST,
            path: "/api/router/v1/neighborhood",
            headers: &headers,
            operations: false,
        };
        assert_eq!(auth.authorize(&post), Err(InspectionError::Forbidden));
        headers.insert(ORIGIN, HeaderValue::from_static("http://127.0.0.1:4040"));
        let post = InspectionHttpAuthContext {
            method: &Method::POST,
            path: "/api/router/v1/neighborhood",
            headers: &headers,
            operations: false,
        };
        assert_eq!(auth.authorize(&post), Ok(()));

        auth.invalidate();
        assert_eq!(auth.authorize(&post), Err(InspectionError::Forbidden));
        assert!(!format!("{auth:?}").contains(parsed.value()));
    }

    #[test]
    fn bootstrap_rejects_wrong_or_ambiguous_metadata_without_consumption() {
        let (auth, secret, _) = authority();
        let mut headers = bootstrap_headers(&secret);
        headers.insert(ORIGIN, HeaderValue::from_static("http://elsewhere.invalid"));
        assert_eq!(auth.bootstrap(&headers), Err(AuthError::Forbidden));
        assert!(!auth.is_bootstrapped());
        headers = bootstrap_headers(&secret);
        headers.append(HOST, HeaderValue::from_static("127.0.0.1:4040"));
        assert_eq!(auth.bootstrap(&headers), Err(AuthError::Forbidden));
        assert!(!auth.is_bootstrapped());
    }

    #[test]
    fn tls_authority_marks_the_session_cookie_secure() {
        let activity = ActivityTracker::new();
        let (auth, secret) = DashboardAuth::new(
            "https://127.0.0.1:4040".into(),
            "127.0.0.1:4040".into(),
            true,
            activity,
        )
        .unwrap();
        let url = secret.launch_url("https://127.0.0.1:4040");
        let nonce = url.split("#bootstrap=").nth(1).unwrap();
        let mut headers = HeaderMap::new();
        headers.insert(HOST, HeaderValue::from_static("127.0.0.1:4040"));
        headers.insert(ORIGIN, HeaderValue::from_static("https://127.0.0.1:4040"));
        headers.insert(
            AUTHORIZATION,
            HeaderValue::from_str(&format!("Bootstrap {nonce}")).unwrap(),
        );
        assert!(
            auth.bootstrap(&headers)
                .unwrap()
                .to_str()
                .unwrap()
                .contains("Secure")
        );
    }
}
