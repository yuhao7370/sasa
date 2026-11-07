use anyhow::{Context, Result};
use cpal::{
    traits::{DeviceTrait, HostTrait},
    DeviceId,
};
use std::{
    sync::{
        atomic::{AtomicBool, Ordering},
        mpsc, Arc,
    },
    thread,
    time::Duration,
};

/// CPAL's WASAPI stream stays bound to its endpoint when Windows changes the default.
/// Query only on this worker; neither the audio callback nor the game loop enumerates devices.
pub(super) struct DefaultOutputMonitor {
    stop: mpsc::Sender<()>,
}

impl DefaultOutputMonitor {
    pub(super) fn new(initial: DeviceId, changed: Arc<AtomicBool>) -> Result<Self> {
        let (stop, receiver) = mpsc::channel();
        thread::Builder::new()
            .name("audio-output-route".into())
            .spawn(move || {
                let host = cpal::default_host();
                let mut previous = Some(initial);
                while matches!(
                    receiver.recv_timeout(Duration::from_millis(500)),
                    Err(mpsc::RecvTimeoutError::Timeout)
                ) {
                    let current = match host.default_output_device() {
                        Some(device) => match device.id() {
                            Ok(id) => Some(id),
                            // A failed identity query is not evidence of a different route.
                            Err(_) => continue,
                        },
                        None => None,
                    };
                    if current != previous {
                        previous = current;
                        changed.store(true, Ordering::Release);
                        eprintln!("[audio/cpal] Windows default output changed");
                    }
                }
            })
            .context("cannot start default audio output monitor")?;
        Ok(Self { stop })
    }
}

impl Drop for DefaultOutputMonitor {
    fn drop(&mut self) {
        // Interrupt the wait without blocking teardown on a driver/COM call.
        let _ = self.stop.send(());
    }
}
