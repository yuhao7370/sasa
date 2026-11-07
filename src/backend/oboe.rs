pub use oboe::{PerformanceMode, Usage};
use parking_lot::Mutex;

use super::BackendSetup;
use crate::{backend::State, Backend};
use anyhow::{ensure, Context, Result};
use oboe::{
    AudioOutputCallback, AudioOutputStreamSafe, AudioStream, AudioStreamAsync, AudioStreamBase,
    AudioStreamBuilder, AudioStreamSafe, DataCallbackResult, Output, SharingMode, Stereo,
};
use std::{
    sync::{
        atomic::{AtomicBool, AtomicI32, AtomicU8, Ordering},
        Arc,
    },
    time::{Duration, Instant},
};

const CLOSING: u8 = 1;
const CLOSED: u8 = 2;

#[derive(Default)]
struct StreamSignals {
    phase: AtomicU8,
    broken: AtomicBool,
    error: AtomicI32,
    xruns: AtomicI32,
    buffer_frames: AtomicI32,
}

pub struct OboeSettings {
    pub buffer_size: Option<u32>,
    pub performance_mode: PerformanceMode,
    pub usage: Usage,
}
impl Default for OboeSettings {
    fn default() -> Self {
        Self {
            buffer_size: None,
            performance_mode: PerformanceMode::None,
            usage: Usage::Media,
        }
    }
}

pub struct OboeBackend {
    settings: OboeSettings,
    stream: Option<AudioStreamAsync<Output, OboeCallback>>,
    state: Option<Arc<Mutex<State>>>,
    signals: Arc<StreamSignals>,
    next_report: Instant,
    reported_xruns: i32,
}

impl OboeBackend {
    pub fn new(settings: OboeSettings) -> Self {
        Self {
            settings,
            stream: None,
            state: None,
            signals: Arc::default(),
            next_report: Instant::now(),
            reported_xruns: 0,
        }
    }
}

impl Backend for OboeBackend {
    fn setup(&mut self, setup: BackendSetup) -> Result<()> {
        self.state = Some(Arc::new(Mutex::new(setup.into())));
        Ok(())
    }

    fn start(&mut self) -> Result<()> {
        ensure!(!self.is_closing(), "previous Oboe stream is still closing");
        // Close before opening: two callbacks must not advance the same mixer.
        drop(self.stream.take());
        if let Some(size) = self.settings.buffer_size {
            ensure!(
                size > 0 && size <= i32::MAX as u32,
                "invalid audio buffer size"
            );
        }
        let signals = Arc::new(StreamSignals::default());
        let mut stream = AudioStreamBuilder::default()
            .set_usage(self.settings.usage)
            .set_performance_mode(self.settings.performance_mode)
            .set_sharing_mode(SharingMode::Exclusive)
            .set_format::<f32>()
            .set_channel_count::<Stereo>()
            .set_callback(OboeCallback::new(
                Arc::clone(self.state.as_ref().unwrap()),
                Arc::clone(&signals),
                self.settings.buffer_size.is_none(),
            ))
            .open_stream()
            .context("failed to open Oboe output")?;
        if let Some(size) = self.settings.buffer_size {
            // Apply once and allow increasing the size; min(current_size) prevented this.
            if let Err(error) = stream.set_buffer_size_in_frames(size as i32) {
                eprintln!("[audio/oboe] buffer request rejected ({error:?}); using device default");
            }
        }
        signals
            .buffer_frames
            .store(stream.get_buffer_size_in_frames(), Ordering::Relaxed);
        eprintln!("[audio/oboe] opening: device={} api={:?} rate={} burst={} buffer={} capacity={} sharing={:?} performance={:?}",
            stream.get_device_id(), stream.get_audio_api(), stream.get_sample_rate(),
            stream.get_frames_per_burst(), stream.get_buffer_size_in_frames(),
            stream.get_buffer_capacity_in_frames(), stream.get_sharing_mode(),
            stream.get_performance_mode());
        self.signals = signals;
        self.reported_xruns = 0;
        self.next_report = Instant::now() + Duration::from_secs(1);
        // Retain ownership even if start fails and an asynchronous error is closing the stream.
        self.stream = Some(stream);
        self.stream
            .as_mut()
            .unwrap()
            .start()
            .context("failed to start Oboe output")?;
        Ok(())
    }

    fn consume_broken(&self) -> bool {
        let broken = self.signals.broken.swap(false, Ordering::Acquire);
        if broken {
            eprintln!(
                "[audio/oboe] disconnected: error={} xruns={} buffer={}",
                self.signals.error.load(Ordering::Relaxed),
                self.signals.xruns.load(Ordering::Relaxed),
                self.signals.buffer_frames.load(Ordering::Relaxed)
            );
        }
        broken
    }

    fn is_closing(&self) -> bool {
        self.signals.phase.load(Ordering::Acquire) == CLOSING
    }

    fn update(&mut self) {
        if Instant::now() < self.next_report {
            return;
        }
        self.next_report = Instant::now() + Duration::from_secs(1);
        let xruns = self.signals.xruns.load(Ordering::Relaxed);
        if xruns != self.reported_xruns {
            eprintln!(
                "[audio/oboe] underruns: total={xruns} delta={} buffer={}",
                xruns - self.reported_xruns,
                self.signals.buffer_frames.load(Ordering::Relaxed)
            );
            self.reported_xruns = xruns;
        }
    }
}

struct OboeCallback {
    state: Arc<Mutex<State>>,
    signals: Arc<StreamSignals>,
    auto_buffer: bool,
    frames_until_poll: usize,
    previous_xruns: Option<i32>,
    xrun_supported: bool,
}

impl OboeCallback {
    pub fn new(state: Arc<Mutex<State>>, signals: Arc<StreamSignals>, auto_buffer: bool) -> Self {
        Self {
            state,
            signals,
            auto_buffer,
            frames_until_poll: 0,
            previous_xruns: None,
            xrun_supported: true,
        }
    }
}

impl AudioOutputCallback for OboeCallback {
    type FrameType = (f32, Stereo);

    fn on_audio_ready(
        &mut self,
        stream: &mut dyn AudioOutputStreamSafe,
        frames: &mut [(f32, f32)],
    ) -> DataCallbackResult {
        let sample_rate = stream.get_sample_rate().max(1) as u32;
        let mut latency = None;
        if self.frames_until_poll <= frames.len() {
            // About 10 polls/second, independent of callback/burst size. No logging or allocation.
            self.frames_until_poll = (sample_rate / 10).max(1) as usize;
            latency = stream.calculate_latency_millis().ok();
            if self.xrun_supported {
                match stream.get_xrun_count() {
                    Ok(xruns) => {
                        self.signals.xruns.store(xruns, Ordering::Relaxed);
                        if self.auto_buffer && self.previous_xruns.is_some_and(|old| xruns > old) {
                            let current = stream.get_buffer_size_in_frames();
                            let next = current
                                .saturating_add(stream.get_frames_per_burst().max(1))
                                .min(stream.get_buffer_capacity_in_frames());
                            if next > current {
                                match stream.set_buffer_size_in_frames(next) {
                                    Ok(actual) => {
                                        self.signals.buffer_frames.store(actual, Ordering::Relaxed)
                                    }
                                    Err(_) => self.auto_buffer = false,
                                }
                            }
                        }
                        self.previous_xruns = Some(xruns);
                    }
                    Err(oboe::Error::Unimplemented) => self.xrun_supported = false,
                    Err(_) => (),
                }
            }
        } else {
            self.frames_until_poll -= frames.len();
        }

        let mut state = self.state.lock();
        if let Some(latency) = latency {
            state.recorder.push((latency / 1000.) as f32);
        }
        state.mixer.sample_rate = sample_rate;
        let raw = frames.as_mut_ptr();
        state.mixer.render_stereo(unsafe {
            std::slice::from_raw_parts_mut(raw as *mut f32, frames.len() * 2)
        });

        DataCallbackResult::Continue
    }

    fn on_error_before_close(
        &mut self,
        _audio_stream: &mut dyn oboe::AudioOutputStreamSafe,
        error: oboe::Error,
    ) {
        self.signals.error.store(error as i32, Ordering::Relaxed);
        self.signals.phase.store(CLOSING, Ordering::Release);
    }

    fn on_error_after_close(
        &mut self,
        _audio_stream: &mut dyn oboe::AudioOutputStreamSafe,
        error: oboe::Error,
    ) {
        self.signals.error.store(error as i32, Ordering::Relaxed);
        self.signals.broken.store(true, Ordering::Release);
        // Publish last: a polling owner may only replace the stream after close completed.
        self.signals.phase.store(CLOSED, Ordering::Release);
    }
}
