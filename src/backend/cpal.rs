use crate::Backend;
use anyhow::{anyhow, Context, Result};
use cpal::{
    traits::{DeviceTrait, HostTrait, StreamTrait},
    BufferSize, FromSample, OutputCallbackInfo, Sample, SampleFormat, Stream, StreamError, I24,
    U24,
};
use std::sync::{
    atomic::{AtomicBool, Ordering},
    Arc,
};

use super::{BackendSetup, StateCell};

#[derive(Debug, Clone, Default)]
pub struct CpalSettings {
    pub buffer_size: Option<u32>,
}

pub struct CpalBackend {
    settings: CpalSettings,
    stream: Option<Stream>,
    broken: Arc<AtomicBool>,
    state: Option<Arc<StateCell>>,
}

impl CpalBackend {
    pub fn new(settings: CpalSettings) -> Self {
        Self {
            settings,
            stream: None,
            broken: Arc::default(),
            state: None,
        }
    }
}

fn write_output<T: Sample + FromSample<f32>>(
    state: &StateCell,
    channels: usize,
    data: &mut [T],
    info: &OutputCallbackInfo,
    stereo_scratch: &mut Vec<f32>,
    out_scratch: &mut Vec<f32>,
) {
    let channels = channels.max(1);
    let frame_count = data.len() / channels;
    let sample_count = frame_count * channels;
    let (data, tail) = data.split_at_mut(sample_count);
    let (mixer, rec) = state.get();

    let rendered: &[f32] = if channels == 1 {
        out_scratch.resize(frame_count, 0.0);
        mixer.render_mono(out_scratch);
        out_scratch.as_slice()
    } else if channels == 2 {
        out_scratch.resize(frame_count * 2, 0.0);
        mixer.render_stereo(out_scratch);
        out_scratch.as_slice()
    } else {
        // Render as stereo first, then expand per frame.
        // This keeps time progression tied to frame count, not channel count.
        stereo_scratch.resize(frame_count * 2, 0.0);
        mixer.render_stereo(stereo_scratch);

        out_scratch.resize(frame_count * channels, 0.0);
        for (stereo, frame) in stereo_scratch
            .chunks_exact(2)
            .zip(out_scratch.chunks_exact_mut(channels))
        {
            let left = stereo[0];
            let right = stereo[1];
            frame[0] = left;
            frame[1] = right;
            let surround = (left + right) * 0.5;
            frame[2..].fill(surround);
        }
        out_scratch.as_slice()
    };

    for (dst, src) in data.iter_mut().zip(rendered.iter().copied()) {
        *dst = T::from_sample(src);
    }
    for sample in tail {
        *sample = T::from_sample(0.0);
    }

    let ts = info.timestamp();
    if let Some(delay) = ts.playback.duration_since(&ts.callback) {
        rec.push(delay.as_secs_f32());
    }
}

fn write_output_f32(
    state: &StateCell,
    channels: usize,
    data: &mut [f32],
    info: &OutputCallbackInfo,
    stereo_scratch: &mut Vec<f32>,
) {
    let channels = channels.max(1);
    let frame_count = data.len() / channels;
    let sample_count = frame_count * channels;
    let (data, tail) = data.split_at_mut(sample_count);
    let (mixer, rec) = state.get();

    if channels == 1 {
        mixer.render_mono(data);
    } else if channels == 2 {
        mixer.render_stereo(data);
    } else {
        // Render as stereo first, then expand per frame.
        // This keeps time progression tied to frame count, not channel count.
        stereo_scratch.resize(frame_count * 2, 0.0);
        mixer.render_stereo(stereo_scratch);

        for (stereo, frame) in stereo_scratch
            .chunks_exact(2)
            .zip(data.chunks_exact_mut(channels))
        {
            let left = stereo[0];
            let right = stereo[1];
            frame[0] = left;
            frame[1] = right;
            let surround = (left + right) * 0.5;
            frame[2..].fill(surround);
        }
    }
    tail.fill(0.0);

    let ts = info.timestamp();
    if let Some(delay) = ts.playback.duration_since(&ts.callback) {
        rec.push(delay.as_secs_f32());
    }
}

fn make_error_callback(broken: Arc<AtomicBool>) -> impl FnMut(StreamError) + Send + 'static {
    move |err| {
        eprintln!("audio error: {err:?}");
        if matches!(err, StreamError::DeviceNotAvailable) {
            broken.store(true, Ordering::Relaxed);
        }
    }
}

impl Backend for CpalBackend {
    fn setup(&mut self, setup: BackendSetup) -> Result<()> {
        self.state = Some(Arc::new(setup.into()));
        Ok(())
    }

    fn start(&mut self) -> Result<()> {
        let host = cpal::default_host();
        let device = host
            .default_output_device()
            .ok_or_else(|| anyhow!("no default output device is found"))?;
        let default_config = device
            .default_output_config()
            .context("cannot get output config")?;
        let sample_format = default_config.sample_format();
        let mut config = default_config.config();
        config.buffer_size = self
            .settings
            .buffer_size
            .map_or(BufferSize::Default, BufferSize::Fixed);
        let channels = config.channels.max(1) as usize;

        let state = Arc::clone(self.state.as_ref().unwrap());
        state.get().0.sample_rate = config.sample_rate;

        macro_rules! build_stream {
            ($sample_ty:ty) => {{
                let state = Arc::clone(&state);
                let mut stereo_scratch = Vec::<f32>::new();
                let mut out_scratch = Vec::<f32>::new();
                device.build_output_stream(
                    &config,
                    move |data: &mut [$sample_ty], info: &OutputCallbackInfo| {
                        write_output(
                            state.as_ref(),
                            channels,
                            data,
                            info,
                            &mut stereo_scratch,
                            &mut out_scratch,
                        );
                    },
                    make_error_callback(Arc::clone(&self.broken)),
                    None,
                )
            }};
        }
        macro_rules! build_stream_f32 {
            () => {{
                let state = Arc::clone(&state);
                let mut stereo_scratch = Vec::<f32>::new();
                device.build_output_stream(
                    &config,
                    move |data: &mut [f32], info: &OutputCallbackInfo| {
                        write_output_f32(state.as_ref(), channels, data, info, &mut stereo_scratch);
                    },
                    make_error_callback(Arc::clone(&self.broken)),
                    None,
                )
            }};
        }

        let stream = match sample_format {
            SampleFormat::I8 => build_stream!(i8),
            SampleFormat::I16 => build_stream!(i16),
            SampleFormat::I24 => build_stream!(I24),
            SampleFormat::I32 => build_stream!(i32),
            SampleFormat::I64 => build_stream!(i64),
            SampleFormat::U8 => build_stream!(u8),
            SampleFormat::U16 => build_stream!(u16),
            SampleFormat::U24 => build_stream!(U24),
            SampleFormat::U32 => build_stream!(u32),
            SampleFormat::U64 => build_stream!(u64),
            SampleFormat::F32 => build_stream_f32!(),
            SampleFormat::F64 => build_stream!(f64),
            _ => Err(cpal::BuildStreamError::StreamConfigNotSupported),
        }
        .context("failed to build stream")?;

        stream.play()?;
        self.stream = Some(stream);
        Ok(())
    }

    fn consume_broken(&self) -> bool {
        self.broken.fetch_and(false, Ordering::Relaxed)
    }
}
