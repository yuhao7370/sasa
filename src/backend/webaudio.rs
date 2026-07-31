// WebAudio 后端(wasm32):AudioContext + AudioWorklet。
//
// 说明:ScriptProcessorNode 回调运行在主线程,游戏渲染繁忙时回调迟到会导致
// 输出碎裂(电流声/爆音/变调)。AudioWorklet 的 processor 运行在独立音频线程,
// 稳定输出;主线程定时器渲染 Mixer 输出后通过 MessagePort 发给 worklet。
#![cfg(all(feature = "webaudio", target_arch = "wasm32"))]

use super::{BackendSetup, State};
use crate::Backend;
use anyhow::{anyhow, Result};
use std::cell::RefCell;
use std::rc::Rc;
use std::sync::{
    atomic::{AtomicBool, Ordering},
    Arc, Mutex,
};

#[derive(Debug, Clone)]
pub struct WebAudioSettings {
    pub buffer_size: Option<usize>,
}

impl Default for WebAudioSettings {
    fn default() -> Self {
        Self { buffer_size: Some(4096) }
    }
}

/// AudioWorklet processor 源码。运行在音频线程:
/// 通过 port 接收主线程渲染好的交错立体声块,按需输出。
const WORKLET_JS: &str = r#"class NRProcessor extends AudioWorkletProcessor {
  constructor() {
    super();
    this.buf = new Float32Array(0);
    this.pos = 0;
    this.port.onmessage = (e) => {
      if (e.data === '__clear__') {
        this.buf = new Float32Array(0);
        this.pos = 0;
        return;
      }
      const block = e.data;
      const rest = this.buf.length - this.pos;
      const nb = new Float32Array(rest + block.length);
      nb.set(this.buf.subarray(this.pos), 0);
      nb.set(block, rest);
      this.buf = nb;
      this.pos = 0;
      // 低延迟:未播放缓冲超过 50ms(交错 stereo)时丢弃旧的,
      // 防止缓冲累积导致延迟增长(主线程偶尔繁忙时允许短暂缓冲)。
      const MAX_PENDING = 48000 * 2 * 0.050;
      if (this.buf.length > MAX_PENDING) {
        this.buf = this.buf.slice(this.buf.length - MAX_PENDING);
      }
    };
  }
  process(inputs, outputs) {
    const out = outputs[0];
    if (!out || out.length < 2) return true;
    const n = out[0].length;
    for (let i = 0; i < n; i++) {
      if (this.pos + 2 <= this.buf.length) {
        out[0][i] = this.buf[this.pos];
        out[1][i] = this.buf[this.pos + 1];
        this.pos += 2;
      } else {
        out[0][i] = 0;
        out[1][i] = 0;
      }
    }
    return true;
  }
}
registerProcessor('notrotaeno', NRProcessor);"#;

/// 通过 `new AudioWorkletNode(ctx, 'notrotaeno', { outputChannelCount: [2] })` 创建节点。
/// 注意:`AudioContext.createAudioWorkletNode` 并不存在,标准方式是该构造函数。
fn create_worklet_node(ctx: &web_sys::AudioContext) -> Result<web_sys::AudioWorkletNode, wasm_bindgen::JsValue> {
    use js_sys::{Array, Function, Object, Reflect};
    use wasm_bindgen::JsCast;
    let ctor: Function = Reflect::get(
        &js_sys::global(),
        &wasm_bindgen::JsValue::from_str("AudioWorkletNode"),
    )?
    .unchecked_into();
    let opts = Object::new();
    Reflect::set(
        &opts,
        &wasm_bindgen::JsValue::from_str("outputChannelCount"),
        &Array::of1(&wasm_bindgen::JsValue::from(2.0f64)),
    )?;
    let args = Array::of3(
        ctx.as_ref(),
        &wasm_bindgen::JsValue::from_str("notrotaeno"),
        &wasm_bindgen::JsValue::from(opts),
    );
    let ret = Reflect::construct(&ctor, &args)?;
    Ok(ret.unchecked_into())
}

pub struct WebAudioBackend {
    settings: WebAudioSettings,
    state: Option<Arc<Mutex<State>>>,
    broken: Arc<AtomicBool>,
    // 保持 JS 对象存活,防止被 GC 回收
    _ctx: Option<web_sys::AudioContext>,
    _node: Option<web_sys::AudioWorkletNode>,
    _closure: Option<wasm_bindgen::closure::ScopedClosure<'static, dyn FnMut(wasm_bindgen::JsValue)>>,
    _timer: Option<wasm_bindgen::closure::ScopedClosure<'static, dyn FnMut()>>,
}

impl WebAudioBackend {
    pub fn new(settings: WebAudioSettings) -> Self {
        Self {
            settings,
            state: None,
            broken: Arc::default(),
            _ctx: None,
            _node: None,
            _closure: None,
            _timer: None,
        }
    }
}

impl Backend for WebAudioBackend {
    fn setup(&mut self, setup: BackendSetup) -> Result<()> {
        self.state = Some(Arc::new(Mutex::new(setup.into())));
        Ok(())
    }

    fn start(&mut self) -> Result<()> {
        use wasm_bindgen::closure::Closure;
        use wasm_bindgen::JsCast;
        use web_sys::{Blob, MessagePort, Url};

        let ctx = web_sys::AudioContext::new().map_err(|e| anyhow!("AudioContext: {e:?}"))?;
        let sample_rate = ctx.sample_rate() as u32;

        {
            let state = self.state.as_ref().unwrap();
            if let Ok(mut guard) = state.lock() {
                guard.mixer.sample_rate = sample_rate;
            }
        }

        // 注册 AudioWorklet processor
        let mut blob_opts = web_sys::BlobPropertyBag::new();
        blob_opts.type_("application/javascript");
        let blob = Blob::new_with_str_sequence_and_options(
            &js_sys::Array::of1(&wasm_bindgen::JsValue::from_str(WORKLET_JS)),
            &blob_opts,
        )
        .map_err(|e| anyhow!("Blob: {e:?}"))?;
        let url = Url::create_object_url_with_blob(&blob)
            .map_err(|e| anyhow!("objectURL: {e:?}"))?;
        let worklet = ctx.audio_worklet().map_err(|e| anyhow!("audioWorklet: {e:?}"))?;
        let add_promise = worklet.add_module(&url).map_err(|e| anyhow!("addModule: {e:?}"))?;

        // addModule 异步完成:创建节点、连接、暴露 port
        let node_port: Rc<RefCell<Option<MessagePort>>> = Rc::new(RefCell::new(None));
        let node_port_done = Rc::clone(&node_port);
        let ctx_done = ctx.clone();
        let done = Closure::wrap(Box::new(move |_: wasm_bindgen::JsValue| {
            let node = create_worklet_node(&ctx_done);
            let node = match node {
                Ok(node) => node,
                Err(err) => {
                    web_sys::console::log_1(&wasm_bindgen::JsValue::from_str(&format!(
                        "[webaudio] worklet node failed: {err:?}"
                    )));
                    return;
                }
            };
            if let Ok(port) = node.port() {
                *node_port_done.borrow_mut() = Some(port);
            }
            if let Ok(dest) = ctx_done.destination().dyn_into::<web_sys::AudioNode>() {
                let _ = node.connect_with_audio_node(&dest);
            }
            let _ = ctx_done.resume();
        }) as Box<dyn FnMut(wasm_bindgen::JsValue)>);
        let _ = add_promise.then(&done);

        // 定时器:渲染量 = 实际经过的时间对应的样本数,保证速率恒为 1.0
        // (setInterval 触发间隔不精确,若固定渲染量会快于/慢于播放速率,
        // 造成音乐变速或缓冲耗尽)。
        let sample_rate_f64 = sample_rate as f64;
        let perf = web_sys::window()
            .ok_or_else(|| anyhow!("no window"))?
            .performance()
            .ok_or_else(|| anyhow!("no performance"))?;
        let perf_timer = perf.clone();
        let mut last_tick_ms = perf.now();
        let state_timer = Arc::clone(self.state.as_ref().unwrap());
        let node_port_timer = Rc::clone(&node_port);
        let ctx_timer = ctx.clone();
        let timer = Closure::wrap(Box::new(move || {
            // AudioContext 未运行(自动播放策略挂起)时不渲染,
            // 避免 worklet 缓冲累积加载期的旧/静音数据,resume 后造成播放延迟。
            if ctx_timer.state() != web_sys::AudioContextState::Running {
                return;
            }
            let now_ms = perf_timer.now();
            let mut elapsed_ms = (now_ms - last_tick_ms).max(0.0);
            last_tick_ms = now_ms;
            // 防止主线程长时间挂起后一次性渲染过多
            elapsed_ms = elapsed_ms.min(250.0);
            let frames = ((elapsed_ms / 1000.0) * sample_rate_f64).max(1.0) as usize;
            let mut data = vec![0.0f32; frames * 2];
            if let Ok(mut guard) = state_timer.lock() {
                guard.mixer.render_stereo(&mut data);
            }
            if let Some(port) = node_port_timer.borrow().as_ref() {
                let arr = js_sys::Float32Array::from(&data[..]);
                let _ = port.post_message(&arr);
            }
        }) as Box<dyn FnMut()>);
        let window = web_sys::window().ok_or_else(|| anyhow!("no window"))?;
        window
            .set_interval_with_callback_and_timeout_and_arguments_0(timer.as_ref().unchecked_ref(), 10)
            .map_err(|e| anyhow!("setInterval: {e:?}"))?;

        // 自动播放策略:初始化时 resume 可能被拒绝,注册交互监听,
        // 用户首次点击/按键时恢复 AudioContext。
        ctx.resume().map_err(|e| anyhow!("resume: {e:?}"))?;
        let resume_ctx = ctx.clone();
        let resume_port = Rc::clone(&node_port);
        let resume_closure = Closure::wrap(Box::new(
            move || {
                let _ = resume_ctx.resume();
                // 清空 worklet 缓冲,避免播放累积的旧/静音数据。
                if let Some(port) = resume_port.borrow().as_ref() {
                    let _ = port.post_message(&wasm_bindgen::JsValue::from_str("__clear__"));
                }
            },
        ) as Box<dyn FnMut()>);
        let document = web_sys::window()
            .and_then(|w| w.document())
            .ok_or_else(|| anyhow!("no document"))?;
        for event in ["pointerdown", "keydown", "touchstart"] {
            let _ = document.add_event_listener_with_callback(event, resume_closure.as_ref().unchecked_ref());
        }
        resume_closure.forget();

        self._closure = Some(done);
        self._timer = Some(timer);
        self._ctx = Some(ctx);
        Ok(())
    }

    fn consume_broken(&self) -> bool {
        self.broken.swap(false, Ordering::Relaxed)
    }
}
