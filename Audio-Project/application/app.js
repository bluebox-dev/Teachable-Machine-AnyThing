// เพิ่มโมเดลอื่นได้โดยวางโฟลเดอร์ไว้ข้าง my-model แล้วเพิ่มรายการในอาร์เรย์นี้
const BUILT_IN_MODELS = [
  { name: 'คำสั่งเสียง (hello, stop, yes, no)', path: './my-model/' },
];
const BACKGROUND_LABELS = ['Background Noise', '_background_noise_'];
const HISTORY_LIMIT = 30;
const LINK_KEY = 'model-link';
const RELEASE_DELAY_MS = 2000;
const REPEAT_MS = 1000; // หนึ่งคำอาจถูกจับได้หลายช่วงเมื่อใช้ overlap จึงไม่บันทึกคำเดิมซ้ำภายในเวลานี้

// ค่าเดียวกับที่ Teachable Machine ใช้สร้างสเปกโตรแกรมตอนเทรน
const SAMPLE_RATE = 44100;
const FRAMES = 43;
const BINS = 232;
const FFT_SIZE = 2048;
const HOP = 1024;
const SILENCE_DB = -160;

const $ = (id) => document.getElementById(id);

let recognizer = null;
let labels = [];
let listening = false;
let source = 'builtin';
let requestedKey = null;
let loadRequest = 0;
let keepSavedLink = false;
let lastWord = null;
let lastWordTime = 0;
let renderEpoch = 0;
let barRows = [];

function initTheme() {
  try {
    const saved = localStorage.getItem('theme');
    if (saved) document.documentElement.dataset.theme = saved;
  } catch (err) { /* ไม่มี localStorage ก็ใช้ธีมตามระบบ */ }
  $('theme-toggle').addEventListener('click', () => {
    const current = document.documentElement.dataset.theme
      || (matchMedia('(prefers-color-scheme: dark)').matches ? 'dark' : 'light');
    const next = current === 'dark' ? 'light' : 'dark';
    document.documentElement.dataset.theme = next;
    try { localStorage.setItem('theme', next); } catch (err) { /* ไม่ต้องจำก็ได้ */ }
  });
}

function setStatus(state, text) {
  $('status-pill').dataset.state = state;
  $('status-text').textContent = text;
}

function setModelMessage(text, isError = false) {
  $('model-message').textContent = text;
  $('model-message').classList.toggle('error', isError);
}

function showEmpty(title, text) {
  $('empty').hidden = false;
  $('empty-title').textContent = title;
  $('empty-text').textContent = text;
  const canvas = $('spectrogram');
  canvas.getContext('2d').clearRect(0, 0, canvas.width, canvas.height);
}

function resetStage() {
  if ($('tab-mic').getAttribute('aria-pressed') === 'true') {
    showEmpty('พร้อมฟัง', 'กดปุ่มเริ่มฟังด้านล่าง แล้วลองพูดคำที่เทรนไว้');
  } else {
    showEmpty('วางไฟล์เสียงที่นี่', 'หรือกดปุ่มเลือกไฟล์เสียงด้านล่าง');
  }
}

function showSource(next) {
  if (next === source) return;
  source = next;
  document.querySelectorAll('[data-source]').forEach((b) => {
    b.setAttribute('aria-pressed', String(b.dataset.source === next));
  });
  $('builtin-field').hidden = next !== 'builtin';
  $('url-field').hidden = next !== 'url';
  $('files-field').hidden = next !== 'files';
  setModelMessage('');
}

// รับลิงก์ได้ทั้งแบบมีและไม่มี / ท้ายลิงก์ และตัด query ออก
function normalizeModelUrl(text) {
  let url;
  try {
    url = new URL(text.trim());
  } catch (err) {
    throw new Error('ลิงก์ไม่ถูกต้อง ให้วางลิงก์เต็มจาก Teachable Machine');
  }
  url.search = '';
  url.hash = '';
  url.pathname = url.pathname.replace(/\/(model\.json|metadata\.json|weights\.bin)$/, '/');
  return url.href.endsWith('/') ? url.href : `${url.href}/`;
}

async function createRecognizerFromFiles(fileList) {
  const files = Array.from(fileList);
  const find = (name) => files.find((f) => f.name.toLowerCase() === name);
  const modelFile = find('model.json');
  const metadataFile = find('metadata.json');
  const weightsFile = files.find((f) => f.name.toLowerCase().endsWith('.bin'));
  if (!modelFile || !metadataFile || !weightsFile) {
    throw new Error('ต้องเลือกครบ 3 ไฟล์: model.json, metadata.json และไฟล์ .bin');
  }
  const metadata = JSON.parse(await metadataFile.text());
  checkModelType(metadata);
  const artifacts = await tf.io.browserFiles([modelFile, weightsFile]).load();
  return speechCommands.create('BROWSER_FFT', undefined, artifacts, metadata);
}

function checkModelType(metadata) {
  if (!metadata.wordLabels && !metadata.words) {
    throw new Error(`โมเดลนี้เป็นชนิด ${metadata.packageName || 'อื่น'} แอปนี้ใช้ได้เฉพาะโมเดล Audio`);
  }
}

async function createRecognizerFromBase(base, query = '') {
  const absolute = new URL(base, window.location.href).href;
  let response;
  try {
    response = await fetch(`${absolute}metadata.json${query}`);
  } catch (err) {
    throw new Error('เชื่อมต่อลิงก์นี้ไม่ได้ ตรวจสอบอินเทอร์เน็ตและลิงก์อีกครั้ง');
  }
  if (!response.ok) throw new Error('ไม่พบโมเดลที่ลิงก์นี้ ตรวจสอบว่าคัดลอกลิงก์มาครบ');
  checkModelType(await response.json());
  return speechCommands.create('BROWSER_FFT', undefined, `${absolute}model.json${query}`, `${absolute}metadata.json${query}`);
}

function buildBars() {
  barRows = labels.map((label) => {
    const row = document.createElement('div');
    row.className = 'bar';
    const head = document.createElement('div');
    head.className = 'bar-head';
    const name = document.createElement('span');
    name.textContent = label;
    const value = document.createElement('span');
    value.textContent = '0%';
    head.append(name, value);
    const track = document.createElement('div');
    track.className = 'bar-track';
    const fill = document.createElement('div');
    fill.className = 'bar-fill';
    track.append(fill);
    row.append(head, track);
    return { row, value, fill };
  });
  $('bars').replaceChildren(...barRows.map((r) => r.row));
}

// คีย์ของโมเดลที่เลือกอยู่ ใช้กันการโหลดซ้ำเมื่อเหตุการณ์เกิดติดกัน
function selectionKey() {
  if (source === 'url') return `url:${$('model-url').value.trim()}`;
  if (source === 'files') {
    return `files:${Array.from($('model-files').files).map((f) => `${f.name}:${f.size}:${f.lastModified}`).join('|')}`;
  }
  return `builtin:${$('builtin-model').value}`;
}

// เปลี่ยนโมเดลทันทีเมื่อผู้ใช้เลือกโมเดลอื่น ปุ่ม "ใช้โมเดลนี้" ยังใช้โหลดซ้ำได้
function loadIfChanged() {
  if (source === 'url' && !$('model-url').value.trim()) return;
  if (source === 'files' && !$('model-files').files.length) return;
  if (selectionKey() !== requestedKey) loadModel();
}

function rememberLink() {
  if (keepSavedLink) return;
  try {
    if (source === 'url') localStorage.setItem(LINK_KEY, $('model-url').value.trim());
    else localStorage.removeItem(LINK_KEY);
  } catch (err) { /* ไม่ต้องจำก็ได้ */ }
}

// เปิดแอปครั้งถัดไปด้วยลิงก์ล่าสุด ถ้าโหลดไม่ได้ให้กลับไปใช้โมเดลในโปรเจกต์
function loadedFromLink() {
  return Boolean(recognizer) && source === 'url' && requestedKey === selectionKey();
}

async function start() {
  let saved = null;
  try { saved = localStorage.getItem(LINK_KEY); } catch (err) { /* ใช้โมเดลในโปรเจกต์ */ }
  if (saved) {
    $('model-url').value = saved;
    showSource('url');
    await loadModel();
    if (loadedFromLink()) return;
    showSource('builtin');
  }
  // โหลดลิงก์ไม่ได้อาจเป็นเพราะเน็ตหลุดชั่วคราว จึงยังเก็บลิงก์ไว้ลองใหม่ตอนเปิดครั้งหน้า
  keepSavedLink = Boolean(saved);
  await loadModel();
  keepSavedLink = false;
  if (saved) setModelMessage('โหลดลิงก์ที่บันทึกไว้ไม่สำเร็จ จึงใช้โมเดลในโปรเจกต์แทน', true);
}

function releaseLayers(layersModel) {
  const variables = layersModel.weights;
  layersModel.dispose();
  variables.forEach((variable) => {
    try {
      variable.dispose();
    } catch (err) { /* ถูกคืนไปแล้วพร้อมโมเดล */ }
  });
}

// คืนหน่วยความจำของโมเดลเก่า โดยรอให้งานที่กำลังประมวลผลจบก่อน
function releaseModel(old) {
  if (!old) return;
  setTimeout(() => {
    try {
      if (old.model) releaseLayers(old.model);
    } catch (err) {
      console.warn(err);
    }
  }, RELEASE_DELAY_MS);
}

async function loadModel() {
  const request = ++loadRequest;
  requestedKey = selectionKey();
  setStatus('loading', 'กำลังโหลดโมเดล');
  setModelMessage('');
  $('load-model').disabled = true;
  try {
    let created;
    if (source === 'files') {
      created = await createRecognizerFromFiles($('model-files').files);
    } else if (source === 'url') {
      created = await createRecognizerFromBase(normalizeModelUrl($('model-url').value));
    } else {
      // เติม query เพื่อให้ได้ไฟล์ล่าสุดเมื่อมีการแทนที่ my-model
      created = await createRecognizerFromBase(BUILT_IN_MODELS[Number($('builtin-model').value)].path, `?t=${Date.now()}`);
    }
    await created.ensureModelLoaded();
    if (request !== loadRequest) {
      releaseModel(created);
      return;
    }
    await stopListening();
    releaseModel(recognizer);
    recognizer = created;
    labels = recognizer.wordLabels();
    buildBars();
    setStatus('ready', `พร้อมใช้งาน · ${labels.length} คลาส`);
    setModelMessage(`เปลี่ยนโมเดลแล้ว: ${labels.join(', ')}`);
    rememberLink();
    $('toggle-mic').disabled = false;
    $('audio-file').disabled = false;
    clearResult();
    resetStage();
  } catch (err) {
    if (request !== loadRequest) return;
    console.error(err);
    setStatus('error', 'โหลดโมเดลไม่สำเร็จ');
    setModelMessage(`${err.message}${recognizer ? ' (ยังใช้โมเดลเดิมอยู่)' : ''}`, true);
  } finally {
    if (request === loadRequest) $('load-model').disabled = false;
  }
}

// เพิ่มค่านี้เมื่อล้างผล เพื่อทิ้งงานที่ยังประมวลผลค้างจากก่อนหน้า
function clearResult() {
  renderEpoch += 1;
  $('top-label').textContent = 'ยังไม่มีผล';
  $('top-label').className = 'top-label waiting';
  $('top-score').textContent = '';
  $('history').replaceChildren();
  $('history-empty').hidden = false;
  barRows.forEach((r) => {
    r.row.classList.remove('top');
    r.value.textContent = '0%';
    r.fill.style.width = '0';
  });
}

function renderScores(scores) {
  let best = 0;
  scores.forEach((s, i) => { if (s > scores[best]) best = i; });
  barRows.forEach((r, i) => {
    const percent = (scores[i] * 100).toFixed(0);
    r.value.textContent = `${percent}%`;
    r.fill.style.width = `${percent}%`;
    r.row.classList.toggle('top', i === best);
  });
  const top = $('top-label');
  const changed = top.textContent !== labels[best];
  top.textContent = labels[best];
  top.className = BACKGROUND_LABELS.includes(labels[best]) ? 'top-label unsure' : 'top-label';
  if (changed && !BACKGROUND_LABELS.includes(labels[best])) {
    void top.offsetWidth; // เริ่มแอนิเมชันใหม่ทุกครั้งที่ได้ยินคำใหม่
    top.classList.add('flash');
  }
  $('top-score').textContent = `${(scores[best] * 100).toFixed(0)}%`;
  return best;
}

function addHistory(label, score, note) {
  const li = document.createElement('li');
  const word = document.createElement('strong');
  word.textContent = label;
  const detail = document.createElement(note ? 'span' : 'time');
  detail.className = note ? 'note' : '';
  detail.textContent = note || new Date().toLocaleTimeString('th-TH');
  const value = document.createElement('span');
  value.className = 'score';
  value.textContent = `${(score * 100).toFixed(0)}%`;
  li.append(word, detail, value);
  $('history').prepend(li);
  $('history-empty').hidden = true;
  while ($('history').children.length > HISTORY_LIMIT) $('history').lastChild.remove();
}

// ไล่สีจากเงียบไปดัง
const COLOR_STOPS = [[11, 10, 31], [74, 26, 110], [181, 54, 122], [247, 111, 92], [253, 226, 163]];
const COLORS = Array.from({ length: 256 }, (_, i) => {
  const position = (i / 255) * (COLOR_STOPS.length - 1);
  const low = Math.floor(position);
  const high = Math.min(low + 1, COLOR_STOPS.length - 1);
  const mix = position - low;
  return COLOR_STOPS[low].map((c, k) => c + (COLOR_STOPS[high][k] - c) * mix);
});

// วาดสเปกโตรแกรม: แกนนอนคือเวลา แกนตั้งคือความถี่ (ความถี่ต่ำอยู่ด้านล่าง)
function drawSpectrogram(data, frameSize) {
  $('empty').hidden = true;
  const canvas = $('spectrogram');
  const ctx = canvas.getContext('2d');
  const frameCount = Math.floor(data.length / frameSize);
  const image = ctx.createImageData(frameCount, frameSize);
  let min = Infinity;
  let max = -Infinity;
  data.forEach((v) => {
    if (v < min) min = v;
    if (v > max) max = v;
  });
  const range = max - min || 1;
  for (let f = 0; f < frameCount; f++) {
    for (let b = 0; b < frameSize; b++) {
      const color = COLORS[Math.round(((data[f * frameSize + b] - min) / range) * 255)];
      const px = ((frameSize - 1 - b) * frameCount + f) * 4;
      image.data[px] = color[0];
      image.data[px + 1] = color[1];
      image.data[px + 2] = color[2];
      image.data[px + 3] = 255;
    }
  }
  const buffer = document.createElement('canvas');
  buffer.width = frameCount;
  buffer.height = frameSize;
  buffer.getContext('2d').putImageData(image, 0, 0);
  ctx.imageSmoothingEnabled = false;
  ctx.drawImage(buffer, 0, 0, canvas.width, canvas.height);
}

async function startListening() {
  const active = recognizer;
  await active.listen((result) => {
    if (active !== recognizer) return;
    const scores = Array.from(result.scores);
    const best = renderScores(scores);
    if (result.spectrogram) drawSpectrogram(result.spectrogram.data, result.spectrogram.frameSize);
    const now = performance.now();
    if (!BACKGROUND_LABELS.includes(labels[best]) && (labels[best] !== lastWord || now - lastWordTime >= REPEAT_MS)) {
      addHistory(labels[best], scores[best]);
    }
    if (!BACKGROUND_LABELS.includes(labels[best])) {
      lastWord = labels[best];
      lastWordTime = now;
    }
  }, {
    includeSpectrogram: true,
    probabilityThreshold: Number($('threshold').value),
    invokeCallbackOnNoiseAndUnknown: true,
    overlapFactor: Number($('overlap').value),
  });
  listening = true;
  $('toggle-mic').textContent = 'หยุดฟัง';
  $('listening').hidden = false;
  $('threshold').disabled = true;
  $('overlap').disabled = true;
  showEmpty('กำลังฟัง...', 'ลองพูดคำที่เทรนไว้');
}

async function stopListening() {
  if (recognizer && listening) await recognizer.stopListening();
  listening = false;
  $('toggle-mic').textContent = 'เริ่มฟัง';
  $('listening').hidden = true;
  $('threshold').disabled = false;
  $('overlap').disabled = false;
}

async function toggleMic() {
  try {
    if (listening) {
      await stopListening();
    } else {
      await startListening();
    }
  } catch (err) {
    console.error(err);
    listening = false;
    showEmpty('เปิดไมโครโฟนไม่ได้', 'ตรวจสอบว่าอนุญาตให้เบราว์เซอร์ใช้ไมโครโฟนแล้ว และเปิดผ่าน localhost');
  }
}

// สร้างสเปกโตรแกรมจากไฟล์เสียงด้วย AnalyserNode ตัวเดียวกับที่ใช้ตอนฟังจากไมค์
async function spectrogramFromFile(file) {
  const encoded = await file.arrayBuffer();
  const decodeContext = new OfflineAudioContext(1, SAMPLE_RATE, SAMPLE_RATE);
  const audio = await decodeContext.decodeAudioData(encoded);
  const frameTotal = Math.max(FRAMES, Math.floor((audio.duration * SAMPLE_RATE) / HOP));
  const context = new OfflineAudioContext(1, (frameTotal + 1) * HOP, SAMPLE_RATE);
  const sourceNode = context.createBufferSource();
  sourceNode.buffer = audio;
  const analyser = context.createAnalyser();
  analyser.fftSize = FFT_SIZE;
  analyser.smoothingTimeConstant = 0;
  sourceNode.connect(analyser);
  analyser.connect(context.destination);
  sourceNode.start();
  const frames = [];
  for (let f = 1; f <= frameTotal; f++) {
    context.suspend((f * HOP) / SAMPLE_RATE).then(() => {
      const data = new Float32Array(FFT_SIZE / 2);
      analyser.getFloatFrequencyData(data);
      frames.push(data.slice(0, BINS).map((v) => (Number.isFinite(v) ? Math.max(v, SILENCE_DB) : SILENCE_DB)));
      context.resume();
    });
  }
  await context.startRendering();
  return frames;
}

// ไฟล์ยาวกว่า 1 วินาที: เลือกช่วง 43 เฟรมที่มีพลังงานมากที่สุด
function loudestWindow(frames) {
  const energy = frames.map((frame) => frame.reduce((sum, v) => sum + 10 ** (v / 10), 0));
  let bestStart = 0;
  let bestEnergy = -1;
  for (let start = 0; start + FRAMES <= frames.length; start++) {
    let total = 0;
    for (let i = start; i < start + FRAMES; i++) total += energy[i];
    if (total > bestEnergy) {
      bestEnergy = total;
      bestStart = start;
    }
  }
  return frames.slice(bestStart, bestStart + FRAMES);
}

async function classifyFile(file) {
  const epoch = renderEpoch;
  const flat = new Float32Array(FRAMES * BINS);
  loudestWindow(await spectrogramFromFile(file)).forEach((frame, i) => flat.set(frame, i * BINS));
  const normalized = tf.tidy(() => {
    const x = tf.tensor4d(flat, [1, FRAMES, BINS, 1]);
    const { mean, variance } = tf.moments(x);
    return x.sub(mean).div(variance.sqrt().add(1e-6));
  });
  const result = await recognizer.recognize(normalized);
  normalized.dispose();
  if (epoch !== renderEpoch) return null;
  const scores = Array.from(result.scores);
  drawSpectrogram(flat, BINS);
  const player = $('player');
  player.src = URL.createObjectURL(file);
  player.hidden = false;
  const best = renderScores(scores);
  addHistory(labels[best], scores[best], file.name);
  return scores;
}

async function classifyFiles(fileList) {
  if (!recognizer) return;
  const audioFiles = Array.from(fileList).filter((f) => f.type.startsWith('audio/') || /\.(wav|mp3|m4a|ogg|webm)$/i.test(f.name));
  if (!audioFiles.length) {
    showEmpty('ไฟล์นี้ไม่ใช่ไฟล์เสียง', 'ลองเลือกไฟล์ .wav หรือ .mp3');
    return;
  }
  try {
    for (const file of audioFiles) {
      if (!await classifyFile(file)) return;
    }
  } catch (err) {
    console.error(err);
    showEmpty('อ่านไฟล์เสียงไม่ได้', err.message);
  }
}

function switchInput(useMic) {
  $('tab-mic').setAttribute('aria-pressed', String(useMic));
  $('tab-file').setAttribute('aria-pressed', String(!useMic));
  $('mic-panel').hidden = !useMic;
  $('file-panel').hidden = useMic;
  $('player').hidden = true;
  stopListening();
  clearResult();
  resetStage();
}

function initDropZone() {
  const viewport = $('viewport');
  const isFileMode = () => $('tab-file').getAttribute('aria-pressed') === 'true';
  viewport.addEventListener('dragover', (e) => {
    e.preventDefault();
    if (isFileMode()) viewport.classList.add('dragging');
  });
  viewport.addEventListener('dragleave', () => viewport.classList.remove('dragging'));
  viewport.addEventListener('drop', (e) => {
    e.preventDefault();
    viewport.classList.remove('dragging');
    if (!isFileMode()) switchInput(false);
    classifyFiles(e.dataTransfer.files);
  });
}

initTheme();
BUILT_IN_MODELS.forEach((m, i) => $('builtin-model').add(new Option(m.name, String(i))));
document.querySelectorAll('[data-source]').forEach((b) => {
  b.addEventListener('click', () => {
    showSource(b.dataset.source);
    loadIfChanged();
  });
});
$('builtin-model').addEventListener('change', loadIfChanged);
$('model-url').addEventListener('change', loadIfChanged);
$('model-url').addEventListener('paste', () => setTimeout(loadIfChanged));
$('model-url').addEventListener('keydown', (e) => {
  if (e.key === 'Enter' && $('model-url').value.trim()) {
    e.preventDefault();
    loadModel();
  }
});
$('model-files').addEventListener('change', (e) => {
  const names = Array.from(e.target.files).map((f) => f.name).join(', ');
  $('model-files-text').textContent = names || 'เลือก model.json, metadata.json และไฟล์ .bin พร้อมกัน';
  loadIfChanged();
});
$('threshold').addEventListener('input', (e) => { $('threshold-value').textContent = Number(e.target.value).toFixed(2); });
$('overlap').addEventListener('input', (e) => { $('overlap-value').textContent = Number(e.target.value).toFixed(2); });
$('load-model').addEventListener('click', loadModel);
$('toggle-mic').addEventListener('click', toggleMic);
$('tab-mic').addEventListener('click', () => switchInput(true));
$('tab-file').addEventListener('click', () => switchInput(false));
$('audio-file').addEventListener('change', (e) => {
  classifyFiles(e.target.files);
  e.target.value = '';
});
initDropZone();
start();
