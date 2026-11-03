// เพิ่มโมเดลอื่นได้โดยวางโฟลเดอร์ไว้ข้าง my-model แล้วเพิ่มรายการในอาร์เรย์นี้
const BUILT_IN_MODELS = [
  { name: 'สิ่งของ 5 ชนิด (pen, phone, card, bag, mouse computer)', path: './my-model/' },
];
const MODEL_PACKAGE = '@teachablemachine/image';
const WEBCAM_SIZE = 480;
const MIN_CONFIDENCE = 0.6;
const MAX_THUMBS = 24;
const LINK_KEY = 'model-link';
const RELEASE_DELAY_MS = 2000;

const $ = (id) => document.getElementById(id);

let model = null;
let webcam = null;
let loopId = null;
let source = 'builtin';
let requestedKey = null;
let loadRequest = 0;
let keepSavedLink = false;
let renderEpoch = 0;
let barRows = [];
let lastFrameTime = 0;

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

function showMedia(node) {
  $('frame').replaceChildren(node);
  $('frame').classList.add('has-media');
  $('empty').hidden = true;
}

function showEmpty(title, text) {
  $('frame').replaceChildren();
  $('frame').classList.remove('has-media');
  $('empty').hidden = false;
  $('empty-title').textContent = title;
  $('empty-text').textContent = text;
  $('live-caption').hidden = true;
  $('fps').textContent = '';
}

function buildBars(labels) {
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
    return { label, row, value, fill };
  });
  $('bars').replaceChildren(...barRows.map((r) => r.row));
}

// เพิ่มค่านี้เมื่อล้างผล เพื่อทิ้งงานที่ยังประมวลผลค้างจากก่อนหน้า
function clearResult() {
  renderEpoch += 1;
  $('top-label').textContent = 'ยังไม่มีผล';
  $('top-label').className = 'top-label waiting';
  $('top-score').textContent = '';
  $('live-caption').hidden = true;
  barRows.forEach((r) => {
    r.row.classList.remove('top');
    r.value.textContent = '0%';
    r.fill.style.width = '0';
  });
}

function renderPredictions(predictions) {
  let best = predictions[0];
  predictions.forEach((p) => { if (p.probability > best.probability) best = p; });
  predictions.forEach((p, i) => {
    const percent = (p.probability * 100).toFixed(0);
    barRows[i].value.textContent = `${percent}%`;
    barRows[i].fill.style.width = `${percent}%`;
    barRows[i].row.classList.toggle('top', p === best);
  });
  const sure = best.probability >= MIN_CONFIDENCE;
  const score = `${(best.probability * 100).toFixed(0)}%`;
  $('top-label').textContent = sure ? best.className : 'ไม่แน่ใจ';
  $('top-label').className = sure ? 'top-label' : 'top-label unsure';
  $('top-score').textContent = score;
  $('live-label').textContent = sure ? best.className : 'ไม่แน่ใจ';
  $('live-score').textContent = score;
  $('live-caption').hidden = false;
  return best;
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

function pickModelFiles(fileList) {
  const files = Array.from(fileList);
  const find = (name) => files.find((f) => f.name.toLowerCase() === name);
  const picked = {
    model: find('model.json'),
    metadata: find('metadata.json'),
    weights: files.find((f) => f.name.toLowerCase().endsWith('.bin')),
  };
  if (!picked.model || !picked.metadata || !picked.weights) {
    throw new Error('ต้องเลือกครบ 3 ไฟล์: model.json, metadata.json และไฟล์ .bin');
  }
  return picked;
}

async function fetchMetadata(url) {
  let response;
  try {
    response = await fetch(url);
  } catch (err) {
    throw new Error('เชื่อมต่อลิงก์นี้ไม่ได้ ตรวจสอบอินเทอร์เน็ตและลิงก์อีกครั้ง');
  }
  if (!response.ok) throw new Error('ไม่พบโมเดลที่ลิงก์นี้ ตรวจสอบว่าคัดลอกลิงก์มาครบ');
  return response.json();
}

// ตรวจจาก metadata ก่อนโหลด เพราะไลบรารีอาจเขียนทับ packageName ตอนโหลดโมเดล
function checkModelType(metadata) {
  const type = metadata.tfjsSpeechCommandsVersion ? '@tensorflow-models/speech-commands' : metadata.packageName;
  if (type && type !== MODEL_PACKAGE) {
    throw new Error(`โมเดลนี้เป็นชนิด ${type} แอปนี้ใช้ได้เฉพาะโมเดล Image`);
  }
}

async function loadFromBase(base, query = '') {
  checkModelType(await fetchMetadata(`${base}metadata.json${query}`));
  return tmImage.load(`${base}model.json${query}`, `${base}metadata.json${query}`);
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
  return Boolean(model) && source === 'url' && requestedKey === selectionKey();
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
      releaseLayers(old.model);
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
    let loaded;
    if (source === 'url') {
      loaded = await loadFromBase(normalizeModelUrl($('model-url').value));
    } else if (source === 'files') {
      const f = pickModelFiles($('model-files').files);
      checkModelType(JSON.parse(await f.metadata.text()));
      loaded = await tmImage.loadFromFiles(f.model, f.weights, f.metadata);
    } else {
      // เติม query เพื่อให้ได้ไฟล์ล่าสุดเมื่อมีการแทนที่ my-model
      loaded = await loadFromBase(BUILT_IN_MODELS[Number($('builtin-model').value)].path, `?t=${Date.now()}`);
    }
    if (request !== loadRequest) {
      releaseModel(loaded);
      return;
    }
    stopWebcam();
    releaseModel(model);
    model = loaded;
    const labels = model.getClassLabels();
    buildBars(labels);
    setStatus('ready', `พร้อมใช้งาน · ${labels.length} คลาส`);
    setModelMessage(`เปลี่ยนโมเดลแล้ว: ${labels.join(', ')}`);
    rememberLink();
    $('toggle-webcam').disabled = false;
    $('image-file').disabled = false;
    clearResult();
    resetStage();
  } catch (err) {
    if (request !== loadRequest) return;
    console.error(err);
    setStatus('error', 'โหลดโมเดลไม่สำเร็จ');
    setModelMessage(`${err.message}${model ? ' (ยังใช้โมเดลเดิมอยู่)' : ''}`, true);
  } finally {
    if (request === loadRequest) $('load-model').disabled = false;
  }
}

async function refreshCameraList() {
  const select = $('camera-select');
  const current = select.value;
  const devices = await navigator.mediaDevices.enumerateDevices();
  const cameras = devices.filter((d) => d.kind === 'videoinput');
  select.replaceChildren(...cameras.map((c, i) => new Option(c.label || `กล้อง ${i + 1}`, c.deviceId)));
  if (cameras.some((c) => c.deviceId === current)) select.value = current;
}

async function startWebcam() {
  webcam = new tmImage.Webcam(WEBCAM_SIZE, WEBCAM_SIZE, $('flip').checked);
  const deviceId = $('camera-select').value;
  await webcam.setup(deviceId ? { deviceId } : {});
  await webcam.play();
  showMedia(webcam.canvas);
  $('toggle-webcam').textContent = 'ปิดกล้อง';
  await refreshCameraList();
  lastFrameTime = performance.now();
  const loop = async () => {
    if (!webcam) return;
    const epoch = renderEpoch;
    webcam.update();
    const predictions = await model.predict(webcam.canvas);
    if (epoch !== renderEpoch) return;
    renderPredictions(predictions);
    const now = performance.now();
    $('fps').textContent = `${Math.round(1000 / Math.max(now - lastFrameTime, 1))} FPS`;
    lastFrameTime = now;
    loopId = requestAnimationFrame(loop);
  };
  loop();
}

function stopWebcam() {
  renderEpoch += 1;
  if (loopId) cancelAnimationFrame(loopId);
  loopId = null;
  try {
    if (webcam) webcam.stop();
  } catch (err) {
    console.warn(err); // กล้องยังเปิดไม่สำเร็จจึงยังไม่มีอะไรให้หยุด
  }
  webcam = null;
  $('toggle-webcam').textContent = 'เปิดกล้อง';
}

async function toggleWebcam() {
  if (webcam) {
    stopWebcam();
    clearResult();
    resetStage();
    return;
  }
  await openWebcam();
}

async function openWebcam() {
  try {
    await startWebcam();
  } catch (err) {
    console.error(err);
    stopWebcam();
    clearResult();
    showEmpty('เปิดกล้องไม่ได้', 'ตรวจสอบว่าอนุญาตให้เบราว์เซอร์ใช้กล้องแล้ว และเปิดผ่าน localhost');
  }
}

// เปลี่ยนกล้องหรือการกลับภาพระหว่างเปิดกล้องอยู่ ต้องเริ่มกล้องใหม่จึงจะมีผล
function restartWebcam() {
  if (!webcam) return;
  stopWebcam();
  openWebcam();
}

function resetStage() {
  if ($('tab-webcam').getAttribute('aria-pressed') === 'true') {
    showEmpty('พร้อมเปิดกล้อง', 'กดปุ่มเปิดกล้องด้านล่าง แล้วถือสิ่งของให้อยู่ในกรอบ');
  } else {
    showEmpty('วางรูปภาพที่นี่', 'หรือกดปุ่มเลือกรูปภาพด้านล่าง');
  }
}

async function classifyFile(file) {
  const epoch = renderEpoch;
  const img = new Image();
  img.src = URL.createObjectURL(file);
  await img.decode();
  img.alt = file.name;
  const predictions = await model.predict(img);
  if (epoch !== renderEpoch) return null;
  showMedia(img);
  return renderPredictions(predictions);
}

function selectThumb(button) {
  $('gallery').querySelectorAll('.thumb').forEach((t) => t.removeAttribute('aria-current'));
  button.setAttribute('aria-current', 'true');
}

function addThumb(file, best) {
  const button = document.createElement('button');
  button.type = 'button';
  button.className = 'thumb';
  button.title = `${file.name}: ${best.className}`;
  const img = document.createElement('img');
  img.src = URL.createObjectURL(file);
  img.alt = file.name;
  const tag = document.createElement('span');
  tag.textContent = best.className;
  button.append(img, tag);
  button.addEventListener('click', () => {
    selectThumb(button);
    classifyFile(file).catch((err) => {
      console.error(err);
      showEmpty('อ่านรูปภาพไม่ได้', err.message);
    });
  });
  $('gallery').prepend(button);
  while ($('gallery').children.length > MAX_THUMBS) $('gallery').lastChild.remove();
  $('gallery').hidden = false;
  selectThumb(button);
}

async function classifyFiles(fileList) {
  const images = Array.from(fileList).filter((f) => f.type.startsWith('image/') || /\.(jpe?g|png|bmp|webp|gif)$/i.test(f.name));
  if (!model) return;
  if (!images.length) {
    showEmpty('ไฟล์นี้ไม่ใช่รูปภาพ', 'ลองเลือกไฟล์ .jpg หรือ .png');
    return;
  }
  try {
    for (const file of images) {
      const best = await classifyFile(file);
      if (!best) return;
      addThumb(file, best);
    }
  } catch (err) {
    console.error(err);
    showEmpty('อ่านรูปภาพไม่ได้', err.message);
  }
}

function switchInput(useWebcam) {
  $('tab-webcam').setAttribute('aria-pressed', String(useWebcam));
  $('tab-file').setAttribute('aria-pressed', String(!useWebcam));
  $('webcam-panel').hidden = !useWebcam;
  $('file-panel').hidden = useWebcam;
  $('gallery').hidden = useWebcam || !$('gallery').children.length;
  stopWebcam();
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
$('load-model').addEventListener('click', loadModel);
$('toggle-webcam').addEventListener('click', toggleWebcam);
$('flip').addEventListener('change', restartWebcam);
$('camera-select').addEventListener('change', restartWebcam);
$('tab-webcam').addEventListener('click', () => switchInput(true));
$('tab-file').addEventListener('click', () => switchInput(false));
$('image-file').addEventListener('change', (e) => {
  classifyFiles(e.target.files);
  e.target.value = '';
});
initDropZone();
start();
