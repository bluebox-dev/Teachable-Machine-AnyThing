// เพิ่มโมเดลอื่นได้โดยวางโฟลเดอร์ไว้ข้าง my-model แล้วเพิ่มรายการในอาร์เรย์นี้
const BUILT_IN_MODELS = [
  { name: 'สิ่งของ 5 ชนิด (pen, phone, card, bag, mouse computer)', path: './my-model/' },
];
const MODEL_PACKAGE = '@teachablemachine/image';
const WEBCAM_SIZE = 480;
const MIN_CONFIDENCE = 0.6;
const MAX_THUMBS = 24;

const $ = (id) => document.getElementById(id);

let model = null;
let webcam = null;
let loopId = null;
let source = 'builtin';
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

function clearResult() {
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

function setSource(next) {
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
  const url = new URL(text.trim());
  url.search = '';
  url.hash = '';
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
    throw new Error('ต้องเลือกครบ 3 ไฟล์: model.json, metadata.json และ weights.bin');
  }
  return picked;
}

async function fetchMetadata(url) {
  const response = await fetch(url);
  if (!response.ok) throw new Error(`ไม่พบไฟล์ ${url}`);
  return response.json();
}

// ตรวจจาก metadata ก่อนโหลด เพราะไลบรารีอาจเขียนทับ packageName ตอนโหลดโมเดล
function checkModelType(metadata) {
  const type = metadata.tfjsSpeechCommandsVersion ? '@tensorflow-models/speech-commands' : metadata.packageName;
  if (type && type !== MODEL_PACKAGE) {
    throw new Error(`โมเดลนี้เป็นชนิด ${type} แอปนี้ใช้ได้เฉพาะโมเดล Image`);
  }
}

async function loadFromBase(base) {
  checkModelType(await fetchMetadata(`${base}metadata.json`));
  return tmImage.load(`${base}model.json`, `${base}metadata.json`);
}

async function loadModel() {
  stopWebcam();
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
      loaded = await loadFromBase(BUILT_IN_MODELS[Number($('builtin-model').value)].path);
    }
    model = loaded;
    const labels = model.getClassLabels();
    buildBars(labels);
    setStatus('ready', `พร้อมใช้งาน · ${labels.length} คลาส`);
    $('toggle-webcam').disabled = false;
    $('image-file').disabled = false;
    clearResult();
    resetStage();
  } catch (err) {
    console.error(err);
    setStatus('error', 'โหลดโมเดลไม่สำเร็จ');
    setModelMessage(`${err.message}${model ? ' (ยังใช้โมเดลเดิมอยู่)' : ''}`, true);
  } finally {
    $('load-model').disabled = false;
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
    webcam.update();
    renderPredictions(await model.predict(webcam.canvas));
    const now = performance.now();
    $('fps').textContent = `${Math.round(1000 / Math.max(now - lastFrameTime, 1))} FPS`;
    lastFrameTime = now;
    loopId = requestAnimationFrame(loop);
  };
  loop();
}

function stopWebcam() {
  if (loopId) cancelAnimationFrame(loopId);
  loopId = null;
  if (webcam) webcam.stop();
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
  try {
    await startWebcam();
  } catch (err) {
    console.error(err);
    stopWebcam();
    showEmpty('เปิดกล้องไม่ได้', 'ตรวจสอบว่าอนุญาตให้เบราว์เซอร์ใช้กล้องแล้ว และเปิดผ่าน localhost');
  }
}

function resetStage() {
  if ($('tab-webcam').getAttribute('aria-pressed') === 'true') {
    showEmpty('พร้อมเปิดกล้อง', 'กดปุ่มเปิดกล้องด้านล่าง แล้วถือสิ่งของให้อยู่ในกรอบ');
  } else {
    showEmpty('วางรูปภาพที่นี่', 'หรือกดปุ่มเลือกรูปภาพด้านล่าง');
  }
}

async function classifyFile(file) {
  const img = new Image();
  img.src = URL.createObjectURL(file);
  await img.decode();
  img.alt = file.name;
  showMedia(img);
  return renderPredictions(await model.predict(img));
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
    classifyFile(file);
  });
  $('gallery').prepend(button);
  while ($('gallery').children.length > MAX_THUMBS) $('gallery').lastChild.remove();
  $('gallery').hidden = false;
  selectThumb(button);
}

async function classifyFiles(fileList) {
  const images = Array.from(fileList).filter((f) => f.type.startsWith('image/'));
  if (!model) return;
  if (!images.length) {
    showEmpty('ไฟล์นี้ไม่ใช่รูปภาพ', 'ลองเลือกไฟล์ .jpg หรือ .png');
    return;
  }
  try {
    for (const file of images) addThumb(file, await classifyFile(file));
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
  b.addEventListener('click', () => setSource(b.dataset.source));
});
$('model-files').addEventListener('change', (e) => {
  const names = Array.from(e.target.files).map((f) => f.name).join(', ');
  $('model-files-text').textContent = names || 'เลือก model.json, metadata.json และ weights.bin พร้อมกัน';
});
$('load-model').addEventListener('click', loadModel);
$('toggle-webcam').addEventListener('click', toggleWebcam);
$('tab-webcam').addEventListener('click', () => switchInput(true));
$('tab-file').addEventListener('click', () => switchInput(false));
$('image-file').addEventListener('change', (e) => {
  classifyFiles(e.target.files);
  e.target.value = '';
});
initDropZone();
loadModel();
