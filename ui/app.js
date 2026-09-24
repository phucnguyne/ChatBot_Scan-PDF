const conversation = document.getElementById('conversation');
const form = document.getElementById('chat-form');
const input = document.getElementById('question');
const sendButton = document.getElementById('send-button');
const fileInput = document.getElementById('file-input');
const importButton = document.getElementById('import-button');
const buildButton = document.getElementById('build-button');
const buildStatus = document.getElementById('build-status');
const buildProgress = document.getElementById('build-progress');
const buildEstimate = document.getElementById('build-estimate');
const allPages = document.getElementById('all-pages');
let selectedFile = null;
let buildTimer = null;

function updateEstimate() {
  const all = allPages.checked;
  const count = Number(document.getElementById('page-count').value) || 0;
  if (all) { buildEstimate.textContent = 'Ước tính: toàn bộ tài liệu, thời gian phụ thuộc số trang'; return; }
  const seconds = Math.max(8, count * 2.2 + 12);
  buildEstimate.textContent = `Ước tính: khoảng ${Math.ceil(seconds / 60) ? `${Math.ceil(seconds / 60)} phút` : `${Math.ceil(seconds)} giây`} cho ${count} trang scan`;
}

function addMessage(role, text, citation = '') {
  const welcome = document.getElementById('welcome');
  if (welcome) welcome.remove();
  const message = document.createElement('div');
  message.className = `message ${role}`;
  const avatar = document.createElement('div');
  avatar.className = 'avatar';
  avatar.textContent = role === 'user' ? 'B' : '✦';
  const body = document.createElement('div');
  body.className = 'bubble';
  body.textContent = text;
  if (citation) {
    const source = document.createElement('div');
    source.className = 'citation';
    source.textContent = citation;
    body.appendChild(source);
  }
  message.append(avatar, body);
  conversation.appendChild(message);
  conversation.scrollTop = conversation.scrollHeight;
  return message;
}

function addTyping() {
  return addMessage('assistant', 'Đang đọc tài liệu...');
}

function resetConversation() {
  conversation.innerHTML = '<div class="welcome" id="welcome"><div class="welcome-icon">✦</div><h2>Đọc tài liệu cùng bạn.</h2><p>Đặt câu hỏi bằng tiếng Việt. Mọi câu trả lời đều bám theo nội dung PDF đã lập chỉ mục.</p></div>';
}

async function loadHistory() {
  const response = await fetch('/api/history');
  const data = await response.json();
  resetConversation();
  data.history.forEach(item => addMessage(item.role === 'user' ? 'user' : 'assistant', item.content));
}

async function sendQuestion(question) {
  question = question.trim();
  if (!question || sendButton.disabled) return;
  addMessage('user', question);
  input.value = '';
  input.style.height = 'auto';
  sendButton.disabled = true;
  const typing = addTyping();
  try {
    const response = await fetch('/api/chat', {
      method: 'POST', headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({question})
    });
    const data = await response.json();
    typing.remove();
    if (!response.ok) addMessage('assistant', data.error || 'Có lỗi xảy ra.', '');
    else addMessage('assistant', data.answer, data.citation);
  } catch (error) {
    typing.remove();
    addMessage('assistant', 'Không thể kết nối tới server local.', 'Kiểm tra terminal đang chạy web_server.py.');
  } finally { sendButton.disabled = false; input.focus(); }
}

form.addEventListener('submit', event => { event.preventDefault(); sendQuestion(input.value); });
input.addEventListener('keydown', event => {
  if (event.key === 'Enter' && !event.shiftKey) { event.preventDefault(); form.requestSubmit(); }
});
input.addEventListener('input', () => { input.style.height = 'auto'; input.style.height = `${Math.min(input.scrollHeight, 140)}px`; });
document.querySelectorAll('.prompt-card').forEach(button => button.addEventListener('click', () => sendQuestion(button.dataset.question)));
document.getElementById('new-chat').addEventListener('click', () => {
  fetch('/api/new-chat', {method:'POST'}).then(() => { resetConversation(); loadStatus(); loadChats(); input.focus(); });
});
document.getElementById('clear-history').addEventListener('click', async () => {
  await fetch('/api/clear', {method: 'POST'});
  resetConversation(); loadStatus();
});

importButton.addEventListener('click', () => fileInput.click());
fileInput.addEventListener('change', async () => {
  selectedFile = fileInput.files[0];
  if (!selectedFile) return;
  if (!selectedFile.name.toLowerCase().endsWith('.pdf')) {
    buildStatus.textContent = 'Chỉ hỗ trợ file PDF';
    selectedFile = null;
    return;
  }
  buildStatus.textContent = 'Đang import tài liệu...';
  const formData = new FormData();
  formData.append('file', selectedFile);
  const response = await fetch('/api/upload', {method: 'POST', body: formData});
  const data = await response.json();
  if (!response.ok) { buildStatus.textContent = data.error || 'Import thất bại'; return; }
  document.getElementById('import-name').textContent = data.filename;
  buildButton.disabled = false;
  buildStatus.textContent = 'Đã import. Chọn phạm vi trang rồi Build.';
  loadStatus();
});

allPages.addEventListener('change', () => {
  document.getElementById('page-count').disabled = allPages.checked;
  if (allPages.checked) document.getElementById('page-count').value = 0;
  updateEstimate();
});
document.getElementById('page-count').addEventListener('input', updateEstimate);
buildButton.addEventListener('click', async () => {
  const startPage = Math.max(1, Number(document.getElementById('start-page').value) || 1);
  const pageCount = allPages.checked ? 0 : Math.max(1, Number(document.getElementById('page-count').value) || 10);
  const response = await fetch('/api/build', {
    method: 'POST', headers: {'Content-Type': 'application/json'},
    body: JSON.stringify({start_page: startPage, page_count: pageCount})
  });
  const data = await response.json();
  if (!response.ok) { buildStatus.textContent = data.error || 'Không thể build'; return; }
  buildButton.disabled = true;
  input.disabled = true;
  sendButton.disabled = true;
  buildStatus.textContent = `Đang build ${pageCount ? `${pageCount} trang` : 'toàn bộ tài liệu'}...`;
  clearInterval(buildTimer);
  buildTimer = setInterval(loadStatus, 1000);
  loadStatus();
});
document.getElementById('clear-documents').addEventListener('click', async () => {
  if (!confirm('Xóa toàn bộ tài liệu và index của cuộc trò chuyện này?')) return;
  await fetch('/api/clear-documents', {method:'POST', headers:{'Content-Type':'application/json'}, body:'{}'});
  resetConversation(); loadStatus();
});

async function loadStatus() {
  try {
    const response = await fetch('/api/status');
    const data = await response.json();
    const dot = document.getElementById('status-dot');
    dot.className = `status-dot ${data.ready ? 'ready' : 'error'}`;
    document.getElementById('status-text').textContent = data.ready ? 'Index sẵn sàng' : 'Chưa có index';
    const build = data.build || {};
    document.getElementById('system-meta').textContent = `${data.pdf_count} PDF · ${data.history_count / 2 | 0} lượt chat`;
    document.getElementById('model-name').textContent = data.model;
    buildProgress.style.width = `${build.progress || 0}%`;
    if (build.running) {
      buildStatus.textContent = build.message || 'Đang build index...';
      buildButton.disabled = true;
    } else if (build.phase === 'done') {
      buildStatus.textContent = build.message || 'Build hoàn tất. Có thể đặt câu hỏi.';
      buildButton.disabled = false;
    } else if (build.phase === 'error') {
      buildStatus.textContent = `Build lỗi: ${build.message}`;
      buildButton.disabled = false;
    }
    buildButton.disabled = build.running || !data.pdf_count;
    const ready = Boolean(data.ready);
    input.disabled = !ready;
    sendButton.disabled = !ready;
    input.placeholder = ready ? 'Hỏi bất cứ điều gì về tài liệu...' : 'Build index trước khi đặt câu hỏi...';
    if (!build.running && buildTimer) { clearInterval(buildTimer); buildTimer = null; }
    const list = document.getElementById('document-list');
    list.innerHTML = data.pdfs.length ? data.pdfs.map(name => `<div class="doc-item active"><span class="doc-icon">▧</span><span class="doc-name" title="${name}">${name}</span><button class="delete-doc" data-file="${encodeURIComponent(name)}" title="Xóa tài liệu">×</button></div>`).join('') : '<div class="muted">Chưa có PDF</div>';
    document.querySelectorAll('.delete-doc').forEach(button => button.addEventListener('click', async () => {
      if (!confirm(`Xóa tài liệu ${decodeURIComponent(button.dataset.file)}?`)) return;
      await fetch('/api/delete-document', {method:'POST', headers:{'Content-Type':'application/json'}, body:JSON.stringify({filename:decodeURIComponent(button.dataset.file)})});
      loadStatus();
    }));
  } catch (_) {
    document.getElementById('status-text').textContent = 'Server chưa sẵn sàng';
  }
}
async function loadChats() {
  const response = await fetch('/api/chats');
  const data = await response.json();
  const list = document.getElementById('chat-list');
  list.innerHTML = data.chats.map(chat => `<div class="chat-row"><button class="chat-item ${chat.id === data.active ? 'active' : ''}" data-chat="${chat.id}">${chat.title}</button>${chat.id === data.active ? '' : `<button class="delete-chat" data-chat="${chat.id}" title="Xóa cuộc trò chuyện">×</button>`}</div>`).join('');
  document.querySelectorAll('.chat-item').forEach(button => button.addEventListener('click', async () => { await fetch('/api/switch-chat', {method:'POST', headers:{'Content-Type':'application/json'}, body:JSON.stringify({chat_id:button.dataset.chat})}); loadHistory(); loadStatus(); loadChats(); }));
  document.querySelectorAll('.delete-chat').forEach(button => button.addEventListener('click', async () => { if (!confirm('Xóa cuộc trò chuyện này?')) return; await fetch('/api/delete-chat', {method:'POST', headers:{'Content-Type':'application/json'}, body:JSON.stringify({chat_id:button.dataset.chat})}); loadChats(); }));
}
async function initializeApp() {
  // A fresh page is a fresh draft; saved chats are opened explicitly from the sidebar.
  try {
    const currentResponse = await fetch('/api/status');
    const current = await currentResponse.json();
    if (current.history_count > 0 || current.pdf_count > 0) {
      await fetch('/api/new-chat', {method:'POST', headers:{'Content-Type':'application/json'}, body:'{}'});
    }
  } catch (_) {
    document.getElementById('status-text').textContent = 'Server chưa sẵn sàng';
  }
  resetConversation();
  await loadStatus();
  await loadChats();
  input.focus();
}

initializeApp();
