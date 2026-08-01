const state = { sessionId: null, busy: false };
const sessionsEl = document.querySelector('#sessions');
const messagesEl = document.querySelector('#messages');
const emptyEl = document.querySelector('#empty');
const questionEl = document.querySelector('#question');
const sendEl = document.querySelector('#send');

async function api(path, options = {}) {
  const response = await fetch(path, {headers:{'Content-Type':'application/json'}, ...options});
  if (!response.ok) throw new Error((await response.json()).error || `请求失败 ${response.status}`);
  return response;
}

function messageNode(role, content, citations = []) {
  const article = document.createElement('article'); article.className = `message ${role}`;
  const avatar = document.createElement('div'); avatar.className = 'avatar'; avatar.textContent = role === 'user' ? '我' : '研';
  const body = document.createElement('div'); const bubble = document.createElement('div'); bubble.className = 'bubble'; bubble.textContent = content;
  body.append(bubble); article.append(avatar, body);
  if (citations.length) {
    const list = document.createElement('div'); list.className = 'citations';
    citations.forEach((item, index) => { const row = document.createElement('div'); row.className = 'citation'; const title = document.createElement('strong'); title.textContent = `资料 ${index + 1} · ${item.title}`; const source = document.createElement('small'); source.textContent = item.source; const excerpt = document.createElement('div'); excerpt.textContent = item.excerpt; row.append(title, source, excerpt); list.append(row); });
    body.append(list);
  }
  messagesEl.append(article); emptyEl.classList.add('hidden'); article.scrollIntoView({behavior:'smooth', block:'end'}); return {article, bubble, body};
}

async function loadSessions(selectNewest = false) {
  const data = await (await api('/api/sessions')).json(); sessionsEl.replaceChildren();
  data.sessions.forEach(session => { const button = document.createElement('button'); button.className = `session ${session.id === state.sessionId ? 'active' : ''}`; button.dataset.sessionId = session.id; button.textContent = session.title; button.onclick = () => selectSession(session); sessionsEl.append(button); });
  if ((selectNewest || !state.sessionId) && data.sessions[0]) await selectSession(data.sessions[0]);
  if (!data.sessions.length) await createSession();
}

async function createSession() {
  const session = await (await api('/api/sessions', {method:'POST', body:JSON.stringify({title:'新的面试练习'})})).json();
  state.sessionId = session.id; await loadSessions(true);
}

async function selectSession(session) {
  state.sessionId = session.id; document.querySelector('#session-title').textContent = session.title;
  const data = await (await api(`/api/sessions/${session.id}/messages`)).json(); messagesEl.replaceChildren(); emptyEl.classList.toggle('hidden', data.messages.length > 0);
  data.messages.forEach(item => messageNode(item.role, item.content, item.citations));
  [...sessionsEl.children].forEach(node => node.classList.toggle('active', node.dataset.sessionId === session.id));
}

async function submit(question) {
  if (state.busy || !question.trim()) return; state.busy = true; sendEl.disabled = true; questionEl.value = '';
  messageNode('user', question); const assistant = messageNode('assistant', '正在检索资料…');
  try {
    const response = await api(`/api/sessions/${state.sessionId}/messages`, {method:'POST', body:JSON.stringify({content:question})});
    const reader = response.body.getReader(); const decoder = new TextDecoder(); let buffer = ''; let answer = '';
    while (true) { const {value, done} = await reader.read(); if (done) break; buffer += decoder.decode(value, {stream:true}); const frames = buffer.split('\n\n'); buffer = frames.pop(); for (const frame of frames) { if (!frame.startsWith('data: ')) continue; const event = JSON.parse(frame.slice(6)); if (event.type === 'citations') { assistant.body.querySelector('.citations')?.remove(); const temp = messageNode('assistant','',event.citations); const list = temp.body.querySelector('.citations'); temp.article.remove(); if (list) assistant.body.append(list); } else if (event.type === 'delta') { answer += event.content; assistant.bubble.textContent = answer; } else if (event.type === 'error') throw new Error(event.error); } }
  } catch (error) { assistant.bubble.textContent = `本轮未完成：${error.message}`; assistant.article.classList.add('error'); }
  finally { state.busy = false; sendEl.disabled = false; questionEl.focus(); await loadSessions(); }
}

document.querySelector('#new-session').onclick = createSession;
document.querySelector('#composer').onsubmit = event => { event.preventDefault(); submit(questionEl.value); };
questionEl.onkeydown = event => { if (event.key === 'Enter' && !event.shiftKey) { event.preventDefault(); submit(questionEl.value); } };
document.querySelectorAll('.starters button').forEach(button => button.onclick = () => submit(button.textContent));

(async () => { try { const health = await (await api('/api/health')).json(); const dot = document.querySelector('#status-dot'); dot.className = health.model_available ? 'ok' : 'bad'; document.querySelector('#status-text').textContent = health.model_available ? '模型服务正常' : '模型服务未连接'; document.querySelector('#knowledge-status').textContent = `${health.knowledge_documents} 份资料 · ${health.knowledge_chunks} 个知识块`; await loadSessions(); } catch (error) { document.querySelector('#status-dot').className = 'bad'; document.querySelector('#status-text').textContent = '应用连接失败'; } })();
