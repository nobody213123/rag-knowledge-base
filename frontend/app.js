// RAG 知识库前端 - API 调用逻辑

const API_BASE = '/chat';
const SYSTEM_API = '/system';

let isLoading = false;

// 会话 ID（浏览器存储，用于多标签隔离）
function getSessionId() {
    let sid = localStorage.getItem('rag_session_id');
    if (!sid) {
        sid = crypto.randomUUID ? crypto.randomUUID() : Date.now().toString(36) + Math.random().toString(36).slice(2);
        localStorage.setItem('rag_session_id', sid);
    }
    return sid;
}

// DOM elements
const chatMessages = document.getElementById('chat-messages');
const questionInput = document.getElementById('question-input');
const sendBtn = document.getElementById('send-btn');
const historyList = document.getElementById('history-list');
const clearHistoryBtn = document.getElementById('clear-history-btn');
const statusText = document.getElementById('status-text');
const statusIndicator = document.getElementById('status-indicator');
const statsQueries = document.getElementById('stats-queries');
const sourcesPanel = document.getElementById('sources-panel');
const sourcesContent = document.getElementById('sources-content');
const sourcesClose = document.getElementById('sources-close');

// Auto resize textarea
questionInput.addEventListener('input', () => {
    questionInput.style.height = 'auto';
    questionInput.style.height = Math.min(questionInput.scrollHeight, 120) + 'px';
    sendBtn.disabled = !questionInput.value.trim();
});

// Send on Enter (Shift+Enter for new line)
questionInput.addEventListener('keydown', (e) => {
    if (e.key === 'Enter' && !e.shiftKey) {
        e.preventDefault();
        if (!sendBtn.disabled) sendMessage();
    }
});

sendBtn.addEventListener('click', sendMessage);

// Clear history
clearHistoryBtn.addEventListener('click', async () => {
    if (!confirm('确定清空所有对话历史？')) return;
    try {
        await fetch(`${API_BASE}/history/clear?session_id=${getSessionId()}`, { method: 'POST' });
        historyList.innerHTML = '';
        chatMessages.innerHTML = `
            <div class="message ai welcome">
                <div class="avatar">🤖</div>
                <div class="bubble">
                    <p>你好！我是知识库助手，请问有什么可以帮助你的？</p>
                    <p class="hint">你可以问关于产品保修、退货政策、售后服务等问题。</p>
                </div>
            </div>`;
    } catch (err) {
        console.error('Failed to clear history:', err);
    }
});

// Sources close
sourcesClose.addEventListener('click', () => {
    sourcesPanel.classList.add('hidden');
});

async function sendMessage() {
    const question = questionInput.value.trim();
    if (!question || isLoading) return;

    // Add user message
    addMessage('user', question);
    questionInput.value = '';
    questionInput.style.height = 'auto';
    sendBtn.disabled = true;

    // Show loading
    const loadingId = addLoadingMessage();

    // Update status
    setStatus('思考中...', true);

    try {
        const res = await fetch(`${API_BASE}/chat`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ question, use_history: true, session_id: getSessionId() }),
        });

        if (!res.ok) throw new Error(`HTTP ${res.status}`);

        const data = await res.json();
        removeMessage(loadingId);
        addAiMessage(data.answer, data.sources_detail || []);
        addHistoryItem(question, data.answer);
        updateStats();
    } catch (err) {
        removeMessage(loadingId);
        addMessage('ai', `❌ 请求失败：${err.message}，请检查服务是否正常运行。`);
    } finally {
        setStatus('就绪', false);
    }
}

function addMessage(role, content) {
    const div = document.createElement('div');
    div.className = `message ${role}`;
    div.innerHTML = `
        <div class="avatar">${role === 'user' ? '👤' : '🤖'}</div>
        <div class="bubble"><p>${escapeHtml(content)}</p></div>
    `;
    chatMessages.appendChild(div);
    scrollToBottom();
    return div;
}

function addAiMessage(content, sources) {
    const div = document.createElement('div');
    div.className = 'message ai';

    let sourcesHtml = '';
    if (sources && sources.length > 0) {
        sourcesHtml = `
            <div class="sources-toggle">
                <button onclick="toggleSources(this)">📎 ${sources.length} 个引用来源</button>
                <div class="source-items" style="display:none">
                    ${sources.map(s => `
                        <div class="source-item">
                            <span class="idx">[${s.index}]</span>
                            <span>${escapeHtml(s.file)}</span>
                        </div>
                    `).join('')}
                </div>
            </div>`;
    }

    div.innerHTML = `
        <div class="avatar">🤖</div>
        <div class="bubble">
            <p>${escapeHtml(content)}</p>
            ${sourcesHtml}
        </div>
    `;
    chatMessages.appendChild(div);
    scrollToBottom();
}

function addLoadingMessage() {
    const id = 'loading-' + Date.now();
    const div = document.createElement('div');
    div.className = 'message ai';
    div.id = id;
    div.innerHTML = `
        <div class="avatar">🤖</div>
        <div class="bubble">
            <div class="typing-indicator">
                <span></span><span></span><span></span>
            </div>
        </div>
    `;
    chatMessages.appendChild(div);
    scrollToBottom();
    return id;
}

function removeMessage(id) {
    const el = document.getElementById(id);
    if (el) el.remove();
}

function addHistoryItem(question, answer) {
    const div = document.createElement('div');
    div.className = 'history-item';
    div.innerHTML = `
        <span class="q">💬 ${escapeHtml(truncate(question, 30))}</span>
        <span class="a">${escapeHtml(truncate(answer, 40))}</span>
    `;
    div.addEventListener('click', () => {
        questionInput.value = question.replace(/<[^>]*>/g, '');
        questionInput.focus();
    });
    historyList.prepend(div);

    // Keep only last 20
    while (historyList.children.length > 20) {
        historyList.removeChild(historyList.lastChild);
    }
}

function toggleSources(btn) {
    const items = btn.nextElementSibling;
    if (items.style.display === 'none') {
        items.style.display = 'block';
        btn.textContent = btn.textContent.replace('📎', '🔽');
    } else {
        items.style.display = 'none';
        btn.textContent = btn.textContent.replace('🔽', '📎');
    }
}

function setStatus(text, loading) {
    statusText.textContent = text;
    isLoading = loading;
    if (loading) {
        statusIndicator.classList.add('loading');
    } else {
        statusIndicator.classList.remove('loading');
    }
}

async function updateStats() {
    try {
        const res = await fetch(`${SYSTEM_API}/stats`);
        const data = await res.json();
        statsQueries.textContent = data.total_queries;
    } catch (err) {
        // ignore
    }
}

async function loadHistory() {
    try {
        const res = await fetch(`${API_BASE}/history?session_id=${getSessionId()}`);
        const data = await res.json();
        historyList.innerHTML = '';
        for (const item of data.history) {
            addHistoryItem(item.question, item.answer);
        }
    } catch (err) {
        console.error('Failed to load history:', err);
    }
}

function scrollToBottom() {
    setTimeout(() => {
        chatMessages.scrollTop = chatMessages.scrollHeight;
    }, 50);
}

function escapeHtml(text) {
    const div = document.createElement('div');
    div.textContent = text;
    return div.innerHTML;
}

function truncate(text, maxLen) {
    if (text.length <= maxLen) return text;
    return text.slice(0, maxLen) + '...';
}

// Init
loadHistory();
updateStats();
