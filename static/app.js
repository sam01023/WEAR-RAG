document.addEventListener('DOMContentLoaded', () => {
  // ── Element refs ──
  const $ = id => document.getElementById(id);
  const questionInput = $('question');
  const askBtn = $('ask-btn');
  const mockToggle = $('mock-toggle');
  const modelSelect = $('model-select');
  const statusDot = $('status-dot');
  const statusText = $('status-text');
  const healthBtn = $('health-btn');
  const fileInput = $('file-input');
  const browseBtn = $('browse-btn');
  const uploadZone = $('upload-zone');
  const docText = $('doc-text');
  const addDocBtn = $('add-doc-btn');
  const clearDocsBtn = $('clear-docs-btn');
  const docsList = $('docs-list');
  const docCounter = $('doc-counter');
  const toastEl = $('toast');
  const answerEmpty = $('answer-empty');
  const loadingState = $('loading-state');
  const answerResult = $('answer-result');
  const evidenceEmpty = $('evidence-empty');
  const evidenceResult = $('evidence-result');
  const historyList = $('history-list');
  const historyCount = $('history-count');
  const clearHistoryBtn = $('clear-history-btn');

  let docs = [];
  const HISTORY_KEY = 'wear_rag_history';

  // ── Helpers ──
  function escHtml(s) {
    return String(s).replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;').replace(/"/g,'&quot;').replace(/'/g,'&#39;');
  }
  function showToast(msg, type = '') {
    toastEl.textContent = msg;
    toastEl.className = `toast show ${type}`;
    setTimeout(() => { toastEl.className = 'toast'; }, 3000);
  }
  function updateRunBtn() {
    const q = questionInput.value.trim();
    const hasDocs = docs.length > 0;
    askBtn.disabled = !(q && hasDocs);
  }

  // ── Tabs ──
  document.querySelectorAll('.tab').forEach(tab => {
    tab.addEventListener('click', () => {
      document.querySelectorAll('.tab').forEach(t => { t.classList.remove('active'); t.setAttribute('aria-selected','false'); });
      document.querySelectorAll('.tab-panel').forEach(p => p.classList.remove('active'));
      tab.classList.add('active');
      tab.setAttribute('aria-selected','true');
      $('tab-' + tab.dataset.tab).classList.add('active');
    });
  });

  function switchToTab(name) {
    document.querySelectorAll('.tab').forEach(t => { t.classList.toggle('active', t.dataset.tab === name); t.setAttribute('aria-selected', t.dataset.tab === name ? 'true' : 'false'); });
    document.querySelectorAll('.tab-panel').forEach(p => p.classList.toggle('active', p.id === 'tab-' + name));
  }

  // ── Health check ──
  async function checkHealth() {
    try {
      const r = await fetch('/api/health');
      const d = await r.json();
      if (d.mistral) {
        statusDot.className = 'status-dot online';
        statusText.textContent = 'Mistral Ready';
      } else if (d.ollama) {
        statusDot.className = 'status-dot offline';
        statusText.textContent = 'No Mistral model';
      } else {
        statusDot.className = 'status-dot offline';
        statusText.textContent = 'Ollama offline';
      }
      modelSelect.innerHTML = '';
      (d.models || []).forEach(m => { const o = document.createElement('option'); o.value = m; o.textContent = m; modelSelect.appendChild(o); });
      if (!d.models || d.models.length === 0) { modelSelect.innerHTML = '<option>No models</option>'; }
    } catch {
      statusDot.className = 'status-dot offline';
      statusText.textContent = 'Server error';
      modelSelect.innerHTML = '<option>Unavailable</option>';
    }
  }
  checkHealth();
  healthBtn.addEventListener('click', checkHealth);

  // ── Document management ──
  function renderDocs() {
    docsList.innerHTML = '';
    docCounter.textContent = `${docs.length} loaded`;
    clearDocsBtn.style.display = docs.length > 0 ? '' : 'none';

    docs.forEach((d, idx) => {
      const li = document.createElement('li');
      li.className = 'doc-item';
      const charCount = d.text.length.toLocaleString();
      li.innerHTML = `
        <div class="doc-item-header">
          <span class="doc-name">${escHtml(d.filename || d.id)}</span>
          <div style="display:flex;align-items:center;gap:8px;">
            <span class="doc-chars">${charCount} chars</span>
            <button class="doc-remove" data-idx="${idx}" title="Remove document">x</button>
          </div>
        </div>
        <div class="doc-preview">${escHtml(d.text.slice(0, 150))}${d.text.length > 150 ? '...' : ''}</div>
      `;
      docsList.appendChild(li);
    });

    docsList.querySelectorAll('.doc-remove').forEach(btn => {
      btn.addEventListener('click', e => {
        const i = Number(e.currentTarget.dataset.idx);
        docs.splice(i, 1);
        renderDocs();
        updateRunBtn();
      });
    });
    updateRunBtn();
  }

  // ── Upload ──
  async function uploadFiles(fileList) {
    if (!fileList || fileList.length === 0) return;
    for (const f of Array.from(fileList)) {
      showToast(`Uploading ${f.name}...`);
      const fd = new FormData();
      fd.append('file', f);
      try {
        const resp = await fetch('/api/upload', { method: 'POST', body: fd });
        const data = await resp.json();
        if (!resp.ok) { showToast(data.error || 'Upload failed', 'error'); }
        else {
          docs.push({ id: `doc${docs.length + 1}`, filename: data.filename, text: data.text });
          showToast(`Loaded: ${data.filename} (${data.chars.toLocaleString()} chars)`, 'success');
        }
      } catch { showToast('Upload request failed', 'error'); }
    }
    renderDocs();
    fileInput.value = '';
  }

  browseBtn.addEventListener('click', () => fileInput.click());
  uploadZone.addEventListener('click', e => { if (e.target === uploadZone || e.target.closest('.upload-zone-inner')) fileInput.click(); });
  fileInput.addEventListener('change', () => uploadFiles(fileInput.files));

  uploadZone.addEventListener('dragover', e => { e.preventDefault(); uploadZone.classList.add('dragover'); });
  uploadZone.addEventListener('dragleave', e => { e.preventDefault(); uploadZone.classList.remove('dragover'); });
  uploadZone.addEventListener('drop', e => { e.preventDefault(); uploadZone.classList.remove('dragover'); uploadFiles(e.dataTransfer.files); });

  addDocBtn.addEventListener('click', () => {
    const text = docText.value.trim();
    if (!text) return showToast('Paste some text first', 'error');
    docs.push({ id: `doc${docs.length + 1}`, filename: `pasted_${docs.length + 1}.txt`, text });
    docText.value = '';
    renderDocs();
    showToast('Document added', 'success');
  });

  clearDocsBtn.addEventListener('click', () => {
    if (!confirm('Remove all documents?')) return;
    docs = [];
    renderDocs();
    showToast('All documents cleared');
  });

  questionInput.addEventListener('input', updateRunBtn);
  questionInput.addEventListener('keydown', e => {
    if (e.key === 'Enter' && (e.ctrlKey || e.metaKey)) { e.preventDefault(); askBtn.click(); }
  });

  // ── Pipeline execution ──
  const STEPS = ['step-chunk','step-embed','step-decomp','step-retrieve','step-rerank','step-agg','step-gen'];
  const LABELS = ['Semantic chunking...','Embedding documents...','Decomposing query...','Retrieving candidates...','Reranking with cross-encoder...','Aggregating evidence...','Generating answer...'];

  function setLoading(on) {
    loadingState.classList.toggle('hidden', !on);
    answerResult.classList.toggle('hidden', on);
    answerEmpty.style.display = on ? 'none' : '';
    askBtn.disabled = on;
    if (on) {
      STEPS.forEach(s => { const el = $(s); if (el) el.className = 'step'; });
    }
  }

  async function runPipeline() {
    const question = questionInput.value.trim();
    if (!question || docs.length === 0) return;

    switchToTab('answer');
    setLoading(true);
    answerEmpty.style.display = 'none';

    // Animate steps
    let stepIdx = 0;
    const stepInterval = setInterval(() => {
      if (stepIdx > 0) {
        const prev = $(STEPS[stepIdx - 1]);
        if (prev) prev.className = 'step done';
      }
      if (stepIdx < STEPS.length) {
        const cur = $(STEPS[stepIdx]);
        if (cur) cur.className = 'step active';
        $('loading-stage').textContent = LABELS[stepIdx];
      }
      stepIdx++;
      if (stepIdx >= STEPS.length) clearInterval(stepInterval);
    }, 700);

    const payload = {
      question,
      documents: docs.map(d => ({ id: d.id, text: d.text })),
      mock: mockToggle.checked
    };
    const model = modelSelect.value;
    if (model && model !== 'No models' && model !== 'Unavailable') payload.model = model;

    try {
      const resp = await fetch('/api/ask', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(payload)
      });

      clearInterval(stepInterval);
      STEPS.forEach(s => { const el = $(s); if (el) el.className = 'step done'; });

      const data = await resp.json();
      setLoading(false);

      if (!resp.ok || data.error) {
        showAnswerError(data.error || 'Unknown error');
        return;
      }

      showAnswer(data);
      showEvidence(data);
      saveHistory({ ts: Date.now(), question, docs: docs.map(d => ({ filename: d.filename, text: d.text })), answer: data.answer, evidence: data.evidence });
      showToast('Answer generated successfully', 'success');

    } catch (e) {
      clearInterval(stepInterval);
      setLoading(false);
      showAnswerError('Network error: ' + e.message);
    }
  }

  askBtn.addEventListener('click', runPipeline);

  // ── Render answer ──
  function showAnswer(d) {
    answerEmpty.style.display = 'none';

    const subqHtml = d.sub_queries && d.sub_queries.length > 1
      ? `<div class="subqueries-wrap">
          <div class="subqueries-label">Sub-queries decomposed</div>
          ${d.sub_queries.map(q => `<span class="subquery-pill">${escHtml(q)}</span>`).join('')}
         </div>` : '';

    answerResult.innerHTML = `
      ${subqHtml}
      <div class="answer-card">
        <div class="answer-label">
          Answer
          <span class="answer-latency">${(d.latency_ms / 1000).toFixed(1)}s &middot; ${d.num_docs} doc${d.num_docs !== 1 ? 's' : ''}</span>
        </div>
        <div class="answer-text">${escHtml(d.answer)}</div>
        <div class="answer-actions">
          <button class="btn btn-outline btn-sm" id="copy-answer-btn">Copy</button>
          <button class="btn btn-outline btn-sm" id="download-answer-btn">Download</button>
        </div>
      </div>
    `;
    answerResult.classList.remove('hidden');

    $('copy-answer-btn').addEventListener('click', () => {
      navigator.clipboard.writeText(d.answer).then(() => showToast('Answer copied', 'success'));
    });
    $('download-answer-btn').addEventListener('click', () => {
      const blob = new Blob([d.answer], { type: 'text/plain' });
      const url = URL.createObjectURL(blob);
      const a = document.createElement('a'); a.href = url; a.download = 'answer.txt';
      document.body.appendChild(a); a.click(); a.remove(); URL.revokeObjectURL(url);
    });
  }

  function showAnswerError(msg) {
    answerEmpty.style.display = 'none';
    answerResult.innerHTML = `<div class="error-card">${escHtml(msg)}</div>`;
    answerResult.classList.remove('hidden');
    showToast(msg, 'error');
  }

  // ── Render evidence ──
  function showEvidence(d) {
    evidenceEmpty.style.display = 'none';
    if (!d.evidence || d.evidence.length === 0) {
      evidenceResult.innerHTML = '<p style="color:var(--text-dim)">No evidence returned.</p>';
      evidenceResult.classList.remove('hidden');
      return;
    }

    const cards = d.evidence.map(ev => {
      const s = ev.scores || {};
      return `
        <div class="evidence-card">
          <div class="evidence-header">
            <div>
              <div class="evidence-rank">Rank ${ev.rank}</div>
              <div class="evidence-source">${escHtml(ev.source)}</div>
            </div>
            <div class="evidence-total">${(s.total || 0).toFixed(3)}</div>
          </div>
          <div class="score-bars">
            <div>
              <div class="score-bar-label"><span>Similarity</span><span style="color:var(--cyan)">${s.similarity || 0}</span></div>
              <div class="score-bar-track"><div class="score-bar-fill" style="width:${(s.similarity || 0) * 100}%;background:var(--cyan)"></div></div>
            </div>
            <div>
              <div class="score-bar-label"><span>Reranker</span><span style="color:var(--purple)">${s.reranker || 0}</span></div>
              <div class="score-bar-track"><div class="score-bar-fill" style="width:${(s.reranker || 0) * 100}%;background:var(--purple)"></div></div>
            </div>
            <div>
              <div class="score-bar-label"><span>Density</span><span style="color:var(--orange)">${s.density || 0}</span></div>
              <div class="score-bar-track"><div class="score-bar-fill" style="width:${(s.density || 0) * 100}%;background:var(--orange)"></div></div>
            </div>
          </div>
          <div class="evidence-text-content">${escHtml(ev.text)}</div>
        </div>
      `;
    }).join('');

    evidenceResult.innerHTML = `<div class="evidence-title">Top Evidence Chunks &mdash; Weighted Aggregation</div>${cards}`;
    evidenceResult.classList.remove('hidden');
  }

  // ── History ──
  function loadHistory() { try { return JSON.parse(localStorage.getItem(HISTORY_KEY) || '[]'); } catch { return []; } }

  function saveHistory(entry) {
    const h = loadHistory();
    h.unshift(entry);
    if (h.length > 50) h.pop();
    localStorage.setItem(HISTORY_KEY, JSON.stringify(h));
    renderHistory();
  }

  function renderHistory() {
    const h = loadHistory();
    historyCount.textContent = `${h.length} entr${h.length !== 1 ? 'ies' : 'y'}`;
    historyList.innerHTML = '';

    if (h.length === 0) {
      historyList.innerHTML = '<div style="text-align:center;color:var(--text-dim);padding:40px;font-size:13px;">No history yet. Run your first query!</div>';
      return;
    }

    h.forEach((it, idx) => {
      const div = document.createElement('div');
      div.className = 'history-item';
      div.innerHTML = `<div class="history-q">${escHtml(it.question)}</div><div class="history-meta">${new Date(it.ts).toLocaleString()}</div>`;
      div.addEventListener('click', () => {
        docs = (it.docs || []).map((d, i) => ({ id: `doc${i + 1}`, filename: d.filename || `doc${i + 1}`, text: d.text }));
        renderDocs();
        questionInput.value = it.question;
        if (it.answer) {
          showAnswer({ answer: it.answer, sub_queries: [], evidence: it.evidence || [], latency_ms: 0, num_docs: docs.length });
          showEvidence({ evidence: it.evidence || [] });
          switchToTab('answer');
        }
        showToast('History entry loaded', 'success');
      });
      historyList.appendChild(div);
    });
  }
  renderHistory();

  clearHistoryBtn.addEventListener('click', () => {
    if (!confirm('Clear all history?')) return;
    localStorage.removeItem(HISTORY_KEY);
    renderHistory();
    showToast('History cleared');
  });
});
