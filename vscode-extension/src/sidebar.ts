import * as vscode from 'vscode';
import * as http from 'http';

export class ThoughtGitSidebarProvider implements vscode.WebviewViewProvider {
    private _view?: vscode.WebviewView;

    constructor(private readonly _extensionUri: vscode.Uri) {}

    public resolveWebviewView(
        webviewView: vscode.WebviewView,
        context: vscode.WebviewViewResolveContext,
        _token: vscode.CancellationToken
    ) {
        this._view = webviewView;

        webviewView.webview.options = {
            enableScripts: true,
            localResourceRoots: [this._extensionUri]
        };

        webviewView.webview.html = this._getHtmlForWebview(webviewView.webview);

        // Handle message signals from Webview
        webviewView.webview.onDidReceiveMessage(async (message) => {
            switch (message.type) {
                case 'search':
                    await this._executeSearch(message.query, message.mode);
                    break;
                case 'ingest':
                    await this._executeIngest(message.content, message.topic, message.timestamp);
                    break;
                case 'ask-mentor':
                    await this._executeMentorQuery(message.context);
                    break;
                case 'load-branches':
                    await this._loadBranches();
                    break;
                case 'switch-branch':
                    await this._switchBranch(message.branch);
                    break;
                case 'open-visualizer':
                    vscode.commands.executeCommand('thoughtgit.showInteractiveMap');
                    break;
            }
        });

        // Load branches immediately on startup
        this._loadBranches();
    }

    public searchTopic(topic: string) {
        if (this._view) {
            this._view.show?.(true);
            this._view.webview.postMessage({ type: 'trigger-search', topic: topic, mode: 'search' });
        }
    }

    public showEvolution(topic: string) {
        if (this._view) {
            this._view.show?.(true);
            this._view.webview.postMessage({ type: 'trigger-search', topic: topic, mode: 'evolution' });
        }
    }

    private async _executeSearch(query: string, mode: string) {
        if (!this._view) return;

        if (mode === 'search') {
            const path = `/recall?query=${encodeURIComponent(query)}&n_results=5`;
            getJSON(path, (err, data) => {
                if (err) {
                    this._view?.webview.postMessage({ type: 'error', message: 'Failed to connect to ThoughtGit server.' });
                    return;
                }
                this._view?.webview.postMessage({ type: 'search-results', results: data });
            });
        } else {
            const path = `/diff?topic=${encodeURIComponent(query)}`;
            getJSON(path, (err, data) => {
                if (err) {
                    this._view?.webview.postMessage({ type: 'error', message: 'Failed to retrieve timeline.' });
                    return;
                }
                this._view?.webview.postMessage({ type: 'evolution-results', data: data });
            });
        }
    }

    private async _executeIngest(content: string, topic: string, timestamp?: string) {
        if (!this._view) return;

        const payload: any = {
            content: content,
            source: 'vscode',
            metadata: { topic_hint: topic }
        };

        if (timestamp) {
            payload.timestamp = timestamp;
        }

        postJSON('/ingest', payload, (err, data) => {
            if (err || data.status !== 'success') {
                this._view?.webview.postMessage({ type: 'ingest-status', status: 'error', message: 'Failed to ingest thought.' });
                return;
            }
            this._view?.webview.postMessage({ type: 'ingest-status', status: 'success', message: 'Thought ingested successfully!' });
        });
    }

    private async _executeMentorQuery(context: string) {
        if (!this._view) return;

        const payload = {
            current_context: context
        };

        postJSON('/mentor/suggest', payload, (err, data) => {
            if (err) {
                this._view?.webview.postMessage({ type: 'error', message: 'Failed to connect to AI Mentor.' });
                return;
            }
            this._view?.webview.postMessage({ type: 'mentor-results', advice: data });
        });
    }

    private async _loadBranches() {
        if (!this._view) return;

        getJSON('/branch', (err, data) => {
            if (!err && data) {
                this._view?.webview.postMessage({
                    type: 'branches-loaded',
                    active: data.active_branch,
                    list: data.branches
                });
            }
        });
    }

    private async _switchBranch(branch: string) {
        if (!this._view) return;

        const payload = { name: branch };
        postJSON('/branch/switch', payload, (err, data) => {
            if (!err && data.status === 'success') {
                this._loadBranches();
            }
        });
    }

    private _getHtmlForWebview(webview: vscode.Webview) {
        return `<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>ThoughtGit Timeline</title>
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;500;600;700;800&display=swap');

        body {
            background-color: #080b11;
            color: #e2e8f0;
            font-family: 'Plus Jakarta Sans', -apple-system, sans-serif;
            padding: 14px;
            margin: 0;
            display: flex;
            flex-direction: column;
            height: 100vh;
            box-sizing: border-box;
        }

        .main-container {
            flex: 1;
            overflow-y: auto;
            padding-bottom: 60px;
        }

        h2 {
            font-size: 18px;
            font-weight: 800;
            margin: 0 0 4px 0;
            background: linear-gradient(135deg, #00f2fe 0%, #4facfe 50%, #8b5cf6 100%);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            letter-spacing: -0.5px;
        }

        .tagline {
            font-size: 11px;
            color: #64748b;
            font-weight: 600;
            margin-bottom: 16px;
            border-bottom: 1px solid rgba(255, 255, 255, 0.05);
            padding-bottom: 8px;
        }

        /* Search input bar styling */
        .search-container {
            position: relative;
            margin-bottom: 12px;
        }

        .search-input {
            width: 100%;
            background-color: rgba(17, 24, 39, 0.8);
            border: 1px solid rgba(255, 255, 255, 0.08);
            border-radius: 8px;
            padding: 8px 10px;
            color: #ffffff;
            font-size: 12.5px;
            box-sizing: border-box;
            outline: none;
            transition: all 0.3s ease;
        }

        .search-input:focus {
            border-color: #00f2fe;
            box-shadow: 0 0 8px rgba(0, 242, 254, 0.2);
        }

        /* Tabs styling */
        .tabs {
            display: flex;
            flex-wrap: wrap;
            gap: 6px;
            margin-bottom: 16px;
            border-bottom: 1px solid rgba(255, 255, 255, 0.05);
            padding-bottom: 10px;
        }

        .tab-btn {
            background: rgba(255, 255, 255, 0.02);
            border: 1px solid rgba(255, 255, 255, 0.06);
            border-radius: 6px;
            padding: 6px 8px;
            color: #94a3b8;
            font-size: 10px;
            font-weight: 600;
            cursor: pointer;
            transition: all 0.2s ease;
            text-align: center;
        }

        .tab-btn:hover {
            border-color: #00f2fe;
            color: #ffffff;
        }

        .tab-btn.active {
            background: linear-gradient(135deg, #00f2fe 0%, #4facfe 100%);
            border-color: transparent;
            color: #080b11;
            font-weight: 700;
            box-shadow: 0 4px 10px rgba(0, 242, 254, 0.25);
        }

        /* Panel Views */
        .panel {
            display: none;
        }

        .panel.active {
            display: block;
        }

        /* Results / Cards */
        .card {
            background: rgba(17, 24, 39, 0.75);
            border: 1px solid rgba(255, 255, 255, 0.06);
            border-radius: 10px;
            padding: 12px;
            margin-bottom: 12px;
            backdrop-filter: blur(8px);
            box-shadow: 0 4px 15px rgba(0, 0, 0, 0.15);
            transition: all 0.3s ease;
        }

        .card:hover {
            border-color: rgba(0, 242, 254, 0.25);
            box-shadow: 0 4px 15px rgba(0, 242, 254, 0.03);
        }

        .card-header {
            display: flex;
            justify-content: space-between;
            align-items: center;
            font-size: 9.5px;
            font-weight: 700;
            color: #94a3b8;
            margin-bottom: 6px;
            text-transform: uppercase;
        }

        .badge {
            background: rgba(0, 242, 254, 0.1);
            color: #00f2fe;
            border: 1px solid rgba(0, 242, 254, 0.2);
            border-radius: 4px;
            padding: 1px 5px;
            font-size: 8.5px;
            font-weight: 700;
        }

        .card-body {
            font-size: 12px;
            line-height: 1.4;
            color: #e2e8f0;
        }

        /* Timeline Layout */
        .timeline {
            border-left: 2px solid rgba(255, 255, 255, 0.05);
            padding-left: 14px;
            margin-left: 8px;
            position: relative;
        }

        .timeline-item {
            position: relative;
            margin-bottom: 20px;
        }

        .timeline-dot {
            width: 8px;
            height: 8px;
            background: #8b5cf6;
            border-radius: 50%;
            position: absolute;
            left: -19px;
            top: 6px;
            box-shadow: 0 0 6px #8b5cf6;
        }

        .timeline-time {
            font-size: 10px;
            font-weight: 700;
            color: #8b5cf6;
            margin-bottom: 4px;
            text-transform: uppercase;
        }

        .timeline-drift {
            display: inline-block;
            font-size: 8.5px;
            font-weight: 800;
            padding: 2px 5px;
            border-radius: 4px;
            margin-top: 6px;
        }

        .drift-reinforced { background: rgba(16, 185, 129, 0.12); color: #34d399; border: 1px solid rgba(16, 185, 129, 0.25); }
        .drift-deepened { background: rgba(59, 130, 246, 0.12); color: #60a5fa; border: 1px solid rgba(59, 130, 246, 0.25); }
        .drift-refined { background: rgba(139, 92, 246, 0.12); color: #a78bfa; border: 1px solid rgba(139, 92, 246, 0.25); }
        .drift-changed_direction { background: rgba(245, 158, 11, 0.12); color: #fbbf24; border: 1px solid rgba(245, 158, 11, 0.25); }
        .drift-major_shift { background: rgba(239, 68, 68, 0.12); color: #f87171; border: 1px solid rgba(239, 68, 68, 0.25); }

        /* Form Ingestion Box */
        .form-group {
            margin-bottom: 12px;
        }

        .form-label {
            font-size: 10.5px;
            font-weight: 700;
            color: #94a3b8;
            margin-bottom: 4px;
            display: block;
        }

        .form-textarea {
            width: 100%;
            height: 100px;
            background-color: rgba(17, 24, 39, 0.8);
            border: 1px solid rgba(255, 255, 255, 0.08);
            border-radius: 8px;
            padding: 8px 10px;
            color: #ffffff;
            font-size: 12.5px;
            box-sizing: border-box;
            outline: none;
            resize: none;
        }

        .form-btn {
            width: 100%;
            background: linear-gradient(135deg, #00f2fe 0%, #4facfe 100%);
            border: none;
            border-radius: 8px;
            color: #080b11;
            font-weight: 700;
            font-size: 12px;
            padding: 8px 12px;
            cursor: pointer;
            box-shadow: 0 4px 10px rgba(0, 242, 254, 0.25);
            transition: all 0.2s ease;
        }

        .form-btn:hover {
            box-shadow: 0 4px 12px rgba(0, 242, 254, 0.4);
            transform: translateY(-0.5px);
        }

        .success-banner {
            background-color: rgba(16, 185, 129, 0.1);
            border: 1px solid rgba(16, 185, 129, 0.25);
            color: #34d399;
            padding: 8px;
            border-radius: 6px;
            font-size: 11px;
            text-align: center;
            margin-bottom: 12px;
            display: none;
        }

        /* Bottom Branch Switcher */
        .footer {
            position: absolute;
            bottom: 0;
            left: 0;
            right: 0;
            background: #080b11;
            border-top: 1px solid rgba(255, 255, 255, 0.05);
            padding: 10px 14px;
            display: flex;
            justify-content: space-between;
            align-items: center;
        }

        .branch-select {
            background: rgba(17, 24, 39, 0.8);
            border: 1px solid rgba(255, 255, 255, 0.08);
            border-radius: 6px;
            color: #00f2fe;
            font-size: 10.5px;
            font-weight: 700;
            padding: 4px 6px;
            outline: none;
        }

        .error-msg {
            color: #f87171;
            font-size: 12px;
            text-align: center;
            margin-top: 20px;
        }
    </style>
</head>
<body>
    <div class="main-container">
        <h2>🧠 ThoughtGit</h2>
        <div class="tagline">Version Control for Human Thinking</div>

        <button class="form-btn" style="margin-bottom: 14px; background: linear-gradient(135deg, #8b5cf6 0%, #00f2fe 100%);" onclick="openVisualizer()">🌐 Open Project Visualizer</button>

        <div class="tabs">
            <button id="btnSearch" class="tab-btn active" onclick="switchPanel('search')">Recall</button>
            <button id="btnEvolution" class="tab-btn" onclick="switchPanel('evolution')">Evolution</button>
            <button id="btnIngest" class="tab-btn" onclick="switchPanel('ingest')">Save Thought</button>
            <button id="btnMentor" class="tab-btn" onclick="switchPanel('mentor')">AI Mentor</button>
        </div>

        <!-- 1. Recall Panel -->
        <div id="panelSearch" class="panel active">
            <div class="search-container">
                <input type="text" id="searchInput" class="search-input" placeholder="Search a topic in memory...">
            </div>
            <div id="recallResults">
                <div style="text-align:center; color:#64748b; font-size:12px; margin-top:30px;">
                    Enter a topic above to query vector memory.
                </div>
            </div>
        </div>

        <!-- 2. Evolution Panel -->
        <div id="panelEvolution" class="panel">
            <div class="search-container">
                <input type="text" id="evolutionInput" class="search-input" placeholder="Enter topic to trace evolution...">
            </div>
            <div id="evolutionResults">
                <div style="text-align:center; color:#64748b; font-size:12px; margin-top:30px;">
                    Enter a topic to generate semantic timelines.
                </div>
            </div>
        </div>

        <!-- 3. Ingest Panel -->
        <div id="panelIngest" class="panel">
            <div id="ingestSuccess" class="success-banner"></div>
            <div class="form-group">
                <span class="form-label">Thought Content</span>
                <textarea id="ingestContent" class="form-textarea" placeholder="Write down your thought, solution, or design idea here..."></textarea>
            </div>
            <div class="form-group">
                <span class="form-label">Topic Tag Hint</span>
                <input type="text" id="ingestTopic" class="search-input" placeholder="e.g. chat-database">
            </div>
            <div class="form-group">
                <span class="form-label">Thought Date (Optional - for testing evolution)</span>
                <input type="date" id="ingestDate" class="search-input" style="color-scheme: dark;">
            </div>
            <button class="form-btn" onclick="triggerIngest()">Save to Memory</button>
        </div>

        <!-- 4. AI Mentor Panel -->
        <div id="panelMentor" class="panel">
            <div class="form-group">
                <span class="form-label">Coding context/question:</span>
                <textarea id="mentorContext" class="form-textarea" placeholder="Example: I am choosing between SQL and NoSQL for scalable messages..."></textarea>
            </div>
            <button class="form-btn" onclick="triggerMentorQuery()">Ask AI Developer Mentor</button>
            <div id="mentorResults" style="margin-top: 16px;"></div>
        </div>
    </div>

    <!-- Active Branch Selector Footer -->
    <div class="footer">
        <span style="font-size: 10px; font-weight:700; color:#64748b;">🌿 ACTIVE BRANCH:</span>
        <select id="branchSelector" class="branch-select" onchange="triggerBranchSwitch()">
            <option value="main">MAIN</option>
        </select>
    </div>

    <script>
        const vscode = acquireVsCodeApi();
        let currentPanel = 'search';

        // Trigger searches on Enter key
        document.getElementById('searchInput').addEventListener('keydown', (e) => {
            if (e.key === 'Enter') triggerSearch('search');
        });
        document.getElementById('evolutionInput').addEventListener('keydown', (e) => {
            if (e.key === 'Enter') triggerSearch('evolution');
        });

        // Listen for messages from extension side
        window.addEventListener('message', event => {
            const message = event.data;
            switch (message.type) {
                case 'trigger-search':
                    if (message.mode === 'search') {
                        document.getElementById('searchInput').value = message.topic;
                    } else {
                        document.getElementById('evolutionInput').value = message.topic;
                    }
                    switchPanel(message.mode);
                    triggerSearch(message.mode);
                    break;
                case 'search-results':
                    renderSearchResults(message.results);
                    break;
                case 'evolution-results':
                    renderEvolutionResults(message.data);
                    break;
                case 'ingest-status':
                    showIngestStatus(message.status, message.message);
                    break;
                case 'mentor-results':
                    renderMentorResults(message.advice);
                    break;
                case 'branches-loaded':
                    updateBranchesDropdown(message.active, message.list);
                    break;
                case 'error':
                    renderError(message.message);
                    break;
            }
        });

        function switchPanel(panel) {
            currentPanel = panel;
            
            // Toggle tab active classes
            document.getElementById('btnSearch').classList.toggle('active', panel === 'search');
            document.getElementById('btnEvolution').classList.toggle('active', panel === 'evolution');
            document.getElementById('btnIngest').classList.toggle('active', panel === 'ingest');
            document.getElementById('btnMentor').classList.toggle('active', panel === 'mentor');

            // Toggle panels
            document.getElementById('panelSearch').classList.toggle('active', panel === 'search');
            document.getElementById('panelEvolution').classList.toggle('active', panel === 'evolution');
            document.getElementById('panelIngest').classList.toggle('active', panel === 'ingest');
            document.getElementById('panelMentor').classList.toggle('active', panel === 'mentor');
        }

        function triggerSearch(mode) {
            const queryInput = mode === 'search' ? document.getElementById('searchInput') : document.getElementById('evolutionInput');
            const query = queryInput.value.trim();
            if (query) {
                vscode.postMessage({ type: 'search', query: query, mode: mode });
            }
        }

        function triggerIngest() {
            const content = document.getElementById('ingestContent').value.trim();
            const topic = document.getElementById('ingestTopic').value.trim();
            const dateVal = document.getElementById('ingestDate').value;
            if (content && topic) {
                let timestamp = undefined;
                if (dateVal) {
                    timestamp = new Date(dateVal).toISOString();
                }
                vscode.postMessage({ type: 'ingest', content: content, topic: topic, timestamp: timestamp });
            }
        }

        function triggerMentorQuery() {
            const context = document.getElementById('mentorContext').value.trim();
            if (context) {
                vscode.postMessage({ type: 'ask-mentor', context: context });
            }
        }

        function triggerBranchSwitch() {
            const selector = document.getElementById('branchSelector');
            vscode.postMessage({ type: 'switch-branch', branch: selector.value });
        }

        function openVisualizer() {
            vscode.postMessage({ type: 'open-visualizer' });
        }

        function showIngestStatus(status, msg) {
            const banner = document.getElementById('ingestSuccess');
            banner.innerText = msg;
            banner.style.display = 'block';
            
            if (status === 'success') {
                banner.style.backgroundColor = 'rgba(16, 185, 129, 0.1)';
                banner.style.color = '#34d399';
                banner.style.borderColor = 'rgba(16, 185, 129, 0.25)';
                document.getElementById('ingestContent').value = '';
                document.getElementById('ingestTopic').value = '';
            } else {
                banner.style.backgroundColor = 'rgba(239, 68, 68, 0.1)';
                banner.style.color = '#f87171';
                banner.style.borderColor = 'rgba(239, 68, 68, 0.25)';
            }
            
            setTimeout(() => {
                banner.style.display = 'none';
            }, 4000);
        }

        function updateBranchesDropdown(active, list) {
            const selector = document.getElementById('branchSelector');
            selector.innerHTML = list.map(b => 
                \`<option value="\${b}" \${b === active ? 'selected' : ''}>\${b.toUpperCase()}</option>\`
            ).join('');
        }

        function renderSearchResults(results) {
            const container = document.getElementById('recallResults');
            if (results.length === 0) {
                container.innerHTML = '<div style="text-align:center; color:#64748b; font-size:12px; margin-top:30px;">No memories found.</div>';
                return;
            }
            container.innerHTML = results.map(r => \`
                <div class="card">
                    <div class="card-header">
                        <span>\${r.collection.replace('thoughts_', '').replace('_', ' ').toUpperCase()}</span>
                        <span class="badge">\${(r.similarity * 100).toFixed(0)}% Match</span>
                    </div>
                    <div class="card-body">\${r.text}</div>
                </div>
            \`).join('');
        }

        function renderEvolutionResults(data) {
            const container = document.getElementById('evolutionResults');
            
            let html = '<div class="timeline">';
            
            const snapshotsMap = {};
            data.snapshots.forEach(s => {
                snapshotsMap[s.time_label] = s;
            });

            let eventsHtml = '';
            if (data.drift_events && data.drift_events.length > 0) {
                eventsHtml = data.drift_events.map(e => {
                    const snap = snapshotsMap[e.to_period] || {};
                    const driftPercent = Math.min(100, Math.max(0, e.distance * 100));
                    const meterColor = driftPercent > 30 ? '#f87171' : (driftPercent > 20 ? '#fbbf24' : '#34d399');
                    
                    return \`
                        <div class="timeline-item">
                            <div class="timeline-dot"></div>
                            <div class="timeline-time">\${e.to_period}</div>
                            <div class="card">
                                <div class="card-body">
                                    <strong>Concept:</strong> \${snap.sample_texts ? snap.sample_texts[0] : 'Ingested note'}
                                    <div class="timeline-drift drift-\${e.drift_type}">
                                        \${e.drift_type.replace('_', ' ').toUpperCase()}
                                    </div>
                                    
                                    <div style="margin-top: 8px;">
                                        <div style="font-size: 9px; color: #64748b; margin-bottom: 3px; display:flex; justify-content:space-between;">
                                            <span>Semantic Shift Magnitude:</span>
                                            <strong>\${(e.distance).toFixed(3)}</strong>
                                        </div>
                                        <div style="background: rgba(255,255,255,0.06); height: 5px; border-radius: 3px; overflow: hidden; width: 100%;">
                                            <div style="width: \${driftPercent}%; background: \${meterColor}; height: 100%; box-shadow: 0 0 6px \${meterColor};"></div>
                                        </div>
                                    </div>

                                    <div style="font-size: 10.5px; color:#94a3b8; margin-top:8px; line-height:1.4;">\${e.summary}</div>
                                </div>
                            </div>
                        </div>
                    \`;
                }).join('');
            } else if (data.snapshots.length > 0) {
                eventsHtml = data.snapshots.map(s => \`
                    <div class="timeline-item">
                        <div class="timeline-dot"></div>
                        <div class="timeline-time">\${s.time_label}</div>
                        <div class="card">
                            <div class="card-body">\${s.sample_texts[0]}</div>
                        </div>
                    </div>
                \`).join('');
            } else {
                eventsHtml = '<div style="text-align:center; color:#64748b; font-size:12px; margin-top:30px;">No timeline snapshots found.</div>';
            }

            html += eventsHtml;
            html += '</div>';
            container.innerHTML = html;
        }

        function renderMentorResults(advice) {
            const container = document.getElementById('mentorResults');
            container.innerHTML = \`
                <div class="card" style="border-color: rgba(0, 242, 254, 0.2); box-shadow: 0 0 10px rgba(0, 242, 254, 0.05);">
                    <div style="font-size: 9.5px; color: #00f2fe; font-weight:700; text-transform:uppercase; margin-bottom:10px;">🎓 Mentor Insight</div>
                    <div style="font-size:11.5px; color:#ffffff; line-height:1.45; margin-bottom:12px;"><strong>Insight:</strong> \${advice.insight}</div>
                    <div style="font-size:11.5px; color:#e2e8f0; line-height:1.45; margin-bottom:12px;"><strong>Reasoning:</strong> \${advice.reason}</div>
                    <div style="font-size:11.5px; color:#94a3b8; line-height:1.45; margin-bottom:12px;"><strong>Reference:</strong> \${advice.past_reference}</div>
                    <div style="font-size:11.5px; color:#34d399; line-height:1.45; font-weight:600;"><strong>Action:</strong> \${advice.action}</div>
                </div>
            \`;
        }

        function renderError(msg) {
            const resultsDiv = currentPanel === 'search' ? 'recallResults' : (currentPanel === 'evolution' ? 'evolutionResults' : 'mentorResults');
            document.getElementById(resultsDiv).innerHTML = \`<div class="error-msg">\${msg}</div>\`;
        }
    </script>
</body>
</html>`;
    }
}

function getJSON(path: string, callback: (err: any, data: any) => void) {
    const workspaceFolder = vscode.workspace.workspaceFolders?.[0]?.uri.fsPath || '';
    const options = {
        hostname: '127.0.0.1',
        port: 8765,
        path: path,
        method: 'GET',
        headers: {
            'X-Workspace-Dir': encodeURIComponent(workspaceFolder)
        }
    };

    const req = http.request(options, (res) => {
        let body = '';
        res.setEncoding('utf-8');
        res.on('data', (chunk) => body += chunk);
        res.on('end', () => {
            try {
                const parsed = JSON.parse(body);
                callback(null, parsed);
            } catch (e) {
                callback(e, null);
            }
        });
    });

    req.on('error', (e) => {
        callback(e, null);
    });

    req.end();
}

function postJSON(path: string, payload: any, callback: (err: any, data: any) => void) {
    const workspaceFolder = vscode.workspace.workspaceFolders?.[0]?.uri.fsPath || '';
    const bodyString = JSON.stringify(payload);
    const options = {
        hostname: '127.0.0.1',
        port: 8765,
        path: path,
        method: 'POST',
        headers: {
            'Content-Type': 'application/json',
            'Content-Length': Buffer.byteLength(bodyString),
            'X-Workspace-Dir': encodeURIComponent(workspaceFolder)
        }
    };

    const req = http.request(options, (res) => {
        let body = '';
        res.setEncoding('utf-8');
        res.on('data', (chunk) => body += chunk);
        res.on('end', () => {
            try {
                const parsed = JSON.parse(body);
                callback(null, parsed);
            } catch (e) {
                callback(e, null);
            }
        });
    });

    req.on('error', (e) => {
        callback(e, null);
    });

    req.write(bodyString);
    req.end();
}
