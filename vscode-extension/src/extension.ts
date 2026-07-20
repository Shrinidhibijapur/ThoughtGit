import * as vscode from 'vscode';
import * as http from 'http';
import * as path from 'path';

// Supported code extensions
const CODE_EXTENSIONS = new Set(['.py', '.ts', '.js', '.rs', '.go', '.java', '.cpp', '.c', '.h', '.cs', '.sh', '.rb']);

export function activate(context: vscode.ExtensionContext) {
    console.log('ThoughtGit extension is now active!');

    // Hook: Ingest on save
    let saveHook = vscode.workspace.onDidSaveTextDocument(async (document: vscode.TextDocument) => {
        await processDocumentSave(document);
    });
    context.subscriptions.push(saveHook);

    // Sidebar View Registration
    const sidebarProvider = new ThoughtGitSidebarProvider(context.extensionUri);
    context.subscriptions.push(
        vscode.window.registerWebviewViewProvider(
            'thoughtgit-sidebar-view',
            sidebarProvider
        )
    );

    // Command: Query Topic
    let queryCmd = vscode.commands.registerCommand('thoughtgit.queryTopic', async () => {
        const topic = await vscode.window.showInputBox({
            prompt: 'Enter a topic to recall from ThoughtGit memory'
        });
        if (topic) {
            sidebarProvider.searchTopic(topic);
        }
    });
    context.subscriptions.push(queryCmd);

    // Command: Show Concept Evolution
    let evolutionCmd = vscode.commands.registerCommand('thoughtgit.showEvolution', async () => {
        const topic = await vscode.window.showInputBox({
            prompt: 'Enter a topic to view its thinking evolution timeline'
        });
        if (topic) {
            sidebarProvider.showEvolution(topic);
        }
    });
    context.subscriptions.push(evolutionCmd);

    // Command: Open Interactive Project Map
    let mapCmd = vscode.commands.registerCommand('thoughtgit.showInteractiveMap', () => {
        InteractiveMapPanel.createOrShow(context.extensionUri, sidebarProvider);
    });
    context.subscriptions.push(mapCmd);
}

async function processDocumentSave(document: vscode.TextDocument) {
    const filePath = document.fileName;
    const fileExt = filePath.substring(filePath.lastIndexOf('.')).toLowerCase();
    
    let contentToIngest = '';
    let isCode = CODE_EXTENSIONS.has(fileExt);

    if (isCode) {
        // Extract comments and docstrings only for code files
        contentToIngest = extractComments(document.getText(), fileExt);
    } else if (fileExt === '.md' || fileExt === '.txt') {
        // Capture full text for markdown and text notes
        contentToIngest = document.getText();
    }

    if (!contentToIngest.trim()) {
        return;
    }

    // Prepare thought metadata
    const source = 'vscode';
    const metadata: any = {
        file_name: filePath.substring(filePath.lastIndexOf('/') + 1),
        file_path: filePath,
        is_code: String(isCode)
    };

    // Parse inline date if present (e.g., "date: 2026-05-15" or "Date: 2026-05-15")
    let timestamp: string | undefined = undefined;
    const dateMatch = contentToIngest.match(/(?:date|Date):\s*(\d{4}-\d{2}-\d{2})/);
    if (dateMatch && dateMatch[1]) {
        try {
            const parsedDate = new Date(dateMatch[1]);
            if (!isNaN(parsedDate.getTime())) {
                timestamp = parsedDate.toISOString();
            }
        } catch (e) {
            console.error('Failed to parse inline date:', e);
        }
    }

    // Parse inline topic if present (e.g., "topic: rag-search" or "Topic: rag-search")
    const topicMatch = contentToIngest.match(/(?:topic|Topic):\s*([a-zA-Z0-9_-]+)/);
    if (topicMatch && topicMatch[1]) {
        metadata.topic_hint = topicMatch[1];
    }

    const payloadObj: any = {
        content: contentToIngest,
        source: source,
        metadata: metadata
    };

    if (timestamp) {
        payloadObj.timestamp = timestamp;
    }

    const payload = JSON.stringify(payloadObj);

    // Send thought ingestion request to FastAPI localhost:8765
    postJSON('/ingest', payload, (err, response) => {
        if (err) {
            console.error('ThoughtGit ingestion failed:', err);
            return;
        }
        console.log('ThoughtGit ingested successfully:', response);
        
        // After successful ingestion, check duplicate thinking
        const checkPayload = JSON.stringify({ text: contentToIngest });
        postJSON('/check_duplicate', checkPayload, (checkErr, checkResponse) => {
            if (checkErr || !checkResponse) {
                return;
            }
            if (checkResponse.is_duplicate) {
                const match = checkResponse.matched_chunk;
                vscode.window.showWarningMessage(
                    `[ThoughtGit] Duplicate Thinking Alert! Very similar concept found in '${match.collection}': "${match.text.substring(0, 60)}..."`,
                    'View Details'
                ).then(selection => {
                    if (selection === 'View Details') {
                        vscode.window.showInformationMessage(`Duplicate Location: ${match.collection}\nText: ${match.text}`);
                    }
                });
            }
        });
    });
}

function extractComments(code: string, ext: string): string {
    const lines = code.split(/\r?\n/);
    const comments: string[] = [];
    
    let inBlockComment = false;

    for (let line of lines) {
        line = line.trim();

        // Python style
        if (ext === '.py') {
            // Check block quotes
            if (line.startsWith('"""') || line.startsWith("'''")) {
                inBlockComment = !inBlockComment;
                comments.push(line.replace(/"""|'''/g, ''));
                continue;
            }
            if (inBlockComment) {
                comments.push(line);
                continue;
            }
            if (line.startsWith('#')) {
                comments.push(line.substring(1).trim());
            }
            continue;
        }

        // C-Style: // and /* */
        if (ext === '.js' || ext === '.ts' || ext === '.rs' || ext === '.go' || ext === '.cpp' || ext === '.c' || ext === '.java' || ext === '.cs') {
            if (line.startsWith('/*')) {
                inBlockComment = true;
                comments.push(line.substring(2).replace(/\*\/$/, '').trim());
                if (line.endsWith('*/')) {
                    inBlockComment = false;
                }
                continue;
            }
            if (inBlockComment) {
                if (line.endsWith('*/')) {
                    inBlockComment = false;
                    comments.push(line.replace(/\*\/$/, '').trim());
                } else {
                    comments.push(line.replace(/^\*+/, '').trim());
                }
                continue;
            }
            if (line.startsWith('//')) {
                comments.push(line.substring(2).trim());
            }
            continue;
        }
    }
    
    return comments.filter(c => c.length > 0).join('\n');
}

// Full-Width Interactive D3/Mermaid Map Webview Panel
class InteractiveMapPanel {
    public static currentPanel: InteractiveMapPanel | undefined;
    private readonly _panel: vscode.WebviewPanel;
    private readonly _extensionUri: vscode.Uri;
    private _disposables: vscode.Disposable[] = [];

    public static createOrShow(extensionUri: vscode.Uri, sidebarProvider: ThoughtGitSidebarProvider) {
        const column = vscode.window.activeTextEditor
            ? vscode.window.activeTextEditor.viewColumn
            : undefined;

        if (InteractiveMapPanel.currentPanel) {
            InteractiveMapPanel.currentPanel._panel.reveal(column);
            return;
        }

        const panel = vscode.window.createWebviewPanel(
            'thoughtgitMap',
            'ThoughtGit Project Visualizer',
            column || vscode.ViewColumn.One,
            {
                enableScripts: true,
                localResourceRoots: [extensionUri]
            }
        );

        InteractiveMapPanel.currentPanel = new InteractiveMapPanel(panel, extensionUri, sidebarProvider);
    }

    private constructor(panel: vscode.WebviewPanel, extensionUri: vscode.Uri, sidebarProvider: ThoughtGitSidebarProvider) {
        this._panel = panel;
        this._extensionUri = extensionUri;

        this._update(sidebarProvider);

        this._panel.onDidDispose(() => this.dispose(), null, this._disposables);
    }

    public dispose() {
        InteractiveMapPanel.currentPanel = undefined;
        this._panel.dispose();
        while (this._disposables.length) {
            const x = this._disposables.pop();
            if (x) {
                x.dispose();
            }
        }
    }

    private _update(sidebarProvider: ThoughtGitSidebarProvider) {
        const webview = this._panel.webview;
        this._panel.webview.html = this._getHtmlForWebview(webview);

        // Handle messages from webview
        const workspaceFolder = vscode.workspace.workspaceFolders?.[0]?.uri.fsPath || '';
        this._panel.webview.onDidReceiveMessage(
            async message => {
                switch (message.type) {
                    case 'open-file':
                        try {
                            const fileUri = vscode.Uri.file(path.join(workspaceFolder, message.path));
                            const doc = await vscode.workspace.openTextDocument(fileUri);
                            await vscode.window.showTextDocument(doc);
                        } catch (e) {
                            vscode.window.showErrorMessage(`Failed to open file: ${message.path}`);
                        }
                        break;
                    case 'open-folder':
                        try {
                            const folderUri = vscode.Uri.file(path.join(workspaceFolder, message.path));
                            await vscode.commands.executeCommand('revealInExplorer', folderUri);
                        } catch (e) {
                            vscode.window.showErrorMessage(`Failed to reveal folder: ${message.path}`);
                        }
                        break;
                    case 'search-topic':
                        sidebarProvider.showEvolution(message.topic);
                        break;
                    case 'load-data':
                        this._fetchAndSendData(workspaceFolder);
                        break;
                }
            },
            null,
            this._disposables
        );
    }

    private _fetchAndSendData(workspaceFolder: string) {
        // Fetch project map
        getJSON('/project_map', (err, mapData) => {
            if (!err) {
                this._panel.webview.postMessage({ type: 'project-map-data', data: mapData });
            }
        });

        // Fetch codebase flow
        const pathEncoded = encodeURIComponent(workspaceFolder);
        getJSON(`/codebase_flow?workspace_dir=${pathEncoded}`, (err, flowData) => {
            if (!err) {
                this._panel.webview.postMessage({ type: 'codebase-flow-data', data: flowData });
            }
        });
    }

    private _getHtmlForWebview(webview: vscode.Webview) {
        return `<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>ThoughtGit Visualizer</title>
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;500;600;700;800&display=swap');

        body {
            background-color: #06090e;
            color: #e2e8f0;
            font-family: 'Plus Jakarta Sans', -apple-system, sans-serif;
            margin: 0;
            padding: 24px;
            display: flex;
            flex-direction: column;
            height: 100vh;
            box-sizing: border-box;
            overflow: hidden;
        }

        .header {
            display: flex;
            justify-content: space-between;
            align-items: center;
            border-bottom: 1px solid rgba(255, 255, 255, 0.05);
            padding-bottom: 14px;
            margin-bottom: 16px;
        }

        h1 {
            font-size: 20px;
            font-weight: 800;
            margin: 0;
            background: linear-gradient(135deg, #00f2fe 0%, #4facfe 50%, #8b5cf6 100%);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            letter-spacing: -0.5px;
        }

        .tabs {
            display: flex;
            gap: 10px;
        }

        .tab-btn {
            background: rgba(255, 255, 255, 0.02);
            border: 1px solid rgba(255, 255, 255, 0.06);
            border-radius: 8px;
            padding: 8px 16px;
            color: #94a3b8;
            font-size: 11px;
            font-weight: 600;
            cursor: pointer;
            transition: all 0.2s ease;
        }

        .tab-btn:hover {
            border-color: #00f2fe;
            color: #ffffff;
        }

        .tab-btn.active {
            background: linear-gradient(135deg, #00f2fe 0%, #4facfe 100%);
            border-color: transparent;
            color: #06090e;
            font-weight: 700;
            box-shadow: 0 4px 12px rgba(0, 242, 254, 0.25);
        }

        .content {
            flex: 1;
            position: relative;
            background-color: #06090e;
            background-image: radial-gradient(rgba(255, 255, 255, 0.03) 1px, transparent 0);
            background-size: 20px 20px;
            border: 1px solid rgba(255, 255, 255, 0.05);
            border-radius: 12px;
            overflow: hidden;
            box-shadow: inset 0 0 30px rgba(0, 0, 0, 0.85);
        }

        .panel {
            position: absolute;
            top: 0;
            left: 0;
            width: 100%;
            height: 100%;
            display: none;
        }

        .panel.active {
            display: block;
        }

        /* D3 Graph Area */
        svg {
            width: 100%;
            height: 100%;
        }

        .node circle {
            cursor: pointer;
            transition: all 0.3s ease;
        }

        .node:hover circle {
            stroke-width: 3.5px;
            filter: drop-shadow(0px 0px 8px rgba(0, 242, 254, 0.7));
        }

        .link {
            stroke-opacity: 0.25;
            stroke-dasharray: 3 3;
            transition: stroke-opacity 0.3s ease;
        }

        /* Tooltip style */
        .tooltip {
            position: absolute;
            background: rgba(8, 12, 20, 0.95);
            border: 1px solid rgba(0, 242, 254, 0.3);
            border-radius: 8px;
            padding: 10px 14px;
            color: #ffffff;
            font-size: 11px;
            pointer-events: none;
            opacity: 0;
            transition: opacity 0.2s ease;
            box-shadow: 0 4px 20px rgba(0, 0, 0, 0.5);
            z-index: 10;
            backdrop-filter: blur(10px);
        }

        /* Folder Files Neumorphic Popover Modal */
        .folder-popup {
            position: absolute;
            background: rgba(8, 12, 20, 0.98);
            border: 1px solid rgba(0, 242, 254, 0.35);
            border-radius: 10px;
            padding: 12px;
            box-shadow: 0 8px 30px rgba(0, 0, 0, 0.65);
            z-index: 20;
            backdrop-filter: blur(12px);
            max-height: 250px;
            overflow-y: auto;
            display: none;
            min-width: 170px;
        }

        .folder-popup h3 {
            margin: 0 0 8px 0;
            font-size: 11.5px;
            color: #00f2fe;
            font-weight: 800;
            border-bottom: 1px solid rgba(255, 255, 255, 0.08);
            padding-bottom: 6px;
            display: flex;
            align-items: center;
            gap: 6px;
        }

        .folder-file-item {
            display: flex;
            align-items: center;
            gap: 6px;
            font-size: 11px;
            color: #cbd5e1;
            padding: 6px 8px;
            border-radius: 6px;
            cursor: pointer;
            transition: all 0.2s ease;
        }

        .folder-file-item:hover {
            background: rgba(0, 242, 254, 0.08);
            color: #ffffff;
        }

        /* HUD Zoom Controls */
        .hud-controls {
            position: absolute;
            top: 20px;
            right: 20px;
            display: flex;
            flex-direction: column;
            gap: 6px;
            background: rgba(8, 12, 20, 0.85);
            border: 1px solid rgba(255, 255, 255, 0.08);
            border-radius: 8px;
            padding: 4px;
            backdrop-filter: blur(10px);
            z-index: 5;
            box-shadow: 0 4px 15px rgba(0,0,0,0.4);
        }

        .hud-btn {
            background: transparent;
            border: none;
            border-radius: 6px;
            color: #94a3b8;
            width: 30px;
            height: 30px;
            font-size: 15px;
            font-weight: 700;
            cursor: pointer;
            display: flex;
            align-items: center;
            justify-content: center;
            transition: all 0.2s ease;
        }

        .hud-btn:hover {
            background: rgba(255, 255, 255, 0.05);
            color: #00f2fe;
        }

        /* Mermaid Scrollable Area */
        .mermaid-container {
            width: 100%;
            height: 100%;
            overflow: hidden;
            cursor: grab;
            display: flex;
            justify-content: center;
            align-items: center;
            position: relative;
        }

        .mermaid {
            transform-origin: center center;
            transition: transform 0.05s ease-out;
        }

        .slider-container {
            position: absolute;
            bottom: 20px;
            left: 50%;
            transform: translateX(-50%);
            background: rgba(8, 12, 20, 0.9);
            border: 1px solid rgba(255, 255, 255, 0.08);
            border-radius: 20px;
            padding: 8px 18px;
            display: flex;
            align-items: center;
            gap: 12px;
            backdrop-filter: blur(10px);
            z-index: 5;
            box-shadow: 0 4px 15px rgba(0,0,0,0.4);
        }

        .slider {
            width: 220px;
            accent-color: #00f2fe;
            cursor: pointer;
        }

        /* D3 Node Labels styling */
        .node text {
            font-size: 10px;
            font-weight: 700;
            fill: #e2e8f0;
            pointer-events: none;
        }

        .node-label-bg {
            fill: rgba(6, 9, 14, 0.85);
            stroke: rgba(0, 242, 254, 0.15);
            stroke-width: 1px;
            rx: 5px;
            ry: 5px;
        }
    </style>
</head>
<body>
    <div class="header">
        <h1>🧠 ThoughtGit Project Visualizer</h1>
        <div class="tabs">
            <button id="btnMap" class="tab-btn active" onclick="switchPanel('map')">Semantic Concept Map</button>
            <button id="btnFlow" class="tab-btn" onclick="switchPanel('flow')">Codebase Dependency Flow</button>
        </div>
    </div>

    <div class="content">
        <!-- Tab 1: Semantic Concept Map -->
        <div id="panelMap" class="panel active">
            <svg id="conceptSvg"></svg>
            
            <div class="hud-controls">
                <button class="hud-btn" onclick="zoomConcept(1.2)">＋</button>
                <button class="hud-btn" onclick="zoomConcept(0.8)">－</button>
                <button class="hud-btn" onclick="resetConceptZoom()">⟲</button>
            </div>

            <div id="sliderContainer" class="slider-container">
                <span style="font-size:10px; font-weight:800; color:#94a3b8; letter-spacing:0.5px;">📅 TIMELINE</span>
                <input type="range" id="timeSlider" class="slider" min="0" max="100" value="100">
                <span id="sliderLabel" style="font-size:10px; font-weight:800; color:#00f2fe; min-width:80px; text-align:center;">All Time</span>
            </div>
        </div>

        <!-- Tab 2: Codebase Flow -->
        <div id="panelFlow" class="panel">
            <div class="mermaid-container" id="flowContainer">
                <div id="mermaidDiv" class="mermaid"></div>
            </div>

            <div class="hud-controls">
                <button class="hud-btn" onclick="zoomFlow(1.2)">＋</button>
                <button class="hud-btn" onclick="zoomFlow(0.8)">－</button>
                <button class="hud-btn" onclick="resetFlowZoom()">⟲</button>
            </div>
        </div>
    </div>

    <div id="tooltip" class="tooltip"></div>
    <div id="folderPopup" class="folder-popup"></div>

    <!-- Graph and Flow Scripts -->
    <script src="https://d3js.org/d3.v7.min.js"></script>
    <script src="https://cdn.jsdelivr.net/npm/mermaid/dist/mermaid.min.js"></script>
    <script>
        const vscode = acquireVsCodeApi();
        
        // Initialize Mermaid
        mermaid.initialize({
            startOnLoad: false,
            theme: 'dark',
            securityLevel: 'loose',
            flowchart: { useMaxWidth: false, htmlLabels: true }
        });

        // Request data immediately
        vscode.postMessage({ type: 'load-data' });

        // Poll for updates every 4 seconds to support real-time data refreshing
        setInterval(() => {
            vscode.postMessage({ type: 'load-data' });
        }, 4000);

        let conceptData = null;
        let flowNodes = [];
        let sortedDates = [];
        let conceptZoomBehavior = null;
        let conceptSvgElement = null;
        
        // Codebase Flow Zoom variables
        let flowScale = 1;
        let flowX = 0;
        let flowY = 0;
        let isDraggingFlow = false;
        let startFlowX, startFlowY;

        window.addEventListener('message', event => {
            const message = event.data;
            switch (message.type) {
                case 'project-map-data':
                    if (JSON.stringify(conceptData) !== JSON.stringify(message.data)) {
                        conceptData = message.data;
                        initConceptSlider();
                        renderConceptMap(conceptData);
                    }
                    break;
                case 'codebase-flow-data':
                    flowNodes = message.data.nodes;
                    renderCodebaseFlow(message.data.mermaid_code);
                    break;
            }
        });

        function switchPanel(panel) {
            document.getElementById('btnMap').classList.toggle('active', panel === 'map');
            document.getElementById('btnFlow').classList.toggle('active', panel === 'flow');
            document.getElementById('panelMap').classList.toggle('active', panel === 'map');
            document.getElementById('panelFlow').classList.toggle('active', panel === 'flow');
        }

        // Initialize Time Slider
        function initConceptSlider() {
            const dates = conceptData.nodes
                .map(n => n.latest_timestamp)
                .filter(Boolean)
                .sort();
            
            sortedDates = [...new Set(dates)];
            const slider = document.getElementById('timeSlider');
            const label = document.getElementById('sliderLabel');

            if (sortedDates.length === 0) {
                document.getElementById('sliderContainer').style.display = 'none';
                return;
            }

            slider.min = 0;
            slider.max = sortedDates.length;
            slider.value = sortedDates.length;
            
            slider.addEventListener('input', () => {
                const val = parseInt(slider.value);
                if (val === sortedDates.length) {
                    label.innerText = 'All Time';
                    filterMapByDate(null);
                } else {
                    const selectedDate = sortedDates[val];
                    try {
                        const dt = new Date(selectedDate);
                        label.innerText = dt.toLocaleDateString('en-IN', { timeZone: 'Asia/Kolkata' });
                    } catch (e) {
                        label.innerText = selectedDate.substring(0, 10);
                    }
                    filterMapByDate(selectedDate);
                }
            });
        }

        function filterMapByDate(cutoffDate) {
            if (!conceptData) return;
            if (!cutoffDate) {
                renderConceptMap(conceptData);
                return;
            }

            const filteredNodes = conceptData.nodes.filter(n => !n.latest_timestamp || n.latest_timestamp <= cutoffDate);
            const nodeIds = new Set(filteredNodes.map(n => n.id));
            const filteredLinks = conceptData.links.filter(l => nodeIds.has(l.source.id || l.source) && nodeIds.has(l.target.id || l.target));

            renderConceptMap({ nodes: filteredNodes, links: filteredLinks });
        }

        // D3 Force-Directed Graph Rendering
        function renderConceptMap(data) {
            const svg = d3.select("#conceptSvg");
            
            // Cache current node coordinate positions to preserve them during re-renders/polling
            const positionCache = {};
            svg.selectAll(".node").each(function(d) {
                if (d && d.id) {
                    positionCache[d.id] = { x: d.x, y: d.y, fx: d.fx, fy: d.fy };
                }
            });
            
            svg.selectAll("*").remove();

            const width = document.getElementById('conceptSvg').clientWidth;
            const height = document.getElementById('conceptSvg').clientHeight;

            const g = svg.append("g");
            conceptSvgElement = svg;

            // Configure D3 zoom
            conceptZoomBehavior = d3.zoom().on("zoom", (event) => {
                g.attr("transform", event.transform);
            });
            svg.call(conceptZoomBehavior);

            // Restore coordinate positions from cache
            data.nodes.forEach(node => {
                const cached = positionCache[node.id];
                if (cached) {
                    node.x = cached.x;
                    node.y = cached.y;
                    node.fx = cached.fx;
                    node.fy = cached.fy;
                }
            });

            const simulation = d3.forceSimulation(data.nodes)
                .force("link", d3.forceLink(data.links).id(d => d.id).distance(140))
                .force("charge", d3.forceManyBody().strength(-280))
                .force("center", d3.forceCenter(width / 2, height / 2))
                .force("collision", d3.forceCollide().radius(d => d.size * 4 + 40));

            // Draw Links
            const link = g.append("g")
                .selectAll("line")
                .data(data.links)
                .join("line")
                .attr("class", "link")
                .attr("stroke", "#8b5cf6")
                .attr("stroke-width", d => d.value * 3);

            // Draw Nodes
            const node = g.append("g")
                .selectAll("g")
                .data(data.nodes)
                .join("g")
                .attr("class", "node")
                .call(d3.drag()
                    .on("start", dragstarted)
                    .on("drag", dragged)
                    .on("end", dragended));

            // Glowing circle
            node.append("circle")
                .attr("r", d => d.size * 4 + 8)
                .attr("fill", "rgba(0, 242, 254, 0.12)")
                .attr("stroke", "#00f2fe")
                .attr("stroke-width", 2.2)
                .style("filter", "drop-shadow(0px 0px 5px rgba(0, 242, 254, 0.35))");

            // Pill Capsule Label: Text size mapping
            node.each(function(d) {
                const nodeGroup = d3.select(this);
                
                // Add a text element temporarily to compute size
                const tempText = nodeGroup.append("text")
                    .attr("text-anchor", "middle")
                    .text(d.label);
                
                const bbox = tempText.node().getBBox();
                tempText.remove(); // Remove temporary text

                const padX = 8;
                const padY = 4;
                const rectW = bbox.width + padX * 2;
                const rectH = bbox.height + padY * 2;
                const offsetY = d.size * 4 + 20;

                // 1. Draw label capsule background
                nodeGroup.append("rect")
                    .attr("class", "node-label-bg")
                    .attr("x", -rectW / 2)
                    .attr("y", offsetY - rectH / 2 - 2)
                    .attr("width", rectW)
                    .attr("height", rectH);

                // 2. Draw label text centered inside capsule
                nodeGroup.append("text")
                    .attr("text-anchor", "middle")
                    .attr("x", 0)
                    .attr("y", offsetY + 2)
                    .text(d.label);
            });

            // Tooltips
            const tooltip = document.getElementById("tooltip");
            node.on("mouseover", (event, d) => {
                tooltip.style.opacity = 1;
                let istTime = 'No Date';
                if (d.latest_timestamp) {
                    try {
                        const dt = new Date(d.latest_timestamp);
                        istTime = dt.toLocaleString('en-IN', { timeZone: 'Asia/Kolkata' });
                    } catch (e) {
                        istTime = d.latest_timestamp;
                    }
                }
                tooltip.innerHTML = \`
                    <strong style="color:#00f2fe; font-size:12px;">Topic: \${d.label}</strong><br/>
                    <span>Notes saved: \${d.size}</span><br/>
                    <span style="color:#94a3b8; font-size:10px;">Latest Save: \${istTime}</span>
                \`;
            })
            .on("mousemove", (event) => {
                tooltip.style.left = (event.pageX + 15) + "px";
                tooltip.style.top = (event.pageY - 15) + "px";
            })
            .on("mouseleave", () => {
                tooltip.style.opacity = 0;
            })
            .on("click", (event, d) => {
                vscode.postMessage({ type: 'search-topic', topic: d.id });
            })
            .on("dblclick", (event, d) => {
                event.stopPropagation();
                d.fx = null;
                d.fy = null;
                simulation.alpha(0.3).restart();
            });

            simulation.on("tick", () => {
                link
                    .attr("x1", d => d.source.x)
                    .attr("y1", d => d.source.y)
                    .attr("x2", d => d.target.x)
                    .attr("y2", d => d.target.y);

                node.attr("transform", d => \`translate(\${d.x},\${d.y})\`);
            });

            function dragstarted(event, d) {
                if (!event.active) simulation.alphaTarget(0.3).restart();
                d.fx = d.x;
                d.fy = d.y;
            }

            function dragged(event, d) {
                d.fx = event.x;
                d.fy = event.y;
            }

            function dragended(event, d) {
                if (!event.active) simulation.alphaTarget(0);
                d.fx = d.x;
                d.fy = d.y;
            }
        }

        // HUD Zoom Helpers for Concept Map
        function zoomConcept(factor) {
            if (conceptSvgElement && conceptZoomBehavior) {
                conceptSvgElement.transition().duration(250).call(conceptZoomBehavior.scaleBy, factor);
            }
        }

        function resetConceptZoom() {
            if (conceptSvgElement && conceptZoomBehavior) {
                conceptSvgElement.transition().duration(250).call(conceptZoomBehavior.transform, d3.zoomIdentity);
            }
        }

        // Render Codebase Flow Diagram dynamically using mermaid.render
        function renderCodebaseFlow(mermaidCode) {
            const container = document.getElementById("mermaidDiv");
            container.innerHTML = "";
            container.className = "mermaid";
            
            try {
                const uniqueId = "mermaid-svg-codebase";
                mermaid.render(uniqueId, mermaidCode).then(({ svg, bindFunctions }) => {
                    container.innerHTML = svg;
                    if (bindFunctions) bindFunctions(container);
                    
                    // Bind click listeners to folder nodes
                    setTimeout(() => {
                        const nodes = container.querySelectorAll('.node');
                        nodes.forEach(n => {
                            n.style.cursor = 'pointer';
                            n.addEventListener('click', (event) => {
                                event.stopPropagation(); // Prevent immediate closing
                                const labelText = n.querySelector('.nodeLabel')?.textContent || '';
                                if (labelText) {
                                    const folderName = labelText.replace(/[^a-zA-Z0-9_-]/g, "").trim();
                                    showFolderFilesPopup(folderName, event.clientX, event.clientY);
                                }
                            });
                        });
                    }, 500);
                }).catch(err => {
                    container.innerHTML = '<div style="color:#f87171; padding:20px; font-size:12px;">Render Error: ' + err.message + '</div>';
                });
            } catch (e) {
                container.innerHTML = '<div style="color:#f87171; padding:20px; font-size:12px;">Mermaid Error: ' + e.message + '</div>';
            }
        }

        // Folder Files Popup Menu Actions
        function showFolderFilesPopup(folderName, x, y) {
            const popup = document.getElementById("folderPopup");
            const files = flowNodes.filter(node => {
                const parts = node.id.split("/");
                const parent = parts.length > 1 ? parts[0] : "root";
                return parent === folderName;
            });

            if (files.length === 0) return;

            let html = \`<h3>📂 \${folderName}/ Files</h3>\`;
            html += files.map(file => {
                const filename = file.id.split("/").pop();
                return \`<div class="folder-file-item" onclick="openFile('\${file.id}')">📄 \${filename}</div>\`;
            }).join('');

            popup.innerHTML = html;
            popup.style.display = "block";
            popup.style.left = (x + 15) + "px";
            popup.style.top = (y - 15) + "px";
        }

        function openFile(filepath) {
            vscode.postMessage({ type: 'open-file', path: filepath });
            document.getElementById("folderPopup").style.display = "none";
        }

        // Hide popup when clicking anywhere else
        document.addEventListener('click', () => {
            const popup = document.getElementById("folderPopup");
            if (popup) popup.style.display = "none";
        });

        // Codebase Flow Zoom/Drag Event Handlers using D3
        let flowZoomBehavior = null;
        let flowSvgElement = null;

        function initFlowZoom() {
            flowSvgElement = d3.select("#flowContainer");
            const flowG = d3.select("#mermaidDiv");

            flowZoomBehavior = d3.zoom().on("zoom", (event) => {
                flowG.style("transform", \`translate(\${event.transform.x}px, \${event.transform.y}px) scale(\${event.transform.k})\`);
            });
            flowSvgElement.call(flowZoomBehavior);
        }

        // Initialize flow zoom immediately
        initFlowZoom();

        function zoomFlow(factor) {
            if (flowSvgElement && flowZoomBehavior) {
                flowSvgElement.transition().duration(250).call(flowZoomBehavior.scaleBy, factor);
            }
        }

        function resetFlowZoom() {
            if (flowSvgElement && flowZoomBehavior) {
                flowSvgElement.transition().duration(250).call(flowZoomBehavior.transform, d3.zoomIdentity);
            }
        }
    </script>
</body>
</html>`;
    }
}

// Built-in HTTP helper to avoid library compilation dependencies
function postJSON(path: string, jsonPayload: string, callback: (err: any, data: any) => void) {
    const options = {
        hostname: '127.0.0.1',
        port: 8765,
        path: path,
        method: 'POST',
        headers: {
            'Content-Type': 'application/json',
            'Content-Length': Buffer.byteLength(jsonPayload)
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

    req.write(jsonPayload);
    req.end();
}

function getJSON(path: string, callback: (err: any, data: any) => void) {
    const options = {
        hostname: '127.0.0.1',
        port: 8765,
        path: path,
        method: 'GET'
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

// Expose Sidebar Provider for Webview
import { ThoughtGitSidebarProvider } from './sidebar';
