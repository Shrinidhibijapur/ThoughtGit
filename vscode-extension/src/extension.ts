import * as vscode from 'vscode';
import * as http from 'http';

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
    const payload = JSON.stringify({
        content: contentToIngest,
        source: source,
        metadata: {
            file_name: filePath.substring(filePath.lastIndexOf('/') + 1),
            file_path: filePath,
            is_code: String(isCode)
        }
    });

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

// Expose Sidebar Provider for Webview
import { ThoughtGitSidebarProvider } from './sidebar';
