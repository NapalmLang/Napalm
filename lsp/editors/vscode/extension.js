'use strict';

// VS Code transport adapter only. Lexing, parsing, type inference and all
// language features execute in the standalone Napalm language server.
const fs = require('node:fs');
const path = require('node:path');
const vscode = require('vscode');
const {LanguageClient} = require('vscode-languageclient/node');

let client;
let output;

function firstDocument() {
    const active = vscode.window.activeTextEditor?.document;
    if (active?.languageId === 'napalm' && active.uri.scheme === 'file') return active;
    return vscode.workspace.textDocuments.find(
        (document) => document.languageId === 'napalm' && document.uri.scheme === 'file',
    );
}

function executableFor(document, extensionPath) {
    const folder = document && vscode.workspace.getWorkspaceFolder(document.uri);
    const root = folder?.uri.fsPath ?? vscode.workspace.workspaceFolders?.[0]?.uri.fsPath ??
        (document ? path.dirname(document.uri.fsPath) : undefined);
    const specified = vscode.workspace.getConfiguration('napalm', document?.uri)
        .get('languageServerPath', '').trim();
    if (specified !== '') {
        const resolved = path.isAbsolute(specified) ? specified : path.resolve(root ?? process.cwd(), specified);
        return {command: resolved, cwd: root ?? path.dirname(resolved)};
    }
    const bundledName = process.platform === 'win32' ? 'napalm-lsp.exe' : 'napalm-lsp';
    const bundled = path.join(extensionPath, 'server', bundledName);
    if (fs.existsSync(bundled) && fs.statSync(bundled).isFile()) {
        return {command: bundled, cwd: root ?? path.dirname(document?.uri.fsPath ?? bundled)};
    }
    // Support source checkouts for local extension development.
    if (!root) return null;
    const binary = process.platform === 'win32' ? 'main.exe' : 'main';
    const standalone = process.platform === 'win32' ? 'napalm-lsp.exe' : 'napalm-lsp';
    for (const candidate of [
        path.join(root, 'lsp', binary),
        path.join(root, standalone),
        path.join(root, 'lsp', standalone),
    ]) {
        if (fs.existsSync(candidate) && fs.statSync(candidate).isFile()) {
            return {command: candidate, cwd: root};
        }
    }
    return null;
}

async function activate(context) {
    output = vscode.window.createOutputChannel('Napalm Language Server');
    context.subscriptions.push(output);
    output.appendLine('Napalm extension activated.');
    const document = firstDocument();
    output.appendLine(`Active Napalm document: ${document?.uri.fsPath ?? '(none)'}`);
    output.appendLine(`Workspace: ${vscode.workspace.workspaceFolders?.[0]?.uri.fsPath ?? '(none)'}`);
    const server = executableFor(document, context.extensionPath);
    if (!server || !fs.existsSync(server.command)) {
        const message = server ? `The configured Napalm LSP executable was not found: ${server.command}` :
            'Napalm LSP executable is missing. Install a platform-specific VSIX with the bundled server or set napalm.languageServerPath.';
        output.appendLine(message);
        vscode.window.showWarningMessage(message);
        return;
    }
    output.appendLine(`Launching LSP: ${server.command}`);
    output.appendLine(`Working directory: ${server.cwd}`);
    const watcher = vscode.workspace.createFileSystemWatcher('**/*.napalm');
    context.subscriptions.push(watcher);
    const options = {
        documentSelector: [{language: 'napalm', scheme: 'file'}],
        synchronize: {fileEvents: watcher},
        outputChannel: output,
    };
    client = new LanguageClient('napalmLanguageServer', 'Napalm Language Server', {
        command: server.command,
        args: [],
        options: {cwd: server.cwd},
    }, options);
    try {
        await client.start();
        output.appendLine(`Napalm language server started: ${server.command}`);
    } catch (error) {
        output.appendLine(`Failed to start Napalm LSP: ${error?.stack ?? String(error)}`);
        vscode.window.showErrorMessage('Could not start the Napalm language server. See the Napalm Language Server output.');
    }
}

async function deactivate() {
    if (client) {
        await client.stop();
        client = undefined;
    }
}

module.exports = {activate, deactivate};
