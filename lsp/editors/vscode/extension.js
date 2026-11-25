'use strict';

// VS Code transport adapter only. Lexing, parsing, type inference and all
// language features execute in the standalone Napalm language server.
const fs = require('node:fs');
const path = require('node:path');
const vscode = require('vscode');

let client;
let output;
let pending = Promise.resolve();
let stopping = false;
let watcher;

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
    // One client serves the window; this setting has window scope, not resource scope.
    const configured = vscode.workspace.getConfiguration('napalm')
        .get('languageServerPath', '');
    if (typeof configured !== 'string') throw new Error('napalm.languageServerPath must be a string.');
    const specified = configured.trim();
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

async function disposeClient() {
    const previous = client;
    client = undefined;
    if (previous) await previous.dispose();
}

async function startServer(context) {
    try {
        await disposeClient();
        if (stopping) return;
        const document = firstDocument();
        const server = executableFor(document, context.extensionPath);
        if (!server) throw new Error('Napalm LSP executable is missing. Install a platform-specific VSIX or set napalm.languageServerPath.');
        if (!fs.existsSync(server.command) || !fs.statSync(server.command).isFile()) {
            throw new Error(`The Napalm LSP path is not an executable file: ${server.command}`);
        }
        output.info(`Launching LSP: ${server.command}`);
        output.info(`Working directory: ${server.cwd}`);
        // Load inside the error boundary so incomplete installs get a useful error.
        const {LanguageClient} = require('vscode-languageclient/node');
        client = new LanguageClient('napalmLanguageServer', 'Napalm Language Server', {
            command: server.command,
            args: [],
            options: {cwd: server.cwd},
        }, {
            documentSelector: [{language: 'napalm', scheme: 'file'}],
            synchronize: {fileEvents: watcher},
            outputChannel: output,
        });
        await client.start();
        output.info(`Napalm language server started: ${server.command}`);
    } catch (error) {
        output.error(`Failed to start Napalm LSP: ${error?.stack ?? String(error)}`);
        try { await disposeClient(); }
        catch (cleanupError) { output.error(`LSP cleanup failed: ${cleanupError}`); }
        if (!stopping) {
            // A notification can remain open indefinitely; never block restarts or
            // deactivation waiting for it to be dismissed.
            void vscode.window.showErrorMessage(
                `Could not start Napalm: ${error?.message ?? String(error)}`, 'Show Output'
            ).then(choice => {
                if (!stopping && choice === 'Show Output') output.show(true);
            });
        }
    }
}

function restart(context) {
    // Configuration changes and explicit restarts must not launch overlapping clients.
    pending = pending.then(() => { if (!stopping) return startServer(context); });
    return pending;
}

async function activate(context) {
    stopping = false;
    output = vscode.window.createOutputChannel('Napalm Language Server', {log: true});
    watcher = vscode.workspace.createFileSystemWatcher('**/*.napalm');
    context.subscriptions.push(output, watcher,
        vscode.commands.registerCommand('napalm.restartLanguageServer', () => restart(context)),
        vscode.workspace.onDidChangeConfiguration(event => {
            if (event.affectsConfiguration('napalm.languageServerPath')) return restart(context);
        }),
        // The server receives workspace roots during initialize; refresh them on changes.
        vscode.workspace.onDidChangeWorkspaceFolders(() => restart(context)),
    );
    output.info('Napalm extension activated.');
    await restart(context);
}

async function deactivate() {
    stopping = true;
    await pending;
    try { await disposeClient(); }
    catch (error) { output?.error(`Failed to stop Napalm LSP: ${error}`); }
}

module.exports = {activate, deactivate};
