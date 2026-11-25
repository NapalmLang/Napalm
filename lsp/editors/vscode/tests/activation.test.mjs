import assert from 'node:assert/strict';
import {readFileSync} from 'node:fs';
import path from 'node:path';
import test from 'node:test';
import vm from 'node:vm';

const source = readFileSync(new URL('../extension.js', import.meta.url), 'utf8');

function setup(options = {}) {
    const extensionPath = path.resolve('extension');
    const workspacePath = path.resolve('workspace');
    const bundled = path.join(extensionPath, 'server', process.platform === 'win32' ? 'napalm-lsp.exe' : 'napalm-lsp');
    const files = new Map([[bundled, true], ...(options.files ?? [])]);
    const log = [];
    const errors = [];
    const clients = [];
    const commands = new Map();
    let configured = options.configured ?? '';
    let configurationChanged;
    let foldersChanged;
    let watcherCount = 0;
    const disposable = () => ({dispose() {}});
    const document = {languageId: 'napalm', uri: {scheme: 'file', fsPath: path.join(workspacePath, 'test.napalm')}};
    const output = {
        ...disposable(), appendLine(message) { log.push(message); },
        info(message) { log.push(message); }, error(message) { log.push(message); },
        warn() {}, debug() {}, trace() {}, show() { log.push('output shown'); },
    };
    const vscode = {
        window: {
            activeTextEditor: options.noDocument ? undefined : {document},
            createOutputChannel(_name, settings) {
                if (settings?.log) {
                    output.logLevel = 3;
                    output.onDidChangeLogLevel = disposable;
                }
                return output;
            },
            showErrorMessage(message) {
                errors.push(message);
                return options.pendingNotice ? new Promise(() => {}) : Promise.resolve('Show Output');
            },
        },
        commands: {registerCommand(name, callback) { commands.set(name, callback); return disposable(); }},
        workspace: {
            textDocuments: options.noDocument ? [] : [document],
            workspaceFolders: options.noWorkspace ? undefined : [{uri: {fsPath: workspacePath}}],
            getWorkspaceFolder() { return this.workspaceFolders?.[0]; },
            getConfiguration() { return {get() { return configured; }}; },
            createFileSystemWatcher() { watcherCount++; return disposable(); },
            onDidChangeConfiguration(callback) { configurationChanged = callback; return disposable(); },
            onDidChangeWorkspaceFolders(callback) { foldersChanged = callback; return disposable(); },
        },
    };
    class LanguageClient {
        constructor(_id, _name, server, settings) {
            if (options.constructorError) throw new Error('constructor failed');
            this.server = server;
            this.settings = settings;
            this.disposed = false;
            clients.push(this);
        }
        async start() {
            // Contract used by the real languageclient 11 during initialization.
            this.settings.outputChannel.onDidChangeLogLevel(() => {});
            if (options.startError) throw new Error('start failed');
            if (options.startGate) await options.startGate;
        }
        async dispose() { this.disposed = true; }
    }
    const context = {extensionPath, subscriptions: []};
    const module = {exports: {}};
    vm.runInNewContext(source, {module, process, require(name) {
        if (name === 'node:fs') return {
            existsSync(file) { return files.has(file); },
            statSync(file) { return {isFile() { return files.get(file); }}; },
        };
        if (name === 'node:path') return path;
        if (name === 'vscode') return vscode;
        if (name === 'vscode-languageclient/node') {
            if (options.loadError) throw new Error('Cannot find module vscode-languageclient/node');
            return {LanguageClient};
        }
        throw new Error(`Unexpected import: ${name}`);
    }}, {filename: 'extension.js'});
    return {
        api: module.exports, context, clients, errors, log, bundled, files, workspacePath,
        restart() { return commands.get('napalm.restartLanguageServer')(); },
        configure(value) { configured = value; return configurationChanged({affectsConfiguration: () => true}); },
        foldersChanged() { return foldersChanged(); },
        watcherCount() { return watcherCount; },
    };
}

test('activation supplies a LogOutputChannel compatible with languageclient 11', async () => {
    const app = setup();
    await app.api.activate(app.context);
    assert.deepEqual(app.errors, []);
    assert.equal(app.clients.length, 1);
    assert.equal(app.clients[0].server.command, app.bundled);
    await app.api.deactivate();
    assert.ok(app.clients[0].disposed);
});

for (const fault of ['loadError', 'constructorError', 'startError']) {
    test(`${fault} is reported with details and can be retried`, async () => {
        const options = {[fault]: true, pendingNotice: true};
        const app = setup(options);
        await app.api.activate(app.context);
        assert.equal(app.errors.length, 1);
        assert.ok(app.errors[0].includes(fault === 'loadError' ? 'Cannot find module' : fault === 'constructorError' ? 'constructor failed' : 'start failed'));
        assert.ok(app.clients.every(client => client.disposed));
        options[fault] = false;
        await app.restart();
        assert.ok(!app.clients.at(-1).disposed);
        assert.equal(app.watcherCount(), 1);
        await app.api.deactivate();
    });
}

test('missing, directory and invalid settings recover when the path is corrected', async () => {
    const app = setup({configured: 'missing.exe'});
    await app.api.activate(app.context);
    assert.match(app.errors[0], /not an executable file/);
    const directory = path.join(app.workspacePath, 'directory');
    app.files.set(directory, false);
    await app.configure('directory');
    assert.match(app.errors[1], /not an executable file/);
    await app.configure(42);
    assert.match(app.errors[2], /must be a string/);
    const custom = path.join(app.workspacePath, 'custom server.exe');
    app.files.set(custom, true);
    await app.configure('custom server.exe');
    assert.equal(app.clients[0].server.command, custom);
    await app.configure('');
    assert.equal(app.clients[1].server.command, app.bundled);
    assert.ok(app.clients[0].disposed);
    await app.api.deactivate();
});

test('standalone files and activation without a document use a valid working directory', async () => {
    for (const noDocument of [false, true]) {
        const app = setup({noWorkspace: true, noDocument});
        await app.api.activate(app.context);
        assert.deepEqual(app.errors, []);
        assert.equal(app.clients[0].server.options.cwd, noDocument ? path.dirname(app.bundled) : app.workspacePath);
        await app.api.deactivate();
    }
});

test('workspace changes restart the client and reuse a single file watcher', async () => {
    const app = setup();
    await app.api.activate(app.context);
    await app.foldersChanged();
    assert.equal(app.clients.length, 2);
    assert.ok(app.clients[0].disposed);
    assert.equal(app.watcherCount(), 1);
    await app.api.deactivate();
});

test('deactivation waits for startup and prevents queued restarts from spawning clients', async () => {
    let release;
    const app = setup({startGate: new Promise(resolve => { release = resolve; })});
    const activating = app.api.activate(app.context);
    // Reach start() before requesting restarts and deactivation.
    await new Promise(resolve => setImmediate(resolve));
    const restart = app.restart();
    const stopping = app.api.deactivate();
    release();
    await Promise.all([activating, restart, stopping]);
    assert.equal(app.clients.length, 1);
    assert.ok(app.clients[0].disposed);
    await app.restart();
    assert.equal(app.clients.length, 1);
});
