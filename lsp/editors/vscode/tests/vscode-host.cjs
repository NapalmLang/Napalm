// Executed by VS Code's Extension Development Host, using the real vscode API.
const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const vscode = require('vscode');

async function until(description, read) {
    const deadline = Date.now() + 15000;
    let last;
    while (Date.now() < deadline) {
        last = await read();
        if (last) return last;
        await new Promise(resolve => setTimeout(resolve, 100));
    }
    throw new Error(`Timed out waiting for ${description}: ${JSON.stringify(last)}`);
}

exports.run = async function () {
    const report = process.env.NAPALM_VSCODE_TEST_REPORT;
    const passed = [];
    try {
        const root = vscode.workspace.workspaceFolders[0].uri.fsPath;
        const uri = vscode.Uri.file(path.join(root, 'editor.napalm'));
        const source = 'const value = 1;\nconst result = value;\nconst broken = missing;\n';
        fs.writeFileSync(uri.fsPath, source);
        const document = await vscode.workspace.openTextDocument(uri);
        await vscode.window.showTextDocument(document);
        assert.equal(document.languageId, 'napalm');
        await vscode.extensions.getExtension('napalm-local.napalm-language').activate();
        await until('semantic diagnostics', () => vscode.languages.getDiagnostics(uri).some(d => d.code === 'NAPALM-SEM-001'));
        passed.push('activation and diagnostics');

        const position = new vscode.Position(1, 16);
        const definitions = await vscode.commands.executeCommand('vscode.executeDefinitionProvider', uri, position);
        assert.equal(definitions.length, 1);
        assert.equal(definitions[0].uri.toString(), uri.toString());
        assert.equal(definitions[0].range.start.line, 0);
        assert.equal(definitions[0].range.start.character, 6);
        const hover = await vscode.commands.executeCommand('vscode.executeHoverProvider', uri, position);
        assert.ok(hover.some(h => h.contents.some(c => c.value.includes('value'))));
        const completions = await vscode.commands.executeCommand('vscode.executeCompletionItemProvider', uri, new vscode.Position(1, 18));
        assert.ok(completions.items.some(item => item.label === 'value'));
        const refs = await vscode.commands.executeCommand('vscode.executeReferenceProvider', uri, position);
        assert.equal(refs.length, 2);
        const rename = await vscode.commands.executeCommand('vscode.executeDocumentRenameProvider', uri, position, 'renamed');
        assert.equal(rename.get(uri).length, 2);
        const tokens = await vscode.commands.executeCommand('vscode.provideDocumentSemanticTokens', uri);
        assert.ok(tokens.data.length > 0);
        passed.push('definition, hover, completion, references, rename and semantic tokens');

        const edit = new vscode.WorkspaceEdit();
        edit.replace(uri, new vscode.Range(2, 15, 2, 22), 'value');
        await vscode.workspace.applyEdit(edit);
        await until('diagnostics cleared after unsaved edit', () => vscode.languages.getDiagnostics(uri).length === 0);
        const edited = await vscode.commands.executeCommand('vscode.executeDefinitionProvider', uri, new vscode.Position(2, 16));
        assert.equal(edited[0].range.start.character, 6);
        passed.push('unsaved incremental edits');

        await vscode.commands.executeCommand('napalm.restartLanguageServer');
        await until('providers after restart', async () => {
            const result = await vscode.commands.executeCommand('vscode.executeDefinitionProvider', uri, position);
            return result?.length === 1;
        });
        passed.push('restart with open unsaved document');

        const config = vscode.workspace.getConfiguration('napalm');
        await config.update('languageServerPath', path.join(root, 'does-not-exist.exe'), vscode.ConfigurationTarget.Workspace);
        await vscode.commands.executeCommand('napalm.restartLanguageServer');
        await config.update('languageServerPath', '', vscode.ConfigurationTarget.Workspace);
        await vscode.commands.executeCommand('napalm.restartLanguageServer');
        await until('recovery after invalid path', async () => {
            const result = await vscode.commands.executeCommand('vscode.executeDefinitionProvider', uri, position);
            return result?.length === 1;
        });
        passed.push('configuration error and recovery');

        // Changes remain unsaved until this point, exercising didChange rather than disk reads.
        await document.save();
        await vscode.commands.executeCommand('workbench.action.closeAllEditors');
        if (report) fs.writeFileSync(report, JSON.stringify({passed}, null, 2));
        console.log(`Napalm VS Code integration passed: ${passed.join('; ')}`);
    } catch (error) {
        if (report) fs.writeFileSync(report, JSON.stringify({passed, error: error.stack}, null, 2));
        throw error;
    }
};
