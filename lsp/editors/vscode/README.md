# Napalm Language for VS Code

This extension provides the original declarative TextMate highlighting and a
minimal VS Code **language client**. The language server itself, including the
lexer, parser, semantic analysis, module resolution, completion and hover, is
written entirely in Napalm. The extension's JavaScript file only launches and
connects to the standalone executable through the LSP protocol.

## Build the language server

From the repository root:

```powershell
napalm.exe --compile .\lsp\main.napalm
```

This produces `lsp/main.exe` on Windows or `lsp/main` on Linux/macOS,
which the extension locates automatically in a workspace rooted at the Napalm
repository. Alternatively set the VS Code setting `napalm.languageServerPath`
to an absolute executable path, or a path relative to your workspace. This
setting is useful for projects outside the Napalm repository. Changing it
restarts the server automatically. **Napalm: Restart Language Server** also
restarts it after a failed launch or an executable rebuild.

The extension synchronizes open `.napalm` documents (including unsaved
changes) and forwards `**/*.napalm` file changes to the server. Standard
`vscode-languageclient` is used as the protocol transport; all language
analysis stays inside the Napalm executable.

## Install for local development

After installing this directory's dependencies and building the server, launch
an Extension Development Host from the repository root:

```powershell
code --extensionDevelopmentPath="$PWD/lsp/editors/vscode" .
```

Then open a `.napalm` source file.

To build and install a VSIX after installing Node.js, from the repository root:

```sh
node lsp/build-vscode.ts --install
```

`publisher` is `napalm-local` as a **local packaging placeholder**. Replace it
with your real Visual Studio Marketplace publisher, set the final license, and
remove `private` when you prepare public distribution.

## Branding and file icons

This extension includes the Napalm artwork supplied by the project author:

- `icons/extension-logo-256.png` is the **extension's logo** in VS Code's
  Extensions view and when published to a Marketplace.
- `icons/flame-16.png`, `flame-32.png`, `flame-48.png`, `flame-64.png`,
  `flame-256.png` are the supplied file icon sizes with transparency.
- `icons/napalm-file.ico` provides native Windows Explorer sizes.

`.napalm`, `.burning`, `.fire` and `.orange` are registered with the same
flame icon. Only `.napalm` receives source syntax highlighting and launches
the LSP. The other formats are not assumed to contain source text.

**Without changing your selected icon theme:** the contributed language icons
are used as fallbacks by themes that support them; a theme with a more
specific association can override them.

**To guarantee the flame in VS Code:** use
`Ctrl+Shift+P` → **Preferences: File Icon Theme** → **Napalm File Icons**.
This optional theme explicitly maps all four extensions. Selecting it will
also replace icons for unrelated files with the theme's neutral defaults;
VS Code does not allow an extension to forcibly inject icons into every
other theme without changing that theme.

You can replace individual icon assets later with
`node scripts/configure-icons.mjs --napalm PATH --fire PATH --burning PATH --orange PATH`.
This script updates both fallback icon contributions and the optional theme.

For Windows Explorer, see `windows/README.md`; it is independent of VS Code.

## What is highlighted?

`.napalm` uses `source.napalm` TextMate grammar: comments, strings, nested
template interpolation, numbers, operators (`|>`), declarations, keywords,
builtin names, and decorators. `.burning` and `.fire` are registered as
independent file types for icon association **without** a syntax grammar. The
source format of `.fire` was not confirmed from the current public manual.

The TextMate grammar provides lexical highlighting even if the server is not
installed. The LSP provides syntax/semantic diagnostics, definition, references,
document symbols, completion and hover when its executable is available.
TextMate provides fallback colors; semantic tokens are also available from the LSP.

## Testing

```sh
npm test
```

Tests validate activation, error recovery, registration, grammar and the production
dependencies selected by VSCE. The builder also runs live protocol tests against
the compiled server on the native platform before packaging it.
Opening the sample in VS Code remains a useful visual smoke test; actual
TextMate tokenization is owned by the editor.

On Windows, run the actual client and server in an isolated VS Code profile:

```powershell
npm run test:editor:windows
# Or test the contents of an extracted VSIX with a specific VS Code installation:
.\scripts\Test-VSCode.ps1 -CodePath 'C:\Program Files\Microsoft VS Code\Code.exe' -ExtensionPath 'C:\path\to\extracted\extension'
```

This verifies activation, diagnostics, navigation, hover, completion, references,
rename, semantic tokens, unsaved edits, restart and recovery from an invalid
server path. Reports and logs remain under the repository's `target` directory.
Language features currently apply to local `file:` documents; a new untitled
buffer must be saved as a `.napalm` file first.

## Production distribution: bundle the native server inside the VSIX

**Important:** `npx vsce package` alone does **not** include the compiled Napalm
server. It only produces the extension client and editor resources. Use the
platform-aware bundled packaging command below for a redistributable release.

Build the standalone LSP from the repository root:

```powershell
.\target\release\napalm.exe --compile .\lsp\main.napalm
```

On Windows x64, with the compiled `lsp/main.exe` present, from the repository root:

```powershell
node .\lsp\build-vscode.ts --skip-compile --target win32-x64
code --install-extension ".\dist\vscode\napalm-language-0.1.3@win32-x64.vsix" --force
```

The packaging script copies `../../main.exe` into
`server/napalm-lsp.exe`, invokes `@vscode/vsce` with `--target win32-x64`, and
**verifies the final VSIX ZIP directory** contains that executable with the
correct uncompressed byte count and contains the language client/protocol modules.
The `files` allowlist must include `node_modules/**`: VSCE collects production
dependencies, then applies this allowlist. Omitting it produces an extension
that colors syntax but fails to activate its language client.
The build also fails if the server is missing, too small, or built for a
different operating system/architecture.

For an executable built separately or for another target:

```sh
node lsp/build-vscode.ts --skip-compile --target linux-arm64 --binary /path/to/napalm-lsp
```

Supported targets: `win32-x64`, `win32-arm64`, `linux-x64`, `linux-arm64`,
`darwin-x64`, and `darwin-arm64`. Build the Napalm standalone executable for
each target before invoking the packaging command. VS Code determines the
platform of the *extension host* (which can be remote) when installing.

The installed extension selects the server in this order:

1. Explicit `napalm.languageServerPath` override, if configured.
2. The executable bundled at `extension/server/napalm-lsp(.exe)`.
3. A workspace-local executable (for developing from the source tree).

For a normal installation, **clear any old `napalm.languageServerPath` setting**
so it does not override the bundled server.

The runtime, LLVM and the Napalm LSP modules are part of the native executable
produced by `napalm --compile`: they are not copied as individual Rust, LLVM,
or `.napalm` files into the extension. Other platform-specific runtime
requirements depend on how that executable was linked; validate it on a clean
machine before publishing.

## If highlighting works but language features do not

Check **Output → Napalm Language Server** and **Output → Log (Extension Host)**.
An activation error such as `Cannot find module 'vscode-languageclient/node'`
means the installed VSIX is incomplete; rebuild/install it with the command above
and run **Developer: Reload Window**. Version 0.1.2 includes the missing client
dependencies and fixes analysis updates after document edits.
Version 0.1.3 uses the log channel required by languageclient 11, reports launch
errors with their cause and lets the client recover without reloading the window.

To test the standalone executable without VS Code (PowerShell, repository root):

```powershell
$env:NAPALM_LSP_BIN = (Resolve-Path .\lsp\main.exe).Path
node --test .\lsp\tests\server.test.mjs
Remove-Item Env:NAPALM_LSP_BIN
```

These tests verify diagnostics, definition, references, hover, completion, rename,
semantic tokens and unsaved edits while keeping the input connection open.
