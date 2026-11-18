# Build the standalone Napalm VS Code extension

The **language server remains entirely written in Napalm**. The TypeScript
builder is only release tooling; the VS Code extension uses JavaScript solely
to launch and communicate with the native server.

## Requirements

- Napalm engine built as `target/release/napalm.exe` on Windows or
  `target/release/napalm` on Linux/macOS.
- Node.js **22 or newer** (with `--experimental-strip-types`), `npm` and a
  working NPM registry. You can run the TypeScript builder directly with Bun
  instead of Node.js.
- A source checkout containing `lsp/main.napalm` and `lsp/editors/vscode`.
- For installation, the VS Code `code` CLI must be on PATH.

## Build and install (Windows x64)

Run **from the Napalm repository root**:

```powershell
node --experimental-strip-types .\lsp\build-vscode.ts --install
```

Alternatively, using Bun:

```powershell
bun run .\lsp\build-vscode.ts --install
```

Or from `lsp/editors/vscode` using NPM:

```powershell
npm run build:vsix -- --install
```

The builder performs the following operations, failing immediately on error:

1. Runs `target/release/napalm.exe --compile ./lsp/main.napalm` from the repo
   root, producing `lsp/main.exe`.
2. Checks that the compiled binary matches the host OS and architecture.
3. Stages the original Napalm artwork in `icons/`, verifies PNG dimensions and
   regenerates the multi-resolution Windows `.ico`. The generic icon-theme
   images are already rasterized to PNG, for Marketplace compatibility.
4. Runs `npm ci` if a lockfile is present, or `npm install` otherwise. Commit
   the generated `package-lock.json` after your first installation so future
   builds are reproducible.
5. Runs the TypeScript typechecker, extension tests, and builder tests.
6. Copies the self-contained native binary to `server/napalm-lsp.exe` (or
   `server/napalm-lsp`), verifying the staged SHA-256.
7. Packages the extension with `@vscode/vsce --target <platform>`, then checks
   the VSIX ZIP directory for the executable, client, syntax and icon assets.
8. If `--install` was supplied, installs it using `code --install-extension`.

The result is saved by default to:

```text
dist/vscode/napalm-language-0.1.1@win32-x64.vsix
```

You can also install the VSIX manually from **Extensions: Install from VSIX...**
inside VS Code. The installed extension automatically launches its bundled
LSP: the end user does **not** need a separate Napalm engine, LLVM install,
Napalm sources, NPM or a custom `napalm.languageServerPath` setting.

Only `.napalm` uses the LSP and TextMate grammar. `.burning`, `.fire`, and
`.orange` are registered for file icons without claiming to be source files.

## CLI switches

```text
--target win32-x64|win32-arm64|linux-x64|linux-arm64|darwin-x64|darwin-arm64
--napalm PATH       Override the engine executable
--out DIR           VSIX output folder (default dist/vscode)
--skip-compile      Package a previously compiled LSP executable
--binary PATH       With --skip-compile: location of that executable
--skip-npm          Do not reinstall NPM dependencies
--skip-tests        Skip the builder/extension tests
--install           Install the VSIX into this machine's VS Code
--publish           Publish the VSIX to Marketplace (explicit only)
--help              Display usage
```

Example to package an executable produced on another machine:

```powershell
bun run .\lsp\build-vscode.ts --skip-compile --binary .\releases\linux-arm64\napalm-lsp --target linux-arm64
```

Cross-compilation is deliberately **not** claimed or attempted: the target
must match the build machine unless `--skip-compile --binary` is given.

## Marketplace publishing (later)

1. Create a publisher at the [Visual Studio Marketplace management portal](https://marketplace.visualstudio.com/manage).
2. Update `lsp/editors/vscode/package.json`: replace `napalm-local` with your
   **actual publisher ID**, add your chosen license and review the metadata,
   screenshots and README. The included license placeholder `UNLICENSED` is
   appropriate for local development but should be deliberately reviewed
   before public distribution.
3. Increase the package version for each release.
4. Configure an authentication token in the environment as `VSCE_PAT` using
   the official VS Code publishing instructions. **Never commit the token**.
5. Package each supported platform using a matching native executable, and
   either upload the VSIX in the publisher portal or invoke:

```powershell
bun run .\lsp\build-vscode.ts --publish
```

`--publish` is opt-in and refuses to run with the placeholder publisher or
without `VSCE_PAT`. Platform-specific VSIX packages can be published under
the **same extension name and version**, each tagged with its platform.

## Diagnosing packaging problems

- **Compiler not found:** run `cargo build --release` or pass `--napalm PATH`.
- **Native binary locked on Windows:** close VS Code windows that are using
  the existing `lsp/main.exe`, or package a copy with `--skip-compile --binary`.
- **Wrong architecture:** use a binary compiled for the specified `--target`.
- **NPM install blocked:** ensure access to `registry.npmjs.org`; do not assume
  a VSIX was generated if dependency installation failed.
- **No LSP after installation:** check VS Code's **Output → Napalm Language
  Server** and remove any stale `napalm.languageServerPath` override.
- **No flame icon:** select **Preferences: File Icon Theme → Napalm File Icons**.

For source-code-only packaging, `npx vsce package` is not sufficient. Always
use this builder so it cannot accidentally ship a VSIX without the native
server executable.
