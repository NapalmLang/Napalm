# Windows Explorer file icons

These scripts are **optional** and configure file type icons in the Windows
Explorer, separately from the icon contributions in VS Code.

The package includes `../icons/napalm-file.ico`, containing the original icon
images in 16, 32, 48, 64 and 256 px. Run PowerShell from this folder:

```powershell
.\Register-NapalmFileIcons.ps1
```

The script registers `.napalm`, `.burning`, `.fire` and `.orange` in the current
user's registry only. It makes a backup at
`%LOCALAPPDATA%\Napalm\file-icons-backup.json` and does **not** change the
default app used to open any extension. You can override individual assets with
`-NapalmIcon`, `-BurningIcon`, `-FireIcon` or `-OrangeIcon` paths.

To restore the previous per-extension icon mappings:

```powershell
.\Restore-NapalmFileIcons.ps1
```

**Important:** Windows may prioritize the current application's ProgID icon
over a per-extension `DefaultIcon` value. These scripts cannot guarantee the
result in every Explorer association configuration; Windows also caches icons.
Running this script is not required for VS Code's own icons.
