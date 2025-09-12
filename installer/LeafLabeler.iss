#define MyAppName "LeafLabeler"
#define MyAppVersion "1.0.0"
#define MyAppPublisher "OS AI Corp"
#define MyAppExeName "LeafLabeler.exe"
#define DistDir "..\dist\LeafLabeler"
#define OutputDir ".\Output"
#define MyAppIcon "..\src\leaf_labeler\assets\logo_os_ai_corp.ico"
#define LicenseFile "..\LICENSE.txt"

[Setup]
AppId={{7F2F7C3C-7D5A-4B0C-8E5A-9B1F8C20F9A1}
AppName={#MyAppName}
AppVersion={#MyAppVersion}
AppPublisher={#MyAppPublisher}
DefaultDirName={autopf}\{#MyAppName}
DefaultGroupName={#MyAppName}
OutputBaseFilename={#MyAppName}-Setup
OutputDir={#OutputDir}
Compression=lzma2
SolidCompression=yes
ArchitecturesInstallIn64BitMode=x64
WizardStyle=modern
SetupIconFile={#MyAppIcon}
LicenseFile={#LicenseFile}
PrivilegesRequired=admin
DisableDirPage=no
UninstallDisplayIcon={app}\{#MyAppExeName}
UninstallDisplayName={#MyAppName}

; Optional code sign (ambil env var dari build.ps1)
SignTool= \
  $q{code}; \
  if (GetEnv('SIGN_PFX') != '') { \
    SignTool sign /fd SHA256 /f "$p" /p "$f" /tr http://timestamp.digicert.com /td SHA256 "$f"; \
  } \
  {code}$q

[Files]
Source: "{#DistDir}\*"; DestDir: "{app}"; Flags: recursesubdirs ignoreversion

[Icons]
Name: "{group}\{#MyAppName}"; Filename: "{app}\{#MyAppExeName}"
Name: "{commondesktop}\{#MyAppName}"; Filename: "{app}\{#MyAppExeName}"; Tasks: desktopicon

[Tasks]
Name: "desktopicon"; Description: "Create a desktop shortcut"; GroupDescription: "Additional icons:"

[Run]
Filename: "{app}\{#MyAppExeName}"; Description: "Launch {#MyAppName}"; Flags: nowait postinstall skipifsilent
