; Inno Setup script for Doc-Diff-Agent
; Requires: Inno Setup 6 (https://jrsoftware.org/isinfo.php)
;
; Build steps:
;   1. pyinstaller build/doc_diff_agent.spec
;   2. iscc build/installer.iss

#define AppName      "Doc-Diff-Agent"
#define AppVersion   "1.0.0"
#define AppPublisher "DocDiffAgent"
#define AppURL       "https://github.com/deceive777xv/doc-diff-agent"
#define AppExeName   "DocDiffAgent.exe"
#define DistDir      "..\dist\DocDiffAgent"

[Setup]
AppId={{A7F3C2D1-9B4E-4F2A-8C6D-1E5B3A7F9C0D}
AppName={#AppName}
AppVersion={#AppVersion}
AppPublisherURL={#AppURL}
AppSupportURL={#AppURL}
AppUpdatesURL={#AppURL}
DefaultDirName={autopf}\{#AppName}
DefaultGroupName={#AppName}
AllowNoIcons=yes
LicenseFile=
; Installer output directory (relative to this .iss file)
OutputDir=..\dist
OutputBaseFilename=DocDiffAgent-v{#AppVersion}-setup
Compression=lzma2/ultra64
SolidCompression=yes
WizardStyle=modern
; Require Windows 10 x64
MinVersion=10.0.17763
ArchitecturesAllowed=x64compatible
ArchitecturesInstallIn64BitMode=x64compatible
; Allow elevation for all users
PrivilegesRequiredOverridesAllowed=dialog

[Languages]
Name: "chinesesimplified"; MessagesFile: "compiler:Languages\ChineseSimplified.isl"
Name: "english"; MessagesFile: "compiler:Default.isl"

[Tasks]
Name: "desktopicon"; Description: "{cm:CreateDesktopIcon}"; GroupDescription: "{cm:AdditionalIcons}"; Flags: unchecked
Name: "quicklaunchicon"; Description: "{cm:CreateQuickLaunchIcon}"; GroupDescription: "{cm:AdditionalIcons}"; Flags: unchecked; OnlyBelowVersion: 6.1; Check: not IsAdminInstallMode

[Files]
; All files from PyInstaller onedir output
Source: "{#DistDir}\*"; DestDir: "{app}"; Flags: ignoreversion recursesubdirs createallsubdirs

[Icons]
Name: "{group}\{#AppName}"; Filename: "{app}\{#AppExeName}"
Name: "{group}\{cm:UninstallProgram,{#AppName}}"; Filename: "{uninstallexe}"
Name: "{autodesktop}\{#AppName}"; Filename: "{app}\{#AppExeName}"; Tasks: desktopicon
Name: "{userappdata}\Microsoft\Internet Explorer\Quick Launch\{#AppName}"; Filename: "{app}\{#AppExeName}"; Tasks: quicklaunchicon

[Run]
Filename: "{app}\{#AppExeName}"; Description: "{cm:LaunchProgram,{#StringChange(AppName, '&', '&&')}}"; Flags: nowait postinstall skipifsilent

[UninstallDelete]
; Remove data directory created by the app
Type: filesandordirs; Name: "{localappdata}\DocDiffAgent"
