; Inno Setup script for the Badminton Studio Windows installer.
;
; Prerequisite: run packaging\build_release.ps1 first so dist\BadmintonStudio exists.
;
; Build (per-user, no admin required):
;   "C:\Users\<you>\AppData\Local\Programs\Inno Setup 6\ISCC.exe" /DAppVersion=0.1.2 packaging\BadmintonStudio.iss
;
; Output: dist\BadmintonStudio-Setup-<version>-win64.exe
;
; Notes:
;  - The payload is the PyInstaller one-folder bundle (portable zip equivalent); the installer only
;    adds Start Menu / desktop shortcuts and an uninstaller.
;  - PrivilegesRequired=lowest keeps installation in %LOCALAPPDATA%, which is what lets a normal
;    user install it without an administrator account.
;  - WebView2 is not bundled: Windows 11 ships it, older Windows 10 machines need the Evergreen
;    Runtime, otherwise pywebview falls back to the system browser (see desktop/app.py).

#ifndef AppVersion
  #define AppVersion "0.1.0"
#endif

#define AppName "Badminton Studio"
#define AppExe "BadmintonStudio.exe"
#define AppPublisher "BadmintonStudio"

[Setup]
AppId={{8E1B6A2C-7D55-4C0E-9C1A-5B6E2F4A9D31}
AppName={#AppName}
AppVersion={#AppVersion}
AppPublisher={#AppPublisher}
DefaultDirName={localappdata}\Programs\Badminton Studio
DefaultGroupName={#AppName}
DisableProgramGroupPage=yes
PrivilegesRequired=lowest
PrivilegesRequiredOverridesAllowed=dialog
ArchitecturesAllowed=x64compatible
ArchitecturesInstallIn64BitMode=x64compatible
MinVersion=10.0
OutputDir=..\dist
OutputBaseFilename=BadmintonStudio-Setup-{#AppVersion}-win64
Compression=lzma2/max
SolidCompression=yes
WizardStyle=modern
UninstallDisplayIcon={app}\{#AppExe}
SetupLogging=yes

[Languages]
Name: "english"; MessagesFile: "compiler:Default.isl"

[Tasks]
Name: "desktopicon"; Description: "{cm:CreateDesktopIcon}"; GroupDescription: "{cm:AdditionalIcons}"; Flags: unchecked

[Files]
Source: "..\dist\BadmintonStudio\*"; DestDir: "{app}"; Flags: ignoreversion recursesubdirs createallsubdirs

[Icons]
Name: "{autoprograms}\{#AppName}"; Filename: "{app}\{#AppExe}"
Name: "{autodesktop}\{#AppName}"; Filename: "{app}\{#AppExe}"; Tasks: desktopicon

[Run]
Filename: "{app}\{#AppExe}"; Description: "{cm:LaunchProgram,{#StringChange(AppName, '&', '&&')}}"; Flags: nowait postinstall skipifsilent
