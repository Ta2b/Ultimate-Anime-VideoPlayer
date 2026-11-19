; Installeur Windows (Inno Setup 6) : tools/release/package-windows.ps1 le prépare, release-windows.yml le lance.
;   ISCC /DSource=<dossier kirari-windows-x64> /DAppVer=0.7.0 /DOutput=<dossier> kirari.iss
; Installation pour l'utilisateur seul (aucun droit d'administrateur), dans son dossier de programmes. « Ouvrir avec » et le lecteur par
; défaut restent le choix du lecteur au premier lancement ; le zip portable reste proposé à côté.
#ifndef Source
  #error "Source : le dossier du lecteur (kirari.exe et ses DLL)"
#endif
#ifndef AppVer
  #define AppVer "0.0.0"
#endif
#ifndef Output
  #define Output "."
#endif

[Setup]
AppId={{6F1D2B7A-3C4E-4D5A-9B8E-1A2C3D4E5F60}
AppName=Kirari Player
AppVersion={#AppVer}
AppPublisher=Ta2b
DefaultDirName={localappdata}\Programs\Kirari Player
; Un ajp installé (même AppId) était dans Programs\AJP : Kirari s'installe dans son propre dossier et retire
; l'ancien (InstallDelete, PrepareToInstall).
UsePreviousAppDir=no
DefaultGroupName=Kirari Player
PrivilegesRequired=lowest
ArchitecturesAllowed=x64compatible
ArchitecturesInstallIn64BitMode=x64compatible
OutputDir={#Output}
OutputBaseFilename=kirari-windows-x64-setup
Compression=lzma2/max
SolidCompression=yes
WizardStyle=modern
UninstallDisplayIcon={app}\kirari.exe
SetupIconFile=kirari.ico
DisableProgramGroupPage=yes

[Languages]
Name: "french"; MessagesFile: "compiler:Languages\French.isl"
Name: "english"; MessagesFile: "compiler:Default.isl"

[Tasks]
Name: "desktopicon"; Description: "Créer un raccourci sur le bureau"; Flags: unchecked

[Files]
Source: "{#Source}\*"; DestDir: "{app}"; Flags: recursesubdirs createallsubdirs ignoreversion

[InstallDelete]
; ajp, le lecteur d'avant Kirari (même AppId) : son dossier et ses raccourcis. Ses réglages et moteurs ne sont pas
; là : le lecteur les reprend lui-même au premier lancement (src/app/play/legacy_dirs.h).
Type: filesandordirs; Name: "{localappdata}\Programs\AJP"
Type: files; Name: "{autoprograms}\Ultimate Anime VideoPlayer.lnk"
Type: files; Name: "{autodesktop}\Ultimate Anime VideoPlayer.lnk"

[Icons]
Name: "{autoprograms}\Kirari Player"; Filename: "{app}\kirari.exe"
Name: "{autodesktop}\Kirari Player"; Filename: "{app}\kirari.exe"; Tasks: desktopicon

[Run]
Filename: "{app}\kirari.exe"; Description: "Lancer Kirari Player"; Flags: nowait postinstall skipifsilent

[UninstallRun]
; Les associations « Ouvrir avec » et le lecteur par défaut posés par le lecteur (HKCU) partent avec lui.
Filename: "{app}\kirari.exe"; Parameters: "--unregister"; Flags: runhidden waituntilterminated skipifdoesntexist; RunOnceId: "kirari-unregister"

[Code]
// Les données du lecteur ne sont pas dans {app} : réglages et journaux (%APPDATA%\kirari), moteurs construits,
// paquet TensorRT, modèles dérivés et caches (%LOCALAPPDATA%\kirari), souvent plusieurs Go. La désinstallation
// le dit et propose de les retirer (A36-07) ; une désinstallation silencieuse les garde.
procedure CurUninstallStepChanged(CurUninstallStep: TUninstallStep);
var
  Roaming, Local: String;
begin
  if (CurUninstallStep <> usPostUninstall) or UninstallSilent then
    exit;
  Roaming := ExpandConstant('{userappdata}\kirari');
  Local := ExpandConstant('{localappdata}\kirari');
  if not DirExists(Roaming) and not DirExists(Local) then
    exit;
  if MsgBox('Supprimer aussi les réglages, les moteurs téléchargés et les caches du lecteur ?' + #13#10#13#10 +
            Roaming + #13#10 + Local + #13#10#13#10 +
            'Ils peuvent occuper plusieurs Go. Non : ils restent pour une prochaine installation.',
            mbConfirmation, MB_YESNO or MB_DEFBUTTON2) = IDYES then
  begin
    DelTree(Roaming, True, True, True);
    DelTree(Local, True, True, True);
  end;
end;

// ajp, installé avant Kirari : son « Ouvrir avec » et son inscription comme lecteur par défaut (HKCU) partent
// avant que son dossier soit retiré, sinon ils pointeraient vers un ajp.exe disparu.
function PrepareToInstall(var NeedsRestart: Boolean): String;
var
  Old: String;
  Code: Integer;
begin
  Result := '';
  Old := ExpandConstant('{localappdata}\Programs\AJP\ajp.exe');
  if FileExists(Old) then
    Exec(Old, '--unregister', '', SW_HIDE, ewWaitUntilTerminated, Code);
end;
