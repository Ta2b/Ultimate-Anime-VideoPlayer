; Installeur Windows (Inno Setup 6) : tools/release/package-windows.ps1 le prépare, release-windows.yml le lance.
;   ISCC /DSource=<dossier ajp-windows-x64> /DAppVer=0.7.0 /DOutput=<dossier> ajp.iss
; Installation pour l'utilisateur seul (aucun droit d'administrateur), dans son dossier de programmes. « Ouvrir avec » et le lecteur par
; défaut restent le choix du lecteur au premier lancement ; le zip portable reste proposé à côté.
#ifndef Source
  #error "Source : le dossier du lecteur (ajp.exe et ses DLL)"
#endif
#ifndef AppVer
  #define AppVer "0.0.0"
#endif
#ifndef Output
  #define Output "."
#endif

[Setup]
AppId={{6F1D2B7A-3C4E-4D5A-9B8E-1A2C3D4E5F60}
AppName=Ultimate Anime VideoPlayer
AppVersion={#AppVer}
AppPublisher=Ta2b
DefaultDirName={localappdata}\Programs\AJP
DefaultGroupName=Ultimate Anime VideoPlayer
PrivilegesRequired=lowest
ArchitecturesAllowed=x64compatible
ArchitecturesInstallIn64BitMode=x64compatible
OutputDir={#Output}
OutputBaseFilename=ajp-windows-x64-setup
Compression=lzma2/max
SolidCompression=yes
WizardStyle=modern
UninstallDisplayIcon={app}\ajp.exe
DisableProgramGroupPage=yes

[Languages]
Name: "french"; MessagesFile: "compiler:Languages\French.isl"
Name: "english"; MessagesFile: "compiler:Default.isl"

[Tasks]
Name: "desktopicon"; Description: "Créer un raccourci sur le bureau"; Flags: unchecked

[Files]
Source: "{#Source}\*"; DestDir: "{app}"; Flags: recursesubdirs createallsubdirs ignoreversion

[Icons]
Name: "{autoprograms}\Ultimate Anime VideoPlayer"; Filename: "{app}\ajp.exe"
Name: "{autodesktop}\Ultimate Anime VideoPlayer"; Filename: "{app}\ajp.exe"; Tasks: desktopicon

[Run]
Filename: "{app}\ajp.exe"; Description: "Lancer Ultimate Anime VideoPlayer"; Flags: nowait postinstall skipifsilent

[UninstallRun]
; Les associations « Ouvrir avec » et le lecteur par défaut posés par le lecteur (HKCU) partent avec lui.
Filename: "{app}\ajp.exe"; Parameters: "--unregister"; Flags: runhidden waituntilterminated skipifdoesntexist; RunOnceId: "ajp-unregister"

[Code]
// Les données du lecteur ne sont pas dans {app} : réglages et journaux (%APPDATA%\ajp\ajp), moteurs construits,
// paquet TensorRT, modèles dérivés et caches (%LOCALAPPDATA%\ajp\ajp), souvent plusieurs Go. La désinstallation
// le dit et propose de les retirer (A36-07) ; une désinstallation silencieuse les garde.
procedure CurUninstallStepChanged(CurUninstallStep: TUninstallStep);
var
  Roaming, Local: String;
begin
  if (CurUninstallStep <> usPostUninstall) or UninstallSilent then
    exit;
  Roaming := ExpandConstant('{userappdata}\ajp\ajp');
  Local := ExpandConstant('{localappdata}\ajp\ajp');
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
