; Installeur Windows (Inno Setup 6) : tools/release/package-windows.ps1 le prépare, release-windows.yml le lance.
;   ISCC /DSource=<dossier ajp-windows-x64> /DAppVer=0.7.0 /DOutput=<dossier> ajp.iss
; Installation pour l'utilisateur seul (aucun droit d'administrateur) : le lecteur se met à jour lui-même
; dans son dossier, ce qu'un dossier de Program Files interdirait. « Ouvrir avec » et le lecteur par
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
