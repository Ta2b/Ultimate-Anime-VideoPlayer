# Signature Authenticode par Azure Trusted Signing (P7.1). Sans AJP_TS_ENDPOINT, ne fait rien :
# les exécutables partent non signés, comme avant (SmartScreen : « éditeur inconnu »).
# Variables : AJP_TS_ENDPOINT (https://<région>.codesigning.azure.net), AJP_TS_ACCOUNT (compte de
# signature), AJP_TS_PROFILE (profil de certificat), et pour l'accès AZURE_TENANT_ID, AZURE_CLIENT_ID,
# AZURE_CLIENT_SECRET (application Entra ID au rôle « Trusted Signing Certificate Profile Signer »).
# Usage : pwsh tools/release/windows/sign.ps1 <fichier ou dossier>...
# Un dossier : chaque .exe et .dll qu'il contient, sous-dossiers compris.
param([Parameter(Mandatory = $true, ValueFromRemainingArguments = $true)][string[]]$Paths)
$ErrorActionPreference = "Stop"

if (-not $env:AJP_TS_ENDPOINT) {
  Write-Host "signature : AJP_TS_ENDPOINT absent, exécutables non signés"
  exit 0
}

$files = foreach ($p in $Paths) {
  if (Test-Path $p -PathType Container) {
    Get-ChildItem $p -Recurse -File -Include *.exe, *.dll | ForEach-Object { $_.FullName }
  } else { (Resolve-Path $p).Path }
}
if (-not $files) { throw "signature : aucun fichier dans $($Paths -join ', ')" }

# Le client de Trusted Signing (la bibliothèque que signtool charge) vient de NuGet, à version épinglée.
$work = Join-Path ($env:RUNNER_TEMP ?? [IO.Path]::GetTempPath()) "ajp-trusted-signing"
$dlib = Join-Path $work "Microsoft.Trusted.Signing.Client/bin/x64/Azure.CodeSigning.Dlib.dll"
if (-not (Test-Path $dlib)) {
  New-Item -ItemType Directory -Force $work | Out-Null
  nuget install Microsoft.Trusted.Signing.Client -Version 1.0.86 -ExcludeVersion -OutputDirectory $work | Out-Null
  if ($LASTEXITCODE -ne 0 -or -not (Test-Path $dlib)) { throw "signature : client Trusted Signing introuvable" }
}
$metadata = Join-Path $work "metadata.json"
@{ Endpoint = $env:AJP_TS_ENDPOINT; CodeSigningAccountName = $env:AJP_TS_ACCOUNT; CertificateProfileName = $env:AJP_TS_PROFILE } |
  ConvertTo-Json | Set-Content -Path $metadata -Encoding utf8NoBOM

$signtool = Get-ChildItem "${env:ProgramFiles(x86)}\Windows Kits\10\bin\*\x64\signtool.exe" |
  Sort-Object FullName -Descending | Select-Object -First 1
if (-not $signtool) { throw "signature : signtool introuvable (Windows SDK)" }

# Par lots : une ligne de commande Windows est bornée à 32 767 caractères.
for ($i = 0; $i -lt $files.Count; $i += 50) {
  $batch = $files[$i..([Math]::Min($i + 49, $files.Count - 1))]
  & $signtool.FullName sign /fd SHA256 /tr "http://timestamp.acs.microsoft.com" /td SHA256 `
    /dlib $dlib /dmdf $metadata @batch
  if ($LASTEXITCODE -ne 0) { throw "signature : signtool a échoué" }
}
& $signtool.FullName verify /pa /q @files
if ($LASTEXITCODE -ne 0) { throw "signature : vérification échouée" }
Write-Host "signature : $($files.Count) fichiers signés"
