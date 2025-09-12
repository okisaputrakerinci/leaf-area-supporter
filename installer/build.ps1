param(
  [ValidateSet("portable","installer")]
  [string]$mode = "installer",
  [switch]$clean,

  # --- optional code signing (pilih salah satu) ---
  [string]$SignPfx,        # path ke .pfx
  [string]$SignPass,       # password .pfx
  [string]$SignSubject,    # alternatif: subject name di Windows Cert Store

  # paths
  [string]$Python = "python",
  [string]$ISCC = "${env:ProgramFiles(x86)}\Inno Setup 6\ISCC.exe"
)

$ErrorActionPreference = "Stop"

# ====== Lokasi proyek & kembali ke root ======
$ThisScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$ProjectRoot   = Split-Path -Parent $ThisScriptDir
Set-Location $ProjectRoot
Write-Host "Project root: $ProjectRoot"
Write-Host "Mode        : $mode"

# ====== Optional clean ======
if ($clean) {
  @(".\build", ".\dist", ".\__pycache__", ".\version_file.txt") | ForEach-Object {
    if (Test-Path $_) { Remove-Item $_ -Recurse -Force -ErrorAction SilentlyContinue }
  }
}

# ====== Python & deps ======
& $Python --version | Out-Null
if ($LASTEXITCODE -ne 0) { throw "Python tidak ditemukan di PATH." }

& $Python -m pip install --upgrade pip | Out-Null
if (Test-Path ".\requirements.txt") {
  Write-Host "==> Installing requirements..."
  & $Python -m pip install -r .\requirements.txt
}

# Pastikan PyInstaller ada
try { & $Python -c "import PyInstaller" 2>$null } catch { & $Python -m pip install pyinstaller | Out-Null }

# ====== Generate version resource ======
$verGen = Join-Path $ProjectRoot "tools\generate_version_file.py"
if (Test-Path $verGen) {
  Write-Host "==> Generating version_file.txt..."
  & $Python $verGen
  if ($LASTEXITCODE -ne 0 -or -not (Test-Path ".\version_file.txt")) {
    throw "Gagal membuat version_file.txt"
  }
} else {
  Write-Warning "tools/generate_version_file.py tidak ditemukan — metadata versi di EXE bisa kosong."
}

# ====== Fallback ANGLE ======
$env:QT_OPENGL = "angle"

# ====== Build portable (one-folder) ======
$specPath = Join-Path $ThisScriptDir "leaf_labeler.spec"
if (-not (Test-Path $specPath)) { throw "Spec file tidak ditemukan: $specPath" }

Write-Host "==> Building portable (one-folder) with PyInstaller..."
& $Python -m PyInstaller -y $specPath --distpath ".\dist" --workpath ".\build"
if ($LASTEXITCODE -ne 0) { throw "PyInstaller failed." }

$distDir = ".\dist\LeafLabeler"
$distExe = Join-Path $distDir "LeafLabeler.exe"
if (-not (Test-Path $distExe)) {
  $fallback = Get-ChildItem $distDir -Filter "*.exe" -ErrorAction SilentlyContinue | Select-Object -First 1
  if ($null -eq $fallback) { throw "Gagal menemukan output EXE di $distDir" }
  else { $distExe = $fallback.FullName }
}
Write-Host "Portable build OK: $distExe"
Write-Host ("SHA256: " + (Get-FileHash $distExe -Algorithm SHA256).Hash)

# ====== Safety check: pastikan TIDAK ada file sensitif di dist ======
$bad = Get-ChildItem $distDir -Recurse -Include *.pem,*.key,*.pfx,*.crt,dev_sign_license.py -ErrorAction SilentlyContinue
if ($bad) {
  $list = ($bad | ForEach-Object { $_.FullName }) -join "`n  "
  throw "Ditemukan file sensitif di dist, build dibatalkan:`n  $list"
}

# ====== Optional: signing EXE/DLL di dist ======
$NeedSign = ($SignPfx -and $SignPass) -or ($SignSubject)
if ($NeedSign) {
  function Find-SignTool {
    $candidates = @(
      "$env:ProgramFiles(x86)\Windows Kits\10\bin\x64\signtool.exe",
      "$env:ProgramFiles(x86)\Windows Kits\10\bin\*\x64\signtool.exe",
      "$env:ProgramFiles\Windows Kits\10\bin\*\x64\signtool.exe",
      "signtool.exe"
    )
    foreach ($c in $candidates) {
      foreach ($p in (Get-Item $c -ErrorAction SilentlyContinue)) { return $p.FullName }
      foreach ($p in (Get-ChildItem $c -ErrorAction SilentlyContinue)) { return $p.FullName }
    }
    return $null
  }

  $signtool = Find-SignTool
  if (-not $signtool) { throw "signtool.exe tidak ditemukan. Install Windows SDK atau tambahkan ke PATH." }

  Write-Host "==> Signing binaries in dist (exe/dll)..."
  $files = Get-ChildItem $distDir -Recurse -Include *.exe,*.dll
  foreach ($f in $files) {
    if ($SignPfx -and $SignPass) {
      & $signtool sign /fd SHA256 /f "$SignPfx" /p "$SignPass" /tr http://timestamp.digicert.com /td SHA256 "$($f.FullName)"
    } else {
      & $signtool sign /fd SHA256 /n "$SignSubject" /tr http://timestamp.digicert.com /td SHA256 "$($f.FullName)"
    }
    if ($LASTEXITCODE -ne 0) { throw "Sign failed: $($f.FullName)" }
  }
} else {
  Write-Host "Skipping dist signing (no SignPfx/SignPass or SignSubject)."
}

# ====== Jika diminta: build Windows Installer (Inno Setup) ======
if ($mode -eq "installer") {
  if (-not (Test-Path $ISCC)) { $ISCC = "${env:ProgramFiles}\Inno Setup 6\ISCC.exe" }
  if (-not (Test-Path $ISCC)) { throw "Inno Setup ISCC.exe tidak ditemukan. Install Inno Setup 6 dulu." }

  $issPath = Join-Path $ThisScriptDir "LeafLabeler.iss"
  if (-not (Test-Path $issPath)) { throw "Inno Setup script tidak ditemukan: $issPath" }

  if ($SignPfx -and $SignPass) {
    $env:SIGN_PFX  = $SignPfx
    $env:SIGN_PASS = $SignPass
    Write-Host "Inno Setup: signing ENABLED (PFX)."
  } else {
    Remove-Item Env:SIGN_PFX  -ErrorAction SilentlyContinue
    Remove-Item Env:SIGN_PASS -ErrorAction SilentlyContinue
    Write-Host "Inno Setup: signing DISABLED."
  }

  Write-Host "==> Building Windows Installer via Inno Setup..."
  & "$ISCC" "$issPath"
  if ($LASTEXITCODE -ne 0) { throw "Inno Setup build failed." }

  $outDir = Join-Path $ThisScriptDir "Output"
  $setup  = Get-ChildItem $outDir -Filter "LeafLabeler-Setup*.exe" | Sort-Object LastWriteTime -Descending | Select-Object -First 1
  if ($setup) {
    Write-Host "Installer OK: $($setup.FullName)"
    $sha = (Get-FileHash $setup.FullName -Algorithm SHA256).Hash
    Write-Host "SHA256: $sha"
  } else {
    Write-Host "Inno Setup selesai, cek folder: $outDir"
  }
}

Write-Host "Selesai."
