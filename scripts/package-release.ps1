param(
    [string]$Config = $(if ($env:MM_BUILD_CONFIG) { $env:MM_BUILD_CONFIG } else { "Release" }),
    [string]$Arch = $env:MM_TARGET_ARCH,
    [string]$BuildDir,
    [string]$DistDir,
    [string]$Version,
    [switch]$SkipBuild,
    [switch]$NoVcpkg,
    [Alias("G")]
    [string]$Generator = $env:CMAKE_GENERATOR,
    [string]$VcpkgRoot = $env:VCPKG_ROOT,
    [string]$Triplet = $env:VCPKG_DEFAULT_TRIPLET,
    [Parameter(ValueFromRemainingArguments = $true)]
    [string[]]$CMakeArgs
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

if ($CMakeArgs -and $CMakeArgs.Count -gt 0 -and $CMakeArgs[0] -eq "--") {
    if ($CMakeArgs.Count -eq 1) {
        $CMakeArgs = @()
    } else {
        $CMakeArgs = $CMakeArgs[1..($CMakeArgs.Count - 1)]
    }
}

$RepoRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path

function Invoke-Native {
    param(
        [string]$FilePath,
        [string[]]$Arguments
    )

    & $FilePath @Arguments
    if ($LASTEXITCODE -ne 0) {
        throw "$FilePath failed with exit code $LASTEXITCODE"
    }
}

function Invoke-Capture {
    param(
        [string]$FilePath,
        [string[]]$Arguments
    )

    $Output = & $FilePath @Arguments 2>$null
    if ($LASTEXITCODE -ne 0) {
        return ""
    }
    return ($Output -join "`n").Trim()
}

function Get-PlatformSlug {
    if ([System.Runtime.InteropServices.RuntimeInformation]::IsOSPlatform([System.Runtime.InteropServices.OSPlatform]::Windows)) {
        return "windows"
    }
    if ([System.Runtime.InteropServices.RuntimeInformation]::IsOSPlatform([System.Runtime.InteropServices.OSPlatform]::OSX)) {
        return "darwin"
    }
    if ([System.Runtime.InteropServices.RuntimeInformation]::IsOSPlatform([System.Runtime.InteropServices.OSPlatform]::Linux)) {
        return "linux"
    }
    return [System.Runtime.InteropServices.RuntimeInformation]::OSDescription.ToLowerInvariant().Replace(" ", "-")
}

function Get-NativeArch {
    return [System.Runtime.InteropServices.RuntimeInformation]::OSArchitecture.ToString().ToLowerInvariant()
}

function Normalize-Arch {
    param([string]$Value)

    if (-not $Value -or $Value.ToLowerInvariant() -eq "native") {
        $Value = Get-NativeArch
    }

    switch ($Value.ToLowerInvariant()) {
        { $_ -in @("x86_64", "amd64", "x64") } { return "x64" }
        { $_ -in @("aarch64", "arm64", "arm64-v8a") } { return "aarch64" }
        { $_ -in @("armv7", "armv7l") } { return "armv7" }
        default { return $Value.ToLowerInvariant() }
    }
}

function Normalize-Generator {
    param([string]$Value)

    if (-not $Value) {
        return ""
    }

    # CMAKE_GENERATOR should be a generator name, but some Windows shells carry
    # architecture text here. Passing that through becomes `cmake -G x64`.
    if ($Value.ToLowerInvariant() -in @("x86", "x64", "amd64", "arm64", "aarch64")) {
        return ""
    }

    return $Value
}

function Get-ProjectVersion {
    $CMakeText = Get-Content -Raw (Join-Path $RepoRoot "CMakeLists.txt")
    if ($CMakeText -match "project\s*\(\s*mantic-mind\s+VERSION\s+([0-9A-Za-z.\-+]+)") {
        return $Matches[1]
    }
    return "0.0.0"
}

function Get-ExecutableName {
    param([string]$BaseName)

    if ((Get-PlatformSlug) -eq "windows") {
        return "$BaseName.exe"
    }
    return $BaseName
}

function Get-BuildOutputDir {
    param(
        [string]$TargetDir,
        [string]$ExecutableName
    )

    $Candidates = @(
        (Join-Path $BuildDir "src/$TargetDir/$Config/$ExecutableName"),
        (Join-Path $BuildDir "src/$TargetDir/$ExecutableName")
    )

    foreach ($Candidate in $Candidates) {
        if (Test-Path $Candidate) {
            return (Resolve-Path (Split-Path -Parent $Candidate)).Path
        }
    }

    $Match = Get-ChildItem -Path (Join-Path $BuildDir "src/$TargetDir") `
        -Recurse -Filter $ExecutableName -File -ErrorAction SilentlyContinue |
        Sort-Object LastWriteTime -Descending |
        Select-Object -First 1

    if (-not $Match) {
        throw "Could not find $ExecutableName under $BuildDir/src/$TargetDir. Build the $Config configuration first."
    }

    return $Match.Directory.FullName
}

function Assert-UnderPath {
    param(
        [string]$Parent,
        [string]$Child
    )

    $ParentFull = [System.IO.Path]::GetFullPath($Parent).TrimEnd("\", "/")
    $ChildFull = [System.IO.Path]::GetFullPath($Child).TrimEnd("\", "/")
    $Comparison = [System.StringComparison]::OrdinalIgnoreCase
    if ($ChildFull.Equals($ParentFull, $Comparison)) {
        return
    }
    if (-not $ChildFull.StartsWith($ParentFull + [System.IO.Path]::DirectorySeparatorChar, $Comparison)) {
        throw "Refusing to modify path outside $ParentFull`: $ChildFull"
    }
}

function Reset-Directory {
    param(
        [string]$Path,
        [string]$AllowedRoot
    )

    Assert-UnderPath $AllowedRoot $Path
    if (Test-Path $Path) {
        Remove-Item -LiteralPath $Path -Recurse -Force
    }
    New-Item -ItemType Directory -Path $Path -Force | Out-Null
}

function Copy-DirectoryContents {
    param(
        [string]$Source,
        [string]$Destination
    )

    if (-not (Test-Path $Source)) {
        return
    }
    New-Item -ItemType Directory -Path $Destination -Force | Out-Null
    Copy-Item -Path (Join-Path $Source "*") -Destination $Destination -Recurse -Force
}

function Copy-MatchingFiles {
    param(
        [string[]]$SourceDirs,
        [string]$Filter,
        [string]$Destination
    )

    New-Item -ItemType Directory -Path $Destination -Force | Out-Null
    foreach ($SourceDir in ($SourceDirs | Where-Object { $_ } | Select-Object -Unique)) {
        if (-not (Test-Path $SourceDir)) {
            continue
        }
        Get-ChildItem -Path $SourceDir -Filter $Filter -File -ErrorAction SilentlyContinue |
            ForEach-Object {
                Copy-Item -LiteralPath $_.FullName -Destination (Join-Path $Destination $_.Name) -Force
            }
    }
}

$Platform = Get-PlatformSlug
$TargetArch = Normalize-Arch $Arch
$Generator = Normalize-Generator $Generator
$ConfigSlug = $Config.ToLowerInvariant()

if (-not $BuildDir) {
    $BuildDir = Join-Path $RepoRoot "build/$Platform-$TargetArch-$ConfigSlug"
}
$BuildDir = [System.IO.Path]::GetFullPath($BuildDir)

if (-not $DistDir) {
    $DistDir = Join-Path $RepoRoot "dist"
}
$DistRoot = [System.IO.Path]::GetFullPath($DistDir)
New-Item -ItemType Directory -Path $DistRoot -Force | Out-Null

if (-not $Version) {
    $Version = Get-ProjectVersion
}

$AssetPlatform = "$Platform-$TargetArch"
$PackageName = "mantic-mind-$Version-$AssetPlatform"
$StagingRoot = Join-Path $DistRoot "staging"
$PackageRoot = Join-Path $StagingRoot $PackageName
$SymbolsRoot = Join-Path $StagingRoot "$PackageName-symbols"
$RuntimeZip = Join-Path $DistRoot "$PackageName.zip"
$SymbolsZip = Join-Path $DistRoot "$PackageName-symbols.zip"

if (-not $SkipBuild) {
    $BuildScript = Join-Path $RepoRoot "scripts/build.ps1"
    $BuildArgs = @{
        Config = $Config
        Arch = $TargetArch
        BuildDir = $BuildDir
        InstallPrefix = $PackageRoot
        Generator = $Generator
    }
    if ($VcpkgRoot) { $BuildArgs.VcpkgRoot = $VcpkgRoot }
    if ($Triplet) { $BuildArgs.Triplet = $Triplet }
    if ($NoVcpkg) { $BuildArgs.NoVcpkg = $true }
    $BuildExtraArgs = @()
    if ($CMakeArgs) { $BuildExtraArgs += @("--") + $CMakeArgs }

    Reset-Directory $PackageRoot $DistRoot
    Write-Host "==> Build and install release package root"
    & $BuildScript @BuildArgs @BuildExtraArgs
    if ($LASTEXITCODE -ne 0) {
        throw "$BuildScript failed with exit code $LASTEXITCODE"
    }
} else {
    Reset-Directory $PackageRoot $DistRoot
    Write-Host "==> Install release package root"
    Invoke-Native "cmake" @("--install", $BuildDir, "--config", $Config, "--prefix", $PackageRoot)
}

$BinDir = Join-Path $PackageRoot "bin"
New-Item -ItemType Directory -Path $BinDir -Force | Out-Null

$NodeExe = Get-ExecutableName "mantic-mind"
$ControlExe = Get-ExecutableName "mantic-mind-control"
$AioExe = Get-ExecutableName "mantic-mind-aio"
$NodeOutputDir = Get-BuildOutputDir "node" $NodeExe
$ControlOutputDir = Get-BuildOutputDir "control" $ControlExe
$AioOutputDir = Get-BuildOutputDir "aio" $AioExe
$OutputDirs = @($NodeOutputDir, $ControlOutputDir, $AioOutputDir)

$VcpkgArch = switch ($TargetArch) {
    "aarch64" { "arm64" }
    default { $TargetArch }
}
$VcpkgPlatform = switch ($Platform) {
    "darwin" { "osx" }
    default { $Platform }
}
$RuntimeTriplet = if ($Triplet) { $Triplet } else { "$VcpkgArch-$VcpkgPlatform" }
$VcpkgTripletRoot = Join-Path $BuildDir "vcpkg_installed/$RuntimeTriplet"
$RuntimeLibraryDirs = @($OutputDirs)
foreach ($RuntimeSubdir in @("bin", "lib")) {
    $RuntimeCandidate = Join-Path $VcpkgTripletRoot $RuntimeSubdir
    if (Test-Path $RuntimeCandidate) {
        $RuntimeLibraryDirs += $RuntimeCandidate
    }
}

Write-Host "==> Copy executables"
# `cmake --install` supplies the normal layout. Copy the selected build outputs
# once more so packaging an older configured tree still uses the requested
# configuration's executables.
$NodeExePath = Join-Path $NodeOutputDir $NodeExe
$ControlExePath = Join-Path $ControlOutputDir $ControlExe
$AioExePath = Join-Path $AioOutputDir $AioExe
if (-not (Test-Path -LiteralPath $NodeExePath)) {
    throw "Node executable not found at $NodeExePath. Build the project before packaging."
}
if (-not (Test-Path -LiteralPath $ControlExePath)) {
    throw "Control executable not found at $ControlExePath. Build the project before packaging."
}
if (-not (Test-Path -LiteralPath $AioExePath)) {
    throw "AIO executable not found at $AioExePath. Build the project before packaging."
}
Copy-Item -LiteralPath $NodeExePath -Destination (Join-Path $BinDir $NodeExe) -Force
Copy-Item -LiteralPath $ControlExePath -Destination (Join-Path $BinDir $ControlExe) -Force
Copy-Item -LiteralPath $AioExePath -Destination (Join-Path $BinDir $AioExe) -Force

Write-Host "==> Copy runtime shared libraries"
switch ($Platform) {
    "windows" { Copy-MatchingFiles $RuntimeLibraryDirs "*.dll" $BinDir }
    "linux" { Copy-MatchingFiles $RuntimeLibraryDirs "*.so*" $BinDir }
    "darwin" { Copy-MatchingFiles $RuntimeLibraryDirs "*.dylib" $BinDir }
}

Write-Host "==> Copy configs, tools, and docs"
Copy-Item -LiteralPath (Join-Path $RepoRoot "tools/mantic-mind.toml") `
    -Destination (Join-Path $PackageRoot "mantic-mind.toml") -Force
Copy-Item -LiteralPath (Join-Path $RepoRoot "tools/mantic-mind-control.toml") `
    -Destination (Join-Path $PackageRoot "mantic-mind-control.toml") -Force
Copy-Item -LiteralPath (Join-Path $RepoRoot "tools/mantic-mind-aio.toml") `
    -Destination (Join-Path $PackageRoot "mantic-mind-aio.toml") -Force
Copy-DirectoryContents (Join-Path $RepoRoot "tools") (Join-Path $PackageRoot "tools")
if (Test-Path (Join-Path $PackageRoot "tools/__pycache__")) {
    Remove-Item -LiteralPath (Join-Path $PackageRoot "tools/__pycache__") -Recurse -Force
}
Copy-Item -LiteralPath (Join-Path $RepoRoot "README.md") -Destination $PackageRoot -Force
Copy-Item -LiteralPath (Join-Path $RepoRoot "LICENSE") -Destination $PackageRoot -Force
$PackageDocsDir = Join-Path $PackageRoot "docs"
New-Item -ItemType Directory -Path $PackageDocsDir -Force | Out-Null
Copy-Item -LiteralPath (Join-Path $RepoRoot "docs/aio.md") `
    -Destination (Join-Path $PackageDocsDir "aio.md") -Force

$Branch = Invoke-Capture "git" @("-C", $RepoRoot, "-c", "core.excludesfile=", "rev-parse", "--abbrev-ref", "HEAD")
$Commit = Invoke-Capture "git" @("-C", $RepoRoot, "-c", "core.excludesfile=", "rev-parse", "HEAD")
$ShortCommit = Invoke-Capture "git" @("-C", $RepoRoot, "-c", "core.excludesfile=", "rev-parse", "--short", "HEAD")
$GeneratedAt = [System.DateTimeOffset]::Now.ToString("o")

$Manifest = @(
    "Mantic-Mind release package",
    "Version: $Version",
    "Asset platform: $AssetPlatform",
    "Build config: $Config",
    "Git branch: $Branch",
    "Git commit: $Commit",
    "Generated at: $GeneratedAt",
    "",
    "Included:",
    "- bin/mantic-mind",
    "- bin/mantic-mind-control",
    "- bin/mantic-mind-aio",
    "- runtime shared libraries copied from the selected build/vcpkg output",
    "- default root config files copied from tools/",
    "- tools/qwen_tts_service.py",
    "- README.md, docs/aio.md, and LICENSE",
    "",
    "Not included:",
    "- managed inference runtime installs and Python package environments",
    "- model weights and model caches",
    "- runtime data, logs, and local agent databases"
)
$Manifest | Set-Content -Path (Join-Path $PackageRoot "release-manifest.txt") -Encoding UTF8

Write-Host "==> Stage symbols"
Reset-Directory $SymbolsRoot $DistRoot
$SymbolsBin = Join-Path $SymbolsRoot "bin"
Copy-MatchingFiles $OutputDirs "*.pdb" $SymbolsBin

$VcpkgBin = Join-Path $BuildDir "vcpkg_installed/$TargetArch-windows/bin"
if (Test-Path $VcpkgBin) {
    $RuntimeDllBaseNames = @{}
    Get-ChildItem -Path $BinDir -Filter "*.dll" -File | ForEach-Object {
        $RuntimeDllBaseNames[$_.BaseName.ToLowerInvariant()] = $true
    }
    Get-ChildItem -Path $VcpkgBin -Filter "*.pdb" -File -ErrorAction SilentlyContinue |
        Where-Object { $RuntimeDllBaseNames.ContainsKey($_.BaseName.ToLowerInvariant()) } |
        ForEach-Object {
            Copy-Item -LiteralPath $_.FullName -Destination (Join-Path $SymbolsBin $_.Name) -Force
        }
}

$SymbolFiles = Get-ChildItem -Path $SymbolsRoot -Recurse -File -ErrorAction SilentlyContinue
if ($SymbolFiles.Count -gt 0) {
    $Manifest | Set-Content -Path (Join-Path $SymbolsRoot "release-manifest.txt") -Encoding UTF8
} else {
    Remove-Item -LiteralPath $SymbolsRoot -Recurse -Force
}

Write-Host "==> Compress release assets"
foreach ($Path in @($RuntimeZip, $SymbolsZip)) {
    Assert-UnderPath $DistRoot $Path
    if (Test-Path $Path) {
        Remove-Item -LiteralPath $Path -Force
    }
}

Compress-Archive -Path $PackageRoot -DestinationPath $RuntimeZip -CompressionLevel Optimal
if (Test-Path $SymbolsRoot) {
    Compress-Archive -Path $SymbolsRoot -DestinationPath $SymbolsZip -CompressionLevel Optimal
}

$ChecksumPath = Join-Path $DistRoot "checksums.txt"
$Assets = @($RuntimeZip)
if (Test-Path $SymbolsZip) {
    $Assets += $SymbolsZip
}

$ChecksumLines = foreach ($Asset in $Assets) {
    $Hash = Get-FileHash -Path $Asset -Algorithm SHA256
    "{0}  {1}" -f $Hash.Hash.ToLowerInvariant(), (Split-Path -Leaf $Asset)
}
$ChecksumLines | Set-Content -Path $ChecksumPath -Encoding ASCII

Write-Host ""
Write-Host "Release assets:"
foreach ($Asset in $Assets) {
    $Info = Get-Item $Asset
    Write-Host ("  {0} ({1:N1} MB)" -f $Info.FullName, ($Info.Length / 1MB))
}
Write-Host "  $ChecksumPath"
if ($ShortCommit) {
    Write-Host "Source: $Branch@$ShortCommit"
}
