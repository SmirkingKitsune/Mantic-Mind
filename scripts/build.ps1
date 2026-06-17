param(
    [string]$Config = $(if ($env:MM_BUILD_CONFIG) { $env:MM_BUILD_CONFIG } else { "Release" }),
    [switch]$DebugBuild,
    [string]$BuildDir,
    [string]$Arch = $env:MM_TARGET_ARCH,
    [Alias("G")]
    [string]$Generator = $env:CMAKE_GENERATOR,
    [string]$VcpkgRoot = $env:VCPKG_ROOT,
    [string]$Triplet = $env:VCPKG_DEFAULT_TRIPLET,
    [switch]$NoVcpkg,
    [string]$InstallPrefix,
    [Parameter(ValueFromRemainingArguments = $true)]
    [string[]]$CMakeArgs
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

if ($DebugBuild) {
    $Config = "Debug"
}

if ($CMakeArgs -and $CMakeArgs.Count -gt 0 -and $CMakeArgs[0] -eq "--") {
    if ($CMakeArgs.Count -eq 1) {
        $CMakeArgs = @()
    } else {
        $CMakeArgs = $CMakeArgs[1..($CMakeArgs.Count - 1)]
    }
}

$SourceDir = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path

function Test-VcpkgRoot {
    param([string]$Path)

    if (-not $Path) {
        return $false
    }

    return Test-Path (Join-Path $Path "scripts/buildsystems/vcpkg.cmake")
}

function Find-VcpkgRoot {
    if (Test-VcpkgRoot $script:VcpkgRoot) {
        return (Resolve-Path $script:VcpkgRoot).Path
    }

    $Command = Get-Command vcpkg -ErrorAction SilentlyContinue
    if ($Command) {
        $Dir = Split-Path -Parent $Command.Source
        $Parent = Split-Path -Parent $Dir

        if (Test-VcpkgRoot $Dir) {
            return $Dir
        }

        if (Test-VcpkgRoot $Parent) {
            return $Parent
        }
    }

    foreach ($Candidate in @("C:\vcpkg", "C:\src\vcpkg", "C:\tools\vcpkg")) {
        if (Test-VcpkgRoot $Candidate) {
            return (Resolve-Path $Candidate).Path
        }
    }

    return $null
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

function Get-VcpkgArchName {
    param([string]$Value)

    switch ($Value) {
        "aarch64" { return "arm64" }
        "x86_64" { return "x64" }
        "x64" { return "x64" }
        "armv7" { return "arm" }
        "armv7l" { return "arm" }
        default { return $Value }
    }
}

function Get-VcpkgOsName {
    param([string]$Value)

    if ($Value -eq "darwin") {
        return "osx"
    }

    return $Value
}

function Test-CMakeArg {
    param([string]$Key)

    if (-not $CMakeArgs) {
        return $false
    }

    foreach ($Arg in $CMakeArgs) {
        if ($Arg -like "-D$Key=*" -or $Arg -like "-D$Key`:*") {
            return $true
        }
    }

    return $false
}

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

function Get-CMakeCacheValue {
    param(
        [string]$CachePath,
        [string]$Name
    )

    if (-not (Test-Path $CachePath)) {
        return $null
    }

    foreach ($Line in Get-Content $CachePath) {
        if ($Line -match "^$([regex]::Escape($Name)):[^=]*=(.*)$") {
            return $Matches[1]
        }
    }

    return $null
}

function Get-CMakeCacheType {
    param(
        [string]$CachePath,
        [string]$Name
    )

    if (-not (Test-Path $CachePath)) {
        return $null
    }

    foreach ($Line in Get-Content $CachePath) {
        if ($Line -match "^$([regex]::Escape($Name)):([^=]*)=") {
            return $Matches[1]
        }
    }

    return $null
}

function Normalize-PathText {
    param([string]$Path)

    if (-not $Path) {
        return ""
    }

    return ([System.IO.Path]::GetFullPath($Path)).TrimEnd("\", "/").Replace("\", "/")
}

$Platform = Get-PlatformSlug
$TargetArch = Normalize-Arch $Arch
$HostArch = Normalize-Arch "native"

if (-not $BuildDir) {
    $ConfigSlug = $Config.ToLowerInvariant()
    $BuildDir = Join-Path $SourceDir "build/$Platform-$TargetArch-$ConfigSlug"
}

$ConfigureArgs = @(
    "-S", $SourceDir,
    "-B", $BuildDir,
    "-DCMAKE_BUILD_TYPE=$Config"
)

if ($Generator) {
    $ConfigureArgs += @("-G", $Generator)
}

if ($Platform -eq "darwin" -and $TargetArch -eq "aarch64") {
    $ConfigureArgs += "-DCMAKE_OSX_ARCHITECTURES=arm64"
}

if ($Platform -eq "linux" -and $TargetArch -eq "aarch64" -and $HostArch -ne "aarch64") {
    $CrossGcc = Get-Command aarch64-linux-gnu-gcc -ErrorAction SilentlyContinue
    $CrossGxx = Get-Command aarch64-linux-gnu-g++ -ErrorAction SilentlyContinue
    if ($CrossGcc -and $CrossGxx) {
        if (-not (Test-CMakeArg "CMAKE_SYSTEM_NAME")) {
            $ConfigureArgs += "-DCMAKE_SYSTEM_NAME=Linux"
        }
        if (-not (Test-CMakeArg "CMAKE_SYSTEM_PROCESSOR")) {
            $ConfigureArgs += "-DCMAKE_SYSTEM_PROCESSOR=aarch64"
        }
        if (-not (Test-CMakeArg "CMAKE_C_COMPILER")) {
            $ConfigureArgs += "-DCMAKE_C_COMPILER=aarch64-linux-gnu-gcc"
        }
        if (-not (Test-CMakeArg "CMAKE_CXX_COMPILER")) {
            $ConfigureArgs += "-DCMAKE_CXX_COMPILER=aarch64-linux-gnu-g++"
        }
    } elseif ((Test-CMakeArg "CMAKE_CXX_COMPILER") -or (Test-CMakeArg "CMAKE_TOOLCHAIN_FILE")) {
        if (-not (Test-CMakeArg "CMAKE_SYSTEM_NAME")) {
            $ConfigureArgs += "-DCMAKE_SYSTEM_NAME=Linux"
        }
        if (-not (Test-CMakeArg "CMAKE_SYSTEM_PROCESSOR")) {
            $ConfigureArgs += "-DCMAKE_SYSTEM_PROCESSOR=aarch64"
        }
    } else {
        throw "AArch64 Linux cross compiler not detected. Install g++-aarch64-linux-gnu or pass a cross toolchain/compiler after --."
    }
}

if ($Platform -eq "windows" -and $TargetArch -eq "aarch64" -and $Generator -like "*Visual Studio*") {
    $ConfigureArgs += @("-A", "ARM64")
}

$ExpectedToolchainFile = $null
if (-not $NoVcpkg) {
    $ResolvedVcpkgRoot = Find-VcpkgRoot
    if ($ResolvedVcpkgRoot) {
        $ExpectedToolchainFile = Join-Path $ResolvedVcpkgRoot "scripts/buildsystems/vcpkg.cmake"
        $ConfigureArgs += "-DCMAKE_TOOLCHAIN_FILE=$ExpectedToolchainFile"
        if (-not $Triplet -and $Arch -and $Arch.ToLowerInvariant() -ne "native") {
            $Triplet = "$(Get-VcpkgArchName $TargetArch)-$(Get-VcpkgOsName $Platform)"
        }
        if ($Triplet) {
            $ConfigureArgs += "-DVCPKG_TARGET_TRIPLET=$Triplet"
        }
    } else {
        Write-Warning "vcpkg not detected; configuring without a toolchain. Dependencies must be discoverable by CMake."
    }
}

if ($CMakeArgs) {
    $ConfigureArgs += $CMakeArgs
}

$Jobs = if ($env:CMAKE_BUILD_PARALLEL_LEVEL) {
    [int]$env:CMAKE_BUILD_PARALLEL_LEVEL
} else {
    [Environment]::ProcessorCount
}

$CachePath = Join-Path $BuildDir "CMakeCache.txt"
$CachedToolchainFile = Get-CMakeCacheValue $CachePath "CMAKE_TOOLCHAIN_FILE"
$CachedToolchainType = Get-CMakeCacheType $CachePath "CMAKE_TOOLCHAIN_FILE"
if (
    (Normalize-PathText $CachedToolchainFile) -ne (Normalize-PathText $ExpectedToolchainFile) -or
    ($ExpectedToolchainFile -and $CachedToolchainType -eq "UNINITIALIZED")
) {
    Write-Host "==> Refreshing CMake cache: toolchain changed"
    $ConfigureArgs = @("--fresh") + $ConfigureArgs
}

Write-Host "==> Configure: $BuildDir"
Invoke-Native "cmake" $ConfigureArgs

Write-Host "==> Build: $BuildDir ($Config, $Jobs jobs)"
Invoke-Native "cmake" @("--build", $BuildDir, "--config", $Config, "--parallel", $Jobs)

if ($InstallPrefix) {
    Write-Host "==> Install: $InstallPrefix"
    Invoke-Native "cmake" @("--install", $BuildDir, "--config", $Config, "--prefix", $InstallPrefix)
}
