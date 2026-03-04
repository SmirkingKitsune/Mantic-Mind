[CmdletBinding()]
param(
    [string]$RepoUrl = $(if ($env:LLAMA_CPP_REPO_URL) { $env:LLAMA_CPP_REPO_URL } else { "https://github.com/ggml-org/llama.cpp.git" }),
    [string]$RepoDir = $(if ($env:LLAMA_CPP_DIR) { $env:LLAMA_CPP_DIR } else { (Join-Path (Resolve-Path "$PSScriptRoot\..") "external\llama.cpp") }),
    [switch]$Build,
    [string]$CudaOverride = $(if ($env:LLAMA_CPP_CUDA) { $env:LLAMA_CPP_CUDA } else { "" }),
    [string]$CudaArchitectures = $(if ($env:LLAMA_CPP_CUDA_ARCHITECTURES) { $env:LLAMA_CPP_CUDA_ARCHITECTURES } else { "all-major" }),
    [string]$CudaFlags = $(if ($env:LLAMA_CPP_CUDA_FLAGS) { $env:LLAMA_CPP_CUDA_FLAGS } else { "--allow-unsupported-compiler" }),
    [string]$CudaDiagSuppress = $(if ($env:LLAMA_CPP_CUDA_DIAG_SUPPRESS) { $env:LLAMA_CPP_CUDA_DIAG_SUPPRESS } else { "177,221" }),
    [string]$PrebuiltCudaVersion = $(if ($env:LLAMA_CPP_PREBUILT_CUDA_VERSION) { $env:LLAMA_CPP_PREBUILT_CUDA_VERSION } else { "" }),
    [string]$PreferPrebuilt = $(if ($env:LLAMA_CPP_PREFER_PREBUILT) { $env:LLAMA_CPP_PREFER_PREBUILT } else { "1" }),
    [string]$PrebuiltAssetRegex = $(if ($env:LLAMA_CPP_PREBUILT_ASSET_REGEX) { $env:LLAMA_CPP_PREBUILT_ASSET_REGEX } else { "" }),
    [string]$Config = $(if ($env:LLAMA_CPP_BUILD_CONFIG) { $env:LLAMA_CPP_BUILD_CONFIG } else { "Release" }),
    [string]$Generator = $(if ($env:LLAMA_CPP_GENERATOR) { $env:LLAMA_CPP_GENERATOR } else { "" }),
    [int]$Jobs = $(if ($env:LLAMA_CPP_JOBS) { [int]$env:LLAMA_CPP_JOBS } else { 0 })
)

$ErrorActionPreference = "Stop"

function Invoke-NativeChecked {
    param(
        [Parameter(Mandatory = $true)][string]$Exe,
        [Parameter()][string[]]$Args = @()
    )

    & $Exe @Args
    if ($LASTEXITCODE -ne 0) {
        throw "[llama.cpp] command failed ($LASTEXITCODE): $Exe $($Args -join ' ')"
    }
}

function Test-Truthy {
    param([string]$Value)
    if (-not $Value) { return $false }
    $v = $Value.Trim().ToLowerInvariant()
    return $v -eq "1" -or $v -eq "true" -or $v -eq "on" -or $v -eq "yes"
}

function Parse-MajorMinorVersion {
    param([string]$Value)
    if (-not $Value) { return $null }
    $m = [regex]::Match($Value, '(?i)(\d+)\.(\d+)')
    if ($m.Success) {
        return [version]::new([int]$m.Groups[1].Value, [int]$m.Groups[2].Value)
    }
    return $null
}

function Get-CudaVersionFromNvcc {
    param([string]$NvccPath)
    if (-not $NvccPath) { return $null }
    try {
        $txt = (& $NvccPath "--version" 2>$null | Out-String)
        return Parse-MajorMinorVersion -Value $txt
    } catch {
        return $null
    }
}

function Get-CudaVersionFromAssetName {
    param([string]$Name)
    if (-not $Name) { return $null }
    $m = [regex]::Match($Name, '(?i)cuda-(\d+)\.(\d+)')
    if ($m.Success) {
        return [version]::new([int]$m.Groups[1].Value, [int]$m.Groups[2].Value)
    }
    return $null
}

Write-Host "[llama.cpp] repo url : $RepoUrl"
Write-Host "[llama.cpp] repo dir : $RepoDir"

if (Test-Path (Join-Path $RepoDir ".git")) {
    Write-Host "[llama.cpp] existing checkout found; pulling latest..."
    Invoke-NativeChecked -Exe "git" -Args @("-C", $RepoDir, "fetch", "--tags", "origin")
    Invoke-NativeChecked -Exe "git" -Args @("-C", $RepoDir, "pull", "--ff-only")
} else {
    Write-Host "[llama.cpp] cloning fresh checkout..."
    New-Item -ItemType Directory -Path (Split-Path -Parent $RepoDir) -Force | Out-Null
    Invoke-NativeChecked -Exe "git" -Args @("clone", $RepoUrl, $RepoDir)
}

$commit = (& git -C $RepoDir rev-parse --short HEAD).Trim()
if ($LASTEXITCODE -ne 0) {
    throw "[llama.cpp] failed to read current git commit"
}
Write-Host "[llama.cpp] at commit $commit"

if (-not $Build) {
    Write-Host "[llama.cpp] update complete (build skipped)"
    exit 0
}

$enableCuda = $false
$detectedCudaToolkitVersion = $null
if ($CudaOverride -eq "1" -or $CudaOverride -eq "true" -or $CudaOverride -eq "on") {
    $enableCuda = $true
} elseif ($CudaOverride -eq "0" -or $CudaOverride -eq "false" -or $CudaOverride -eq "off") {
    $enableCuda = $false
} else {
    $nvcc = Get-Command nvcc -ErrorAction SilentlyContinue
    if ($nvcc) {
        $enableCuda = $true
        $detectedCudaToolkitVersion = Get-CudaVersionFromNvcc -NvccPath $nvcc.Source
        Write-Host "[llama.cpp] CUDA toolkit detected (nvcc: $($nvcc.Source)) - enabling CUDA automatically"
        if ($detectedCudaToolkitVersion) {
            Write-Host "[llama.cpp] CUDA toolkit version detected: $($detectedCudaToolkitVersion.Major).$($detectedCudaToolkitVersion.Minor)"
        }
    }
}

$preferredCudaToolkitVersion = Parse-MajorMinorVersion -Value $PrebuiltCudaVersion
if ($preferredCudaToolkitVersion) {
    Write-Host "[llama.cpp] prebuilt CUDA version preference override: $($preferredCudaToolkitVersion.Major).$($preferredCudaToolkitVersion.Minor)"
} elseif ($detectedCudaToolkitVersion) {
    $preferredCudaToolkitVersion = $detectedCudaToolkitVersion
}

$preferPrebuiltEnabled = Test-Truthy -Value $PreferPrebuilt
if ($preferPrebuiltEnabled) {
    Write-Host "[llama.cpp] attempting prebuilt release install before source build..."
    try {
        $releaseApi = "https://api.github.com/repos/ggml-org/llama.cpp/releases/latest"
        $headers = @{
            "User-Agent" = "mantic-mind-llama-updater"
            "Accept" = "application/vnd.github+json"
        }
        $release = Invoke-RestMethod -Uri $releaseApi -Headers $headers -Method Get
        if (-not $release -or -not $release.assets) {
            throw "release assets not found"
        }

        $assets = @($release.assets)
        if ($PrebuiltAssetRegex) {
            $assets = @($assets | Where-Object { $_.name -match $PrebuiltAssetRegex })
        } else {
            $assets = @($assets | Where-Object {
                    $_.name -match "(?i)\.zip$" -and
                    $_.name -match "(?i)(win|windows)" -and
                    $_.name -match "(?i)(x64|amd64)" -and
                    $_.name -notmatch "(?i)arm64"
                })
            if ($assets.Count -gt 0) {
                if ($enableCuda) {
                    $assets = @($assets | Where-Object { $_.name -match "(?i)(cuda|cudart|cu1[23])" -and $_.name -notmatch "(?i)vulkan" })
                    $fullCudaAssets = @($assets | Where-Object { $_.name -match "(?i)^llama-b[0-9]+-bin-win-cuda-.*\.zip$" })
                    if ($fullCudaAssets.Count -gt 0) {
                        $assets = $fullCudaAssets
                    }
                    $rankedAssets = @($assets | ForEach-Object {
                            $assetCudaVer = Get-CudaVersionFromAssetName -Name $_.name
                            $familyRank = if ($_.name -match "(?i)^llama-b[0-9]+-bin-win-cuda-") { 0 } elseif ($_.name -match "(?i)^llama-.*-bin-win-cuda-") { 1 } elseif ($_.name -match "(?i)^cudart-llama-bin-win-cuda-") { 2 } else { 3 }
                            $distanceScore = 9999
                            $versionScore = -1
                            if ($assetCudaVer) {
                                $versionScore = ($assetCudaVer.Major * 100) + $assetCudaVer.Minor
                                if ($preferredCudaToolkitVersion) {
                                    if ($assetCudaVer.Major -eq $preferredCudaToolkitVersion.Major) {
                                        $distanceScore = [math]::Abs($assetCudaVer.Minor - $preferredCudaToolkitVersion.Minor)
                                    } else {
                                        $distanceScore = 100 + ([math]::Abs($assetCudaVer.Major - $preferredCudaToolkitVersion.Major) * 10) + $assetCudaVer.Minor
                                    }
                                } else {
                                    $distanceScore = 500 - $versionScore
                                }
                            }
                            [pscustomobject]@{
                                asset = $_
                                name = $_.name
                                family_rank = $familyRank
                                distance_score = $distanceScore
                                version_score = $versionScore
                            }
                        })
                    $assets = @($rankedAssets |
                            Sort-Object -Property `
                                family_rank, `
                                distance_score, `
                                @{ Expression = { $_.version_score }; Descending = $true }, `
                                name |
                            ForEach-Object { $_.asset })
                } else {
                    $cpuAssets = @($assets | Where-Object { $_.name -match "(?i)cpu" -and $_.name -notmatch "(?i)(cuda|cu1[23]|vulkan)" })
                    if ($cpuAssets.Count -gt 0) {
                        $assets = $cpuAssets
                    } else {
                        $assets = @($assets | Where-Object { $_.name -notmatch "(?i)(cuda|cu1[23]|vulkan)" })
                    }
                    $assets = @($assets | Sort-Object -Property `
                            @{Expression = {
                                    if ($_.name -match "(?i)^llama-b[0-9]+-bin-win-cpu-") { 0 }
                                    else { 1 }
                                } },
                            @{Expression = { $_.name } })
                }
            }
        }

        if ($assets.Count -eq 0) {
            throw "no matching prebuilt asset found"
        }

        $asset = $assets[0]
        $tag = if ($release.tag_name) { "$($release.tag_name)" } else { "latest" }
        $prebuiltRoot = Join-Path $RepoDir "prebuilt"
        $targetDir = Join-Path $prebuiltRoot $tag
        $zipPath = Join-Path $targetDir $asset.name
        New-Item -ItemType Directory -Force -Path $targetDir | Out-Null

        Write-Host "[llama.cpp] downloading prebuilt asset: $($asset.name)"
        Invoke-WebRequest -Uri $asset.browser_download_url -Headers $headers -OutFile $zipPath

        $extractDir = Join-Path $targetDir "extracted"
        if (Test-Path $extractDir) {
            Remove-Item -Recurse -Force $extractDir
        }
        Expand-Archive -Path $zipPath -DestinationPath $extractDir -Force

        $server = Get-ChildItem -Path $extractDir -Recurse -Filter "llama-server.exe" -File | Select-Object -First 1
        if ($server) {
            $outPath = $server.FullName
            Write-Host "[llama.cpp] prebuilt install complete"
            Write-Host "[llama.cpp] llama-server path: $outPath"
            Write-Host "[llama.cpp] set MM_LLAMA_PATH to this path for mantic-mind node startup"
            exit 0
        }

        throw "downloaded asset did not contain llama-server.exe"
    } catch {
        Write-Host "[llama.cpp] prebuilt install unavailable: $($_.Exception.Message)"
        Write-Host "[llama.cpp] falling back to source build..."
    }
}

$buildDir = Join-Path $RepoDir "build"
Write-Host "[llama.cpp] configuring CMake in $buildDir..."

$cmakeConfigArgs = @("-S", $RepoDir, "-B", $buildDir, "-DLLAMA_BUILD_SERVER=ON")
if ($enableCuda) {
    $cmakeConfigArgs += "-DGGML_CUDA=ON"
    $cmakeConfigArgs += "-DCMAKE_CUDA_ARCHITECTURES=$CudaArchitectures"
    $effectiveCudaFlags = $CudaFlags
    if ($CudaDiagSuppress) {
        if ($effectiveCudaFlags) { $effectiveCudaFlags += " " }
        $effectiveCudaFlags += "-diag-suppress=$CudaDiagSuppress"
    }
    if ($effectiveCudaFlags) {
        $cmakeConfigArgs += "-DCMAKE_CUDA_FLAGS=$effectiveCudaFlags"
    }
    Write-Host "[llama.cpp] CUDA enabled"
    Write-Host "[llama.cpp] CUDA architectures: $CudaArchitectures"
    if ($effectiveCudaFlags) {
        Write-Host "[llama.cpp] CUDA flags: $effectiveCudaFlags"
    }
} else {
    Write-Host "[llama.cpp] CUDA not enabled (no CUDA toolkit found; pass -CudaOverride 1 or set LLAMA_CPP_CUDA=1 to force)"
}
if ($Generator) {
    $cmakeConfigArgs += @("-G", $Generator)
}
Invoke-NativeChecked -Exe "cmake" -Args $cmakeConfigArgs

$cmakeBuildArgs = @("--build", $buildDir, "--config", $Config, "--target", "llama-server")
if ($Jobs -gt 0) {
    $cmakeBuildArgs += @("--parallel", "$Jobs")
}

Write-Host "[llama.cpp] building llama-server..."
Invoke-NativeChecked -Exe "cmake" -Args $cmakeBuildArgs

$candidates = @(
    (Join-Path $buildDir "bin\llama-server.exe"),
    (Join-Path $buildDir "bin\$Config\llama-server.exe"),
    (Join-Path $buildDir "bin\llama-server"),
    (Join-Path $buildDir "bin\$Config\llama-server")
)

$outPath = "(not found in expected build/bin paths)"
foreach ($p in $candidates) {
    if (Test-Path $p) {
        $outPath = $p
        break
    }
}

Write-Host "[llama.cpp] build complete"
Write-Host "[llama.cpp] llama-server path: $outPath"
Write-Host "[llama.cpp] set MM_LLAMA_PATH to this path for mantic-mind node startup"

if ($outPath -eq "(not found in expected build/bin paths)") {
    Write-Error "[llama.cpp] error: build completed but llama-server binary was not found"
    exit 3
}
