param(
    [string]$ConfigPath = "scripts/model_urls.json",
    [switch]$Force
)

$ErrorActionPreference = "Stop"

function Download-File($url, $destPath) {
    $destDir = Split-Path -Parent $destPath
    if (-not (Test-Path $destDir)) { New-Item -ItemType Directory -Force -Path $destDir | Out-Null }

    if ((Test-Path $destPath) -and (-not $Force)) {
        Write-Host "Exists: $destPath (skip). Use -Force to re-download." -ForegroundColor Yellow
        return
    }

    Write-Host "Downloading: $url -> $destPath" -ForegroundColor Cyan
    try {
        Invoke-WebRequest -Uri $url -OutFile $destPath
        Write-Host "Saved: $destPath" -ForegroundColor Green
    }
    catch {
        Write-Error "Failed to download $url: $($_.Exception.Message)"
        throw
    }
}

if (-not (Test-Path $ConfigPath)) {
    Write-Error "Config file not found: $ConfigPath"
    exit 1
}

try {
    $json = Get-Content $ConfigPath -Raw | ConvertFrom-Json
}
catch {
    Write-Error "Invalid JSON in $ConfigPath: $($_.Exception.Message)"
    exit 1
}

# $json is a PSCustomObject; enumerate its properties
$json.PSObject.Properties | ForEach-Object {
    $dest = $_.Name
    $url = $_.Value
    if ([string]::IsNullOrWhiteSpace($url) -or $url -like "CHANGE_ME*") {
        Write-Host "Missing URL for $dest. Update $ConfigPath." -ForegroundColor Yellow
        return
    }
    Download-File -url $url -destPath $dest
}
