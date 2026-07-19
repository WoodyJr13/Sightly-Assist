param(
    [int]$Port = 8765
)

$ErrorActionPreference = "Stop"
$RepoRoot = Split-Path -Parent $PSScriptRoot
$MobileRoot = Join-Path $RepoRoot "mobile"

Write-Host "Preparing Sightly Assist preview..." -ForegroundColor Cyan
Push-Location $RepoRoot
try {
    python -m pip install -e ".[dev,visualization,mobile]"

    $LanAddress = Get-NetIPAddress -AddressFamily IPv4 |
        Where-Object {
            $_.AddressState -eq "Preferred" -and
            $_.IPAddress -notlike "127.*" -and
            $_.IPAddress -notlike "169.254.*"
        } |
        Sort-Object InterfaceMetric |
        Select-Object -First 1 -ExpandProperty IPAddress

    if (-not $LanAddress) {
        throw "No usable LAN IPv4 address was found. Connect to Wi-Fi and retry."
    }

    $BackendUrl = "http://${LanAddress}:$Port"
    $env:EXPO_PUBLIC_SIGHTLY_API_URL = $BackendUrl
    Write-Host "Phone backend URL: $BackendUrl" -ForegroundColor Green

    $Bridge = Start-Process `
        -FilePath "python" `
        -ArgumentList @("-m", "sightly_assist.mobile_cli", "serve", "--port", "$Port", "--demo") `
        -WorkingDirectory $RepoRoot `
        -PassThru

    try {
        Push-Location $MobileRoot
        npm install
        npm start
    }
    finally {
        Pop-Location
        if ($Bridge -and -not $Bridge.HasExited) {
            Stop-Process -Id $Bridge.Id
        }
    }
}
finally {
    Pop-Location
}
