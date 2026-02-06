# load-env.ps1
# Loads variables from a .env file into the current PowerShell session.
# Usage (must be dot-sourced):
#   . .\load-env.ps1
# Optional:
#   . .\load-env.ps1 -Path ".env"

param (
    [string]$Path = ".env"
)

if (-not (Test-Path $Path)) {
    Write-Error "Env file not found: $Path"
    return
}

Get-Content $Path | ForEach-Object {
    $line = $_.Trim()

    # Skip empty lines and comments
    if ($line -eq "" -or $line.StartsWith("#")) {
        return
    }

    # Split on first '=' only
    $parts = $line -split "=", 2
    if ($parts.Count -ne 2) {
        return
    }

    $name = $parts[0].Trim()
    $value = $parts[1].Trim()

    # Remove optional surrounding quotes
    if (
        ($value.StartsWith('"') -and $value.EndsWith('"')) -or
        ($value.StartsWith("'") -and $value.EndsWith("'"))
    ) {
        $value = $value.Substring(1, $value.Length - 2)
    }

    Set-Item -Path "Env:$name" -Value $value
}
