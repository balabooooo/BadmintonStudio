# Recover a UTF-8 file that was accidentally read as the ANSI codepage and
# rewritten as UTF-8.  Try a few candidate codepages and emit one candidate
# file per codepage so an out-of-band checker can pick the one that parses.
param(
    [Parameter(Mandatory = $true)][string]$In,
    [Parameter(Mandatory = $true)][string]$OutDir
)

$ErrorActionPreference = 'Stop'
New-Item -ItemType Directory -Force -Path $OutDir | Out-Null

$raw = [System.IO.File]::ReadAllBytes($In)
$text = [System.Text.Encoding]::UTF8.GetString($raw)
$text = $text.TrimStart([char]0xFEFF)

foreach ($cp in 936, 950, 1252, 20127, 28591, 54936) {
    try {
        $enc = [System.Text.Encoding]::GetEncoding($cp)
        $bytes = $enc.GetBytes($text)
        $dest = Join-Path $OutDir ("recovered_cp{0}.py" -f $cp)
        [System.IO.File]::WriteAllBytes($dest, $bytes)
        Write-Output ("cp{0}: wrote {1} bytes -> {2}" -f $cp, $bytes.Length, $dest)
    } catch {
        Write-Output ("cp{0}: FAILED {1}" -f $cp, $_.Exception.Message)
    }
}
