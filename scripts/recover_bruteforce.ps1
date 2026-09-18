# Brute-force every .NET encoding to find the one whose inverse turns the
# mojibake file back into valid UTF-8 bytes.  Emits one candidate per encoding
# that yields a strictly-decodable UTF-8 result.
param(
    [Parameter(Mandatory = $true)][string]$In,
    [Parameter(Mandatory = $true)][string]$OutDir
)

$ErrorActionPreference = 'Stop'
New-Item -ItemType Directory -Force -Path $OutDir | Out-Null

$raw = [System.IO.File]::ReadAllBytes($In)
$text = [System.Text.Encoding]::UTF8.GetString($raw)
$text = $text.TrimStart([char]0xFEFF)

$strict = New-Object System.Text.UTF8Encoding($false, $true)
$tried = 0
$hit = 0
foreach ($info in [System.Text.Encoding]::GetEncodings()) {
    $cp = $info.CodePage
    $tried++
    try {
        $enc = [System.Text.Encoding]::GetEncoding($cp)
        $bytes = $enc.GetBytes($text)
    } catch {
        continue
    }
    try {
        $null = $strict.GetString($bytes)      # throws unless valid UTF-8
    } catch {
        continue
    }
    $dest = Join-Path $OutDir ("ok_cp{0}.py" -f $cp)
    [System.IO.File]::WriteAllBytes($dest, $bytes)
    $hit++
    Write-Output ("cp{0,-8} {1}  -> {2}" -f $cp, $info.Name, $dest)
}
Write-Output ("tried {0} encodings, {1} produced strict UTF-8" -f $tried, $hit)
