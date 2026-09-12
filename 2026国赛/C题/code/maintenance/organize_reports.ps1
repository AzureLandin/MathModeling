param([string]$ProjectRoot = 'E:/MathModeling/2026国赛/C题')
$ErrorActionPreference = 'Stop'
$rootPath = [IO.Path]::GetFullPath($ProjectRoot)
$recordDir = Join-Path $rootPath 'results/report_organization_20260912'
$mapPath = Join-Path $recordDir 'path_map.json'
$mapping = Get-Content -LiteralPath $mapPath -Raw | ConvertFrom-Json -AsHashtable
$backupRoot = Join-Path $recordDir 'before'
if (Test-Path -LiteralPath $backupRoot) { throw 'Migration backup already exists; do not run twice.' }
$utf8 = [Text.UTF8Encoding]::new($false)
$moves = @()
$edits = @()
foreach ($old in $mapping.Keys) {
    $source = [IO.Path]::GetFullPath((Join-Path $rootPath $old))
    $target = [IO.Path]::GetFullPath((Join-Path $rootPath $mapping[$old]))
    $reportBoundary = [IO.Path]::GetFullPath((Join-Path $rootPath 'reports')) + [IO.Path]::DirectorySeparatorChar
    if (-not $source.StartsWith($reportBoundary) -or -not $target.StartsWith($reportBoundary)) {
        throw "Report path outside workspace: $source -> $target"
    }
    if (-not (Test-Path -LiteralPath $source -PathType Leaf)) { throw "Missing source: $source" }
    if (Test-Path -LiteralPath $target) { throw "Target already exists: $target" }
}
foreach ($old in $mapping.Keys) {
    $source = Join-Path $rootPath $old
    $target = Join-Path $rootPath $mapping[$old]
    $backup = Join-Path $backupRoot $old
    [IO.Directory]::CreateDirectory([IO.Path]::GetDirectoryName($backup)) | Out-Null
    Copy-Item -LiteralPath $source -Destination $backup
    $hash = (Get-FileHash -LiteralPath $source -Algorithm SHA256).Hash
    [IO.Directory]::CreateDirectory([IO.Path]::GetDirectoryName($target)) | Out-Null
    Move-Item -LiteralPath $source -Destination $target
    if ((Get-FileHash -LiteralPath $target -Algorithm SHA256).Hash -ne $hash) { throw "Move hash mismatch: $old" }
    $moves += [ordered]@{old=$old; new=$mapping[$old]; original_sha256=$hash}
}
$files = @((Get-ChildItem -LiteralPath (Join-Path $rootPath 'code') -File -Filter '*.py'))
$files += @(Get-ChildItem -LiteralPath (Join-Path $rootPath 'reports') -File -Recurse -Filter '*.md' |
    Where-Object { $_.FullName -notmatch '[\\/]archive[\\/](?!早期思路[\\/])' })
$files += Get-Item -LiteralPath (Join-Path $rootPath '建模上下文记忆.md')
foreach ($file in $files) {
    $original = [IO.File]::ReadAllText($file.FullName)
    $text = $original
    foreach ($old in $mapping.Keys) {
        $new = $mapping[$old]
        $name = [IO.Path]::GetFileName($old)
        $text = $text.Replace($old, $new).Replace($old.Replace('/', '\'), $new.Replace('/', '\'))
        if ($file.Extension -eq '.py') {
            foreach ($quote in @("'", '"')) {
                $pattern = [regex]::Escape($quote + 'reports' + $quote) + '\s*/\s*' + [regex]::Escape($quote + $name + $quote)
                $text = [regex]::Replace($text, $pattern, $quote + $new + $quote)
            }
        } else {
            $text = $text.Replace('](' + $name + ')', '](' + $rootPath.Replace('\','/') + '/' + $new + ')')
        }
    }
    if ($text -ne $original) {
        $relative = [IO.Path]::GetRelativePath($rootPath, $file.FullName).Replace('\','/')
        $backup = Join-Path $backupRoot $relative
        if (-not (Test-Path -LiteralPath $backup)) {
            [IO.Directory]::CreateDirectory([IO.Path]::GetDirectoryName($backup)) | Out-Null
            Copy-Item -LiteralPath $file.FullName -Destination $backup
        }
        $beforeHash = (Get-FileHash -LiteralPath $file.FullName -Algorithm SHA256).Hash
        [IO.File]::WriteAllText($file.FullName, $text, $utf8)
        $edits += [ordered]@{path=$relative; before_sha256=$beforeHash; after_sha256=(Get-FileHash -LiteralPath $file.FullName -Algorithm SHA256).Hash}
    }
}
$result = [ordered]@{scope='Report relocation and literal reference updates only; signed result manifests and source snapshots unchanged'; moves=$moves; reference_edits=$edits}
[IO.File]::WriteAllText((Join-Path $recordDir 'migration.json'), ($result | ConvertTo-Json -Depth 8), $utf8)
Write-Output "Moved $($moves.Count) reports; updated references in $($edits.Count) files."
