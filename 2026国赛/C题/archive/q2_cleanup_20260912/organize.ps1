$ErrorActionPreference = 'Stop'
$root = [IO.Path]::GetFullPath('E:/MathModeling/2026国赛/C题')
$archive = Join-Path $root 'archive/q2_cleanup_20260912'
$keepCode = @('05_q2_baseline.py','08_q2_quantile_experiment.py','14_q2_ridge_forecast_experiment.py','19_q2_lightgbm_residual_experiment.py','21_q2_quantile_level_scan.py','29_q2_time_mapping_experiment.py','30_q2_report_renderer.py','fill_result2_frozen.py')
$keepResults = @('q2_time_mapping','q2_frozen_delivery','q2_revision_audit_20260911')
$targets = @()
$targets += Get-ChildItem -LiteralPath (Join-Path $root 'code') -File | Where-Object { $_.Name -match 'q2' -and $_.Name -notin $keepCode }
$targets += Get-ChildItem -LiteralPath (Join-Path $root 'results') | Where-Object { $_.Name -like 'q2*' -and $_.Name -notin $keepResults }
$targets += Get-ChildItem -LiteralPath (Join-Path $root 'figures') | Where-Object { $_.Name -like 'q2*' -and $_.Name -ne 'q2_time_mapping' }
$targets += Get-ChildItem -LiteralPath (Join-Path $root 'reports/问题二') -Directory
$cache = Join-Path $root 'code/__pycache__'
if (Test-Path -LiteralPath $cache) { $targets += Get-ChildItem -LiteralPath $cache -File | Where-Object { $_.Name -match 'q2' -and -not ($keepCode | Where-Object { $_.BaseName -eq '' }) } }
$mapping = @()
$files = @()
foreach ($target in $targets) {
    $src = [IO.Path]::GetFullPath($target.FullName)
    if (-not $src.StartsWith($root + '\', [StringComparison]::OrdinalIgnoreCase)) { throw "Outside workspace: $src" }
    $rel = [IO.Path]::GetRelativePath($root, $src).Replace('\','/')
    $dest = [IO.Path]::GetFullPath((Join-Path $archive $rel))
    if (-not $dest.StartsWith($archive + '\', [StringComparison]::OrdinalIgnoreCase)) { throw "Outside archive: $dest" }
    if (Test-Path -LiteralPath $dest) { throw "Archive destination exists: $dest" }
    $mapping += [pscustomobject]@{old=$rel; new=('archive/q2_cleanup_20260912/' + $rel)}
    $leaves = if ($target.PSIsContainer) { Get-ChildItem -LiteralPath $src -Recurse -File } else { @($target) }
    foreach ($file in $leaves) {
        $frel = [IO.Path]::GetRelativePath($root,$file.FullName).Replace('\','/')
        $files += [pscustomobject]@{old=$frel; new=('archive/q2_cleanup_20260912/' + $frel); bytes=$file.Length; sha256=(Get-FileHash -LiteralPath $file.FullName -Algorithm SHA256).Hash}
    }
}
$protectedPaths = @('附件/附件5/result2.xlsx','results/q2_time_mapping/archive_float.npz','results/q2_time_mapping/N_free/natural_dispatch.csv','results/q2_time_mapping/N_free/template_plan.csv')
$protected = @($protectedPaths | ForEach-Object { [pscustomobject]@{path=$_; sha256=(Get-FileHash -LiteralPath (Join-Path $root $_) -Algorithm SHA256).Hash} })
$files | ConvertTo-Json -Depth 5 | Set-Content -LiteralPath (Join-Path $archive 'file_manifest.json') -Encoding utf8
$mapping | ConvertTo-Json -Depth 5 | Set-Content -LiteralPath (Join-Path $archive 'path_map.json') -Encoding utf8
foreach ($item in $mapping) {
    $dest = Join-Path $root $item.new
    New-Item -ItemType Directory -Path (Split-Path $dest -Parent) -Force | Out-Null
    Move-Item -LiteralPath (Join-Path $root $item.old) -Destination $dest
}
foreach ($file in $files) {
    if ((Get-FileHash -LiteralPath (Join-Path $root $file.new) -Algorithm SHA256).Hash -ne $file.sha256) { throw "Hash mismatch: $($file.new)" }
}
# Update live documentation links mechanically; frozen output and archived prose stay unchanged.
$docs = @(Get-ChildItem -LiteralPath (Join-Path $root 'reports') -Recurse -File -Filter '*.md' | Where-Object { $_.FullName -notmatch '\\archive\\' })
$docs += Get-Item -LiteralPath (Join-Path $root '建模上下文记忆.md')
$updated = @()
foreach ($doc in $docs) {
    $text = [IO.File]::ReadAllText($doc.FullName)
    $newText = $text
    foreach ($m in ($mapping | Sort-Object { $_.old.Length } -Descending)) { $newText = $newText.Replace($m.old,$m.new) }
    if ($newText -ne $text) {
        $rel = [IO.Path]::GetRelativePath($root,$doc.FullName)
        $backup = Join-Path $archive ('docs_before/' + $rel)
        New-Item -ItemType Directory -Path (Split-Path $backup -Parent) -Force | Out-Null
        Copy-Item -LiteralPath $doc.FullName -Destination $backup
        [IO.File]::WriteAllText($doc.FullName,$newText,[Text.UTF8Encoding]::new($false))
        $updated += $rel
    }
}
foreach ($p in $protected) { if ((Get-FileHash -LiteralPath (Join-Path $root $p.path) -Algorithm SHA256).Hash -ne $p.sha256) { throw "Core modified: $($p.path)" } }
$summary = [pscustomobject]@{moved_items=$mapping.Count; moved_files=$files.Count; moved_bytes=($files | Measure-Object bytes -Sum).Sum; updated_documents=$updated; protected=$protected; hashes_verified=$true; deleted_files=0}
$summary | ConvertTo-Json -Depth 6 | Set-Content -LiteralPath (Join-Path $archive 'cleanup_summary.json') -Encoding utf8
$summary | Select-Object moved_items,moved_files,moved_bytes,hashes_verified,deleted_files | Format-List
