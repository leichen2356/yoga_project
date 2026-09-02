param(
    [string]$InputDirectory = "csv_data/normalized_nofilter",
    [string]$OutputDirectory = "csv_data/normalized_noempty"
)

$ErrorActionPreference = "Stop"

$inputPath = (Resolve-Path -LiteralPath $InputDirectory).Path
New-Item -ItemType Directory -Path $OutputDirectory -Force | Out-Null
$outputPath = (Resolve-Path -LiteralPath $OutputDirectory).Path

$csvFiles = @(Get-ChildItem -LiteralPath $inputPath -File -Filter "*.csv" | Sort-Object Name)
if ($csvFiles.Count -eq 0) {
    throw "No CSV files were found in '$inputPath'."
}

$countsByDataset = @{}
$allPoses = @{}
$processingSummary = @()

foreach ($file in $csvFiles) {
    $rows = @(Import-Csv -LiteralPath $file.FullName)
    if ($rows.Count -eq 0) {
        throw "'$($file.FullName)' has no data rows."
    }

    $columns = @($rows[0].PSObject.Properties.Name)
    if ($columns -notcontains "label") {
        throw "'$($file.FullName)' does not contain a 'label' column."
    }

    $cleanRows = @(
        foreach ($row in $rows) {
            $hasEmptyValue = $false
            foreach ($column in $columns) {
                if ([string]::IsNullOrWhiteSpace([string]$row.$column)) {
                    $hasEmptyValue = $true
                    break
                }
            }

            if (-not $hasEmptyValue) {
                $row
            }
        }
    )

    $destination = Join-Path $outputPath $file.Name
    $cleanRows | Export-Csv -LiteralPath $destination -NoTypeInformation -Encoding UTF8

    $dataset = [IO.Path]::GetFileNameWithoutExtension($file.Name) -replace '^keypoints_', ''
    $poseCounts = @{}
    foreach ($group in ($cleanRows | Group-Object -Property label)) {
        $poseCounts[$group.Name] = $group.Count
        $allPoses[$group.Name] = $true
    }
    $countsByDataset[$dataset] = $poseCounts

    $processingSummary += [pscustomobject]@{
        File = $file.Name
        OriginalRows = $rows.Count
        RemovedRows = $rows.Count - $cleanRows.Count
        CleanRows = $cleanRows.Count
    }
}

$poseSummary = @(
    foreach ($pose in ($allPoses.Keys | Sort-Object)) {
        $trainCount = if ($countsByDataset.ContainsKey("train") -and $countsByDataset["train"].ContainsKey($pose)) {
            $countsByDataset["train"][$pose]
        } else { 0 }

        $testCount = if ($countsByDataset.ContainsKey("test") -and $countsByDataset["test"].ContainsKey($pose)) {
            $countsByDataset["test"][$pose]
        } else { 0 }

        [pscustomobject]@{
            pose = $pose
            train_samples = $trainCount
            test_samples = $testCount
            total_samples = $trainCount + $testCount
        }
    }
)

$summaryPath = Join-Path $outputPath "pose_sample_summary.csv"
$poseSummary | Export-Csv -LiteralPath $summaryPath -NoTypeInformation -Encoding UTF8

$processingSummary | Format-Table -AutoSize
Write-Output "Summary: $summaryPath"
