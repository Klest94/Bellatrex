param(
    [string]$CoverageXml = "coverage.xml"
)

$ErrorActionPreference = "Stop"

if (-not (Test-Path -LiteralPath $CoverageXml)) {
    throw "Coverage XML not found: $CoverageXml. Generate it first with pytest --cov-report=xml."
}

$cov = [xml](Get-Content -LiteralPath $CoverageXml)
$partials = @(
    $cov.coverage.packages.package.classes.class.lines.line |
        Where-Object {
            $_.branch -eq "true" -and
            $_."condition-coverage" -and
            $_."condition-coverage" -notmatch "^100%" -and
            [int]$_.hits -gt 0
        }
).Count

$covered = [int]$cov.coverage."lines-covered"
$valid = [int]$cov.coverage."lines-valid"
$hits = $covered - $partials
$coverage = ($hits / $valid) * 100

"Codecov-style coverage: {0:N2}% ({1}/{2} hits, {3} partials)" -f $coverage, $hits, $valid, $partials
