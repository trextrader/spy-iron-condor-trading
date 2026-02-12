# ================================================================
#  Download Dec 13-31 2025 SPY Options Bars from Alpaca
#  Generates full OPRA symbol matrix from confirmed expirations
#  and strike range, then batches API calls.
# ================================================================

$apiKey    = "PKWCQL536DJKE7EJCP5OEETWE2"
$apiSecret = "8PK6xfXx13Hqna2ryjHyQCMAf6D1zj6kNGE96CjnzmKM"
$outFile   = "dec_2025_options_bars.csv"

$headers = @{
    "accept"              = "application/json"
    "APCA-API-KEY-ID"     = $apiKey
    "APCA-API-SECRET-KEY" = $apiSecret
}

# ================================================================
#  CONFIRMED EXPIRATION DATES (YYMMDD)
#  Daily/Weekly + Monthly (251219) + Year-End (251231)
# ================================================================
$expirations = @(
    "251215", "251216", "251217", "251218", "251219",
    "251222", "251223", "251224", "251226",
    "251229", "251230", "251231"
)

# ================================================================
#  LEAP EXPIRATIONS traded in late Dec 2025
# ================================================================
$leapExpirations = @(
    "260116",   # Jan 2026 monthly
    "260320",   # Mar 2026 quarterly
    "271219"    # Dec 2027
)

# ================================================================
#  STRIKE GENERATION
#  SPY traded $681.92 - $697.00 in this window
#  Near-term: $1 strikes from $670-$710 (covers +/- ~$20 from range)
#  LEAPs:     $5 strikes from $650-$730 (wider spread for long-dated)
# ================================================================

# Build near-term symbols: 41 strikes x 12 expirations x 2 (C+P) = 984 symbols
$nearTermSymbols = @()
foreach ($exp in $expirations) {
    for ($strike = 670; $strike -le 710; $strike++) {
        $strikeStr = ($strike * 1000).ToString("D8")
        $nearTermSymbols += "SPY${exp}C${strikeStr}"
        $nearTermSymbols += "SPY${exp}P${strikeStr}"
    }
}

# Build LEAP symbols: 17 strikes x 3 expirations x 2 (C+P) = 102 symbols
$leapSymbols = @()
foreach ($exp in $leapExpirations) {
    for ($strike = 650; $strike -le 730; $strike += 5) {
        $strikeStr = ($strike * 1000).ToString("D8")
        $leapSymbols += "SPY${exp}C${strikeStr}"
        $leapSymbols += "SPY${exp}P${strikeStr}"
    }
}

$allSymbols = $nearTermSymbols + $leapSymbols
Write-Host "================================================================"
Write-Host "  SPY Dec 2025 Options Bar Downloader"
Write-Host "  Near-term symbols:  $($nearTermSymbols.Count)"
Write-Host "  LEAP symbols:       $($leapSymbols.Count)"
Write-Host "  Total symbols:      $($allSymbols.Count)"
Write-Host "================================================================"

# Write CSV header (fresh file)
"symbol,timestamp,open,high,low,close,volume,trade_count,vwap" | Out-File -FilePath $outFile -Encoding utf8

# ================================================================
#  BATCH DOWNLOAD -- 10 symbols per request
#  Free Alpaca: 200 req/min limit -> 1 req per 350ms minimum
#  Using 750ms base delay + exponential backoff on 429
# ================================================================
$batchSize    = 10
$totalBatches = [math]::Ceiling($allSymbols.Count / $batchSize)
$totalBars    = 0
$symbolsWithData = 0
$symbolsEmpty    = 0
$errors          = 0
$rateLimitHits   = 0
$batchNum        = 0
$baseDelay       = 750    # ms between requests (safe for 200 req/min)

for ($i = 0; $i -lt $allSymbols.Count; $i += $batchSize) {

    $batchNum++
    $end = [math]::Min($i + $batchSize - 1, $allSymbols.Count - 1)
    $batch = $allSymbols[$i..$end]
    $symbolList = $batch -join ","

    $pct = [math]::Round(($batchNum / $totalBatches) * 100, 1)
    Write-Host "Batch $batchNum/$totalBatches ($pct%)  symbols $($i+1)-$($end+1)..."

    $url = "https://data.alpaca.markets/v1beta1/options/bars?symbols=$symbolList&timeframe=1Min&start=2025-12-13T00:00:00Z&end=2025-12-31T23:59:59Z&limit=10000"

    # Retry loop with exponential backoff for rate limits
    $maxRetries = 4
    $parsed = $null
    for ($attempt = 1; $attempt -le $maxRetries; $attempt++) {
        try {
            $parsed = Invoke-RestMethod -Uri $url -Headers $headers -Method Get
            break  # success
        }
        catch {
            $status = $_.Exception.Response.StatusCode.value__
            if ($status -eq 429) {
                $rateLimitHits++
                $backoff = $baseDelay * [math]::Pow(2, $attempt)
                Write-Host "  429 Rate limited - backing off ${backoff}ms (attempt $attempt/$maxRetries)"
                Start-Sleep -Milliseconds $backoff
            }
            else {
                Write-Host "  HTTP $status error: $($_.Exception.Message)"
                $errors++
                break
            }
        }
    }

    if ($null -eq $parsed) {
        Write-Host "  Skipping batch after $maxRetries retries"
        continue
    }

    # Process each symbol in the batch response
    foreach ($sym in $batch) {
        $bars = $parsed.bars.$sym

        if ($null -eq $bars -or $bars.Count -eq 0) {
            $symbolsEmpty++
            continue
        }

        $symbolsWithData++
        $totalBars += $bars.Count
        $first = $bars[0]
        Write-Host "  $sym  $($bars.Count) bars  first=$($first.t) o=$($first.o) h=$($first.h) l=$($first.l) c=$($first.c) v=$($first.v)"

        foreach ($bar in $bars) {
            $line = "$sym,$($bar.t),$($bar.o),$($bar.h),$($bar.l),$($bar.c),$($bar.v),$($bar.n),$($bar.vw)"
            Add-Content -Path $outFile -Value $line
        }
    }

    Start-Sleep -Milliseconds $baseDelay
}

# ================================================================
#  SUMMARY
# ================================================================
Write-Host ""
Write-Host "================================================================"
Write-Host "  DOWNLOAD COMPLETE"
Write-Host "  Symbols with data:  $symbolsWithData"
Write-Host "  Symbols empty:      $symbolsEmpty"
Write-Host "  Rate limit hits:    $rateLimitHits"
Write-Host "  Errors:             $errors"
Write-Host "  Total bars saved:   $totalBars"
Write-Host "  Output file:        $outFile"
Write-Host "================================================================"
