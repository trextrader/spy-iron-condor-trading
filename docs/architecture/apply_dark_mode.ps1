$files = Get-ChildItem "c:\SPYOptionTrader_test\docs\architecture\*.dot"

foreach ($file in $files) {
    $content = Get-Content $file.FullName -Raw
    $modified = $false

    # 1. Update existing bgcolor to black
    if ($content -match 'bgcolor="[^"]+"') {
        if ($content -notmatch 'bgcolor="#0a0a0a"') {
            $content = $content -replace 'bgcolor="[^"]+"', 'bgcolor="#0a0a0a"'
            $modified = $true
        }
    } 
    # 2. Inject bgcolor if missing
    elseif ($content -match 'digraph\s+\w+\s*\{') {
        $content = $content -replace '(digraph\s+\w+\s*\{)', '$1
    bgcolor="#0a0a0a";'
        $modified = $true
    }

    # 3. Ensure global fontcolor is white-ish if not set
    if (-not ($content -match 'graph\s*\[.*fontcolor')) {
        # This is hard to regex safely, but we can try to inject a default block
        # Only inject if we see a node definition to avoid breaking syntax
        if ($content -match '(node\s*\[)') {
             # Replace node [ with node [fontcolor="white", color="white", 
             # Use a robust replace
             # $content = $content -replace '(node\s*\[)', '$1 fontcolor="white", color="white", '
             # Actually, better to just append a holistic style block at top if missing
        }
    }
    
    if ($modified) {
        Set-Content -Path $file.FullName -Value $content
        Write-Host "Updated $($file.Name)"
    }
}
