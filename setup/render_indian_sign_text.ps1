param([Parameter(Mandatory=$true)][string]$Layout)
$ErrorActionPreference = 'Stop'
Add-Type -AssemblyName System.Drawing
$signRows = Get-Content -LiteralPath $Layout -Raw -Encoding UTF8 | ConvertFrom-Json
$signFolder = Split-Path -Parent $Layout
foreach ($signRow in $signRows) {
    if (-not $signRow.background) { continue }
    $signFile = Join-Path $signFolder $signRow.file
    $signOriginal = [Drawing.Image]::FromFile($signFile)
    $signBitmap = New-Object Drawing.Bitmap($signOriginal)
    $signOriginal.Dispose()
    $signGraphics = [Drawing.Graphics]::FromImage($signBitmap)
    $signGraphics.TextRenderingHint = [Drawing.Text.TextRenderingHint]::AntiAliasGridFit
    $signBackground = New-Object Drawing.SolidBrush([Drawing.Color]::FromArgb($signRow.background[0],$signRow.background[1],$signRow.background[2]))
    $signForeground = New-Object Drawing.SolidBrush([Drawing.Color]::FromArgb(255,242,200))
    $signFormat = New-Object Drawing.StringFormat
    $signFormat.Alignment = [Drawing.StringAlignment]::Center
    $signFormat.LineAlignment = [Drawing.StringAlignment]::Center
    if ($signRow.kind -eq 'shopfront') {
        $signRectangle = New-Object Drawing.RectangleF(17,22,734,100)
        $signFont = New-Object Drawing.Font('Nirmala UI',43,[Drawing.FontStyle]::Regular,[Drawing.GraphicsUnit]::Pixel)
    } else {
        $signRectangle = New-Object Drawing.RectangleF(17,35,734,127)
        $signFont = New-Object Drawing.Font('Nirmala UI',52,[Drawing.FontStyle]::Regular,[Drawing.GraphicsUnit]::Pixel)
    }
    $signGraphics.FillRectangle($signBackground,$signRectangle)
    $signGraphics.DrawString([string]$signRow.text[0],$signFont,$signForeground,$signRectangle,$signFormat)
    $signGraphics.Dispose()
    $signBitmap.Save($signFile,[Drawing.Imaging.ImageFormat]::Png)
    $signBitmap.Dispose()
    $signFont.Dispose()
    $signFormat.Dispose()
    $signBackground.Dispose()
    $signForeground.Dispose()
}
