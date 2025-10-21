$ErrorActionPreference = "Stop"

Write-Host "=" -NoNewline
Write-Host ("=" * 79)
Write-Host "IDS-ML Complete Setup and Execution"
Write-Host "=" -NoNewline
Write-Host ("=" * 79)
Write-Host ""

function Write-Section {
	param([string]$Title)
	Write-Host ""
	Write-Host "=" -NoNewline
	Write-Host ("=" * 79)
	Write-Host $Title
	Write-Host "=" -NoNewline
	Write-Host ("=" * 79)
	Write-Host ""
}

function Write-Status {
	param([string]$Message)
	Write-Host "[*] $Message" -ForegroundColor Cyan
}

function Write-Success {
	param([string]$Message)
	Write-Host "[+] $Message" -ForegroundColor Green
}

function Write-Error-Custom {
	param([string]$Message)
	Write-Host "[!] $Message" -ForegroundColor Red
}

Write-Section "Step 1: Checking for uv installation"

$uvInstalled = Get-Command uv -ErrorAction SilentlyContinue

if ($uvInstalled) {
	Write-Success "uv is already installed at: $($uvInstalled.Source)"
	Write-Host "Version: " -NoNewline
	& uv --version
}
else {
	Write-Status "uv not found. Downloading and installing uv..."

	try {
		Write-Status "Running uv installer..."
		irm https://astral.sh/uv/install.ps1 | iex

		$env:Path = [System.Environment]::GetEnvironmentVariable("Path", "Machine") + ";" + [System.Environment]::GetEnvironmentVariable("Path", "User")

		Write-Success "uv installed successfully!"
		Write-Host "Version: " -NoNewline
		& uv --version
	}
 catch {
		Write-Error-Custom "Failed to install uv: $_"
		exit 1
	}
}

Write-Section "Step 2: Installing Python dependencies"

Write-Status "Installing dependencies with uv..."
try {
	& uv sync
	Write-Success "Dependencies installed successfully!"
}
catch {
	Write-Error-Custom "Failed to install dependencies: $_"
	exit 1
}

Write-Section "Step 3: Downloading CIC-IDS-2017 dataset"

Write-Status "Checking if dataset already exists..."
$datasetPath = "data\raw\dataset.csv"

if (Test-Path $datasetPath) {
	Write-Host "Dataset already exists at: $datasetPath"
	$response = Read-Host "Do you want to re-download? (y/N)"
	if ($response -eq 'y' -or $response -eq 'Y') {
		Write-Status "Running download script..."
		& uv run ids-download
	}
 else {
		Write-Success "Skipping download, using existing dataset"
	}
}
else {
	Write-Status "Running download script..."
	Write-Host "Note: This will download approximately 500MB of data and may take several minutes..."
	& uv run ids-download

	if ($LASTEXITCODE -ne 0) {
		Write-Error-Custom "Dataset download failed!"
		exit 1
	}
	Write-Success "Dataset downloaded successfully!"
}

Write-Section "Step 4: Preprocessing the dataset"

Write-Status "Checking if preprocessing is needed..."

if (Test-Path $datasetPath) {
	Write-Host "Preprocessed dataset already exists at: $datasetPath"
	$response = Read-Host "Do you want to re-preprocess? (y/N)"
	if ($response -eq 'y' -or $response -eq 'Y') {
		Write-Status "Running preprocessing..."
		& uv run ids-preprocess
		if ($LASTEXITCODE -ne 0) {
			Write-Error-Custom "Preprocessing failed!"
			exit 1
		}
		Write-Success "Preprocessing completed!"
	}
 else {
		Write-Success "Skipping preprocessing, using existing dataset"
	}
}
else {
	Write-Status "Running preprocessing..."
	& uv run ids-preprocess

	if ($LASTEXITCODE -ne 0) {
		Write-Error-Custom "Preprocessing failed!"
		exit 1
	}
	Write-Success "Preprocessing completed!"
}

Write-Section "Step 5: Training the model"

Write-Status "Starting model training..."
Write-Host "Note: Training may take a considerable amount of time depending on your hardware..."
Write-Host "The model will use GPU acceleration if available."
Write-Host ""

try {
	& uv run ids-train

	if ($LASTEXITCODE -ne 0) {
		Write-Error-Custom "Model training failed!"
		exit 1
	}
	Write-Success "Model training completed successfully!"
}
catch {
	Write-Error-Custom "Model training encountered an error: $_"
	exit 1
}

Write-Section "Step 6: Generating sample data for inference testing"

Write-Status "Creating sample datasets with different attack scenarios..."

$samplesDir = "samples"
if (-not (Test-Path $samplesDir)) {
	New-Item -ItemType Directory -Path $samplesDir | Out-Null
	Write-Status "Created samples directory: $samplesDir"
}

$scenarios = @(
	@{Name = "small_test"; Description = "Small test sample (100 flows)" },
	@{Name = "mostly_benign"; Description = "Mostly benign traffic (95% benign)" },
	@{Name = "mixed_attacks"; Description = "Mixed attack types (90% benign)" },
	@{Name = "all_attack_types"; Description = "All attack types represented" }
)

foreach ($scenario in $scenarios) {
	$outputFile = "$samplesDir\$($scenario.Name).csv"
	Write-Status "Generating $($scenario.Description)..."

	try {
		& uv run ids-sample-gen --scenario $scenario.Name --output $outputFile

		if ($LASTEXITCODE -eq 0) {
			Write-Success "Created: $outputFile"
		}
		else {
			Write-Error-Custom "Failed to create sample: $($scenario.Name)"
		}
	}
 catch {
		Write-Error-Custom "Error generating sample $($scenario.Name): $_"
	}
}

Write-Section "Setup Complete!"

Write-Host "All steps completed successfully!" -ForegroundColor Green
Write-Host ""
Write-Host "Summary:" -ForegroundColor Yellow
Write-Host "  [+] uv installed and configured"
Write-Host "  [+] Python dependencies installed"
Write-Host "  [+] Dataset downloaded and preprocessed"
Write-Host "  [+] Model trained and saved to: models/"
Write-Host "  [+] Sample data generated in: $samplesDir/"
Write-Host ""
Write-Host "Next steps:" -ForegroundColor Yellow
Write-Host "  - Test inference with: " -NoNewline
Write-Host "uv run ids <sample_file.csv>" -ForegroundColor Cyan
Write-Host "  - Evaluate model with: " -NoNewline
Write-Host "uv run ids-evaluate" -ForegroundColor Cyan
Write-Host "  - Generate custom samples with: " -NoNewline
Write-Host "uv run ids-sample-gen --help" -ForegroundColor Cyan
Write-Host ""
Write-Host "Sample files available for testing:"
Get-ChildItem -Path $samplesDir -Filter "*.csv" | ForEach-Object {
	Write-Host "  - $($_.Name)" -ForegroundColor Gray
}
Write-Host ""
Write-Host "=" -NoNewline
Write-Host ("=" * 79)
