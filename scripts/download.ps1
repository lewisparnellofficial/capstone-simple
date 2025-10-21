$zip_url = "http://205.174.165.80/CICDataset/CIC-IDS-2017/Dataset/CIC-IDS-2017/CSVs/MachineLearningCSV.zip"
$md5_url = "http://205.174.165.80/CICDataset/CIC-IDS-2017/Dataset/CIC-IDS-2017/CSVs/MachineLearningCSV.md5"

$zip_path = "..\data\MachineLearningCSV.zip"
$md5_path = "..\data\MachineLearningCSV.md5"

if (-not (Test-Path $zip_path)) {
	Invoke-WebRequest -Uri $zip_url -OutFile $zip_path
}

Invoke-WebRequest -Uri $md5_url -OutFile $md5_path

$zip_hash = (Get-FileHash -Path $zip_path -Algorithm MD5).Hash
$expected_hash = ((Get-Content -Path $md5_path) -split '\s')[0]
if ($zip_hash -ne $expected_hash) {
	Write-Host "The zip file is corrupt. Redownload the file."
	exit 1
}

Expand-Archive -Force -Path $zip_path -DestinationPath ".\data\"
