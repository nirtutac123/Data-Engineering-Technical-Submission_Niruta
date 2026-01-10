# Airflow Docker Management Script for Windows PowerShell
# This script simplifies common Airflow Docker operations

param(
    [Parameter(Position=0)]
    [string]$Command
)

$ProjectDir = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $ProjectDir

function Print-Header {
    param([string]$Message)
    Write-Host ""
    Write-Host "========================================================================"
    Write-Host "  $Message"
    Write-Host "========================================================================"
    Write-Host ""
}

function Print-Info {
    param([string]$Message)
    Write-Host "ℹ $Message" -ForegroundColor Blue
}

function Print-Success {
    param([string]$Message)
    Write-Host "✓ $Message" -ForegroundColor Green
}

function Print-Warning {
    param([string]$Message)
    Write-Host "⚠ $Message" -ForegroundColor Yellow
}

function Print-Error {
    param([string]$Message)
    Write-Host "✗ $Message" -ForegroundColor Red
}

function Setup-Airflow {
    Print-Header "Setting up Airflow Environment"
    
    # Create required directories
    Print-Info "Creating required directories..."
    $dirs = @("dags", "logs", "plugins", "config")
    foreach ($dir in $dirs) {
        if (-not (Test-Path $dir)) {
            New-Item -ItemType Directory -Path $dir -Force | Out-Null
        }
    }
    Print-Success "Directories created successfully"
    
    # Initialize Airflow database
    Print-Info "Initializing Airflow database (this may take a few minutes)..."
    docker-compose up airflow-init
    
    Print-Success "Airflow setup completed!"
}

function Start-Airflow {
    Print-Header "Starting Airflow Services"
    
    # Check if setup has been run
    if (-not (Test-Path "./logs")) {
        Print-Warning "Airflow not set up yet. Running setup first..."
        Setup-Airflow
    }
    
    Print-Info "Starting Airflow webserver and scheduler..."
    docker-compose up -d
    
    Print-Success "Airflow is starting up!"
    Write-Host ""
    Write-Host "Web UI will be available at: http://localhost:8080"
    Write-Host "Default credentials:"
    Write-Host "  Username: airflow"
    Write-Host "  Password: airflow"
    Write-Host ""
    Print-Info "Waiting for services to be healthy (this may take 30-60 seconds)..."
    Start-Sleep -Seconds 10
    
    # Wait for webserver to be ready
    $maxAttempts = 30
    $attempt = 0
    while ($attempt -lt $maxAttempts) {
        try {
            $response = Invoke-WebRequest -Uri "http://localhost:8080/health" -UseBasicParsing -TimeoutSec 2 -ErrorAction SilentlyContinue
            if ($response.StatusCode -eq 200) {
                Print-Success "Airflow webserver is ready!"
                break
            }
        }
        catch {
            # Continue waiting
        }
        Write-Host "." -NoNewline
        Start-Sleep -Seconds 2
        $attempt++
    }
    Write-Host ""
    
    Print-Info "To view logs, run: docker-compose logs -f"
}

function Stop-Airflow {
    Print-Header "Stopping Airflow Services"
    
    Print-Info "Stopping all Airflow containers..."
    docker-compose down
    
    Print-Success "Airflow services stopped"
}

function Restart-Airflow {
    Print-Header "Restarting Airflow Services"
    
    Stop-Airflow
    Start-Sleep -Seconds 2
    Start-Airflow
}

function Show-Logs {
    Print-Header "Airflow Logs"
    
    Print-Info "Showing logs from all services (press Ctrl+C to exit)..."
    docker-compose logs -f
}

function Show-Status {
    Print-Header "Airflow Services Status"
    
    docker-compose ps
}

function Clean-Airflow {
    Print-Header "Cleaning Airflow Environment"
    
    Print-Warning "This will remove all containers, volumes, and data!"
    $confirm = Read-Host "Are you sure? (yes/no)"
    
    if ($confirm -eq "yes") {
        Print-Info "Stopping services..."
        docker-compose down -v
        
        Print-Info "Removing logs..."
        if (Test-Path "./logs") {
            Remove-Item -Path "./logs/*" -Recurse -Force -ErrorAction SilentlyContinue
        }
        
        Print-Success "Cleanup completed!"
    }
    else {
        Print-Info "Cleanup cancelled"
    }
}

function Trigger-DAG {
    Print-Header "Triggering DAG Execution"
    
    Print-Info "Triggering 'schema_matching_pipeline' DAG..."
    docker-compose exec airflow-scheduler airflow dags trigger schema_matching_pipeline
    
    Print-Success "DAG triggered! Check the web UI for execution status."
}

function Show-Usage {
    Write-Host "Airflow Docker Management Script for PowerShell"
    Write-Host ""
    Write-Host "Usage: .\airflow.ps1 <command>"
    Write-Host ""
    Write-Host "Commands:"
    Write-Host "  setup      - Initial setup of Airflow environment"
    Write-Host "  start      - Start Airflow services"
    Write-Host "  stop       - Stop Airflow services"
    Write-Host "  restart    - Restart Airflow services"
    Write-Host "  logs       - Show and follow service logs"
    Write-Host "  status     - Show status of all services"
    Write-Host "  trigger    - Trigger the schema matching DAG"
    Write-Host "  clean      - Clean up all containers and volumes"
    Write-Host "  help       - Show this help message"
    Write-Host ""
}

# Main script logic
switch ($Command) {
    "setup" { Setup-Airflow }
    "start" { Start-Airflow }
    "stop" { Stop-Airflow }
    "restart" { Restart-Airflow }
    "logs" { Show-Logs }
    "status" { Show-Status }
    "trigger" { Trigger-DAG }
    "clean" { Clean-Airflow }
    "help" { Show-Usage }
    default {
        if ($Command) {
            Print-Error "Unknown command: $Command"
            Write-Host ""
        }
        Show-Usage
        if ($Command -and $Command -ne "help") {
            exit 1
        }
    }
}
