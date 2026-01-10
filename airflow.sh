#!/bin/bash
# Airflow Docker Management Script
# This script simplifies common Airflow Docker operations

set -e

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$PROJECT_DIR"

# Colors for output
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

function print_info() {
    echo -e "${BLUE}ℹ ${1}${NC}"
}

function print_success() {
    echo -e "${GREEN}✓ ${1}${NC}"
}

function print_warning() {
    echo -e "${YELLOW}⚠ ${1}${NC}"
}

function print_error() {
    echo -e "${RED}✗ ${1}${NC}"
}

function print_header() {
    echo ""
    echo "========================================================================"
    echo "  $1"
    echo "========================================================================"
    echo ""
}

function setup_airflow() {
    print_header "Setting up Airflow Environment"
    
    # Create required directories
    print_info "Creating required directories..."
    mkdir -p ./dags ./logs ./plugins ./config
    
    # Set permissions (Unix-like systems)
    if [ "$(uname)" != "MINGW64_NT"* ] && [ "$(uname)" != "MSYS_NT"* ]; then
        print_info "Setting directory permissions..."
        chmod -R 755 ./dags ./logs ./plugins ./config 2>/dev/null || true
    fi
    
    print_success "Directories created successfully"
    
    # Initialize Airflow database
    print_info "Initializing Airflow database (this may take a few minutes)..."
    docker-compose up airflow-init
    
    print_success "Airflow setup completed!"
}

function start_airflow() {
    print_header "Starting Airflow Services"
    
    # Check if setup has been run
    if [ ! -d "./logs" ]; then
        print_warning "Airflow not set up yet. Running setup first..."
        setup_airflow
    fi
    
    print_info "Starting Airflow webserver and scheduler..."
    docker-compose up -d
    
    print_success "Airflow is starting up!"
    echo ""
    echo "Web UI will be available at: http://localhost:8080"
    echo "Default credentials:"
    echo "  Username: airflow"
    echo "  Password: airflow"
    echo ""
    print_info "Waiting for services to be healthy (this may take 30-60 seconds)..."
    sleep 10
    
    # Wait for webserver to be ready
    for i in {1..30}; do
        if curl -s http://localhost:8080/health > /dev/null 2>&1; then
            print_success "Airflow webserver is ready!"
            break
        fi
        echo -n "."
        sleep 2
    done
    echo ""
    
    print_info "To view logs, run: docker-compose logs -f"
}

function stop_airflow() {
    print_header "Stopping Airflow Services"
    
    print_info "Stopping all Airflow containers..."
    docker-compose down
    
    print_success "Airflow services stopped"
}

function restart_airflow() {
    print_header "Restarting Airflow Services"
    
    stop_airflow
    sleep 2
    start_airflow
}

function show_logs() {
    print_header "Airflow Logs"
    
    print_info "Showing logs from all services (press Ctrl+C to exit)..."
    docker-compose logs -f
}

function show_status() {
    print_header "Airflow Services Status"
    
    docker-compose ps
}

function clean_airflow() {
    print_header "Cleaning Airflow Environment"
    
    print_warning "This will remove all containers, volumes, and data!"
    read -p "Are you sure? (yes/no): " confirm
    
    if [ "$confirm" = "yes" ]; then
        print_info "Stopping services..."
        docker-compose down -v
        
        print_info "Removing logs..."
        rm -rf ./logs/*
        
        print_success "Cleanup completed!"
    else
        print_info "Cleanup cancelled"
    fi
}

function trigger_dag() {
    print_header "Triggering DAG Execution"
    
    print_info "Triggering 'schema_matching_pipeline' DAG..."
    docker-compose exec airflow-scheduler airflow dags trigger schema_matching_pipeline
    
    print_success "DAG triggered! Check the web UI for execution status."
}

function print_usage() {
    echo "Airflow Docker Management Script"
    echo ""
    echo "Usage: $0 <command>"
    echo ""
    echo "Commands:"
    echo "  setup      - Initial setup of Airflow environment"
    echo "  start      - Start Airflow services"
    echo "  stop       - Stop Airflow services"
    echo "  restart    - Restart Airflow services"
    echo "  logs       - Show and follow service logs"
    echo "  status     - Show status of all services"
    echo "  trigger    - Trigger the schema matching DAG"
    echo "  clean      - Clean up all containers and volumes"
    echo "  help       - Show this help message"
    echo ""
}

# Main script logic
case "${1:-}" in
    setup)
        setup_airflow
        ;;
    start)
        start_airflow
        ;;
    stop)
        stop_airflow
        ;;
    restart)
        restart_airflow
        ;;
    logs)
        show_logs
        ;;
    status)
        show_status
        ;;
    trigger)
        trigger_dag
        ;;
    clean)
        clean_airflow
        ;;
    help|--help|-h)
        print_usage
        ;;
    *)
        print_error "Unknown command: ${1:-}"
        echo ""
        print_usage
        exit 1
        ;;
esac
