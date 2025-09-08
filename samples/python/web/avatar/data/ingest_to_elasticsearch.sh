#!/bin/bash

# Clinical Patient Data Elasticsearch Ingestion Script
# This script loads clinical patient data from JSON file into Elasticsearch

set -e  # Exit on any error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Configuration
INDEX_NAME="clinical-patient-data"
DATA_FILE="clinical-patient-data"
ENV_FILE="../variables.env"

echo -e "${YELLOW}Starting Clinical Patient Data Ingestion to Elasticsearch...${NC}"

# Check if environment file exists
if [ ! -f "$ENV_FILE" ]; then
    echo -e "${RED}Error: Environment file $ENV_FILE not found!${NC}"
    exit 1
fi

# Load environment variables
source "$ENV_FILE"

# Check if required environment variables are set
if [ -z "$ELASTIC_URL" ] || [ -z "$ELASTIC_API_KEY" ]; then
    echo -e "${RED}Error: ELASTIC_URL or ELASTIC_API_KEY not set in environment file!${NC}"
    exit 1
fi

# Check if data file exists
if [ ! -f "$DATA_FILE" ]; then
    echo -e "${RED}Error: Data file $DATA_FILE not found!${NC}"
    exit 1
fi

echo -e "${GREEN}Environment variables loaded successfully${NC}"
echo -e "${GREEN}Elasticsearch URL: $ELASTIC_URL${NC}"
echo -e "${GREEN}Index name: $INDEX_NAME${NC}"

# Create Python script for Elasticsearch ingestion
ELASTIC_URL="$ELASTIC_URL" ELASTIC_API_KEY="$ELASTIC_API_KEY" python3 << 'EOF'
import json
import sys
import os
import requests
from datetime import datetime
import base64

# Configuration
ELASTIC_URL = os.environ.get('ELASTIC_URL')
ELASTIC_API_KEY = os.environ.get('ELASTIC_API_KEY')
INDEX_NAME = 'clinical-patient-data'
DATA_FILE = 'clinical-patient-data'

def print_status(message, status="INFO"):
    colors = {
        "INFO": "\033[0;32m",    # Green
        "WARNING": "\033[1;33m", # Yellow
        "ERROR": "\033[0;31m",   # Red
        "SUCCESS": "\033[0;32m"  # Green
    }
    print(f"{colors.get(status, '')}{message}\033[0m")

def test_elasticsearch_connection():
    """Test connection to Elasticsearch"""
    try:
        headers = {
            'Authorization': f'ApiKey {ELASTIC_API_KEY}',
            'Content-Type': 'application/json'
        }
        
        # Test connection with a simple info request (works for both regular and serverless)
        response = requests.get(f"{ELASTIC_URL}/", headers=headers, timeout=10)
        
        if response.status_code == 200:
            print_status("✓ Elasticsearch connection successful", "SUCCESS")
            return True
        else:
            print_status(f"✗ Elasticsearch connection failed: {response.status_code} - {response.text}", "ERROR")
            return False
            
    except Exception as e:
        print_status(f"✗ Elasticsearch connection error: {str(e)}", "ERROR")
        return False

def delete_index_if_exists():
    """Delete the index if it exists"""
    try:
        headers = {
            'Authorization': f'ApiKey {ELASTIC_API_KEY}',
            'Content-Type': 'application/json'
        }
        
        # Check if index exists and delete it
        response = requests.head(f"{ELASTIC_URL}/{INDEX_NAME}", headers=headers)
        
        if response.status_code == 200:
            # Index exists, delete it
            delete_response = requests.delete(f"{ELASTIC_URL}/{INDEX_NAME}", headers=headers)
            if delete_response.status_code in [200, 404]:  # 404 means already deleted
                print_status(f"✓ Existing index '{INDEX_NAME}' deleted successfully", "SUCCESS")
            else:
                print_status(f"✗ Failed to delete existing index: {delete_response.status_code} - {delete_response.text}", "ERROR")
                return False
        else:
            print_status(f"✓ Index '{INDEX_NAME}' does not exist, no deletion needed", "INFO")
            
        return True
        
    except Exception as e:
        print_status(f"✗ Error deleting index: {str(e)}", "ERROR")
        return False

def create_index():
    """Create index with proper mapping"""
    try:
        headers = {
            'Authorization': f'ApiKey {ELASTIC_API_KEY}',
            'Content-Type': 'application/json'
        }
        
        # Create index with mapping
        mapping = {
            "mappings": {
                "properties": {
                    "patient_name": {
                        "type": "text",
                        "fields": {
                            "keyword": {
                                "type": "keyword"
                            }
                        }
                    },
                    "date_of_visit": {
                        "type": "date",
                        "format": "yyyy-MM-dd"
                    },
                    "patient_complaint": {
                        "type": "text"
                    },
                    "diagnosis": {
                        "type": "text",
                        "fields": {
                            "keyword": {
                                "type": "keyword"
                            }
                        }
                    },
                    "doctor_notes": {
                        "type": "text"
                    },
                    "drugs_prescribed": {
                        "type": "keyword"
                    },
                    "patient_age_at_visit": {
                        "type": "integer"
                    },
                    "ingestion_timestamp": {
                        "type": "date"
                    }
                }
            }
        }
        
        response = requests.put(f"{ELASTIC_URL}/{INDEX_NAME}", 
                              headers=headers, 
                              data=json.dumps(mapping))
        
        if response.status_code in [200, 201]:
            print_status(f"✓ Index '{INDEX_NAME}' created successfully", "SUCCESS")
            return True
        else:
            print_status(f"✗ Failed to create index: {response.status_code} - {response.text}", "ERROR")
            return False
        
    except Exception as e:
        print_status(f"✗ Error creating index: {str(e)}", "ERROR")
        return False

def load_and_ingest_data():
    """Load data from file and ingest into Elasticsearch"""
    try:
        # Read and parse JSON data
        with open(DATA_FILE, 'r') as f:
            data = json.load(f)
        
        print_status(f"✓ Loaded {len(data)} records from {DATA_FILE}", "SUCCESS")
        
        headers = {
            'Authorization': f'ApiKey {ELASTIC_API_KEY}',
            'Content-Type': 'application/json'
        }
        
        # Prepare bulk request
        bulk_data = []
        current_time = datetime.utcnow().isoformat()
        
        for i, record in enumerate(data):
            # Add ingestion timestamp
            record['ingestion_timestamp'] = current_time
            
            # Create bulk index operation
            index_op = {
                "index": {
                    "_index": INDEX_NAME,
                    "_id": f"patient_{i+1}_{record['date_of_visit'].replace('-', '')}"
                }
            }
            
            bulk_data.append(json.dumps(index_op))
            bulk_data.append(json.dumps(record))
        
        # Send bulk request
        bulk_payload = '\n'.join(bulk_data) + '\n'
        
        print_status("Uploading data to Elasticsearch...", "INFO")
        
        response = requests.post(f"{ELASTIC_URL}/_bulk", 
                               headers=headers, 
                               data=bulk_payload)
        
        if response.status_code == 200:
            result = response.json()
            
            if result.get('errors', False):
                print_status("⚠ Some documents failed to index:", "WARNING")
                for item in result.get('items', []):
                    if 'index' in item and 'error' in item['index']:
                        print_status(f"  - {item['index']['error']}", "WARNING")
            else:
                print_status(f"✓ Successfully indexed {len(data)} documents", "SUCCESS")
                
            # Print summary
            indexed = sum(1 for item in result.get('items', []) 
                         if 'index' in item and item['index'].get('status') in [200, 201])
            print_status(f"✓ Total documents indexed: {indexed}/{len(data)}", "SUCCESS")
            
        else:
            print_status(f"✗ Bulk indexing failed: {response.status_code} - {response.text}", "ERROR")
            return False
            
        return True
        
    except FileNotFoundError:
        print_status(f"✗ Data file '{DATA_FILE}' not found", "ERROR")
        return False
    except json.JSONDecodeError as e:
        print_status(f"✗ Invalid JSON in data file: {str(e)}", "ERROR")
        return False
    except Exception as e:
        print_status(f"✗ Error during data ingestion: {str(e)}", "ERROR")
        return False

def main():
    """Main execution function"""
    print_status("Starting Elasticsearch ingestion process...", "INFO")
    
    # Test connection
    if not test_elasticsearch_connection():
        sys.exit(1)
    
    # Delete existing index if it exists
    if not delete_index_if_exists():
        sys.exit(1)
    
    # Create new index
    if not create_index():
        sys.exit(1)
    
    # Load and ingest data
    if not load_and_ingest_data():
        sys.exit(1)
    
    print_status("✓ Data ingestion completed successfully!", "SUCCESS")
    print_status(f"You can now search your data at: {ELASTIC_URL}/{INDEX_NAME}/_search", "INFO")

if __name__ == "__main__":
    main()
EOF

# Check if Python script executed successfully
if [ $? -eq 0 ]; then
    echo -e "${GREEN}✓ Clinical patient data successfully ingested into Elasticsearch!${NC}"
    echo -e "${YELLOW}You can now query your data using the Elasticsearch API or Kibana.${NC}"
else
    echo -e "${RED}✗ Data ingestion failed. Please check the error messages above.${NC}"
    exit 1
fi
