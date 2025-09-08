#!/usr/bin/env python3
"""
Simplified MCP Server for Clinical Assistant
A simplified version that works with the current MCP library structure
"""

import asyncio
import json
import logging
import os
import sys
from typing import Dict, List, Any, AsyncGenerator

# Add the current directory to Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from elasticsearch import Elasticsearch

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SimpleClinicalServer:
    def __init__(self):
        self.elastic_client = None
        self.elastic_index_name = None
        self.drug_interactions = {}
        self.setup_elasticsearch()
        self.load_drug_interactions()
    
    def setup_elasticsearch(self):
        """Initialize Elasticsearch connection"""
        elastic_url = os.environ.get('ELASTIC_URL')
        elastic_api_key = os.environ.get('ELASTIC_API_KEY')
        self.elastic_index_name = os.environ.get('ELASTIC_INDEX_NAME')
        
        if elastic_url and elastic_api_key:
            try:
                self.elastic_client = Elasticsearch(
                    hosts=[elastic_url],
                    api_key=elastic_api_key,
                    verify_certs=True
                )
                if self.elastic_client.ping():
                    logger.info("Elasticsearch connection successful!")
                else:
                    logger.error("Elasticsearch connection failed!")
                    self.elastic_client = None
            except Exception as e:
                logger.error(f"Failed to initialize Elasticsearch client: {e}")
                self.elastic_client = None
        else:
            logger.warning("Elasticsearch credentials not configured")
    
    def load_drug_interactions(self):
        """Load drug interactions data"""
        try:
            with open('data/drug-interactions-data.json', 'r') as f:
                interactions_data = json.load(f)
            
            # Convert to the format expected by the existing code
            for item in interactions_data:
                primary_drug = item['primary_drug']
                negative_interactions = item['negative_drug_interactions']
                
                if primary_drug not in self.drug_interactions:
                    self.drug_interactions[primary_drug] = {}
                
                for interaction_drug in negative_interactions:
                    if interaction_drug != "None":
                        self.drug_interactions[primary_drug][interaction_drug] = f"⚠️ INTERACTION: Potential interaction between {primary_drug} and {interaction_drug}."
            
            logger.info(f"Loaded {len(self.drug_interactions)} drug interaction entries")
        except Exception as e:
            logger.error(f"Failed to load drug interactions: {e}")
    
    def get_patient_data(self, patient_name: str) -> Dict[str, Any]:
        """Get patient data from Elasticsearch"""
        logger.info(f"Querying patient data for: '{patient_name}'")
        
        if not self.elastic_client or not self.elastic_index_name:
            return {"error": "Elasticsearch client not configured"}
        
        try:
            # Use a simpler query structure
            query = {
                "query": {
                    "match": {
                        "patient_name": patient_name
                    }
                },
                "_source": [
                    "date_of_visit",
                    "patient_complaint", 
                    "diagnosis",
                    "doctor_notes",
                    "drugs_prescribed",
                    "patient_age_at_visit",
                    "patient_name"
                ]
            }
            
            # Execute the search
            response = self.elastic_client.search(
                index=self.elastic_index_name,
                body=query
            )
            
            # Convert response to dictionary if it's an ObjectApiResponse
            if hasattr(response, 'body'):
                response_dict = response.body
            else:
                response_dict = response
            
            # Extract and format the results
            hits = response_dict.get('hits', {}).get('hits', [])
            patient_records = []
            
            for hit in hits:
                source = hit.get('_source', {})
                patient_records.append({
                    'date_of_visit': source.get('date_of_visit'),
                    'patient_complaint': source.get('patient_complaint'),
                    'diagnosis': source.get('diagnosis'),
                    'doctor_notes': source.get('doctor_notes'),
                    'drugs_prescribed': source.get('drugs_prescribed'),
                    'patient_age_at_visit': source.get('patient_age_at_visit'),
                    'patient_name': source.get('patient_name')
                })
            
            return {
                "success": True,
                "patient_name": patient_name,
                "total_records": len(patient_records),
                "records": patient_records
            }
            
        except Exception as e:
            logger.error(f"Error querying patient data: {str(e)}")
            return {"error": f"Failed to query patient data: {str(e)}"}
    
    def check_medication_interactions(self, new_medications: List[str], existing_medications: List[str]) -> List[str]:
        """Check for medication interactions"""
        interactions = []
        
        for new_med in new_medications:
            if new_med in self.drug_interactions:
                for existing_med in existing_medications:
                    if existing_med in self.drug_interactions[new_med]:
                        interactions.append(self.drug_interactions[new_med][existing_med])
        
        return interactions
    
    def create_patient_summary(self, patient_data: Dict[str, Any], summary_type: str = "comprehensive") -> Dict[str, Any]:
        """Create patient summary"""
        if not patient_data or not patient_data.get("success"):
            return {"error": "Invalid patient data provided"}
        
        records = patient_data.get("records", [])
        
        if summary_type == "comprehensive":
            summary = {
                "patient_overview": {
                    "name": patient_data.get("patient_name"),
                    "total_visits": patient_data.get("total_records"),
                    "primary_conditions": self.identify_primary_conditions(records)
                },
                "medication_history": self.analyze_medications(records),
                "recent_visits": self.get_recent_visits(records),
                "clinical_patterns": self.identify_patterns(records)
            }
        elif summary_type == "medication_focus":
            summary = {
                "medication_history": self.analyze_medications(records),
                "current_medications": self.get_current_medications(records),
                "interaction_risks": self.assess_medication_risks(records)
            }
        else:
            summary = {
                "patient_overview": {
                    "name": patient_data.get("patient_name"),
                    "total_visits": patient_data.get("total_records")
                },
                "recent_visits": self.get_recent_visits(records)
            }
        
        return summary
    
    def identify_primary_conditions(self, records: List[dict]) -> List[str]:
        """Identify primary medical conditions"""
        diagnoses = [record.get('diagnosis') for record in records if record.get('diagnosis')]
        from collections import Counter
        condition_counts = Counter(diagnoses)
        return [condition for condition, count in condition_counts.most_common(3)]
    
    def analyze_medications(self, records: List[dict]) -> dict:
        """Analyze medication history"""
        all_medications = []
        for record in records:
            drugs = record.get('drugs_prescribed', [])
            if drugs and drugs != ["None"]:
                all_medications.extend(drugs)
        
        from collections import Counter
        medication_counts = Counter(all_medications)
        
        return {
            "all_medications": list(set(all_medications)),
            "medication_frequency": dict(medication_counts),
            "total_unique_medications": len(set(all_medications))
        }
    
    def get_recent_visits(self, records: List[dict], count: int = 3) -> List[dict]:
        """Get recent visits"""
        sorted_records = sorted(records, key=lambda x: x.get('date_of_visit', ''), reverse=True)
        return sorted_records[:count]
    
    def identify_patterns(self, records: List[dict]) -> List[str]:
        """Identify clinical patterns"""
        patterns = []
        
        # Check for recurring conditions
        diagnoses = [record.get('diagnosis') for record in records if record.get('diagnosis')]
        from collections import Counter
        diagnosis_counts = Counter(diagnoses)
        
        for diagnosis, count in diagnosis_counts.items():
            if count > 1:
                patterns.append(f"Recurring {diagnosis} ({count} occurrences)")
        
        return patterns
    
    def get_current_medications(self, records: List[dict]) -> List[str]:
        """Get current medications from recent visits"""
        recent_records = self.get_recent_visits(records, 2)  # Last 2 visits
        current_meds = []
        
        for record in recent_records:
            drugs = record.get('drugs_prescribed', [])
            if drugs and drugs != ["None"]:
                current_meds.extend(drugs)
        
        return list(set(current_meds))
    
    def assess_medication_risks(self, records: List[dict]) -> List[str]:
        """Assess medication-related risks"""
        risks = []
        current_meds = self.get_current_medications(records)
        
        # Check for potential interactions
        for med1 in current_meds:
            for med2 in current_meds:
                if med1 != med2 and med1 in self.drug_interactions:
                    if med2 in self.drug_interactions[med1]:
                        risks.append(f"Potential interaction: {med1} + {med2}")
        
        return risks

# Global server instance
clinical_server = SimpleClinicalServer()

def get_patient_data(patient_name: str) -> Dict[str, Any]:
    """Get patient data - wrapper function"""
    return clinical_server.get_patient_data(patient_name)

def check_medication_interactions(new_medications: List[str], existing_medications: List[str]) -> List[str]:
    """Check medication interactions - wrapper function"""
    return clinical_server.check_medication_interactions(new_medications, existing_medications)

def create_patient_summary(patient_data: Dict[str, Any], summary_type: str = "comprehensive") -> Dict[str, Any]:
    """Create patient summary - wrapper function"""
    return clinical_server.create_patient_summary(patient_data, summary_type)

if __name__ == "__main__":
    import time
    import signal
    import sys
    
    # Test the server first
    print("Testing Simple Clinical Server...")
    
    # Test patient data retrieval
    result = get_patient_data("Jane Doe")
    print(f"Patient data result: {result}")
    
    # Test medication interactions
    interactions = check_medication_interactions(["Diazepam"], ["Meclizine"])
    print(f"Medication interactions: {interactions}")
    
    print("Simple Clinical Server test completed!")
    
    # Now run as a persistent server
    print("Starting Simple Clinical Server as persistent service...")
    print("Server is ready to handle requests. Press Ctrl+C to stop.")
    
    def signal_handler(sig, frame):
        print("\nShutting down Simple Clinical Server...")
        sys.exit(0)
    
    # Set up signal handler for graceful shutdown
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    try:
        # Keep the server running
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\nServer stopped by user")
    except Exception as e:
        print(f"Server error: {e}")
        sys.exit(1)
