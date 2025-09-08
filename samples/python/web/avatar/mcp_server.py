#!/usr/bin/env python3
"""
MCP Server for Clinical Assistant
Provides streaming HTTP tools for patient data retrieval and analysis
"""

import asyncio
import json
import logging
from typing import Dict, List, Any, AsyncGenerator
from datetime import datetime, timedelta
import os
import sys

# Add the current directory to Python path to import from app.py
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from elasticsearch import Elasticsearch
from mcp.server import Server
from mcp.server.stdio import stdio_server

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ClinicalMCPServer:
    def __init__(self):
        self.server = Server("clinical-assistant")
        self.elastic_client = None
        self.elastic_index_name = None
        self.drug_interactions = {}
        self.setup_elasticsearch()
        self.load_drug_interactions()
        self.setup_tools()
    
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
    
    def setup_tools(self):
        """Setup MCP tools"""
        
        @self.server.tool("get_patient_data")
        async def get_patient_data(patient_name: str) -> str:
            """
            Retrieve raw patient medical records from Elasticsearch.
            Returns complete, unprocessed patient data.
            
            Args:
                patient_name: Full name of the patient to retrieve records for
            """
            async for chunk in self.stream_patient_data(patient_name):
                yield chunk
        
        @self.server.tool("summarize_patient_data")
        async def summarize_patient_data(
            patient_data: dict,
            summary_type: str = "comprehensive"
        ) -> str:
            """
            Analyze and summarize patient medical data.
            
            Args:
                patient_data: Raw patient data from get_patient_data
                summary_type: Type of summary (comprehensive, medication_focus, 
                             recent_visits, risk_assessment, treatment_history)
            """
            async for chunk in self.stream_patient_summary(patient_data, summary_type):
                yield chunk
        
        @self.server.tool("get_patient_summary")
        async def get_patient_summary(
            patient_name: str,
            summary_type: str = "comprehensive"
        ) -> str:
            """
            Convenience tool that combines data retrieval and summarization.
            Use this for one-step patient summary requests.
            
            Args:
                patient_name: Full name of the patient
                summary_type: Type of summary to generate
            """
            # First get the data
            patient_data = None
            async for chunk in self.stream_patient_data(patient_name):
                if chunk.get("status") == "complete":
                    patient_data = chunk.get("data")
                    break
            
            if patient_data and patient_data.get("success"):
                # Then summarize it
                async for chunk in self.stream_patient_summary(patient_data, summary_type):
                    yield chunk
            else:
                yield {"status": "error", "message": "Failed to retrieve patient data"}
        
        @self.server.tool("check_medication_interactions")
        async def check_medication_interactions(
            new_medications: List[str],
            existing_medications: List[str]
        ) -> str:
            """
            Check for potential drug interactions between medications.
            
            Args:
                new_medications: List of new medications being considered
                existing_medications: List of patient's current medications
            """
            async for chunk in self.stream_drug_interaction_check(new_medications, existing_medications):
                yield chunk
    
    async def stream_patient_data(self, patient_name: str) -> AsyncGenerator[Dict[str, Any], None]:
        """Stream patient data retrieval from Elasticsearch"""
        logger.info(f"Querying patient data for: '{patient_name}'")
        
        yield {"status": "searching", "message": f"Looking up patient: {patient_name}"}
        
        if not self.elastic_client or not self.elastic_index_name:
            yield {"status": "error", "message": "Elasticsearch client not configured"}
            return
        
        try:
            # Use a simpler query structure that's more compatible with standard Elasticsearch
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
            
            yield {"status": "querying", "message": "Searching Elasticsearch database..."}
            
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
            
            yield {"status": "processing", "message": f"Processing {len(hits)} records..."}
            
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
            
            result = {
                "success": True,
                "patient_name": patient_name,
                "total_records": len(patient_records),
                "records": patient_records
            }
            
            yield {"status": "complete", "data": result, "message": f"Found {len(patient_records)} records for {patient_name}"}
            
        except Exception as e:
            logger.error(f"Error querying patient data: {str(e)}")
            yield {"status": "error", "message": f"Failed to query patient data: {str(e)}"}
    
    async def stream_patient_summary(self, patient_data: dict, summary_type: str) -> AsyncGenerator[Dict[str, Any], None]:
        """Stream patient data summarization"""
        logger.info(f"Creating {summary_type} summary for patient data")
        
        yield {"status": "analyzing", "message": f"Analyzing patient data for {summary_type} summary..."}
        
        if not patient_data or not patient_data.get("success"):
            yield {"status": "error", "message": "Invalid patient data provided"}
            return
        
        try:
            if summary_type == "comprehensive":
                summary = await self.create_comprehensive_summary(patient_data)
            elif summary_type == "medication_focus":
                summary = await self.create_medication_summary(patient_data)
            elif summary_type == "recent_visits":
                summary = await self.create_recent_visits_summary(patient_data)
            elif summary_type == "risk_assessment":
                summary = await self.create_risk_assessment(patient_data)
            elif summary_type == "treatment_history":
                summary = await self.create_treatment_history(patient_data)
            else:
                summary = await self.create_comprehensive_summary(patient_data)
            
            yield {"status": "complete", "data": summary, "message": f"Generated {summary_type} summary"}
            
        except Exception as e:
            logger.error(f"Error creating summary: {str(e)}")
            yield {"status": "error", "message": f"Failed to create summary: {str(e)}"}
    
    async def create_comprehensive_summary(self, patient_data: dict) -> dict:
        """Create comprehensive patient summary"""
        records = patient_data.get("records", [])
        
        summary = {
            "patient_overview": {
                "name": patient_data.get("patient_name"),
                "total_visits": patient_data.get("total_records"),
                "age_range": self.calculate_age_range(records),
                "primary_conditions": self.identify_primary_conditions(records)
            },
            "medication_history": self.analyze_medications(records),
            "recent_visits": self.get_recent_visits(records),
            "clinical_patterns": self.identify_patterns(records),
            "risk_factors": self.assess_risks(records)
        }
        return summary
    
    async def create_medication_summary(self, patient_data: dict) -> dict:
        """Create medication-focused summary"""
        records = patient_data.get("records", [])
        
        summary = {
            "medication_history": self.analyze_medications(records),
            "current_medications": self.get_current_medications(records),
            "medication_timeline": self.create_medication_timeline(records),
            "interaction_risks": self.assess_medication_risks(records)
        }
        return summary
    
    async def create_recent_visits_summary(self, patient_data: dict) -> dict:
        """Create recent visits summary"""
        records = patient_data.get("records", [])
        
        # Sort by date and get recent visits
        sorted_records = sorted(records, key=lambda x: x.get('date_of_visit', ''), reverse=True)
        recent_records = sorted_records[:3]  # Last 3 visits
        
        summary = {
            "recent_visits": recent_records,
            "visit_frequency": self.calculate_visit_frequency(records),
            "trending_conditions": self.identify_trending_conditions(records)
        }
        return summary
    
    async def create_risk_assessment(self, patient_data: dict) -> dict:
        """Create risk assessment summary"""
        records = patient_data.get("records", [])
        
        summary = {
            "health_risks": self.assess_risks(records),
            "medication_risks": self.assess_medication_risks(records),
            "chronic_conditions": self.identify_chronic_conditions(records),
            "preventive_recommendations": self.generate_preventive_recommendations(records)
        }
        return summary
    
    async def create_treatment_history(self, patient_data: dict) -> dict:
        """Create treatment history summary"""
        records = patient_data.get("records", [])
        
        summary = {
            "treatment_timeline": self.create_treatment_timeline(records),
            "treatment_effectiveness": self.assess_treatment_effectiveness(records),
            "ongoing_treatments": self.identify_ongoing_treatments(records),
            "treatment_recommendations": self.generate_treatment_recommendations(records)
        }
        return summary
    
    async def stream_drug_interaction_check(self, new_medications: List[str], existing_medications: List[str]) -> AsyncGenerator[Dict[str, Any], None]:
        """Stream drug interaction checking"""
        yield {"status": "checking", "message": "Analyzing medication interactions..."}
        
        interactions = []
        
        for new_med in new_medications:
            if new_med in self.drug_interactions:
                for existing_med in existing_medications:
                    if existing_med in self.drug_interactions[new_med]:
                        interactions.append(self.drug_interactions[new_med][existing_med])
        
        if interactions:
            yield {"status": "interactions_found", "data": interactions, "message": f"Found {len(interactions)} potential interactions"}
        else:
            yield {"status": "safe", "message": "No interactions detected"}
    
    # Helper methods for data analysis
    def calculate_age_range(self, records: List[dict]) -> str:
        """Calculate patient age range from records"""
        ages = [record.get('patient_age_at_visit') for record in records if record.get('patient_age_at_visit')]
        if ages:
            return f"{min(ages)}-{max(ages)} years"
        return "Unknown"
    
    def identify_primary_conditions(self, records: List[dict]) -> List[str]:
        """Identify primary medical conditions"""
        diagnoses = [record.get('diagnosis') for record in records if record.get('diagnosis')]
        # Count frequency and return most common
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
    
    def assess_risks(self, records: List[dict]) -> List[str]:
        """Assess health risks"""
        risks = []
        
        # Check for chronic conditions
        chronic_conditions = self.identify_chronic_conditions(records)
        if chronic_conditions:
            risks.extend([f"Chronic condition: {condition}" for condition in chronic_conditions])
        
        # Check for frequent visits
        if len(records) > 5:
            risks.append("High healthcare utilization")
        
        return risks
    
    def identify_chronic_conditions(self, records: List[dict]) -> List[str]:
        """Identify chronic conditions"""
        diagnoses = [record.get('diagnosis') for record in records if record.get('diagnosis')]
        from collections import Counter
        diagnosis_counts = Counter(diagnoses)
        
        chronic_conditions = []
        for diagnosis, count in diagnosis_counts.items():
            if count > 2:  # Appears in more than 2 visits
                chronic_conditions.append(diagnosis)
        
        return chronic_conditions
    
    def get_current_medications(self, records: List[dict]) -> List[str]:
        """Get current medications from recent visits"""
        recent_records = self.get_recent_visits(records, 2)  # Last 2 visits
        current_meds = []
        
        for record in recent_records:
            drugs = record.get('drugs_prescribed', [])
            if drugs and drugs != ["None"]:
                current_meds.extend(drugs)
        
        return list(set(current_meds))
    
    def create_medication_timeline(self, records: List[dict]) -> List[dict]:
        """Create medication timeline"""
        timeline = []
        for record in records:
            if record.get('drugs_prescribed') and record.get('drugs_prescribed') != ["None"]:
                timeline.append({
                    "date": record.get('date_of_visit'),
                    "medications": record.get('drugs_prescribed'),
                    "condition": record.get('diagnosis')
                })
        return sorted(timeline, key=lambda x: x.get('date', ''), reverse=True)
    
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
    
    def calculate_visit_frequency(self, records: List[dict]) -> str:
        """Calculate visit frequency"""
        if len(records) < 2:
            return "Insufficient data"
        
        # Parse dates and calculate frequency
        dates = []
        for record in records:
            date_str = record.get('date_of_visit')
            if date_str and date_str != "3-DAYS-AGO":
                try:
                    dates.append(datetime.strptime(date_str, "%Y-%m-%d"))
                except:
                    pass
        
        if len(dates) < 2:
            return "Insufficient date data"
        
        dates.sort()
        time_span = (dates[-1] - dates[0]).days
        if time_span > 0:
            frequency = len(dates) / (time_span / 365.25)  # visits per year
            return f"{frequency:.1f} visits per year"
        
        return "Unknown"
    
    def identify_trending_conditions(self, records: List[dict]) -> List[str]:
        """Identify trending conditions"""
        # Get recent vs older conditions
        sorted_records = sorted(records, key=lambda x: x.get('date_of_visit', ''), reverse=True)
        recent_count = len(sorted_records) // 2
        recent_records = sorted_records[:recent_count]
        older_records = sorted_records[recent_count:]
        
        recent_diagnoses = [r.get('diagnosis') for r in recent_records if r.get('diagnosis')]
        older_diagnoses = [r.get('diagnosis') for r in older_records if r.get('diagnosis')]
        
        from collections import Counter
        recent_counts = Counter(recent_diagnoses)
        older_counts = Counter(older_diagnoses)
        
        trending = []
        for diagnosis in recent_counts:
            recent_freq = recent_counts[diagnosis] / len(recent_records) if recent_records else 0
            older_freq = older_counts[diagnosis] / len(older_records) if older_records else 0
            
            if recent_freq > older_freq * 1.5:  # 50% increase
                trending.append(diagnosis)
        
        return trending
    
    def create_treatment_timeline(self, records: List[dict]) -> List[dict]:
        """Create treatment timeline"""
        timeline = []
        for record in records:
            timeline.append({
                "date": record.get('date_of_visit'),
                "condition": record.get('diagnosis'),
                "treatment": record.get('drugs_prescribed'),
                "notes": record.get('doctor_notes')
            })
        return sorted(timeline, key=lambda x: x.get('date', ''), reverse=True)
    
    def assess_treatment_effectiveness(self, records: List[dict]) -> List[str]:
        """Assess treatment effectiveness"""
        effectiveness = []
        
        # Look for recurring conditions that might indicate ineffective treatment
        chronic_conditions = self.identify_chronic_conditions(records)
        for condition in chronic_conditions:
            effectiveness.append(f"Ongoing treatment for {condition} - monitor effectiveness")
        
        return effectiveness
    
    def identify_ongoing_treatments(self, records: List[dict]) -> List[str]:
        """Identify ongoing treatments"""
        return self.get_current_medications(records)
    
    def generate_treatment_recommendations(self, records: List[dict]) -> List[str]:
        """Generate treatment recommendations"""
        recommendations = []
        
        # Based on patterns and risks
        patterns = self.identify_patterns(records)
        if patterns:
            recommendations.append("Consider preventive measures for recurring conditions")
        
        risks = self.assess_risks(records)
        if risks:
            recommendations.append("Monitor identified risk factors closely")
        
        return recommendations
    
    def generate_preventive_recommendations(self, records: List[dict]) -> List[str]:
        """Generate preventive recommendations"""
        recommendations = []
        
        # Based on conditions and patterns
        conditions = [r.get('diagnosis') for r in records if r.get('diagnosis')]
        if any('BPPV' in condition for condition in conditions):
            recommendations.append("Consider vestibular rehabilitation exercises")
        
        if any('GERD' in condition for condition in conditions):
            recommendations.append("Maintain dietary modifications for GERD management")
        
        return recommendations

async def main():
    """Run the MCP server"""
    server = ClinicalMCPServer()
    async with stdio_server() as (read_stream, write_stream):
        await server.run(read_stream, write_stream)

if __name__ == "__main__":
    asyncio.run(main())
