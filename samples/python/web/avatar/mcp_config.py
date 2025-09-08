#!/usr/bin/env python3
"""
MCP Configuration for Clinical Assistant
Configuration settings for MCP server and client
"""

import os
from typing import Dict, List, Any

class MCPConfig:
    """Configuration class for MCP Clinical Assistant"""
    
    def __init__(self):
        self.server_name = "clinical-assistant"
        self.server_version = "1.0.0"
        self.server_description = "Clinical Assistant MCP Server for patient data management"
        
        # MCP Server Configuration
        self.server_config = {
            "name": self.server_name,
            "version": self.server_version,
            "description": self.server_description,
            "capabilities": {
                "tools": True,
                "resources": True,
                "prompts": False,
                "logging": True
            }
        }
        
        # Tool Definitions
        self.tools = [
            {
                "name": "get_patient_data",
                "description": "Retrieve raw patient medical records from Elasticsearch",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "patient_name": {
                            "type": "string",
                            "description": "Full name of the patient to retrieve records for"
                        }
                    },
                    "required": ["patient_name"]
                }
            },
            {
                "name": "summarize_patient_data",
                "description": "Analyze and summarize patient medical data",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "patient_data": {
                            "type": "object",
                            "description": "Raw patient data from get_patient_data"
                        },
                        "summary_type": {
                            "type": "string",
                            "enum": ["comprehensive", "medication_focus", "recent_visits", "risk_assessment", "treatment_history"],
                            "description": "Type of summary to generate",
                            "default": "comprehensive"
                        }
                    },
                    "required": ["patient_data", "summary_type"]
                }
            },
            {
                "name": "get_patient_summary",
                "description": "Convenience tool that combines data retrieval and summarization",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "patient_name": {
                            "type": "string",
                            "description": "Full name of the patient"
                        },
                        "summary_type": {
                            "type": "string",
                            "enum": ["comprehensive", "medication_focus", "recent_visits", "risk_assessment", "treatment_history"],
                            "description": "Type of summary to generate",
                            "default": "comprehensive"
                        }
                    },
                    "required": ["patient_name"]
                }
            },
            {
                "name": "check_medication_interactions",
                "description": "Check for potential drug interactions between medications",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "new_medications": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "List of new medications being considered"
                        },
                        "existing_medications": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "List of patient's current medications"
                        }
                    },
                    "required": ["new_medications", "existing_medications"]
                }
            }
        ]
        
        # Resource Definitions
        self.resources = [
            {
                "uri": "patient_database://elasticsearch",
                "name": "Patient Database",
                "description": "Elasticsearch database containing patient medical records",
                "mimeType": "application/json"
            },
            {
                "uri": "drug_interactions://database",
                "name": "Drug Interactions Database",
                "description": "Database of known drug interactions",
                "mimeType": "application/json"
            }
        ]
        
        # Streaming Configuration
        self.streaming_config = {
            "enabled": True,
            "chunk_size": 1024,
            "timeout": 30,
            "max_retries": 3,
            "retry_delay": 1.0
        }
        
        # Logging Configuration
        self.logging_config = {
            "level": os.getenv("MCP_LOG_LEVEL", "INFO"),
            "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
            "file": os.getenv("MCP_LOG_FILE", "mcp_server.log")
        }
        
        # Performance Configuration
        self.performance_config = {
            "max_concurrent_requests": int(os.getenv("MCP_MAX_CONCURRENT", "10")),
            "request_timeout": int(os.getenv("MCP_REQUEST_TIMEOUT", "30")),
            "cache_size": int(os.getenv("MCP_CACHE_SIZE", "100")),
            "cache_ttl": int(os.getenv("MCP_CACHE_TTL", "300"))  # 5 minutes
        }
        
        # Security Configuration
        self.security_config = {
            "enable_auth": os.getenv("MCP_ENABLE_AUTH", "false").lower() == "true",
            "api_key": os.getenv("MCP_API_KEY"),
            "allowed_origins": os.getenv("MCP_ALLOWED_ORIGINS", "*").split(","),
            "rate_limit": int(os.getenv("MCP_RATE_LIMIT", "100"))  # requests per minute
        }
    
    def get_server_config(self) -> Dict[str, Any]:
        """Get server configuration"""
        return self.server_config
    
    def get_tools(self) -> List[Dict[str, Any]]:
        """Get tool definitions"""
        return self.tools
    
    def get_resources(self) -> List[Dict[str, Any]]:
        """Get resource definitions"""
        return self.resources
    
    def get_streaming_config(self) -> Dict[str, Any]:
        """Get streaming configuration"""
        return self.streaming_config
    
    def get_logging_config(self) -> Dict[str, Any]:
        """Get logging configuration"""
        return self.logging_config
    
    def get_performance_config(self) -> Dict[str, Any]:
        """Get performance configuration"""
        return self.performance_config
    
    def get_security_config(self) -> Dict[str, Any]:
        """Get security configuration"""
        return self.security_config
    
    def validate_config(self) -> bool:
        """Validate configuration"""
        try:
            # Check required environment variables
            required_env_vars = [
                "ELASTIC_URL",
                "ELASTIC_API_KEY", 
                "ELASTIC_INDEX_NAME"
            ]
            
            for var in required_env_vars:
                if not os.getenv(var):
                    print(f"Warning: Required environment variable {var} not set")
            
            # Validate tool definitions
            for tool in self.tools:
                if not tool.get("name") or not tool.get("description"):
                    print(f"Invalid tool definition: {tool}")
                    return False
            
            # Validate resource definitions
            for resource in self.resources:
                if not resource.get("uri") or not resource.get("name"):
                    print(f"Invalid resource definition: {resource}")
                    return False
            
            return True
            
        except Exception as e:
            print(f"Configuration validation error: {e}")
            return False

# Global configuration instance
config = MCPConfig()

def get_config() -> MCPConfig:
    """Get the global configuration instance"""
    return config

def validate_and_get_config() -> MCPConfig:
    """Validate configuration and return instance"""
    if config.validate_config():
        return config
    else:
        raise ValueError("Invalid MCP configuration")
