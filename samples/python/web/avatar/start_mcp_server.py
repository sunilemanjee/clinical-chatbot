#!/usr/bin/env python3
"""
Start MCP Server for Clinical Assistant
Script to start the MCP server with proper configuration
"""

import asyncio
import sys
import os
import logging
from pathlib import Path

# Add the current directory to Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from mcp_server import ClinicalMCPServer
from mcp_config import validate_and_get_config

def setup_logging():
    """Setup logging configuration"""
    config = validate_and_get_config()
    logging_config = config.get_logging_config()
    
    logging.basicConfig(
        level=getattr(logging, logging_config["level"]),
        format=logging_config["format"],
        handlers=[
            logging.FileHandler(logging_config["file"]),
            logging.StreamHandler(sys.stdout)
        ]
    )
    
    logger = logging.getLogger(__name__)
    logger.info("MCP Server logging configured")
    return logger

async def main():
    """Main function to start the MCP server"""
    logger = setup_logging()
    
    try:
        # Validate configuration
        config = validate_and_get_config()
        logger.info("Configuration validated successfully")
        
        # Create and start the MCP server
        server = ClinicalMCPServer()
        logger.info("MCP Server created successfully")
        
        # Start the server
        logger.info("Starting MCP Server...")
        await server.run()
        
    except Exception as e:
        logger.error(f"Failed to start MCP server: {e}")
        sys.exit(1)

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\nMCP Server stopped by user")
    except Exception as e:
        print(f"Error starting MCP server: {e}")
        sys.exit(1)
