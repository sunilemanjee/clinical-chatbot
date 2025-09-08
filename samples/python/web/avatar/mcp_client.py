#!/usr/bin/env python3
"""
MCP Client for Clinical Assistant
Handles communication with MCP server and streaming responses
"""

import asyncio
import json
import logging
from typing import Dict, List, Any, AsyncGenerator, Optional
import uuid
import os
import sys

# Add the current directory to Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from mcp.client.session import ClientSession
from mcp.client.stdio import stdio_client

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class StreamingClinicalMCPClient:
    def __init__(self):
        self.session = None
        self.connected = False
        self.available_tools = []
    
    async def connect(self, server_command: List[str] = None):
        """Connect to MCP server"""
        try:
            if server_command is None:
                # Default server command
                server_command = ["python", "mcp_server.py"]
            
            self.session = ClientSession("clinical-assistant")
            await self.session.connect(server_command)
            self.connected = True
            
            # Get available tools
            await self.discover_tools()
            logger.info(f"Connected to MCP server with {len(self.available_tools)} tools")
            
        except Exception as e:
            logger.error(f"Failed to connect to MCP server: {e}")
            self.connected = False
            raise
    
    async def discover_tools(self):
        """Discover available tools from MCP server"""
        try:
            if self.session:
                # Get list of available tools
                tools_response = await self.session.list_tools()
                self.available_tools = [tool.name for tool in tools_response.tools]
                logger.info(f"Discovered tools: {self.available_tools}")
        except Exception as e:
            logger.error(f"Failed to discover tools: {e}")
            self.available_tools = []
    
    async def stream_chat(self, messages: List[Dict[str, str]], tools: List[str] = None) -> AsyncGenerator[Dict[str, Any], None]:
        """Stream chat responses with tool calls"""
        if not self.connected:
            raise Exception("MCP client not connected")
        
        try:
            # Filter tools to only include available ones
            if tools:
                available_tools = [tool for tool in tools if tool in self.available_tools]
            else:
                available_tools = self.available_tools
            
            # Start chat completion with streaming
            async for chunk in self.session.stream_chat_completion(
                messages=messages,
                tools=available_tools
            ):
                yield chunk
                
        except Exception as e:
            logger.error(f"Error in stream_chat: {e}")
            yield {"type": "error", "content": f"Error: {str(e)}"}
    
    async def handle_user_query_streaming(self, user_query: str, client_id: uuid.UUID) -> AsyncGenerator[str, None]:
        """Handle user query with streaming responses"""
        if not self.connected:
            yield "❌ MCP client not connected. Please check server status."
            return
        
        try:
            messages = [{"role": "user", "content": user_query}]
            
            async for chunk in self.stream_chat(messages):
                if chunk.get("type") == "text":
                    # Stream text response
                    yield chunk.get("content", "")
                elif chunk.get("type") == "tool_call":
                    # Handle tool call
                    tool_result = await self.execute_tool_call(chunk)
                    if tool_result:
                        yield f"🔍 {tool_result.get('message', 'Processing...')}"
                elif chunk.get("type") == "tool_result":
                    # Stream tool results
                    result_data = chunk.get("result", {})
                    if result_data.get("status") == "complete":
                        yield f"✅ {result_data.get('message', 'Completed')}"
                    elif result_data.get("status") == "error":
                        yield f"❌ {result_data.get('message', 'Error occurred')}"
                elif chunk.get("type") == "error":
                    yield f"❌ {chunk.get('content', 'Unknown error')}"
                    
        except Exception as e:
            logger.error(f"Error handling user query: {e}")
            yield f"❌ Error processing request: {str(e)}"
    
    async def execute_tool_call(self, tool_call: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Execute tool call and return result"""
        try:
            tool_name = tool_call.get("name")
            tool_args = tool_call.get("arguments", {})
            
            if tool_name == "get_patient_data":
                return await self.get_patient_data(tool_args.get("patient_name"))
            elif tool_name == "summarize_patient_data":
                return await self.summarize_patient_data(
                    tool_args.get("patient_data"),
                    tool_args.get("summary_type", "comprehensive")
                )
            elif tool_name == "get_patient_summary":
                return await self.get_patient_summary(
                    tool_args.get("patient_name"),
                    tool_args.get("summary_type", "comprehensive")
                )
            elif tool_name == "check_medication_interactions":
                return await self.check_medication_interactions(
                    tool_args.get("new_medications", []),
                    tool_args.get("existing_medications", [])
                )
            else:
                logger.warning(f"Unknown tool: {tool_name}")
                return {"message": f"Unknown tool: {tool_name}"}
                
        except Exception as e:
            logger.error(f"Error executing tool call: {e}")
            return {"message": f"Error executing tool: {str(e)}"}
    
    async def get_patient_data(self, patient_name: str) -> Dict[str, Any]:
        """Get patient data using MCP tool"""
        try:
            result = await self.session.call_tool("get_patient_data", {"patient_name": patient_name})
            return result
        except Exception as e:
            logger.error(f"Error getting patient data: {e}")
            return {"status": "error", "message": str(e)}
    
    async def summarize_patient_data(self, patient_data: dict, summary_type: str = "comprehensive") -> Dict[str, Any]:
        """Summarize patient data using MCP tool"""
        try:
            result = await self.session.call_tool("summarize_patient_data", {
                "patient_data": patient_data,
                "summary_type": summary_type
            })
            return result
        except Exception as e:
            logger.error(f"Error summarizing patient data: {e}")
            return {"status": "error", "message": str(e)}
    
    async def get_patient_summary(self, patient_name: str, summary_type: str = "comprehensive") -> Dict[str, Any]:
        """Get patient summary using MCP tool"""
        try:
            result = await self.session.call_tool("get_patient_summary", {
                "patient_name": patient_name,
                "summary_type": summary_type
            })
            return result
        except Exception as e:
            logger.error(f"Error getting patient summary: {e}")
            return {"status": "error", "message": str(e)}
    
    async def check_medication_interactions(self, new_medications: List[str], existing_medications: List[str]) -> Dict[str, Any]:
        """Check medication interactions using MCP tool"""
        try:
            result = await self.session.call_tool("check_medication_interactions", {
                "new_medications": new_medications,
                "existing_medications": existing_medications
            })
            return result
        except Exception as e:
            logger.error(f"Error checking medication interactions: {e}")
            return {"status": "error", "message": str(e)}
    
    async def disconnect(self):
        """Disconnect from MCP server"""
        try:
            if self.session and self.connected:
                await self.session.disconnect()
                self.connected = False
                logger.info("Disconnected from MCP server")
        except Exception as e:
            logger.error(f"Error disconnecting from MCP server: {e}")

# Global MCP client instance
mcp_client = StreamingClinicalMCPClient()

async def initialize_mcp_client():
    """Initialize the global MCP client"""
    try:
        await mcp_client.connect()
        return True
    except Exception as e:
        logger.error(f"Failed to initialize MCP client: {e}")
        return False

async def get_mcp_client() -> StreamingClinicalMCPClient:
    """Get the global MCP client instance"""
    if not mcp_client.connected:
        await initialize_mcp_client()
    return mcp_client

# Utility functions for integration with Flask app
async def handle_user_query_with_mcp(user_query: str, client_id: uuid.UUID) -> AsyncGenerator[str, None]:
    """Handle user query using MCP client"""
    client = await get_mcp_client()
    async for chunk in client.handle_user_query_streaming(user_query, client_id):
        yield chunk

async def get_patient_data_with_mcp(patient_name: str) -> Dict[str, Any]:
    """Get patient data using MCP"""
    client = await get_mcp_client()
    return await client.get_patient_data(patient_name)

async def get_patient_summary_with_mcp(patient_name: str, summary_type: str = "comprehensive") -> Dict[str, Any]:
    """Get patient summary using MCP"""
    client = await get_mcp_client()
    return await client.get_patient_summary(patient_name, summary_type)

async def check_medication_interactions_with_mcp(new_medications: List[str], existing_medications: List[str]) -> Dict[str, Any]:
    """Check medication interactions using MCP"""
    client = await get_mcp_client()
    return await client.check_medication_interactions(new_medications, existing_medications)
