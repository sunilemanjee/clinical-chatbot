# MCP Clinical Assistant Implementation

This implementation adds Model Context Protocol (MCP) support to the clinical chatbot, enabling intelligent function calling for patient data retrieval and analysis.

## Overview

The MCP implementation provides:
- **Streaming HTTP support** for real-time responses
- **Intelligent function calling** for patient data operations
- **Modular architecture** with separate server and client components
- **Enhanced clinical intelligence** with advanced data analysis

## Architecture

```
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   Flask App     │    │   MCP Client     │    │   MCP Server    │
│   (Chat UI)     │◄──►│   (Integration)  │◄──►│   (Data Layer)  │
└─────────────────┘    └──────────────────┘    └─────────────────┘
                              │                         │
                              ▼                         ▼
                       ┌──────────────────┐    ┌─────────────────┐
                       │   Azure OpenAI   │    │  Elasticsearch  │
                       │   (LLM)          │    │  (Patient Data) │
                       └──────────────────┘    └─────────────────┘
```

## Components

### 1. MCP Server (`mcp_server.py`)
- **Purpose**: Provides clinical data tools via MCP protocol
- **Tools Available**:
  - `get_patient_data`: Retrieve raw patient records
  - `summarize_patient_data`: Analyze and summarize patient data
  - `get_patient_summary`: Combined data retrieval and summarization
  - `check_medication_interactions`: Check for drug interactions

### 2. MCP Client (`mcp_client.py`)
- **Purpose**: Handles communication with MCP server
- **Features**:
  - Streaming response support
  - Automatic tool discovery
  - Error handling and fallback mechanisms

### 3. Configuration (`mcp_config.py`)
- **Purpose**: Centralized configuration management
- **Features**:
  - Tool and resource definitions
  - Performance and security settings
  - Environment variable validation

### 4. Flask Integration (Updated `app.py`)
- **Purpose**: Integrates MCP with existing Flask application
- **Changes**:
  - Added function calling to LLM chat completions
  - Integrated MCP tools with conversation flow
  - Maintained backward compatibility

## Installation

### 1. Install MCP Dependencies
```bash
pip install -r requirements_mcp.txt
```

### 2. Environment Variables
Ensure these environment variables are set:
```bash
# Elasticsearch Configuration
ELASTIC_URL=your_elasticsearch_url
ELASTIC_API_KEY=your_elasticsearch_api_key
ELASTIC_INDEX_NAME=clinical-patient-data

# MCP Configuration (Optional)
MCP_LOG_LEVEL=INFO
MCP_MAX_CONCURRENT=10
MCP_REQUEST_TIMEOUT=30
```

### 3. Start MCP Server
```bash
python start_mcp_server.py
```

### 4. Start Flask Application
```bash
python app.py
```

## Usage

### Basic Patient Data Retrieval
```
User: "Get me Jane Doe's medical records"
LLM: I'll retrieve Jane Doe's medical records for you.
🔍 Calling get_patient_data...
✅ get_patient_data completed
Found 10 records for Jane Doe. Recent visits include BPPV treatment and GERD management.
```

### Patient Summary
```
User: "Summarize Jane Doe's medication history"
LLM: I'll get a comprehensive summary of Jane Doe's medication history.
🔍 Calling get_patient_summary...
✅ get_patient_summary completed
Jane Doe's medication history shows: Current medications include Meclizine and Omeprazole. 
Previous medications: Diazepam, Ondansetron, Promethazine. No significant interactions detected.
```

### Drug Interaction Check
```
User: "Check if I can prescribe Diazepam for Jane Doe"
LLM: I'll check for potential drug interactions with Jane Doe's current medications.
🔍 Calling check_medication_interactions...
✅ check_medication_interactions completed
⚠️ INTERACTION: Potential interaction between Diazepam and Meclizine. Both cause drowsiness. Risk of excessive sedation.
```

## Function Definitions

### get_patient_data
- **Purpose**: Retrieve raw patient medical records
- **Parameters**: `patient_name` (string)
- **Returns**: Complete patient data with visit history, diagnoses, medications

### get_patient_summary
- **Purpose**: Get comprehensive patient summary
- **Parameters**: 
  - `patient_name` (string)
  - `summary_type` (enum: comprehensive, medication_focus, recent_visits, risk_assessment, treatment_history)
- **Returns**: Analyzed and summarized patient information

### check_medication_interactions
- **Purpose**: Check for drug interactions
- **Parameters**:
  - `new_medications` (array of strings)
  - `existing_medications` (array of strings)
- **Returns**: List of potential interactions and warnings

## Streaming Features

### Real-time Updates
- Function calls show progress indicators (🔍, ✅, ❌)
- Data retrieval streams as it's processed
- Users see immediate feedback on operations

### Error Handling
- Graceful fallback to direct Elasticsearch queries
- Comprehensive error messages
- Automatic retry mechanisms

## Performance Considerations

### Caching
- Patient data cached for session duration
- Function results cached to reduce redundant calls
- Configurable cache TTL and size

### Concurrency
- Configurable maximum concurrent requests
- Request timeout settings
- Rate limiting for API protection

## Security

### Authentication
- Optional API key authentication
- Configurable allowed origins
- Rate limiting protection

### Data Privacy
- Patient data remains in secure Elasticsearch
- No persistent storage of sensitive information
- Audit logging for compliance

## Monitoring and Logging

### Logging Levels
- INFO: General operation information
- WARNING: Non-critical issues
- ERROR: Function failures and errors
- DEBUG: Detailed operation tracing

### Metrics
- Function call success rates
- Response times
- Cache hit rates
- Error frequencies

## Troubleshooting

### Common Issues

1. **MCP Server Connection Failed**
   - Check if MCP server is running
   - Verify environment variables
   - Check network connectivity

2. **Function Call Timeouts**
   - Increase `MCP_REQUEST_TIMEOUT`
   - Check Elasticsearch connectivity
   - Verify data availability

3. **Tool Discovery Issues**
   - Restart MCP server
   - Check tool definitions in `mcp_config.py`
   - Verify MCP client connection

### Debug Mode
Enable debug logging:
```bash
export MCP_LOG_LEVEL=DEBUG
python start_mcp_server.py
```

## Future Enhancements

### Planned Features
- Additional clinical analysis tools
- Integration with external medical databases
- Advanced pattern recognition
- Predictive analytics capabilities

### Extensibility
- Easy addition of new tools
- Plugin architecture for custom functions
- Support for multiple data sources
- Custom summary types

## Support

For issues or questions:
1. Check the logs in `mcp_server.log`
2. Verify configuration with `python mcp_config.py`
3. Test MCP server independently
4. Review function call parameters and responses

## Migration from Direct Elasticsearch

The MCP implementation maintains backward compatibility:
- Existing direct Elasticsearch calls still work
- Gradual migration path available
- Fallback mechanisms ensure reliability
- No breaking changes to existing functionality
