# Clinical Chat Bot

An intelligent clinical assistant powered by Azure AI services, featuring a conversational avatar for healthcare interactions.

## Features

- 🎤 **Voice Input**: Speech-to-text for natural conversation
- 🤖 **AI-Powered Responses**: Azure OpenAI for intelligent clinical assistance
- 👤 **Animated Avatar**: Realistic talking avatar with synchronized speech
- 📊 **Performance Monitoring**: Real-time latency metrics for optimization
- 🔄 **WebSocket Support**: Low-latency communication
- 🎨 **Customizable Interface**: Configurable avatar and voice settings

## Use Cases

- **Clinical Consultations**: Interactive patient consultations
- **Medical Education**: Training and educational scenarios
- **Patient Support**: 24/7 clinical assistance
- **Telemedicine**: Remote healthcare interactions

## Quick Start

### Prerequisites
- Python 3.7 or later
- Azure AI Services resources:
  - **Speech Service** (with TTS Avatar support)
  - **Azure OpenAI Service** 
  - **Cognitive Search** (optional, for medical knowledge base)

### Setup

1. **Clone and setup environment:**
```bash
# Create virtual environment
python3 -m venv avatar-env

# Activate virtual environment
source avatar-env/bin/activate

# Install dependencies
pip install -r requirements.txt
```

2. **Configure environment variables:**
Create a `variables.env` file with your Azure credentials:
```bash
# Speech Service
SPEECH_REGION=eastus2
SPEECH_KEY=your_speech_key_here
SPEECH_RESOURCE_URL=/subscriptions/your_subscription/resourceGroups/your_rg/providers/Microsoft.CognitiveServices/accounts/your_account

# Azure OpenAI
AZURE_OPENAI_ENDPOINT=https://your-openai.openai.azure.com/
AZURE_OPENAI_API_KEY=your_openai_key_here
AZURE_OPENAI_DEPLOYMENT_NAME=gpt-4o

# Elastic
ELASTIC_URL="https://xxxxx.elastic.cloud:443"
ELASTIC_API_KEY="xxx"
```

3. **Run the setup script:**
```bash
# This script will load environment variables and prepare the application
source ./setup_env.sh
```

4. **Run the application:**
```bash
python -m flask run -h 0.0.0.0 -p 5000
```

5. **Access the application:**
- Clinical chat: `http://localhost:5000/chat`
- Basic demo: `http://localhost:5000/basic`

### Quick Test
To verify your setup is working correctly:
1. Open `http://localhost:5000/basic` in your browser
2. Click the microphone button and speak a simple phrase
3. Check that the avatar responds with speech and animation

### Available Regions
TTS Avatar is available in: Southeast Asia, North Europe, West Europe, Sweden Central, South Central US, East US 2, and West US 2.

### Key Components

**setup_env.sh**: Environment setup script that loads your Azure credentials and prepares the application environment.

**vad_iterator.py**: Voice Activity Detection module that processes real-time audio streams to detect when users are speaking, enabling more natural conversation flow.

## Data Ingestion

### Elasticsearch Integration

The application includes a data ingestion script to load clinical patient data into Elasticsearch for enhanced search capabilities and knowledge base functionality.

#### Ingest Clinical Data

1. **Prepare your data**: Place your clinical patient data in JSON format in the `data/` directory
2. **Configure Elasticsearch**: Ensure your `variables.env` file contains:
   ```bash
   ELASTIC_URL="https://your-cluster.elastic.cloud:443"
   ELASTIC_API_KEY="your_api_key_here"
   ```

3. **Run the ingestion script**:
   ```bash
   cd data/
   ./ingest_to_elasticsearch.sh
   ```

#### Script Features

- **Serverless Compatible**: Works with both regular and serverless Elasticsearch instances
- **Index Management**: Automatically deletes existing index and creates a fresh one
- **Field Mapping**: Properly maps clinical data fields for optimal search performance
- **Bulk Operations**: Efficiently ingests large datasets using Elasticsearch bulk API
- **Error Handling**: Comprehensive error checking and colored output for troubleshooting

#### Data Structure

The script expects JSON data with the following structure:
```json
[
  {
    "patient_name": "Jane Doe",
    "date_of_visit": "2023-01-15",
    "patient_complaint": "Persistent cough, low-grade fever, and fatigue",
    "diagnosis": "Acute bronchitis",
    "doctor_notes": "Patient presents with productive cough...",
    "drugs_prescribed": ["Mucinex"],
    "patient_age_at_visit": 35
  }
]
```

#### Elasticsearch Index Mapping

The script creates an index with optimized field mappings:
- **Text Fields**: `patient_complaint`, `doctor_notes` for full-text search
- **Keyword Fields**: `patient_name`, `diagnosis` for exact matching and aggregations
- **Date Fields**: `date_of_visit` with proper date formatting
- **Array Fields**: `drugs_prescribed` for medication searches
- **Numeric Fields**: `patient_age_at_visit` for age-based queries

#### Querying Your Data

After ingestion, you can search your clinical data using:
- **Elasticsearch API**: `GET /clinical-patient-data/_search`
- **Kibana**: Access through your Elasticsearch cluster dashboard
- **Application Integration**: Use the indexed data to enhance AI responses

## Clinical Configuration

### System Prompt for Clinical Use
Configure the system prompt in the chat interface for clinical scenarios:

```
You are a professional clinical assistant. Provide accurate, evidence-based medical information while always reminding users to consult with healthcare professionals for medical decisions. Focus on:

- Symptom assessment guidance
- General health information
- Medication information
- Preventive care recommendations
- When to seek immediate medical attention

Always emphasize that this is for informational purposes only and not a substitute for professional medical advice.
```

### Avatar Settings for Clinical Environment
- **Character**: Professional appearance (e.g., `lisa` with professional styling)
- **Voice**: Clear, professional tone
- **Background**: Clean, medical environment

## Performance Monitoring

The application provides real-time performance metrics:

- **AOAI Latency**: Time for Azure OpenAI to generate first sentence
- **TTS Latency**: Time from response received to avatar speaking
- **STT Latency**: Speech-to-text processing time
- **App Service Latency**: Application overhead

## Deployment

### Docker Deployment
```bash
# Build image
docker build -t clinical-chat-bot .

# Run container
docker run -p 5000:5000 --env-file variables.env clinical-chat-bot
```

### Azure Container Apps
1. Build and push Docker image to Azure Container Registry
2. Create Container App with environment variables
3. Deploy with WebSocket support enabled
4. Configure for HIPAA compliance if handling PHI

## File Structure

```
├── app.py                 # Main Flask application
├── chat.html             # Clinical chat interface
├── basic.html            # Basic demo interface
├── data/
│   ├── clinical-patient-data    # Sample clinical data in JSON format
│   └── ingest_to_elasticsearch.sh  # Data ingestion script for Elasticsearch
├── static/
│   ├── js/
│   │   ├── chat.js       # Chat functionality
│   │   └── basic.js      # Basic demo functionality
│   ├── css/
│   │   └── styles.css    # Styling
│   └── image/            # Static assets
├── requirements.txt      # Python dependencies
├── Dockerfile           # Container configuration
├── setup_env.sh         # Environment setup script
├── vad_iterator.py      # Voice activity detection for real-time audio processing
└── variables.env        # Environment variables (configure with your Azure credentials)
```

## Clinical Considerations

### Privacy and Security
- Ensure HIPAA compliance for patient data
- Use secure communication protocols
- Implement proper access controls
- Regular security audits

### Medical Disclaimer
This application is for educational and informational purposes only. It is not intended to:
- Replace professional medical advice
- Diagnose medical conditions
- Provide treatment recommendations
- Handle emergency medical situations

Always consult with qualified healthcare professionals for medical decisions.

## Troubleshooting

### Common Issues

**Azure Service Connection Issues:**
- Verify all environment variables are set correctly in `variables.env`
- Check that your Azure services are in supported regions
- Ensure your API keys have the correct permissions
- Verify your Azure OpenAI deployment is active and accessible

**Audio/Video Issues:**
- Check microphone permissions in your browser
- Ensure you're using HTTPS in production (required for microphone access)
- Try refreshing the page if audio doesn't work initially
- Check browser console for WebSocket connection errors

**Performance Issues:**
- Monitor latency logs in the application interface
- Check Azure service quotas and limits
- Verify your internet connection stability
- Consider using a region closer to your users

**Setup Issues:**
- Ensure Python 3.7+ is installed
- Verify virtual environment is activated
- Check that all dependencies are installed correctly
- Run `source ./setup_env.sh` before starting the application

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Test thoroughly
5. Submit a pull request

## License

This project is licensed under the MIT License - see the LICENSE file for details.


