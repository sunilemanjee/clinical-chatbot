#!/bin/bash

# Azure TTS Talking Avatar - Environment Setup Script
# This script activates the virtual environment and sets environment variables
# Edit the .env file with your Azure credentials, then source this script

echo "=========================================="
echo "Azure TTS Talking Avatar - Environment Setup"
echo "=========================================="
echo ""

# Check if virtual environment exists
if [ ! -d "avatar-env" ]; then
    echo "❌ Virtual environment 'avatar-env' not found!"
    echo "Please create it first with: python3 -m venv avatar-env"
    echo ""
    echo "Then edit the .env file with your Azure credentials and source this script again."
    return 1
fi

# Check if variables.env file exists
if [ ! -f "variables.env" ]; then
    echo "❌ variables.env file not found!"
    echo ""
    if [ -f "variables.env.template" ]; then
        echo "📋 A template file is available. To create your variables.env file:"
        echo "   cp variables.env.template variables.env"
        echo "   # Then edit variables.env with your actual Azure credentials"
    else
        echo "Please create a variables.env file with your Azure credentials:"
        echo ""
        echo "SPEECH_REGION=your-region-here"
        echo "SPEECH_KEY=your-api-key-here"
        echo "SPEECH_RESOURCE_URL=your-speech-resource-url-here"
        echo "AZURE_OPENAI_ENDPOINT=your-openai-endpoint-here"
        echo "AZURE_OPENAI_API_KEY=your-openai-api-key-here"
        echo "AZURE_OPENAI_DEPLOYMENT_NAME=your-deployment-name-here"
        echo "PORT=5005  # Optional: Flask port (default: 5005)"
    fi
    echo ""
    echo "Then source this script again."
    return 1
fi

# Deactivate any existing virtual environment first
if [ -n "$VIRTUAL_ENV" ]; then
    echo "🔄 Deactivating existing virtual environment: $VIRTUAL_ENV"
    deactivate
fi

# Activate virtual environment
echo "🐍 Activating virtual environment..."
source avatar-env/bin/activate

# Install requirements if requirements.txt exists
if [ -f "requirements.txt" ]; then
    echo "📦 Installing requirements from requirements.txt..."
    pip install -r requirements.txt
    if [ $? -eq 0 ]; then
        echo "✅ Requirements installed successfully!"
    else
        echo "❌ Failed to install requirements. Please check requirements.txt and try again."
        return 1
    fi
else
    echo "⚠️  requirements.txt not found. Skipping dependency installation."
fi

# Load environment variables from variables.env file
echo "🔑 Loading environment variables from variables.env file..."
export $(cat variables.env | grep -v '^#' | xargs)

# Verify required variables are set
if [ -z "$SPEECH_REGION" ]; then
    echo "❌ SPEECH_REGION is not set in variables.env file"
    return 1
fi

if [ -z "$SPEECH_KEY" ]; then
    echo "❌ SPEECH_KEY is not set in variables.env file"
    return 1
fi

echo ""
echo "✅ Environment setup complete!"
echo "✅ Virtual environment activated: $(which python)"
echo "✅ Environment variables loaded:"
echo "   SPEECH_REGION: $SPEECH_REGION"
echo "   SPEECH_KEY: ${SPEECH_KEY:0:8}..."  # Only show first 8 characters for security
if [ -n "$SPEECH_PRIVATE_ENDPOINT" ]; then
    echo "   SPEECH_PRIVATE_ENDPOINT: $SPEECH_PRIVATE_ENDPOINT"
fi
echo ""
# Set default port if not specified
if [ -z "$PORT" ]; then
    PORT=5005
fi

echo "🚀 You can now run the sample with:"
echo "   python -m flask run -h 0.0.0.0 -p $PORT"
echo ""
echo "📱 IMPORTANT: Access the application at:"
echo "   http://localhost:$PORT/chat    # ← USE THIS ENDPOINT (/chat)"
echo ""
echo "Other available endpoints:"
echo "   http://localhost:$PORT/basic   # Basic interface"
echo "   http://localhost:$PORT/        # Root (redirects to basic)"
echo ""
echo "Or specify a different port:"
echo "   python -m flask run -h 0.0.0.0 -p 8080  # Then use: http://localhost:8080/chat"
echo "   python -m flask run -h 0.0.0.0 -p 3000  # Then use: http://localhost:3000/chat"
echo ""
echo "Note: Port 5000 is often used by AirPlay on macOS. Use a different port to avoid conflicts."
echo ""
