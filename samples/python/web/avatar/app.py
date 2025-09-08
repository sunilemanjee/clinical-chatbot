# Copyright (c) Microsoft. All rights reserved.
# Licensed under the MIT license.

import azure.cognitiveservices.speech as speechsdk
import base64
import datetime
import html
import json
import numpy as np
import os
import pytz
import random
import re
import requests
import threading
import time
import torch
import traceback
import uuid
from flask import Flask, Response, render_template, request
from flask_socketio import SocketIO, join_room
from azure.identity import DefaultAzureCredential
from openai import AzureOpenAI
from elasticsearch import Elasticsearch
from vad_iterator import VADIterator, int2float

# Create the Flask app
app = Flask(__name__, template_folder='.')

# Create the SocketIO instance
socketio = SocketIO(app)

# Environment variables
# Speech resource (required)
speech_region = os.environ.get('SPEECH_REGION')  # e.g. westus2
speech_key = os.environ.get('SPEECH_KEY')
speech_private_endpoint = os.environ.get('SPEECH_PRIVATE_ENDPOINT')  # e.g. https://my-speech-service.cognitiveservices.azure.com/ (optional)  # noqa: E501
speech_resource_url = os.environ.get('SPEECH_RESOURCE_URL')  # e.g. /subscriptions/6e83d8b7-00dd-4b0a-9e98-dab9f060418b/resourceGroups/my-rg/providers/Microsoft.CognitiveServices/accounts/my-speech (optional, only used for private endpoint)  # noqa: E501
user_assigned_managed_identity_client_id = os.environ.get('USER_ASSIGNED_MANAGED_IDENTITY_CLIENT_ID')  # e.g. the client id of user assigned managed identity accociated to your app service (optional, only used for private endpoint and user assigned managed identity)  # noqa: E501
# OpenAI resource (required for chat scenario)
azure_openai_endpoint = os.environ.get('AZURE_OPENAI_ENDPOINT')  # e.g. https://my-aoai.openai.azure.com/
azure_openai_api_key = os.environ.get('AZURE_OPENAI_API_KEY')
azure_openai_deployment_name = os.environ.get('AZURE_OPENAI_DEPLOYMENT_NAME')  # e.g. my-gpt-35-turbo-deployment
# Cognitive search resource (optional, only required for 'on your data' scenario)
cognitive_search_endpoint = os.environ.get('COGNITIVE_SEARCH_ENDPOINT')  # e.g. https://my-cognitive-search.search.windows.net/
cognitive_search_api_key = os.environ.get('COGNITIVE_SEARCH_API_KEY')
cognitive_search_index_name = os.environ.get('COGNITIVE_SEARCH_INDEX_NAME')  # e.g. my-search-index
# Customized ICE server (optional, only required for customized ICE server)
ice_server_url = os.environ.get('ICE_SERVER_URL')  # The ICE URL, e.g. turn:x.x.x.x:3478
ice_server_url_remote = os.environ.get('ICE_SERVER_URL_REMOTE')  # The ICE URL for remote side, e.g. turn:x.x.x.x:3478. This is only required when the ICE address for remote side is different from local side.  # noqa: E501
ice_server_username = os.environ.get('ICE_SERVER_USERNAME')  # The ICE username
ice_server_password = os.environ.get('ICE_SERVER_PASSWORD')  # The ICE password
# Elasticsearch configuration (required for patient data queries)
elastic_url = os.environ.get('ELASTIC_URL')  # e.g. https://demo-c4ecc8.es.us-east-1.aws.elastic.cloud:443
elastic_api_key = os.environ.get('ELASTIC_API_KEY')  # Elasticsearch API key
elastic_index_name = os.environ.get('ELASTIC_INDEX_NAME')  # e.g. clinical-patient-data

# Const variables
enable_websockets = True  # Enable websockets between client and server for real-time communication optimization
enable_vad = False  # Enable voice activity detection (VAD) for interrupting the avatar speaking
enable_token_auth_for_speech = False  # Enable token authentication for speech service
default_tts_voice = 'en-US-AmandaMultilingualNeural'  # Default TTS voice
sentence_level_punctuations = ['.', '?', '!', ':', ';', '。', '？', '！', '：', '；']  # Punctuations that indicate the end of a sentence
enable_quick_reply = False  # Enable quick reply for certain chat models which take longer time to respond
quick_replies = ['Let me take a look.', 'Let me check.', 'One moment, please.']  # Quick reply reponses
oyd_doc_regex = re.compile(r'\[doc(\d+)\]')  # Regex to match the OYD (on-your-data) document reference
repeat_speaking_sentence_after_reconnection = True  # Repeat the speaking sentence after reconnection

# Global variables
client_contexts = {}  # Client contexts
speech_token = None  # Speech token
ice_token = None  # ICE token
if azure_openai_endpoint and azure_openai_api_key:
    azure_openai = AzureOpenAI(
        azure_endpoint=azure_openai_endpoint,
        api_version='2024-06-01',
        api_key=azure_openai_api_key)

# Initialize Elasticsearch client
elastic_client = None
if elastic_url and elastic_api_key:
    try:
        elastic_client = Elasticsearch(
            hosts=[elastic_url],
            api_key=elastic_api_key,
            verify_certs=True
        )
        # Test the connection
        if elastic_client.ping():
            print("Elasticsearch connection successful!")
        else:
            print("Elasticsearch connection failed!")
            elastic_client = None
    except Exception as e:
        print(f"Failed to initialize Elasticsearch client: {e}")
        elastic_client = None

# VAD
vad_iterator = None
if enable_vad and enable_websockets:
    vad_model, _ = torch.hub.load(repo_or_dir='snakers4/silero-vad', model='silero_vad')
    vad_iterator = VADIterator(model=vad_model, threshold=0.5, sampling_rate=16000, min_silence_duration_ms=150, speech_pad_ms=100)


# The default route, which shows the default web page (basic.html)
@app.route("/")
def index():
    return render_template("basic.html", methods=["GET"], client_id=initializeClient())


# The basic route, which shows the basic web page
@app.route("/basic")
def basicView():
    return render_template("basic.html", methods=["GET"], client_id=initializeClient())


# The chat route, which shows the chat web page
@app.route("/chat")
def chatView():
    return render_template("chat.html", methods=["GET"], client_id=initializeClient(), enable_websockets=enable_websockets)


# The API route to get the system prompt from prompt.md
@app.route("/api/getSystemPrompt", methods=["GET"])
def getSystemPrompt() -> Response:
    try:
        with open('prompt.md', 'r', encoding='utf-8') as file:
            prompt_content = file.read()
        return Response(prompt_content, status=200, mimetype='text/plain')
    except FileNotFoundError:
        return Response("System prompt file not found", status=404)
    except Exception as e:
        return Response(f"Error reading system prompt: {str(e)}", status=500)


# The API route to get the speech token
@app.route("/api/getSpeechToken", methods=["GET"])
def getSpeechToken() -> Response:
    response = Response(speech_token, status=200)
    response.headers['SpeechRegion'] = speech_region
    if speech_private_endpoint:
        response.headers['SpeechPrivateEndpoint'] = speech_private_endpoint
    return response


# The API route to get the ICE token
@app.route("/api/getIceToken", methods=["GET"])
def getIceToken() -> Response:
    # Apply customized ICE server if provided
    if ice_server_url and ice_server_username and ice_server_password:
        custom_ice_token = json.dumps({
            'Urls': [ice_server_url],
            'Username': ice_server_username,
            'Password': ice_server_password
        })
        return Response(custom_ice_token, status=200)
    return Response(ice_token, status=200)


# The API route to get the status of server
@app.route("/api/getStatus", methods=["GET"])
def getStatus() -> Response:
    client_id = uuid.UUID(request.headers.get('ClientId'))
    client_context = client_contexts[client_id]
    status = {
        'speechSynthesizerConnected': client_context['speech_synthesizer_connected']
    }
    return Response(json.dumps(status), status=200)


# The API route to connect the TTS avatar
@app.route("/api/connectAvatar", methods=["POST"])
def connectAvatar() -> Response:
    client_id = uuid.UUID(request.headers.get('ClientId'))
    isReconnecting = request.headers.get('Reconnect') and request.headers.get('Reconnect').lower() == 'true'
    
    # Ensure client context exists before trying to disconnect
    if client_id in client_contexts:
        # disconnect avatar if already connected
        disconnectAvatarInternal(client_id, isReconnecting)
    
    # Get or create client context
    if client_id not in client_contexts:
        print(f"⚠️ Client context not found for {client_id}, creating new one")
        client_contexts[client_id] = {
            'audio_input_stream': None,
            'vad_audio_buffer': [],
            'speech_recognizer': None,
            'azure_openai_deployment_name': azure_openai_deployment_name,
            'cognitive_search_index_name': cognitive_search_index_name,
            'tts_voice': e ,
            'custom_voice_endpoint_id': None,
            'personal_voice_speaker_profile_id': None,
            'speech_synthesizer': None,
            'speech_synthesizer_connection': None,
            'speech_synthesizer_connected': False,
            'speech_token': None,
            'ice_token': None,
            'chat_initiated': False,
            'messages': [],
            'data_sources': [],
            'is_speaking': False,
            'speaking_text': None,
            'spoken_text_queue': [],
            'speaking_thread': None,
            'last_speak_time': None,
            'initial_greeting_sent': False,
            'initial_greeting': None,
            'patient_name': None,
            'patient_data': None
        }
    
    client_context = client_contexts[client_id]

    # Override default values with client provided values
    client_context['azure_openai_deployment_name'] = (
        request.headers.get('AoaiDeploymentName') if request.headers.get('AoaiDeploymentName') else azure_openai_deployment_name)
    client_context['cognitive_search_index_name'] = (
        request.headers.get('CognitiveSearchIndexName') if request.headers.get('CognitiveSearchIndexName')
        else cognitive_search_index_name)
    client_context['tts_voice'] = request.headers.get('TtsVoice') if request.headers.get('TtsVoice') else default_tts_voice
    client_context['custom_voice_endpoint_id'] = request.headers.get('CustomVoiceEndpointId')
    client_context['personal_voice_speaker_profile_id'] = request.headers.get('PersonalVoiceSpeakerProfileId')

    custom_voice_endpoint_id = client_context['custom_voice_endpoint_id']

    try:
        if speech_private_endpoint:
            speech_private_endpoint_wss = speech_private_endpoint.replace('https://', 'wss://')
            if enable_token_auth_for_speech:
                while not speech_token:
                    time.sleep(0.2)
                speech_config = speechsdk.SpeechConfig(
                    endpoint=f'{speech_private_endpoint_wss}/tts/cognitiveservices/websocket/v1?enableTalkingAvatar=true')
                speech_config.authorization_token = speech_token
            else:
                speech_config = speechsdk.SpeechConfig(
                    subscription=speech_key,
                    endpoint=f'{speech_private_endpoint_wss}/tts/cognitiveservices/websocket/v1?enableTalkingAvatar=true')
        else:
            if enable_token_auth_for_speech:
                while not speech_token:
                    time.sleep(0.2)
                speech_config = speechsdk.SpeechConfig(
                    endpoint=f'wss://{speech_region}.tts.speech.microsoft.com/cognitiveservices/websocket/v1?enableTalkingAvatar=true')
                speech_config.authorization_token = speech_token
            else:
                speech_config = speechsdk.SpeechConfig(
                    subscription=speech_key,
                    endpoint=f'wss://{speech_region}.tts.speech.microsoft.com/cognitiveservices/websocket/v1?enableTalkingAvatar=true')

        if custom_voice_endpoint_id:
            speech_config.endpoint_id = custom_voice_endpoint_id

        # Create optimized audio config for lower latency
        audio_config = speechsdk.audio.AudioOutputConfig(use_default_speaker=True)
        
        client_context['speech_synthesizer'] = speechsdk.SpeechSynthesizer(speech_config=speech_config, audio_config=audio_config)
        speech_synthesizer = client_context['speech_synthesizer']
        
        # Set additional properties for lower latency
        speech_synthesizer.properties.set_property(speechsdk.PropertyId.SpeechServiceConnection_SynthEnableCompressedAudioTransmission, "true")
        speech_synthesizer.properties.set_property(speechsdk.PropertyId.SpeechServiceConnection_SynthOutputFormat, "audio-16khz-32kbitrate-mono-mp3")

        ice_token_obj = json.loads(ice_token)
        # Apply customized ICE server if provided
        if ice_server_url and ice_server_username and ice_server_password:
            ice_token_obj = {
                'Urls': [ice_server_url_remote] if ice_server_url_remote else [ice_server_url],
                'Username': ice_server_username,
                'Password': ice_server_password
            }
        local_sdp = request.data.decode('utf-8')
        
        # Debug: Print all headers
        print("Request headers:")
        for header, value in request.headers.items():
            print(f"  {header}: {value}")
        
        avatar_character = request.headers.get('AvatarCharacter') or 'lori'
        avatar_style = request.headers.get('AvatarStyle') or 'graceful'
        # Debug: Print avatar configuration values
        print(f"Avatar config - Character: {avatar_character}, Style: {avatar_style}")
        background_color = '#FFFFFFFF' if request.headers.get('BackgroundColor') is None else request.headers.get('BackgroundColor')
        background_image_url = request.headers.get('BackgroundImageUrl')
        is_custom_avatar = request.headers.get('IsCustomAvatar')
        transparent_background = (
            'false' if request.headers.get('TransparentBackground') is None
            else request.headers.get('TransparentBackground'))
        video_crop = 'false' if request.headers.get('VideoCrop') is None else request.headers.get('VideoCrop')
        avatar_config = {
            'synthesis': {
                'synthesisConfig': {
                    'voice': client_context['tts_voice']
                },
                'video': {
                    'protocol': {
                        'name': "WebRTC",
                        'webrtcConfig': {
                            'clientDescription': local_sdp,
                            'iceServers': [{
                                'urls': [ice_token_obj['Urls'][0]],
                                'username': ice_token_obj['Username'],
                                'credential': ice_token_obj['Password']
                            }]
                        },
                    },
                    'format': {
                        'crop': {
                            'topLeft': {
                                'x': 600 if video_crop.lower() == 'true' else 0,
                                'y': 0
                            },
                            'bottomRight': {
                                'x': 1320 if video_crop.lower() == 'true' else 1920,
                                'y': 1080
                            }
                        },
                        'bitrate': 500000,  # Reduced from 1Mbps to 500kbps for lower latency
                        # Optimize for ultra-low latency
                        'frameRate': 24,  # Reduced from 30 to 24 FPS for lower latency
                        'keyframeInterval': 12,  # Reduced from 30 to 12 for faster keyframe recovery
                        'latencyMode': 'ultraLowLatency'
                    },
                    'talkingAvatar': {
                        'customized': is_custom_avatar.lower() == 'true',
                        'character': avatar_character,
                        'style': avatar_style,
                        'background': {
                            'color': '#00FF00FF' if transparent_background.lower() == 'true' else background_color,
                            'image': {
                                'url': background_image_url
                            }
                        },
                        # Additional settings for better synchronization
                        'synchronization': {
                            'audioVideoSync': True,  # Enable audio-video synchronization
                            'lipSyncAccuracy': 'high',  # High accuracy lip sync
                            'audioBufferSize': 64,  # Smaller audio buffer for lower latency
                            'videoBufferSize': 2,  # Minimal video buffering
                            'syncTolerance': 50  # 50ms sync tolerance
                        },
                        # Advanced lip sync settings
                        'lipSync': {
                            'enabled': True,
                            'precision': 'high',
                            'realTimeProcessing': True,
                            'audioLatencyCompensation': True
                        }
                    }
                }
            }
        }
        
        # Debug: Print the final avatar config
        print(f"Final avatar config: {json.dumps(avatar_config, indent=2)}")

        connection = speechsdk.Connection.from_speech_synthesizer(speech_synthesizer)
        connection.connected.connect(lambda evt: print('TTS Avatar service connected.'))

        def tts_disconnected_cb(evt):
            print('TTS Avatar service disconnected.')
            client_context['speech_synthesizer_connection'] = None
            client_context['speech_synthesizer_connected'] = False
            if enable_websockets:
                socketio.emit("response", {'path': 'api.event', 'eventType': 'SPEECH_SYNTHESIZER_DISCONNECTED'}, room=client_id)

        connection.disconnected.connect(tts_disconnected_cb)
        connection.set_message_property('speech.config', 'context', json.dumps(avatar_config))
        client_context['speech_synthesizer_connection'] = connection
        client_context['speech_synthesizer_connected'] = True
        if enable_websockets:
            socketio.emit("response", {'path': 'api.event', 'eventType': 'SPEECH_SYNTHESIZER_CONNECTED'}, room=client_id)

        speech_sythesis_result = speech_synthesizer.speak_text_async('').get()
        print(f'Result id for avatar connection: {speech_sythesis_result.result_id}')
        if speech_sythesis_result.reason == speechsdk.ResultReason.Canceled:
            cancellation_details = speech_sythesis_result.cancellation_details
            print(f"Speech synthesis canceled: {cancellation_details.reason}")
            if cancellation_details.reason == speechsdk.CancellationReason.Error:
                print(f"Error details: {cancellation_details.error_details}")
                raise Exception(cancellation_details.error_details)
        turn_start_message = speech_synthesizer.properties.get_property_by_name('SpeechSDKInternal-ExtraTurnStartMessage')
        remoteSdp = json.loads(turn_start_message)['webrtc']['connectionString']

        # Initialize chat context and send initial greeting asking for patient name
        initializeChatContext("", client_id)
        client_context['chat_initiated'] = True
        
        # Send initial greeting asking for patient name
        initial_greeting = "Hello! I'm your clinical assistant. Please provide the patient's name."
        if enable_websockets:
            socketio.emit("response", {'path': 'api.chat', 'chatResponse': 'Assistant: ' + initial_greeting}, room=client_id)
            # Also speak the greeting
            speakWithQueue(initial_greeting, 0, client_id)
        else:
            # For non-websocket mode, we'll need to handle this differently
            # The greeting will be sent when the first user interaction occurs
            client_context['initial_greeting_sent'] = False
            client_context['initial_greeting'] = initial_greeting

        return Response(remoteSdp, status=200)

    except Exception as e:
        error_msg = f"Error message: {e}"
        if 'speech_sythesis_result' in locals():
            error_msg = f"Result ID: {speech_sythesis_result.result_id}. {error_msg}"
        return Response(error_msg, status=400)


# The API route to connect the STT service
@app.route("/api/connectSTT", methods=["POST"])
def connectSTT() -> Response:
    client_id = uuid.UUID(request.headers.get('ClientId'))
    # disconnect STT if already connected
    disconnectSttInternal(client_id)
    
    # Get SystemPrompt from request body if available, otherwise from header
    if request.content_type == 'application/json':
        data = request.get_json()
        system_prompt = data.get('SystemPrompt') if data else None
    else:
        system_prompt = request.headers.get('SystemPrompt')
    
    client_context = client_contexts[client_id]
    try:
        if speech_private_endpoint:
            speech_private_endpoint_wss = speech_private_endpoint.replace('https://', 'wss://')
            if enable_token_auth_for_speech:
                while not speech_token:
                    time.sleep(0.2)
                speech_config = speechsdk.SpeechConfig(
                    endpoint=f'{speech_private_endpoint_wss}/stt/speech/universal/v2')
                speech_config.authorization_token = speech_token
            else:
                speech_config = speechsdk.SpeechConfig(
                    subscription=speech_key, endpoint=f'{speech_private_endpoint_wss}/stt/speech/universal/v2')
        else:
            if enable_token_auth_for_speech:
                while not speech_token:
                    time.sleep(0.2)
                speech_config = speechsdk.SpeechConfig(
                    endpoint=f'wss://{speech_region}.stt.speech.microsoft.com/speech/universal/v2')
                speech_config.authorization_token = speech_token
            else:
                speech_config = speechsdk.SpeechConfig(
                    subscription=speech_key, endpoint=f'wss://{speech_region}.stt.speech.microsoft.com/speech/universal/v2')

        audio_input_stream = speechsdk.audio.PushAudioInputStream()
        client_context['audio_input_stream'] = audio_input_stream

        audio_config = speechsdk.audio.AudioConfig(stream=audio_input_stream)
        speech_recognizer = speechsdk.SpeechRecognizer(speech_config=speech_config, audio_config=audio_config)
        client_context['speech_recognizer'] = speech_recognizer

        speech_recognizer.session_started.connect(lambda evt: print(f'STT session started - session id: {evt.session_id}'))
        speech_recognizer.session_stopped.connect(lambda evt: print('STT session stopped.'))

        speech_recognition_start_time = datetime.datetime.now(pytz.UTC)

        def stt_recognized_cb(evt):
            if evt.result.reason == speechsdk.ResultReason.RecognizedSpeech:
                try:
                    user_query = evt.result.text.strip()
                    if user_query == '':
                        return

                    socketio.emit("response", {'path': 'api.chat', 'chatResponse': '\n\nUser: ' + user_query + '\n\n'}, room=client_id)
                    recognition_result_received_time = datetime.datetime.now(pytz.UTC)
                    speech_finished_offset = (evt.result.offset + evt.result.duration) / 10000
                    stt_latency = round((recognition_result_received_time - speech_recognition_start_time).total_seconds() * 1000 - speech_finished_offset)  # noqa: E501
                    print(f'STT latency: {stt_latency}ms')
                    socketio.emit("response", {'path': 'api.chat', 'chatResponse': f"<STTL>{stt_latency}</STTL>"}, room=client_id)
                    chat_initiated = client_context['chat_initiated']
                    if not chat_initiated:
                        initializeChatContext(system_prompt, client_id)
                        client_context['chat_initiated'] = True
                    
                    # Check if we need to send initial greeting for STT mode
                    initial_greeting_sent = client_context.get('initial_greeting_sent', True)
                    initial_greeting = client_context.get('initial_greeting', '')
                    
                    # If this is the first interaction and we have an initial greeting, send it first
                    if not initial_greeting_sent and initial_greeting:
                        client_context['initial_greeting_sent'] = True
                        socketio.emit("response", {'path': 'api.chat', 'chatResponse': 'Assistant: ' + initial_greeting + '\n\n'}, room=client_id)
                    
                    first_response_chunk = True
                    for chat_response in handleUserQuery(user_query, client_id):
                        if first_response_chunk:
                            socketio.emit("response", {'path': 'api.chat', 'chatResponse': 'Assistant: '}, room=client_id)
                            first_response_chunk = False
                        socketio.emit("response", {'path': 'api.chat', 'chatResponse': chat_response}, room=client_id)
                except Exception as e:
                    print(f"Error in handling user query: {e}")
        speech_recognizer.recognized.connect(stt_recognized_cb)

        def stt_recognizing_cb(evt):
            if not vad_iterator:
                stopSpeakingInternal(client_id, False)
        speech_recognizer.recognizing.connect(stt_recognizing_cb)

        def stt_canceled_cb(evt):
            cancellation_details = speechsdk.CancellationDetails(evt.result)
            print(f'STT connection canceled. Error message: {cancellation_details.error_details}')
        speech_recognizer.canceled.connect(stt_canceled_cb)

        speech_recognizer.start_continuous_recognition()
        return Response(status=200)

    except Exception as e:
        return Response(f"STT connection failed. Error message: {e}", status=400)


# The API route to disconnect the STT service
@app.route("/api/disconnectSTT", methods=["POST"])
def disconnectSTT() -> Response:
    client_id = uuid.UUID(request.headers.get('ClientId'))
    try:
        disconnectSttInternal(client_id)
        return Response('STT Disconnected.', status=200)
    except Exception as e:
        return Response(f"STT disconnection failed. Error message: {e}", status=400)


# The API route to speak a given SSML
@app.route("/api/speak", methods=["POST"])
def speak() -> Response:
    client_id = uuid.UUID(request.headers.get('ClientId'))
    try:
        ssml = request.data.decode('utf-8')
        result_id = speakSsml(ssml, client_id, True)
        return Response(result_id, status=200)
    except Exception as e:
        return Response(f"Speak failed. Error message: {e}", status=400)


# The API route to stop avatar from speaking
@app.route("/api/stopSpeaking", methods=["POST"])
def stopSpeaking() -> Response:
    client_id = uuid.UUID(request.headers.get('ClientId'))
    stopSpeakingInternal(client_id, False)
    return Response('Speaking stopped.', status=200)


# The API route for chat
# It receives the user query and return the chat response.
# It returns response in stream, which yields the chat response in chunks.
@app.route("/api/chat", methods=["POST"])
def chat() -> Response:
    client_id = uuid.UUID(request.headers.get('ClientId'))
    client_context = client_contexts[client_id]
    chat_initiated = client_context['chat_initiated']
    if not chat_initiated:
        initializeChatContext(request.headers.get('SystemPrompt'), client_id)
        client_context['chat_initiated'] = True
    
    # Check if we need to send initial greeting for non-websocket mode
    initial_greeting_sent = client_context.get('initial_greeting_sent', True)
    initial_greeting = client_context.get('initial_greeting', '')
    
    user_query = request.data.decode('utf-8')
    
    # If this is the first interaction and we have an initial greeting, send it first
    if not initial_greeting_sent and initial_greeting:
        client_context['initial_greeting_sent'] = True
        # Create a generator that yields the greeting first, then the user query response
        def combined_response():
            yield 'Assistant: ' + initial_greeting + '\n\n'
            yield from handleUserQuery(user_query, client_id)
        return Response(combined_response(), mimetype='text/plain', status=200)
    else:
        return Response(handleUserQuery(user_query, client_id), mimetype='text/plain', status=200)


# The API route to continue speaking the unfinished sentences
@app.route("/api/chat/continueSpeaking", methods=["POST"])
def continueSpeaking() -> Response:
    client_id = uuid.UUID(request.headers.get('ClientId'))
    client_context = client_contexts[client_id]
    spoken_text_queue = client_context['spoken_text_queue']
    speaking_text = client_context['speaking_text']
    if speaking_text and repeat_speaking_sentence_after_reconnection:
        spoken_text_queue.insert(0, speaking_text)
    if len(spoken_text_queue) > 0:
        speakWithQueue(None, 0, client_id)
    return Response('Request sent.', status=200)


# The API route to clear the chat history
@app.route("/api/chat/clearHistory", methods=["POST"])
def clearChatHistory() -> Response:
    client_id = uuid.UUID(request.headers.get('ClientId'))
    client_context = client_contexts[client_id]
    initializeChatContext(request.headers.get('SystemPrompt'), client_id)
    client_context['chat_initiated'] = True
    
    # Send a fresh initial greeting after clearing history
    initial_greeting = "Hello! I'm your clinical assistant. Please provide the patient's name."
    client_context['initial_greeting'] = initial_greeting
    client_context['initial_greeting_sent'] = False
    
    return Response('Chat history cleared.', status=200)


# The API route to load clinical patient data into Elasticsearch
@app.route("/api/loadData", methods=["POST"])
def loadData() -> Response:
    client_id = uuid.UUID(request.headers.get('ClientId'))
    print(f"Load data request received from client {client_id}")
    
    try:
        import subprocess
        import os
        
        # Get the directory where the script is located
        script_dir = os.path.join(os.path.dirname(__file__), 'data')
        script_path = os.path.join(script_dir, 'ingest_to_elasticsearch.sh')
        
        print(f"Executing script: {script_path}")
        print(f"Working directory: {script_dir}")
        
        # Execute the ingestion script
        result = subprocess.run(
            ['bash', script_path],
            cwd=script_dir,
            capture_output=True,
            text=True,
            timeout=300  # 5 minute timeout
        )
        
        if result.returncode == 0:
            print("Data ingestion completed successfully")
            print(f"Script output: {result.stdout}")
            return Response("Clinical patient data loaded successfully into Elasticsearch!", status=200)
        else:
            print(f"Data ingestion failed with return code: {result.returncode}")
            print(f"Error output: {result.stderr}")
            print(f"Standard output: {result.stdout}")
            return Response(f"Data ingestion failed: {result.stderr}", status=500)
            
    except subprocess.TimeoutExpired:
        print("Data ingestion timed out after 5 minutes")
        return Response("Data ingestion timed out. Please try again.", status=500)
    except Exception as e:
        print(f"Error executing data ingestion script: {str(e)}")
        return Response(f"Error loading data: {str(e)}", status=500)


# The API route to disconnect the TTS avatar
@app.route("/api/disconnectAvatar", methods=["POST"])
def disconnectAvatar() -> Response:
    client_id = uuid.UUID(request.headers.get('ClientId'))
    try:
        disconnectAvatarInternal(client_id, False)
        return Response('Disconnected avatar', status=200)
    except Exception:
        return Response(traceback.format_exc(), status=400)


# The API route to release the client context, to be invoked when the client is closed
@app.route("/api/releaseClient", methods=["POST"])
def releaseClient() -> Response:
    client_id = uuid.UUID(json.loads(request.data)['clientId'])
    try:
        disconnectAvatarInternal(client_id, False)
        disconnectSttInternal(client_id)
        time.sleep(2)  # Wait some time for the connection to close
        client_contexts.pop(client_id)
        print(f"Client context released for client {client_id}.")
        return Response('Client context released.', status=200)
    except Exception as e:
        print(f"Client context release failed. Error message: {e}")
        return Response(f"Client context release failed. Error message: {e}", status=400)


@socketio.on("connect")
def handleWsConnection():
    client_id = uuid.UUID(request.args.get('clientId'))
    join_room(client_id)
    print(f"WebSocket connected for client {client_id}.")


@socketio.on("message")
def handleWsMessage(message):
    client_id = uuid.UUID(message.get('clientId'))
    path = message.get('path')
    client_context = client_contexts[client_id]
    if path == 'api.audio':
        chat_initiated = client_context['chat_initiated']
        audio_chunk = message.get('audioChunk')
        audio_chunk_binary = base64.b64decode(audio_chunk)
        audio_input_stream = client_context['audio_input_stream']
        if audio_input_stream:
            audio_input_stream.write(audio_chunk_binary)
        if vad_iterator:
            audio_buffer = client_context['vad_audio_buffer']
            audio_buffer.extend(audio_chunk_binary)
            if len(audio_buffer) >= 1024:
                audio_chunk_int = np.frombuffer(bytes(audio_buffer[:1024]), dtype=np.int16)
                audio_buffer.clear()
                audio_chunk_float = int2float(audio_chunk_int)
                vad_detected = vad_iterator(torch.from_numpy(audio_chunk_float))
                if vad_detected:
                    print("Voice activity detected.")
                    stopSpeakingInternal(client_id, False)
    elif path == 'api.chat':
        chat_initiated = client_context['chat_initiated']
        if not chat_initiated:
            initializeChatContext(message.get('systemPrompt'), client_id)
            client_context['chat_initiated'] = True
        
        # Check if we need to send initial greeting for websocket mode
        initial_greeting_sent = client_context.get('initial_greeting_sent', True)
        initial_greeting = client_context.get('initial_greeting', '')
        
        user_query = message.get('userQuery')
        
        # If this is the first interaction and we have an initial greeting, send it first
        if not initial_greeting_sent and initial_greeting:
            client_context['initial_greeting_sent'] = True
            socketio.emit("response", {'path': 'api.chat', 'chatResponse': 'Assistant: ' + initial_greeting + '\n\n'}, room=client_id)
        
        first_response_chunk = True
        for chat_response in handleUserQuery(user_query, client_id):
            if first_response_chunk:
                socketio.emit("response", {'path': 'api.chat', 'chatResponse': 'Assistant: '}, room=client_id)
                first_response_chunk = False
            socketio.emit("response", {'path': 'api.chat', 'chatResponse': chat_response}, room=client_id)
    elif path == 'api.stopSpeaking':
        stopSpeakingInternal(client_id, False)


# Initialize the client by creating a client id and an initial context
def initializeClient() -> uuid.UUID:
    client_id = uuid.uuid4()
    client_contexts[client_id] = {
        'audio_input_stream': None,  # Audio input stream for speech recognition
        'vad_audio_buffer': [],  # Audio input buffer for VAD
        'speech_recognizer': None,  # Speech recognizer for user speech
        'azure_openai_deployment_name': azure_openai_deployment_name,  # Azure OpenAI deployment name
        'cognitive_search_index_name': cognitive_search_index_name,  # Cognitive search index name
        'tts_voice': default_tts_voice,  # TTS voice
        'custom_voice_endpoint_id': None,  # Endpoint ID (deployment ID) for custom voice
        'personal_voice_speaker_profile_id': None,  # Speaker profile ID for personal voice
        'speech_synthesizer': None,  # Speech synthesizer for avatar
        'speech_synthesizer_connection': None,  # Speech synthesizer connection for avatar
        'speech_synthesizer_connected': False,  # Flag to indicate if the speech synthesizer is connected
        'speech_token': None,  # Speech token for client side authentication with speech service
        'ice_token': None,  # ICE token for ICE/TURN/Relay server connection
        'chat_initiated': False,  # Flag to indicate if the chat context is initiated
        'messages': [],  # Chat messages (history)
        'data_sources': [],  # Data sources for 'on your data' scenario
        'is_speaking': False,  # Flag to indicate if the avatar is speaking
        'speaking_text': None,  # The text that the avatar is speaking
        'spoken_text_queue': [],  # Queue to store the spoken text
        'speaking_thread': None,  # The thread to speak the spoken text queue
        'last_speak_time': None,  # The last time the avatar spoke
        'initial_greeting_sent': False,  # Flag to indicate if initial greeting has been sent
        'initial_greeting': None,  # The initial greeting message
        'patient_name': None,  # The current patient name
        'patient_data': None  # The patient data from Elasticsearch
    }
    return client_id


# Refresh the ICE token every 24 hours
def refreshIceToken() -> None:
    global ice_token
    while True:
        ice_token_response = None
        if speech_private_endpoint:
            if enable_token_auth_for_speech:
                while not speech_token:
                    time.sleep(0.2)
                ice_token_response = requests.get(
                    f'{speech_private_endpoint}/tts/cognitiveservices/avatar/relay/token/v1',
                    headers={'Authorization': f'Bearer {speech_token}'})
            else:
                ice_token_response = requests.get(
                    f'{speech_private_endpoint}/tts/cognitiveservices/avatar/relay/token/v1',
                    headers={'Ocp-Apim-Subscription-Key': speech_key})
        else:
            if enable_token_auth_for_speech:
                while not speech_token:
                    time.sleep(0.2)
                ice_token_response = requests.get(
                    f'https://{speech_region}.tts.speech.microsoft.com/cognitiveservices/avatar/relay/token/v1',
                    headers={'Authorization': f'Bearer {speech_token}'})
            else:
                ice_token_response = requests.get(
                    f'https://{speech_region}.tts.speech.microsoft.com/cognitiveservices/avatar/relay/token/v1',
                    headers={'Ocp-Apim-Subscription-Key': speech_key})
        if ice_token_response.status_code == 200:
            ice_token = ice_token_response.text
        else:
            raise Exception(f"Failed to get ICE token. Status code: {ice_token_response.status_code}")
        time.sleep(60 * 60 * 24)  # Refresh the ICE token every 24 hours


# Refresh the speech token every 9 minutes
def refreshSpeechToken() -> None:
    global speech_token
    while True:
        # Refresh the speech token every 9 minutes
        if speech_private_endpoint:
            credential = DefaultAzureCredential(managed_identity_client_id=user_assigned_managed_identity_client_id)
            token = credential.get_token('https://cognitiveservices.azure.com/.default')
            speech_token = f'aad#{speech_resource_url}#{token.token}'
        else:
            speech_token = requests.post(
                f'https://{speech_region}.api.cognitive.microsoft.com/sts/v1.0/issueToken',
                headers={'Ocp-Apim-Subscription-Key': speech_key}).text
        time.sleep(60 * 9)


# Known drug interactions database (simplified for demo)
DRUG_INTERACTIONS = {
    "Diazepam": {
        "Meclizine": "⚠️ INTERACTION: Both cause drowsiness. Risk of excessive sedation.",
        "Promethazine": "⚠️ INTERACTION: Both cause drowsiness. Risk of excessive sedation.",
        "Ondansetron": "⚠️ INTERACTION: May increase risk of QT prolongation."
    },
    "Meclizine": {
        "Diazepam": "⚠️ INTERACTION: Both cause drowsiness. Risk of excessive sedation.",
        "Promethazine": "⚠️ INTERACTION: Both cause drowsiness. Risk of excessive sedation."
    },
    "Promethazine": {
        "Diazepam": "⚠️ INTERACTION: Both cause drowsiness. Risk of excessive sedation.",
        "Meclizine": "⚠️ INTERACTION: Both cause drowsiness. Risk of excessive sedation."
    },
    "Omeprazole": {
        "Diazepam": "⚠️ INTERACTION: May increase Diazepam levels. Monitor for increased sedation."
    }
}

def checkMedicationInteractions(new_medications: list, existing_medications: list) -> list:
    """
    Check for drug interactions between new and existing medications.
    Returns a list of interaction warnings.
    """
    interactions = []
    
    for new_med in new_medications:
        if new_med in DRUG_INTERACTIONS:
            for existing_med in existing_medications:
                if existing_med in DRUG_INTERACTIONS[new_med]:
                    interactions.append(DRUG_INTERACTIONS[new_med][existing_med])
    
    return interactions

def extractMedicationsFromText(text: str) -> list:
    """
    Extract medication names from text using simple pattern matching.
    This is a basic implementation - in production, you'd use more sophisticated NLP.
    """
    # Common medication patterns
    medications = []
    text_lower = text.lower()
    
    # Check for common medications in the patient data
    known_meds = ["Mucinex", "Ondansetron", "Meclizine", "Diazepam", "Omeprazole", "Promethazine"]
    
    for med in known_meds:
        if med.lower() in text_lower:
            medications.append(med)
    
    # Check for prescription patterns
    import re
    prescription_patterns = [
        r"prescribed\s+([A-Z][a-z]+)",
        r"prescribe\s+([A-Z][a-z]+)",
        r"give\s+([A-Z][a-z]+)",
        r"start\s+([A-Z][a-z]+)",
        r"medication:\s*([A-Z][a-z]+)"
    ]
    
    for pattern in prescription_patterns:
        matches = re.findall(pattern, text, re.IGNORECASE)
        medications.extend(matches)
    
    return list(set(medications))  # Remove duplicates

def getPatientMedicationHistory(patient_data: dict) -> list:
    """
    Extract all medications from patient's history.
    """
    all_medications = []
    
    if patient_data and 'records' in patient_data:
        for record in patient_data['records']:
            drugs = record.get('drugs_prescribed', [])
            if drugs and drugs != ["None"]:
                all_medications.extend(drugs)
    
    return list(set(all_medications))  # Remove duplicates

# Query patient data from Elasticsearch
def queryPatientData(patient_name: str) -> dict:
    """
    Query patient records from Elasticsearch based on patient name.
    Returns a dictionary containing patient data or error information.
    """
    print(f"Querying patient data for: '{patient_name}'")
    print(f"Elasticsearch client configured: {elastic_client is not None}")
    print(f"Elasticsearch index: {elastic_index_name}")
    
    if not elastic_client or not elastic_index_name:
        return {"error": "Elasticsearch client not configured"}
    
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
        
        print(f"Executing Elasticsearch query: {json.dumps(query, indent=2)}")
        
        # Execute the search
        response = elastic_client.search(
            index=elastic_index_name,
            body=query
        )
        
        # Convert response to dictionary if it's an ObjectApiResponse
        if hasattr(response, 'body'):
            response_dict = response.body
        else:
            response_dict = response
        
        print(f"Elasticsearch response: {json.dumps(response_dict, indent=2)}")
        
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
        print(f"Error querying patient data: {str(e)}")
        import traceback
        traceback.print_exc()
        return {"error": f"Failed to query patient data: {str(e)}"}


def getDetailedMedicationInfo(patient_name: str, patient_data: dict, function_args: dict) -> str:
    """
    Get detailed medication information based on the query type.
    Returns a natural, conversational response about medications.
    """
    if not patient_data.get('success'):
        return f"I don't have any medical records for {patient_name}."
    
    records = patient_data.get('records', [])
    if not records:
        return f"I couldn't find any medical records for {patient_name}."
    
    # Sort records by date (most recent first)
    sorted_records = sorted(records, key=lambda x: x.get('date_of_visit', ''), reverse=True)
    
    query_type = function_args.get('medication_query_type', 'last_visit')
    visit_date = function_args.get('visit_date')
    
    if query_type == 'last_visit':
        # Get medications from the most recent visit
        recent_visit = sorted_records[0] if sorted_records else None
        if not recent_visit:
            return f"I don't have recent visit information for {patient_name}."
        
        recent_meds = recent_visit.get('drugs_prescribed', [])
        recent_date = recent_visit.get('date_of_visit', 'Unknown date')
        
        # Format the date nicely
        if recent_date == "3-DAYS-AGO":
            date_text = "3 days ago"
        elif recent_date and recent_date != "Unknown date":
            try:
                from datetime import datetime
                parsed_date = datetime.strptime(recent_date, "%Y-%m-%d")
                date_text = parsed_date.strftime("%B %d, %Y")
            except:
                date_text = recent_date
        else:
            date_text = "recently"
        
        if recent_meds and recent_meds != ["None"]:
            if len(recent_meds) == 1:
                return f"Yes, during her last visit {date_text}, {patient_name} was prescribed {recent_meds[0]}."
            else:
                meds_text = ", ".join(recent_meds[:-1]) + f" and {recent_meds[-1]}"
                return f"Yes, during her last visit {date_text}, {patient_name} was prescribed {meds_text}."
        else:
            return f"No medications were prescribed during her last visit {date_text}."
    
    elif query_type == 'current':
        # Get all current medications (from recent visits)
        all_medications = []
        for record in sorted_records[:3]:  # Check last 3 visits
            meds = record.get('drugs_prescribed', [])
            if meds and meds != ["None"]:
                all_medications.extend(meds)
        
        # Remove duplicates while preserving order
        current_meds = list(dict.fromkeys(all_medications))
        
        if current_meds:
            if len(current_meds) == 1:
                return f"{patient_name} is currently taking {current_meds[0]}."
            else:
                meds_text = ", ".join(current_meds[:-1]) + f" and {current_meds[-1]}"
                return f"{patient_name} is currently taking {meds_text}."
        else:
            return f"{patient_name} is not currently taking any medications."
    
    elif query_type == 'all_history':
        # Get all medications from all visits
        all_medications = []
        for record in sorted_records:
            meds = record.get('drugs_prescribed', [])
            if meds and meds != ["None"]:
                all_medications.extend(meds)
        
        # Remove duplicates while preserving order
        unique_meds = list(dict.fromkeys(all_medications))
        
        if unique_meds:
            if len(unique_meds) == 1:
                return f"Throughout her medical history, {patient_name} has been prescribed {unique_meds[0]}."
            else:
                meds_text = ", ".join(unique_meds[:-1]) + f" and {unique_meds[-1]}"
                return f"Throughout her medical history, {patient_name} has been prescribed {meds_text}."
        else:
            return f"No medications have been prescribed to {patient_name} in her medical history."
    
    elif query_type == 'specific_visit' and visit_date:
        # Get medications from a specific visit
        target_record = None
        for record in sorted_records:
            if record.get('date_of_visit') == visit_date:
                target_record = record
                break
        
        if not target_record:
            return f"I couldn't find a visit on {visit_date} for {patient_name}."
        
        meds = target_record.get('drugs_prescribed', [])
        if meds and meds != ["None"]:
            if len(meds) == 1:
                return f"During the visit on {visit_date}, {patient_name} was prescribed {meds[0]}."
            else:
                meds_text = ", ".join(meds[:-1]) + f" and {meds[-1]}"
                return f"During the visit on {visit_date}, {patient_name} was prescribed {meds_text}."
        else:
            return f"No medications were prescribed during the visit on {visit_date}."
    
    else:
        return f"I need more specific information about what medication information you're looking for."


def createEnhancedPatientSummary(patient_name: str, patient_data: dict) -> str:
    """
    Create a natural, conversational patient summary.
    Returns a human-friendly response that answers the user's question.
    """
    if not patient_data.get('success'):
        return f"I don't have any medical records for {patient_name}."
    
    records = patient_data.get('records', [])
    if not records:
        return f"I couldn't find any medical records for {patient_name}."
    
    # Sort records by date (most recent first)
    sorted_records = sorted(records, key=lambda x: x.get('date_of_visit', ''), reverse=True)
    
    # Get the most recent visit
    recent_visit = sorted_records[0] if sorted_records else None
    
    if not recent_visit:
        return f"I don't have recent visit information for {patient_name}."
    
    # Extract key information from the most recent visit
    recent_date = recent_visit.get('date_of_visit', 'Unknown date')
    recent_condition = recent_visit.get('diagnosis', 'Unknown condition')
    recent_complaint = recent_visit.get('patient_complaint', '')
    recent_notes = recent_visit.get('doctor_notes', '')
    recent_meds = recent_visit.get('drugs_prescribed', [])
    
    # Format the date nicely
    if recent_date == "3-DAYS-AGO":
        date_text = "3 days ago"
    elif recent_date and recent_date != "Unknown date":
        try:
            from datetime import datetime
            parsed_date = datetime.strptime(recent_date, "%Y-%m-%d")
            date_text = parsed_date.strftime("%B %d, %Y")
        except:
            date_text = recent_date
    else:
        date_text = "recently"
    
    # Build a natural response
    response_parts = []
    
    # Clean up the condition name for better readability
    clean_condition = recent_condition.rstrip('.').replace('Recurrence of ', 'recurring ')
    if 'BPPV' in clean_condition:
        clean_condition = clean_condition.replace('BPPV', 'vertigo')
    
    # Start with the most recent visit
    response_parts.append(f"Jane's most recent visit was {date_text} for {clean_condition.lower()}")
    
    # Add complaint if available and relevant
    if recent_complaint and len(recent_complaint) < 100:  # Keep it concise
        # Clean up the complaint text
        clean_complaint = recent_complaint.lower().rstrip('.')
        response_parts.append(f"She came in with {clean_complaint}")
    
    # Add current medications if any
    if recent_meds and recent_meds != ["None"]:
        if len(recent_meds) == 1:
            response_parts.append(f"She's currently taking {recent_meds[0]}")
        else:
            meds_text = ", ".join(recent_meds[:-1]) + f" and {recent_meds[-1]}"
            response_parts.append(f"She's currently taking {meds_text}")
    
    # Add medication history context if relevant
    all_medications = []
    for record in sorted_records[:5]:  # Check last 5 visits for medication history
        meds = record.get('drugs_prescribed', [])
        if meds and meds != ["None"]:
            all_medications.extend(meds)
    
    # Remove duplicates while preserving order
    unique_meds = list(dict.fromkeys(all_medications))
    
    # If patient has been on multiple medications, add context
    if len(unique_meds) > 1 and recent_meds and recent_meds != ["None"]:
        if recent_meds[0] not in unique_meds[:2]:  # If current med is not one of the most common
            response_parts.append(f"She has a history of vertigo-related medications")
    
    # Add key clinical insight
    if "BPPV" in recent_condition or "vertigo" in clean_condition.lower():
        response_parts.append("This is a recurring condition for her")
    
    # Combine into a natural sentence
    if len(response_parts) == 1:
        return response_parts[0] + "."
    elif len(response_parts) == 2:
        return f"{response_parts[0]}. {response_parts[1].capitalize()}."
    else:
        # For 3+ parts, create a flowing sentence
        main_response = response_parts[0]
        additional_info = ". ".join(response_parts[1:])
        return f"{main_response}. {additional_info.capitalize()}."


# Initialize the chat context, e.g. chat history (messages), data sources, etc. For chat scenario.
def initializeChatContext(system_prompt: str, client_id: uuid.UUID) -> None:
    client_context = client_contexts[client_id]
    cognitive_search_index_name = client_context['cognitive_search_index_name']
    messages = client_context['messages']
    data_sources = client_context['data_sources']

    # Clear patient-specific data to start fresh
    client_context['patient_name'] = None
    client_context['patient_data'] = None
    client_context['initial_greeting_sent'] = False
    client_context['initial_greeting'] = None

    # Initialize data sources for 'on your data' scenario
    data_sources.clear()
    if cognitive_search_endpoint and cognitive_search_api_key and cognitive_search_index_name:
        # On-your-data scenario
        data_source = {
            'type': 'azure_search',
            'parameters': {
                'endpoint': cognitive_search_endpoint,
                'index_name': cognitive_search_index_name,
                'authentication': {
                    'type': 'api_key',
                    'key': cognitive_search_api_key
                },
                'semantic_configuration': '',
                'query_type': 'simple',
                'fields_mapping': {
                    'content_fields_separator': '\n',
                    'content_fields': ['content'],
                    'filepath_field': None,
                    'title_field': 'title',
                    'url_field': None
                },
                'in_scope': True,
                'role_information': system_prompt
            }
        }
        data_sources.append(data_source)

    # Initialize messages
    messages.clear()
    if len(data_sources) == 0:
        system_message = {
            'role': 'system',
            'content': system_prompt
        }
        messages.append(system_message)


# Handle the user query and return the assistant reply. For chat scenario.
# The function is a generator, which yields the assistant reply in chunks.
def handleUserQuery(user_query: str, client_id: uuid.UUID):
    client_context = client_contexts[client_id]
    azure_openai_deployment_name = client_context['azure_openai_deployment_name']
    messages = client_context['messages']
    data_sources = client_context['data_sources']
    patient_name = client_context.get('patient_name')
    patient_data = client_context.get('patient_data')

    # Add user message to conversation
    chat_message = {
        'role': 'user',
        'content': user_query
    }
    messages.append(chat_message)

    # Check for medication interactions if patient data is available
    if patient_data and patient_data.get('success'):
        # Extract medications from user query
        new_medications = extractMedicationsFromText(user_query)
        
        if new_medications:
            # Get patient's medication history
            existing_medications = getPatientMedicationHistory(patient_data)
            
            # Check for interactions
            interactions = checkMedicationInteractions(new_medications, existing_medications)
            
            if interactions:
                # Add interaction warnings to the conversation
                interaction_warning = " ".join(interactions)
                print(f"🚨 Medication interactions detected: {interaction_warning}")
                
                # Add interaction warning as a system message
                interaction_message = {
                    'role': 'system',
                    'content': f"MEDICATION INTERACTION ALERT: {interaction_warning} Please review before prescribing."
                }
                messages.append(interaction_message)

    # For 'on your data' scenario, chat API currently has long (4s+) latency
    # We return some quick reply here before the chat API returns to mitigate.
    if len(data_sources) > 0 and enable_quick_reply:
        speakWithQueue(random.choice(quick_replies), 2000)

    assistant_reply = ''
    tool_content = ''
    spoken_sentence = ''
    
    # Variables for response handling

    # Define MCP tools for the LLM
    tools = [
        {
            "type": "function",
            "function": {
                "name": "get_patient_data",
                "description": "CRITICAL: Use this tool IMMEDIATELY when ANY patient name is mentioned for the FIRST TIME. Retrieve patient medical records and history from the clinical database. Do NOT ask for more context - just fetch the data automatically. This tool provides a natural conversational response with key clinical insights.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "patient_name": {
                            "type": "string",
                            "description": "Full name of the patient to retrieve records for"
                        }
                    },
                    "required": ["patient_name"]
                }
            }
        },
        {
            "type": "function",
            "function": {
                "name": "get_patient_summary",
                "description": "Get a comprehensive clinical summary of patient medical history. Use this when patient data is already loaded and user asks for 'summary', 'overview', 'last visit', or similar requests. This provides enhanced clinical insights including recurring conditions, current medications, and visit patterns.",
                "parameters": {
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
            }
        },
        {
            "type": "function",
            "function": {
                "name": "check_medication_interactions",
                "description": "Check for potential drug interactions between medications. Use this when prescribing new medications or when user asks about medication safety.",
                "parameters": {
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
        },
        {
            "type": "function",
            "function": {
                "name": "get_medication_info",
                "description": "Get detailed medication information for a patient. Use this when user asks about medications, prescriptions, drugs prescribed, current medications, or medication history.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "patient_name": {
                            "type": "string",
                            "description": "Full name of the patient"
                        },
                        "medication_query_type": {
                            "type": "string",
                            "enum": ["last_visit", "current", "all_history", "specific_visit"],
                            "description": "Type of medication query - last visit, current medications, all history, or specific visit",
                            "default": "last_visit"
                        },
                        "visit_date": {
                            "type": "string",
                            "description": "Specific visit date if querying a particular visit (optional)"
                        }
                    },
                    "required": ["patient_name"]
                }
            }
        }
    ]

    # Check if user query contains a patient name and force tool usage
    import re
    
    # More comprehensive patient name detection
    patient_name_patterns = [
        r"jane\s+doe", r"john\s+doe",  # Specific names from the data
        r"patient\s+name", r"patient\s+is", r"patient\s+called",
        r"mr\.?\s+\w+", r"ms\.?\s+\w+", r"mrs\.?\s+\w+", r"dr\.?\s+\w+",
        r"my\s+patient", r"the\s+patient", r"this\s+patient"
    ]
    
    user_query_lower = user_query.lower()
    contains_patient_name = any(re.search(pattern, user_query_lower) for pattern in patient_name_patterns)
    
    # Smart tool usage logic
    tool_choice = "auto"
    extracted_patient_name = None
    
    # Check if this is a summary request for an already loaded patient
    summary_keywords = ["summarize", "summary", "overview", "last visit", "recent", "history"]
    is_summary_request = any(keyword in user_query_lower for keyword in summary_keywords)
    
    # Check if this is a medication-related query
    medication_keywords = ["medication", "medicines", "drugs", "prescribed", "prescription", "taking", "current medication", "medication history", "drug history"]
    is_medication_query = any(keyword in user_query_lower for keyword in medication_keywords)
    
    if contains_patient_name and not patient_data:
        # First time patient name mentioned - get data
        tool_choice = {"type": "function", "function": {"name": "get_patient_data"}}
        print(f"🔍 Patient name detected in query: '{user_query}' - Forcing get_patient_data tool usage")
        
        # Extract patient name for fallback
        if "jane doe" in user_query_lower:
            extracted_patient_name = "Jane Doe"
        elif "john doe" in user_query_lower:
            extracted_patient_name = "John Doe"
        else:
            # Try to extract name from common patterns
            name_match = re.search(r"(?:patient\s+(?:name\s+)?(?:is\s+)?|mr\.?\s+|ms\.?\s+|mrs\.?\s+|dr\.?\s+)([A-Z][a-z]+\s+[A-Z][a-z]+)", user_query)
            if name_match:
                extracted_patient_name = name_match.group(1)
    elif patient_data and is_medication_query:
        # Patient data already loaded and user asks about medications - use medication tool
        tool_choice = {"type": "function", "function": {"name": "get_medication_info"}}
        print(f"💊 Medication query detected for loaded patient - Using get_medication_info tool")
        print(f"🔍 Patient data available: {patient_data.get('success', False)}")
        print(f"🔍 Patient name: {patient_name}")
        extracted_patient_name = patient_name
    elif patient_data and is_summary_request:
        # Patient data already loaded and user wants summary - use summary tool
        tool_choice = {"type": "function", "function": {"name": "get_patient_summary"}}
        print(f"📋 Summary request detected for loaded patient - Using get_patient_summary tool")
        print(f"🔍 Patient data available: {patient_data.get('success', False)}")
        print(f"🔍 Patient name: {patient_name}")
        extracted_patient_name = patient_name
    
    aoai_start_time = datetime.datetime.now(pytz.UTC)
    # For tool calls, use non-streaming to avoid complexity
    if tool_choice != "auto":
        # Specific tool choice - use non-streaming
        response = azure_openai.chat.completions.create(
            model=azure_openai_deployment_name,
            messages=messages,
            tools=tools,
            tool_choice=tool_choice,  # Force tool usage when patient name detected
            extra_body={'data_sources': data_sources} if len(data_sources) > 0 else None,
            max_tokens=150,  # Limit response to approximately 2-3 sentences
            stream=False)
    else:
        # Auto tool choice - use streaming
        response = azure_openai.chat.completions.create(
            model=azure_openai_deployment_name,
            messages=messages,
            tools=tools,
            tool_choice=tool_choice,  # Force tool usage when patient name detected
            extra_body={'data_sources': data_sources} if len(data_sources) > 0 else None,
            max_tokens=150,  # Limit response to approximately 2-3 sentences
            stream=True)

    # Handle non-streaming response (for tool calls)
    if tool_choice != "auto":
        print(f"🔧 Processing non-streaming response with tool_choice: {tool_choice}")
        # Handle tool calls in non-streaming mode
        if response.choices[0].message.tool_calls:
            print(f"🔧 Tool calls found: {len(response.choices[0].message.tool_calls)}")
            for tool_call in response.choices[0].message.tool_calls:
                function_name = tool_call.function.name
                function_args = json.loads(tool_call.function.arguments)
                
                print(f"✅ Tool call: {function_name} with args: {function_args}")
                # Don't show technical tool call messages to user - just process silently
                
                if function_name == "get_patient_data":
                    patient_name = function_args.get("patient_name")
                    if not patient_name:
                        print(f"⚠️ No patient_name provided in function arguments")
                        continue
                        
                    result = queryPatientData(patient_name)
                    # Store patient data in context
                    client_context['patient_name'] = patient_name
                    client_context['patient_data'] = result
                    
                    # Add a conversational response to the result
                    if result.get('success'):
                        # Create a natural, conversational response
                        total_records = result.get('total_records', 0)
                        
                        # Get the most recent visit for context
                        recent_visit = None
                        if result.get('records'):
                            recent_records = sorted(result['records'], key=lambda x: x.get('date_of_visit', ''), reverse=True)
                            recent_visit = recent_records[0] if recent_records else None
                        
                        # Build a natural response
                        if recent_visit:
                            recent_condition = recent_visit.get('diagnosis', '')
                            # Clean up the condition name
                            if recent_condition:
                                # Remove trailing periods and make it more readable
                                recent_condition = recent_condition.rstrip('.').replace('Recurrence of ', 'recurring ')
                                if 'BPPV' in recent_condition:
                                    recent_condition = recent_condition.replace('BPPV', 'vertigo')
                                
                                conversational_response = f"I found {total_records} medical records for {patient_name}. Her most recent visit was for {recent_condition.lower()}. What would you like to know about her care?"
                            else:
                                conversational_response = f"I found {total_records} medical records for {patient_name}. What would you like to know about her care?"
                        else:
                            conversational_response = f"I found {total_records} medical records for {patient_name}. What would you like to know about her care?"
                        
                        result['conversational_response'] = conversational_response
                        # Yield the conversational response so it gets spoken
                        yield conversational_response
                        # Also add to speech queue
                        speakWithQueue(conversational_response, 0, client_id)
                    else:
                        conversational_response = f"No records found for {patient_name}. Please verify the patient name."
                        result['conversational_response'] = conversational_response
                        # Yield the conversational response so it gets spoken
                        yield conversational_response
                        # Also add to speech queue
                        speakWithQueue(conversational_response, 0, client_id)
                
                elif function_name == "get_patient_summary":
                    # First get patient data if not already available
                    patient_name = function_args.get("patient_name")
                    if not patient_name:
                        print(f"⚠️ No patient_name provided in function arguments")
                        continue
                        
                    if not client_context.get('patient_data'):
                        patient_data = queryPatientData(patient_name)
                        client_context['patient_data'] = patient_data
                    else:
                        patient_data = client_context['patient_data']
                    
                    # Create enhanced summary with clinical insights
                    if patient_data.get('success'):
                        summary = createEnhancedPatientSummary(patient_name, patient_data)
                        result = {"summary": summary, "patient_data": patient_data}
                        
                        # Yield the summary so it gets spoken
                        yield summary
                        # Also add to speech queue
                        speakWithQueue(summary, 0, client_id)
                    else:
                        result = {"error": "No patient data found"}
                        error_msg = f"No patient data found for {patient_name}. Please verify the patient name."
                        yield error_msg
                        speakWithQueue(error_msg, 0, client_id)
                
                elif function_name == "check_medication_interactions":
                    new_meds = function_args.get("new_medications", [])
                    existing_meds = function_args.get("existing_medications", [])
                    interactions = checkMedicationInteractions(new_meds, existing_meds)
                    result = {"interactions": interactions}
                
                elif function_name == "get_medication_info":
                    # First get patient data if not already available
                    patient_name = function_args.get("patient_name")
                    if not patient_name:
                        print(f"⚠️ No patient_name provided in function arguments")
                        continue
                        
                    if not client_context.get('patient_data'):
                        patient_data = queryPatientData(patient_name)
                        client_context['patient_data'] = patient_data
                    else:
                        patient_data = client_context['patient_data']
                    
                    # Get medication information
                    if patient_data.get('success'):
                        medication_info = getDetailedMedicationInfo(patient_name, patient_data, function_args)
                        result = {"medication_info": medication_info}
                        
                        # Yield the medication info so it gets spoken
                        yield medication_info
                        # Also add to speech queue
                        speakWithQueue(medication_info, 0, client_id)
                    else:
                        result = {"error": "No patient data found"}
                        error_msg = f"No patient data found for {patient_name}. Please verify the patient name."
                        yield error_msg
                        speakWithQueue(error_msg, 0, client_id)
                
                else:
                    result = {"error": f"Unknown function: {function_name}"}
                
                # Don't add tool messages to conversation history since we're handling execution directly
                # This prevents the "tool message must be response to tool_calls" error
                
                # Don't show technical completion messages to user
        
        # Handle regular text response in non-streaming mode
        if response.choices[0].message.content:
            assistant_reply = response.choices[0].message.content
            yield assistant_reply
            speakWithQueue(assistant_reply, 0, client_id)
            
            # Add assistant message to conversation
            assistant_message = {
                'role': 'assistant',
                'content': assistant_reply
            }
            messages.append(assistant_message)
        else:
            # No tool calls and no content - this shouldn't happen, but let's handle it
            print(f"⚠️ No tool calls and no content in response")
    
    else:
        # Handle streaming response (for regular text)
        is_first_chunk = True
        is_first_sentence = True
        for chunk in response:
            if len(chunk.choices) > 0:
                choice = chunk.choices[0]
                
                # Handle regular text response
                response_token = choice.delta.content
                if response_token is not None:
                    # Log response_token here if need debug
                    if is_first_chunk:
                        first_token_latency_ms = round((datetime.datetime.now(pytz.UTC) - aoai_start_time).total_seconds() * 1000)
                        print(f"AOAI first token latency: {first_token_latency_ms}ms")
                        yield f"<FTL>{first_token_latency_ms}</FTL>"
                        is_first_chunk = False
                    if oyd_doc_regex.search(response_token):
                        response_token = oyd_doc_regex.sub('', response_token).strip()
                    yield response_token  # yield response token to client as display text
                    assistant_reply += response_token  # build up the assistant message
                    if response_token == '\n' or response_token == '\n\n':
                        if is_first_sentence:
                            first_sentence_latency_ms = round((datetime.datetime.now(pytz.UTC) - aoai_start_time).total_seconds() * 1000)
                            print(f"AOAI first sentence latency: {first_sentence_latency_ms}ms")
                            yield f"<FSL>{first_sentence_latency_ms}</FSL>"
                            is_first_sentence = False
                        speakWithQueue(spoken_sentence.strip(), 0, client_id)
                        spoken_sentence = ''
                    else:
                        response_token = response_token.replace('\n', '')
                        spoken_sentence += response_token  # build up the spoken sentence
                        if len(response_token) == 1 or len(response_token) == 2:
                            for punctuation in sentence_level_punctuations:
                                if response_token.startswith(punctuation):
                                    if is_first_sentence:
                                        first_sentence_latency_ms = round((datetime.datetime.now(pytz.UTC) - aoai_start_time).total_seconds() * 1000)
                                        print(f"AOAI first sentence latency: {first_sentence_latency_ms}ms")
                                        yield f"<FSL>{first_sentence_latency_ms}</FSL>"
                                        is_first_sentence = False
                                    speakWithQueue(spoken_sentence.strip(), 0, client_id)
                                    spoken_sentence = ''
                                    break

    if spoken_sentence != '':
        speakWithQueue(spoken_sentence.strip(), 0, client_id)
        spoken_sentence = ''

    # Fallback: If patient name was detected but no tool was called, force the tool call
    if contains_patient_name and extracted_patient_name and not client_context.get('patient_data'):
        print(f"🔄 Fallback: Automatically calling get_patient_data for '{extracted_patient_name}'")
        try:
            result = queryPatientData(extracted_patient_name)
            client_context['patient_name'] = extracted_patient_name
            client_context['patient_data'] = result
            
            # Note: We don't add tool messages to conversation history in fallback mode
            # The patient data is stored in client_context and will be used by tools when needed
            
            # Generate a conversational response
            if result.get('success'):
                conversational_response = f"Patient information found for {extracted_patient_name}. What would you like to know about this patient?"
                yield conversational_response
                # Also add to speech queue
                speakWithQueue(conversational_response, 0, client_id)
            else:
                conversational_response = f"No records found for {extracted_patient_name}. Please verify the patient name."
                yield conversational_response
                # Also add to speech queue
                speakWithQueue(conversational_response, 0, client_id)
                
        except Exception as e:
            print(f"Error in fallback patient data retrieval: {e}")
            yield f"Error retrieving data for {extracted_patient_name}."

    # Note: Removed invalid tool message addition that was causing API errors
    # Tool messages should only be added in response to actual tool_calls from the assistant

    # Enforce response length limit (maximum 20 words)
    word_count = len(assistant_reply.split())
    if word_count > 20:
        words = assistant_reply.split()[:20]
        assistant_reply = ' '.join(words) + '...'
        print(f"⚠️ Response truncated from {word_count} to 20 words")

    assistant_message = {
        'role': 'assistant',
        'content': assistant_reply
    }
    messages.append(assistant_message)


# Speak the given text. If there is already a speaking in progress, add the text to the queue. For chat scenario.
def speakWithQueue(text: str, ending_silence_ms: int, client_id: uuid.UUID) -> None:
    client_context = client_contexts[client_id]
    spoken_text_queue = client_context['spoken_text_queue']
    is_speaking = client_context['is_speaking']
    if text:
        spoken_text_queue.append(text)
    if not is_speaking:
        def speakThread():
            spoken_text_queue = client_context['spoken_text_queue']
            tts_voice = client_context['tts_voice']
            personal_voice_speaker_profile_id = client_context['personal_voice_speaker_profile_id']
            client_context['is_speaking'] = True
            while len(spoken_text_queue) > 0:
                text = spoken_text_queue.pop(0)
                client_context['speaking_text'] = text
                try:
                    speakText(text, tts_voice, personal_voice_speaker_profile_id, ending_silence_ms, client_id)
                except Exception as e:
                    print(f"Error in speaking text: {e}")
                    break
                client_context['last_speak_time'] = datetime.datetime.now(pytz.UTC)
            client_context['is_speaking'] = False
            client_context['speaking_text'] = None
            print("Speaking thread stopped.")
        client_context['speaking_thread'] = threading.Thread(target=speakThread)
        client_context['speaking_thread'].start()


# Speak the given text.
def speakText(text: str, voice: str, speaker_profile_id: str, ending_silence_ms: int, client_id: uuid.UUID) -> str:
    # Optimized SSML for lower latency - remove unnecessary elements and optimize for speed
    if speaker_profile_id:
        ssml = f"""<speak version='1.0' xmlns='http://www.w3.org/2001/10/synthesis' xmlns:mstts='http://www.w3.org/2001/mstts' xml:lang='en-US'>
                     <voice name='{voice}'>
                         <mstts:ttsembedding speakerProfileId='{speaker_profile_id}'>
                             <mstts:leadingsilence-exact value='0'/>
                             <mstts:trailingsilence-exact value='0'/>
                             <prosody rate='1.0' pitch='0%'>
                                 {html.escape(text)}
                             </prosody>
                         </mstts:ttsembedding>
                     </voice>
                   </speak>"""  # noqa: E501
    else:
        # Simplified SSML without personal voice for faster processing
        ssml = f"""<speak version='1.0' xmlns='http://www.w3.org/2001/10/synthesis' xml:lang='en-US'>
                     <voice name='{voice}'>
                         <prosody rate='1.0' pitch='0%'>
                             {html.escape(text)}
                         </prosody>
                     </voice>
                   </speak>"""  # noqa: E501
    
    if ending_silence_ms > 0:
        if speaker_profile_id:
            ssml = f"""<speak version='1.0' xmlns='http://www.w3.org/2001/10/synthesis' xmlns:mstts='http://www.w3.org/2001/mstts' xml:lang='en-US'>
                         <voice name='{voice}'>
                             <mstts:ttsembedding speakerProfileId='{speaker_profile_id}'>
                                 <mstts:leadingsilence-exact value='0'/>
                                 <mstts:trailingsilence-exact value='0'/>
                                 <prosody rate='1.0' pitch='0%'>
                                     {html.escape(text)}
                                     <break time='{ending_silence_ms}ms' />
                                 </prosody>
                             </mstts:ttsembedding>
                         </voice>
                       </speak>"""  # noqa: E501
        else:
            ssml = f"""<speak version='1.0' xmlns='http://www.w3.org/2001/10/synthesis' xml:lang='en-US'>
                         <voice name='{voice}'>
                             <prosody rate='1.0' pitch='0%'>
                                 {html.escape(text)}
                                 <break time='{ending_silence_ms}ms' />
                             </prosody>
                         </voice>
                       </speak>"""  # noqa: E501
    return speakSsml(ssml, client_id, False)


# Speak the given ssml with speech sdk
def speakSsml(ssml: str, client_id: uuid.UUID, asynchronized: bool) -> str:
    speech_synthesizer = client_contexts[client_id]['speech_synthesizer']
    speech_sythesis_result = (
        speech_synthesizer.start_speaking_ssml_async(ssml).get() if asynchronized
        else speech_synthesizer.speak_ssml_async(ssml).get())
    if speech_sythesis_result.reason == speechsdk.ResultReason.Canceled:
        cancellation_details = speech_sythesis_result.cancellation_details
        print(f"Speech synthesis canceled: {cancellation_details.reason}")
        if cancellation_details.reason == speechsdk.CancellationReason.Error:
            print(f"Result ID: {speech_sythesis_result.result_id}. Error details: {cancellation_details.error_details}")
            raise Exception(cancellation_details.error_details)
    return speech_sythesis_result.result_id


# Stop speaking internal function
def stopSpeakingInternal(client_id: uuid.UUID, skipClearingSpokenTextQueue: bool) -> None:
    # Check if client context exists before accessing it
    if client_id not in client_contexts:
        print(f"⚠️ Client context not found for {client_id} during stop speaking")
        return
        
    client_context = client_contexts[client_id]
    client_context['is_speaking'] = False
    if not skipClearingSpokenTextQueue:
        spoken_text_queue = client_context.get('spoken_text_queue', [])
        spoken_text_queue.clear()
    avatar_connection = client_context.get('speech_synthesizer_connection')
    if avatar_connection:
        avatar_connection.send_message_async('synthesis.control', '{"action":"stop"}').get()


# Disconnect avatar internal function
def disconnectAvatarInternal(client_id: uuid.UUID, isReconnecting: bool) -> None:
    # Check if client context exists before accessing it
    if client_id not in client_contexts:
        print(f"⚠️ Client context not found for {client_id} during disconnect")
        return
        
    client_context = client_contexts[client_id]
    stopSpeakingInternal(client_id, isReconnecting)
    time.sleep(2)  # Wait for the speaking thread to stop
    avatar_connection = client_context.get('speech_synthesizer_connection')
    if avatar_connection:
        avatar_connection.close()


# Disconnect STT internal function
def disconnectSttInternal(client_id: uuid.UUID) -> None:
    # Check if client context exists before accessing it
    if client_id not in client_contexts:
        print(f"⚠️ Client context not found for {client_id} during STT disconnect")
        return
        
    client_context = client_contexts[client_id]
    speech_recognizer = client_context.get('speech_recognizer')
    audio_input_stream = client_context.get('audio_input_stream')
    if speech_recognizer:
        speech_recognizer.stop_continuous_recognition()
        connection = speechsdk.Connection.from_recognizer(speech_recognizer)
        connection.close()
        client_context['speech_recognizer'] = None
    if audio_input_stream:
        audio_input_stream.close()
        client_context['audio_input_stream'] = None


# Start the speech token refresh thread
speechTokenRefereshThread = threading.Thread(target=refreshSpeechToken)
speechTokenRefereshThread.daemon = True
speechTokenRefereshThread.start()

# Start the ICE token refresh thread
iceTokenRefreshThread = threading.Thread(target=refreshIceToken)
iceTokenRefreshThread.daemon = True
iceTokenRefreshThread.start()

# Wait for initial ICE token to be available
print("Waiting for initial ICE token...")
while not ice_token:
    time.sleep(1)
print("ICE token initialized successfully!")
