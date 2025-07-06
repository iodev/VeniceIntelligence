# Venice.ai API Documentation - COMPLETE KNOWLEDGE BASE
*All information verified from official Swagger specification and documentation*

## Overview

Venice.ai is a privacy-first AI platform that provides OpenAI-compatible APIs for text and image generation. The platform emphasizes user privacy, open-source models, and uncensored AI responses.

### Key Features
- **Privacy-First Architecture**: Venice does not utilize or store user data for any purposes
- **Open-Source Models**: Only uses open-source models for full transparency
- **OpenAI API Compatible**: Seamless integration with existing OpenAI clients and tools
- **Uncensored Responses**: Default system prompts designed for natural, uncensored model responses
- **No Data Storage**: Venice does not store user conversations or generated content
- **Transparent Operations**: Full visibility into model operations and capabilities

## Base Configuration

### Required Base URL
All API requests must use Venice's base URL:
```
https://api.venice.ai/api/v1
```

### Authentication
Venice uses Bearer token authentication with JWT format:
```
Authorization: Bearer <your-api-key>
```

### Client Setup (OpenAI Compatible)
```javascript
import OpenAI from "openai";

const client = new OpenAI({
  apiKey: "your-api-key",
  baseURL: "https://api.venice.ai/api/v1",
});
```

## Complete API Endpoints Reference

### 1. Chat Completions API

**Endpoint**: `POST /api/v1/chat/completions`

**Purpose**: Generate text responses in a chat-like format with full OpenAI compatibility

**Description**: Run text inference based on the supplied parameters. Long running requests should use the streaming API by setting stream=true in your request.

**Headers**:
- `Authorization: Bearer <token>` (required)
- `Content-Type: application/json` (required)
- `Accept-Encoding: gzip, br` (optional, for compression - only applied when stream is false)

**Request Parameters** (Complete & Verified):

| Parameter | Type | Required | Default | Min | Max | Description |
|-----------|------|----------|---------|-----|-----|-------------|
| `model` | string | Yes | - | - | - | Model ID. May also be a model trait, or a model compatibility mapping. You can use feature suffixes to enable features from the venice_parameters object |
| `messages` | array | Yes | - | 1+ | - | Array of conversation messages comprising the conversation so far |
| `temperature` | number | No | 0.3 | 0 | 2 | Sampling temperature. Higher values make output more random |
| `max_tokens` | integer | No | - | - | - | **DEPRECATED** - Use max_completion_tokens instead |
| `max_completion_tokens` | integer | No | - | - | - | Upper bound for completion tokens including visible output and reasoning tokens |
| `stream` | boolean | No | false | - | - | Enable streaming responses |
| `frequency_penalty` | number | No | 0 | -2 | 2 | Penalize token frequency |
| `presence_penalty` | number | No | 0 | -2 | 2 | Penalize token presence |
| `repetition_penalty` | number | No | - | 0+ | - | Discourage repetition (>1.0) |
| `top_p` | number | No | 1 | 0 | 1 | Nucleus sampling |
| `top_k` | integer | No | - | 0+ | - | Top-k filtering |
| `min_p` | number | No | - | 0 | 1 | Minimum probability threshold |
| `min_temp` | number | No | - | 0 | 2 | Minimum temperature for dynamic scaling |
| `max_temp` | number | No | - | 0 | 2 | Maximum temperature for dynamic scaling |
| `n` | integer | No | 1 | - | - | Number of completions to generate |
| `seed` | integer | No | - | 1+ | - | Random seed for reproducibility |
| `stop` | string/array | No | null | - | 4 items | Stop sequences (up to 4) |
| `stop_token_ids` | array | No | - | - | - | Array of token IDs to stop on |
| `user` | string | No | - | - | - | User identifier (discarded but supported for compatibility) |
| `logprobs` | boolean | No | false | - | - | Include log probabilities (not supported by all models) |
| `top_logprobs` | integer | No | - | 0+ | - | Number of top logprobs to return |
| `stream_options` | object | No | - | - | - | Streaming configuration |
| `response_format` | object | No | - | - | - | Response format specification |
| `tool_choice` | object/string | No | - | - | - | Tool selection strategy |
| `tools` | array | No | - | - | - | Available tools for function calling |
| `parallel_tool_calls` | boolean | No | true | - | - | Enable parallel tool calls |
| `venice_parameters` | object | No | - | - | - | Venice-specific parameters |

**Venice Parameters** (`venice_parameters` object) - **VERIFIED**:

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `character_slug` | string | - | The character slug of a public Venice character |
| `strip_thinking_response` | boolean | false | Strip <think></think> blocks from response (reasoning models only) |
| `disable_thinking` | boolean | false | Disable thinking on reasoning models and strip blocks |
| `enable_web_search` | string | "off" | Web search mode: "auto", "on", "off" |
| `enable_web_citations` | boolean | false | Request LLM to cite sources using [REF]0[/REF] format |
| `include_venice_system_prompt` | boolean | true | Include Venice's default system prompts |

**Message Types** (Complete Schema):

1. **User Message**:
```json
{
  "role": "user",
  "content": "string or array of content objects"
}
```

Content can include:
- Simple string
- Array with text and image_url objects (for vision models):
```json
{
  "role": "user",
  "content": [
    {
      "type": "text",
      "text": "Describe this image"
    },
    {
      "type": "image_url",
      "image_url": {
        "url": "data:image/jpeg;base64,... or public URL"
      }
    }
  ]
}
```

2. **Assistant Message**:
```json
{
  "role": "assistant",
  "content": "string, array, or null",
  "name": "string (optional)",
  "reasoning_content": "string (optional, nullable)",
  "tool_calls": "array (optional, nullable)"
}
```

3. **System Message**:
```json
{
  "role": "system",
  "content": "string or array",
  "name": "string (optional)"
}
```

4. **Tool Message**:
```json
{
  "role": "tool",
  "content": "string",
  "tool_call_id": "string",
  "name": "string (optional)",
  "reasoning_content": "string (optional, nullable)",
  "tool_calls": "array (optional, nullable)"
}
```

**Response Structure** (Verified):
```json
{
  "id": "chatcmpl-123",
  "object": "chat.completion",
  "created": 1713833628,
  "model": "venice-uncensored",
  "choices": [
    {
      "index": 0,
      "message": {
        "role": "assistant",
        "content": "Response text",
        "reasoning_content": "Thinking process (if applicable)"
      },
      "finish_reason": "stop",
      "stop_reason": "string (nullable)",
      "logprobs": {
        "content": [
          {
            "token": "hello",
            "logprob": -0.34,
            "bytes": [104, 101, 108, 108, 111],
            "top_logprobs": [
              {
                "token": "hello",
                "logprob": -0.34,
                "bytes": [104, 101, 108, 108, 111]
              }
            ]
          }
        ]
      }
    }
  ],
  "usage": {
    "prompt_tokens": 20,
    "completion_tokens": 50,
    "total_tokens": 70
  }
}
```

### 2. Models API

**Endpoint**: `GET /api/v1/models`

**Purpose**: Returns a list of available models supported by the Venice.ai API for both text and image inference

**Authentication**: Optional (supports both authenticated and unauthenticated requests)

**Parameters**:
- `type` (optional): Filter by model type
  - Values: "embedding", "image", "text", "tts", "upscale", "all", "code"
  - Example: `?type=text`

**Response Structure** (Verified):
```json
{
  "object": "list",
  "type": "text",
  "data": [
    {
      "id": "model-id",
      "object": "model",
      "created": 1727966436,
      "owned_by": "venice.ai",
      "type": "text",
      "model_spec": {
        "availableContextTokens": 131072,
        "capabilities": {
          "optimizedForCode": false,
          "quantization": "fp16",
          "supportsFunctionCalling": true,
          "supportsReasoning": false,
          "supportsResponseSchema": true,
          "supportsVision": false,
          "supportsWebSearch": true,
          "supportsLogProbs": true
        },
        "constraints": {
          "temperature": {
            "default": 0.8
          },
          "top_p": {
            "default": 0.9
          }
        }
      },
      "modelSource": "https://huggingface.co/...",
      "offline": false,
      "pricing": {
        "input": {
          "usd": 0.15,
          "vcu": 1.5
        },
        "output": {
          "usd": 0.6,
          "vcu": 6
        }
      },
      "traits": ["fastest"]
    }
  ]
}
```

**Model Capabilities** (Complete List):
- `optimizedForCode`: Boolean - Optimized for code generation
- `quantization`: String - Quantization method (e.g., "fp16")
- `supportsFunctionCalling`: Boolean - Tool use support
- `supportsReasoning`: Boolean - Thinking/reasoning capabilities
- `supportsResponseSchema`: Boolean - JSON schema validation
- `supportsVision`: Boolean - Image input support
- `supportsWebSearch`: Boolean - Web search integration
- `supportsLogProbs`: Boolean - Log probability output

**Model Constraints**:
- **Text Models**: temperature (default), top_p (default)
- **Image Models**: promptCharacterLimit, steps (default, max), widthHeightDivisor

### 3. Model Traits API

**Endpoint**: `GET /api/v1/models/traits`

**Purpose**: Returns a list of model traits and the associated model

**Authentication**: Optional

**Parameters**:
- `type` (optional): Filter by model type ("embedding", "image", "text", "tts", "upscale")
- Default: "text"

**Response**: List of available traits like "fastest", "smartest", etc.

### 4. Model Compatibility Mapping API

**Endpoint**: `GET /api/v1/models/compatibility_mapping`

**Purpose**: Get OpenAI to Venice model mapping

**Authentication**: Optional

**Parameters**:
- `type` (optional): Filter by model type

**Response**: Mapping of OpenAI model names to Venice equivalents

### 5. Image Generation API (Venice Native)

**Endpoint**: `POST /api/v1/image/generate`

**Purpose**: Generate an image based on input parameters

**Description**: Venice's native image generation API provides comprehensive control over image generation with advanced parameters not available in the OpenAI-compatible endpoint.

**Headers**:
- `Authorization: Bearer <token>` (required)
- `Content-Type: application/json` (required)
- `Accept-Encoding: gzip, br` (optional, only applied when return_binary is false)

**Request Parameters** (Complete & Verified):

| Parameter | Type | Required | Default | Min | Max | Description |
|-----------|------|----------|---------|-----|-----|-------------|
| `model` | string | Yes | - | - | - | Model to use for image generation |
| `prompt` | string | Yes | - | 1 | 1500 | Description for the image |
| `negative_prompt` | string | No | - | - | 1500 | What should NOT be in the image |
| `width` | integer | No | 1024 | 1+ | 1280 | Width of generated image |
| `height` | integer | No | 1024 | 1+ | 1280 | Height of generated image |
| `steps` | integer | No | 20 | 1+ | 30 | Number of inference steps |
| `cfg_scale` | number | No | 7.5 | 1 | 20 | Guidance scale for prompt adherence |
| `seed` | integer | No | 0 | -999999999 | 999999999 | Random seed for generation |
| `format` | string | No | "webp" | - | - | Image format: "jpeg", "png", "webp" |
| `safe_mode` | boolean | No | true | - | - | Enable content safety filtering |
| `hide_watermark` | boolean | No | false | - | - | Hide Venice watermark (may be ignored) |
| `return_binary` | boolean | No | false | - | - | Return binary data instead of base64 |
| `embed_exif_metadata` | boolean | No | false | - | - | Embed generation info in EXIF |
| `lora_strength` | integer | No | - | 0 | 100 | LoRA strength (if model supports) |
| `inpaint` | - | No | - | - | - | **DEPRECATED** (disabled May 19th, 2025) |

**Response Structure** (Verified):
```json
{
  "id": "generate-image-1234567890",
  "images": [
    "base64-encoded-image-data"
  ],
  "request": {
    "model": "venice-sd35",
    "prompt": "A beautiful sunset",
    "width": 1024,
    "height": 1024
  },
  "timing": {
    "inferenceDuration": 2500,
    "inferencePreprocessingTime": 100,
    "inferenceQueueTime": 50,
    "total": 2650
  }
}
```

**Response Headers** (Verified):
- `Content-Encoding`: gzip, br (when compression is used)
- `x-venice-is-blurred`: boolean - Indicates if image was blurred due to content policy
- `x-venice-is-content-violation`: boolean - Indicates content policy violation

### 6. Image Generation API (OpenAI Compatible)

**Endpoint**: `POST /api/v1/images/generations`

**Purpose**: Generate images using OpenAI-compatible endpoint

**Request Parameters** (Verified):

| Parameter | Type | Required | Default | Min | Max | Description |
|-----------|------|----------|---------|-----|-----|-------------|
| `prompt` | string | Yes | - | 1 | 1500 | Description for the image |
| `model` | string | No | "default" | - | - | Model to use (defaults to Venice's default if non-existent) |
| `n` | integer | No | 1 | 1 | 1 | Number of images (Venice only supports 1) |
| `size` | string | No | "1024x1024" | - | - | Image size |
| `response_format` | string | No | "url" | - | - | "url" or "b64_json" |
| `moderation` | string | No | "auto" | - | - | "auto" (safe mode) or "low" (disabled) |
| `background` | string | No | "auto" | - | - | **Compatibility only** - not used in Venice |
| `output_compression` | integer | No | 100 | 0 | 100 | **Compatibility only** - not used in Venice |
| `output_format` | string | No | "webp" | - | - | Output format |

### 7. Image Styles API

**Endpoint**: `GET /api/v1/image/styles`

**Purpose**: List available image styles that can be used with the generate API

**Authentication**: Optional (supports both authenticated and unauthenticated requests)

**Parameters**: None

**Response Structure** (Verified):
```json
{
  "object": "list",
  "data": [
    "3D Model",
    "Analog Film", 
    "Anime",
    "Cinematic",
    "Comic Book",
    "Digital Art",
    "Fantasy Art",
    "Isometric",
    "Line Art",
    "Low Poly",
    "Neon Punk",
    "Origami",
    "Photographic",
    "Pixel Art",
    "Texture"
  ]
}
```

**Error Responses**:
- `401`: Authentication failed
- `500`: An unknown error occurred

### 8. Image Upscale API

**Endpoint**: `POST /api/v1/image/upscale`

**Purpose**: Upscale or enhance an image based on the supplied parameters

**Description**: Using a scale of 1 with enhance enabled will only run the enhancer. The image can be provided either as a multipart form-data file upload or as a base64-encoded string in a JSON request.

**Content Types**: 
- `application/json` (for base64 images)
- `multipart/form-data` (for file uploads)

**Request Parameters** (Complete & Verified):

| Parameter | Type | Required | Default | Min | Max | Description |
|-----------|------|----------|---------|-----|-----|-------------|
| `image` | file/string | Yes | - | - | - | Image to upscale (file upload or base64). Min 65536 pixels, max final 16777216 pixels |
| `scale` | number | No | 2 | 1 | 4 | Scale factor for upscaling. Scale of 1 requires enhance=true |
| `enhance` | boolean/string | No | "false" | - | - | Whether to enhance using Venice's image engine. Must be true if scale is 1 |
| `enhanceCreativity` | number | No | 0.5 | 0 | 1 | Higher values let enhancement AI change image more. 1 = entirely new image |
| `enhancePrompt` | string | No | - | - | 1500 | Text-to-image style for enhancement (e.g., "gold", "marble", "angry") |
| `replication` | number | No | 0.35 | 0.1 | 1 | How strongly lines/noise are preserved. Higher = noisier but less AI-generated |

**Response**: Binary image data (PNG format)

**Response Headers**:
- `Content-Type: image/png`

**Error Responses**:
- `400`: Invalid request parameters
- `401`: Authentication failed  
- `402`: Insufficient USD or VCU balance
- `415`: Invalid request content-type
- `429`: Rate limit exceeded

**Example Request**:
```json
{
  "image": "iVBORw0KGgoAAAANSUhEUgAAAgAAAAIACAIAAAB7GkOt...",
  "scale": 2,
  "enhance": true,
  "enhanceCreativity": 0.5,
  "enhancePrompt": "gold",
  "replication": 0.35
}
```

### 9. Embeddings API

**Endpoint**: `POST /api/v1/embeddings`

**Purpose**: Create embeddings for the supplied input

**Description**: Convert text into high-dimensional vector representations for semantic search, clustering, and similarity analysis. This is a beta model accessible to Venice beta testers.

**Headers**:
- `Authorization: Bearer <token>` (required)
- `Content-Type: application/json` (required)
- `Accept-Encoding: gzip, br` (optional)

**Request Parameters** (Complete & Verified):

| Parameter | Type | Required | Default | Min | Max | Description |
|-----------|------|----------|---------|-----|-----|-------------|
| `input` | string/array | Yes | - | 1 | 2048 | Text to embed. Can be string, array of strings, or array of integers |
| `model` | string | Yes | - | - | - | Model ID for embedding generation |
| `encoding_format` | string | No | "float" | - | - | Format: "float" or "base64" |
| `dimensions` | integer | No | - | 1+ | - | Number of dimensions for output embeddings |

**Input Types**:
1. **Single String**:
```json
{
  "input": "This is a test.",
  "model": "text-embedding-bge-m3"
}
```

2. **Array of Strings** (max 2048 items):
```json
{
  "input": ["This is a test.", "Another sentence."],
  "model": "text-embedding-bge-m3"
}
```

3. **Array of Integers** (max 2048 items):
```json
{
  "input": [1212, 318, 257, 1332, 13],
  "model": "text-embedding-bge-m3"
}
```

**Response Structure** (Verified):
```json
{
  "object": "list",
  "model": "text-embedding-bge-m3",
  "data": [
    {
      "object": "embedding",
      "index": 0,
      "embedding": [
        0.0023064255,
        -0.009327292,
        0.015797377
      ]
    }
  ],
  "usage": {
    "prompt_tokens": 8,
    "total_tokens": 8
  }
}
```

**Response Headers**:
- `Content-Encoding`: gzip, br (when compression is used)

**Error Responses**:
- `400`: Invalid request parameters
- `401`: Authentication failed
- `402`: Insufficient USD or VCU balance
- `403`: Unauthorized access
- `429`: Rate limit exceeded

### 10. Audio Speech API

**Endpoint**: `POST /api/v1/audio/speech`

**Purpose**: Generate speech from text (Text-to-Speech)

**Description**: Converts text to speech using various voice models and formats.

**Request Parameters** (Complete & Verified):

| Parameter | Type | Required | Default | Min | Max | Description |
|-----------|------|----------|---------|-----|-----|-------------|
| `input` | string | Yes | - | 1 | 4096 | Text to generate audio for |
| `model` | string | No | "tts-kokoro" | - | - | TTS model ID |
| `voice` | string | Yes | - | - | - | Voice to use for generation |
| `response_format` | string | No | "mp3" | - | - | Audio format |
| `speed` | number | No | 1.0 | 0.25 | 4.0 | Speed of generated audio |
| `streaming` | boolean | No | false | - | - | Stream back sentence by sentence |

**Available Models**:
- `tts-kokoro`

**Available Voices** (Complete List):
**Female Voices (af_)**:
- `af_alloy`, `af_aoede`, `af_bella`, `af_heart`, `af_jadzia`, `af_jessica`, `af_kore`, `af_nicole`, `af_nova`, `af_river`, `af_sarah`, `af_sky`

**Male Voices (am_)**:
- `am_adam`, `am_echo`, `am_eric`, `am_fenrir`, `am_liam`, `am_michael`, `am_onyx`, `am_puck`, `am_santa`

**British Female (bf_)**:
- `bf_alice`, `bf_emma`, `bf_lily`

**British Male (bm_)**:
- `bm_daniel`, `bm_fable`, `bm_george`

**Response Formats**:
- `mp3` (default)
- `opus`
- `aac` 
- `flac`
- `wav`
- `pcm`

**Response**: Binary audio data in specified format

**Response Content Types**:
- `audio/mpeg` (mp3)
- `audio/opus` (opus)
- `audio/aac` (aac)
- `audio/flac` (flac)
- `audio/wav` (wav)
- `audio/pcm` (pcm)

**Example Request**:
```json
{
  "input": "Hello, this is a test of the text to speech system.",
  "model": "tts-kokoro",
  "voice": "af_alloy",
  "response_format": "mp3",
  "speed": 1.0,
  "streaming": false
}
```

**Error Responses**:
- `400`: Invalid request parameters
- `401`: Authentication failed
- `402`: Insufficient USD or VCU balance
- `403`: Unauthorized access
- `415`: Invalid request content-type
- `429`: Rate limit exceeded

### 11. Characters API

**Endpoint**: `GET /api/v1/characters`

**Purpose**: Get available Venice characters

**Description**: This is a preview API and may change. Returns a list of characters supported in the API. Venice provides character-based interactions where AI models can adopt specific personalities and conversation styles.

**Authentication**: Required

**Parameters**: None

**Response Structure** (Complete & Verified):
```json
{
  "object": "list",
  "data": [
    {
      "slug": "alan-watts",
      "name": "Alan Watts",
      "description": "Alan Watts (6 January 1915 – 16 November 1973) was a British and American writer, speaker, and self-styled \"philosophical entertainer\", known for interpreting and popularizing Buddhist, Taoist, and Hindu philosophy for a Western audience.",
      "shareUrl": "https://venice.ai/c/alan-watts",
      "adult": false,
      "webEnabled": true,
      "createdAt": "2024-12-20T21:28:08.934Z",
      "updatedAt": "2025-02-09T03:23:53.708Z",
      "tags": [
        "AlanWatts",
        "Philosophy", 
        "Buddhism",
        "Taoist",
        "Hindu"
      ],
      "stats": {
        "imports": 112
      }
    }
  ]
}
```

**Character Object Properties**:
- `slug`: Character identifier for use in chat completions API
- `name`: Display name of the character
- `description`: Character background and description
- `shareUrl`: Public URL to character page
- `adult`: Whether character contains adult content
- `webEnabled`: Whether character is available for web use
- `createdAt`: Character creation timestamp
- `updatedAt`: Last update timestamp
- `tags`: Associated tags and categories
- `stats.imports`: Number of times character has been imported

**Using Characters in Chat**:
```json
{
  "model": "venice-uncensored",
  "messages": [...],
  "venice_parameters": {
    "character_slug": "alan-watts"
  }
}
```

**Error Responses**:
- `401`: Authentication failed
- `500`: An unknown error occurred

### 12. API Keys Management

#### 12.1 List API Keys
**Endpoint**: `GET /api/v1/api_keys`

**Purpose**: Return a list of API keys

**Response Structure** (Complete & Verified):
```json
{
  "object": "list",
  "data": [
    {
      "id": "e28e82dc-9df2-4b47-b726-d0a222ef2ab5",
      "description": "Example API Key",
      "last6Chars": "2V2jNW",
      "apiKeyType": "ADMIN",
      "createdAt": "2023-10-01T12:00:00Z",
      "expiresAt": "2023-10-01T12:00:00Z",
      "lastUsedAt": "2023-10-01T12:00:00Z",
      "consumptionLimits": {
        "usd": 50,
        "vcu": 100
      },
      "usage": {
        "trailingSevenDays": {
          "usd": "10.2424",
          "vcu": "42.2315"
        }
      }
    }
  ]
}
```

**API Key Types**:
- `INFERENCE`: Standard inference API key
- `ADMIN`: Administrative API key with additional permissions

#### 12.2 Delete API Key
**Endpoint**: `DELETE /api/v1/api_keys`

**Parameters**:
- `id` (query, optional): The ID of the API key to delete

#### 12.3 Create API Key
**Endpoint**: `POST /api/v1/api_keys`

**Purpose**: Create a new API key

#### 12.4 Get API Key Rate Limits
**Endpoint**: `GET /api/v1/api_keys/rate_limits`

**Purpose**: Get rate limit information for API keys

#### 12.5 Get Rate Limit Logs
**Endpoint**: `GET /api/v1/api_keys/rate_limits/log`

**Purpose**: Get usage logs and rate limit history

#### 12.6 Generate Web3 API Key
**Endpoint**: `POST /api/v1/api_keys/generate_web3_key`

**Purpose**: Generate a Web3-based API key

### 13. Billing and Usage API

**Endpoint**: `GET /api/v1/billing/usage`

**Purpose**: Get billing and usage information

**Description**: Retrieve detailed billing information, usage statistics, and account balance data.

**Authentication**: Required

**Parameters**: May include date ranges and filtering options

**Response**: Detailed billing and usage information including:
- Current USD and VCU balances
- Usage statistics by time period
- Cost breakdowns by service type
- Rate limit consumption
- Historical usage data

## Advanced Features

### Function Calling / Tool Use

Venice supports OpenAI-compatible function calling with full parallel execution:

```json
{
  "model": "venice-uncensored",
  "messages": [...],
  "tools": [
    {
      "type": "function",
      "function": {
        "name": "get_weather",
        "description": "Get weather information",
        "parameters": {
          "type": "object",
          "properties": {
            "location": {
              "type": "string",
              "description": "City name"
            }
          },
          "required": ["location"]
        }
      }
    }
  ],
  "tool_choice": "auto",
  "parallel_tool_calls": true
}
```

### Structured Responses

Venice supports JSON schema validation for structured outputs:

```json
{
  "model": "venice-uncensored",
  "messages": [...],
  "response_format": {
    "type": "json_schema",
    "json_schema": {
      "type": "object",
      "properties": {
        "name": {"type": "string"},
        "age": {"type": "number"}
      },
      "required": ["name", "age"]
    }
  }
}
```

**Note**: `json_object` type is deprecated; use `json_schema` instead.

### Streaming Responses

Enable streaming for real-time responses:

```json
{
  "model": "venice-uncensored",
  "messages": [...],
  "stream": true,
  "stream_options": {
    "include_usage": true
  }
}
```

### Web Search Integration

Venice provides built-in web search capabilities:

```json
{
  "model": "venice-uncensored",
  "messages": [...],
  "venice_parameters": {
    "enable_web_search": "auto",
    "enable_web_citations": true
  }
}
```

Web search modes:
- `"off"`: Disable web search
- `"on"`: Always use web search  
- `"auto"`: Use web search when beneficial

Citations are returned either in the first chunk of a streaming result, or in the non-streaming response.

## System Prompts (Official Behavior)

Venice provides default system prompts designed to ensure uncensored and natural model responses. You have two options for handling system prompts:

1. **Default Behavior**: Your system prompts are appended to Venice's defaults
2. **Custom Behavior**: Disable Venice's system prompts entirely

### Disabling Venice System Prompts

Use the `venice_parameters` option to remove Venice's default system prompts:

```javascript
const completionStream = await openAI.chat.completions.create({
  model: "default",
  messages: [
    {
      role: "system",
      content: "Your system prompt",
    },
    {
      role: "user", 
      content: "Why is the sky blue?",
    },
  ],
  // @ts-expect-error Venice.ai parameters are unique to Venice.
  venice_parameters: {
    include_venice_system_prompt: false,
  },
});
```

## Rate Limits (Current as of API Version 20250604.155856)

### Paid Tier Rate Limits

Rate limits apply to users who have purchased API credits or staked VVV to gain VCU.

**Important Note**: Venice continuously monitors usage and adds compute capacity to the network. Rate limits are reviewed and adjusted as needed. Contact support@venice.ai if you consistently hit rate limits.

### Paid Tier - LLMs

| Model | Model ID | Req/Min | Req/Day | Tokens/Min |
|-------|----------|---------|---------|------------|
| Llama 3.2 3B | llama-3.2-3b | 500 | 288,000 | 1,000,000 |
| Qwen 3 4B | qwen3-4b | 500 | 288,000 | 1,000,000 |
| Deepseek Coder V2 | deepseek-coder-v2-lite | 75 | 54,000 | 750,000 |
| Qwen 2.5 Coder 32B | qwen-2.5-coder-32b | 75 | 54,000 | 750,000 |
| Qwen 2.5 QWQ 32B | qwen-2.5-qwq-32b | 75 | 54,000 | 750,000 |
| Dolphin 72B | dolphin-2.9.2-qwen2-72b | 50 | 36,000 | 750,000 |
| Llama 3.3 70B | llama-3.3-70b | 50 | 36,000 | 750,000 |
| Mistral Small 3.1 24B | mistral-31-24b | 50 | 36,000 | 750,000 |
| Qwen 2.5 VL 72B | qwen-2.5-vl | 50 | 36,000 | 750,000 |
| Qwen 3 235B | qwen3-235b | 50 | 36,000 | 750,000 |
| Llama 3.1 405B | llama-3.1-405b | 20 | 15,000 | 750,000 |
| Deepseek R1 671B | deepseek-r1-671b | 15 | 10,000 | 200,000 |

### Paid Tier - Image Models

| Model | Model ID | Req/Min | Req/Day |
|-------|----------|---------|---------|
| Flux | flux-dev / flux-dev-uncensored | 20 | 14,400 |
| All others | All | 20 | 28,800 |

### Paid Tier - Audio Models

| Model | Model ID | Req/Min | Req/Day |
|-------|----------|---------|---------|
| All Audio Models | All | 60 | 86,400 |

### Rate Limit and Consumption Headers

Monitor your API utilization and remaining requests by evaluating the following headers:

| Header | Description |
|--------|-------------|
| `x-ratelimit-limit-requests` | The number of requests you've made in the current evaluation period |
| `x-ratelimit-remaining-requests` | The remaining requests you can make in the current evaluation period |
| `x-ratelimit-reset-requests` | The unix time stamp when the rate limit will reset |
| `x-ratelimit-limit-tokens` | The number of total (prompt + completion) tokens used within a 1 minute sliding window |
| `x-ratelimit-remaining-tokens` | The remaining number of total tokens that can be used during the evaluation period |
| `x-ratelimit-reset-tokens` | The duration of time in seconds until the token rate limit resets |
| `x-venice-balance-vcu` | The user's VCU balance before the request has been processed |
| `x-venice-balance-usd` | The user's USD balance before the request has been processed |

## Error Handling (Complete & Verified)

### HTTP Status Codes

| Code | Description |
|------|-------------|
| 200 | Success |
| 400 | Bad Request - Invalid parameters |
| 401 | Unauthorized - Invalid API key |
| 402 | Payment Required - Insufficient credits |
| 403 | Forbidden - Unauthorized access |
| 404 | Not Found - Resource not found |
| 413 | Payload Too Large - File size exceeds limit |
| 415 | Unsupported Media Type |
| 429 | Too Many Requests - Rate limit exceeded |
| 500 | Internal Server Error |
| 503 | Service Unavailable |
| 504 | Gateway Timeout |

### Detailed Error Codes (Verified from Swagger)

Venice returns predictable error codes with consistent error response format that includes an error code, HTTP status code, and descriptive message.

| Error Code | HTTP Status | Message | Log Level |
|------------|-------------|---------|-----------|
| `AUTHENTICATION_FAILED` | 401 | Authentication failed | - |
| `AUTHENTICATION_FAILED_INACTIVE_KEY` | 401 | Authentication failed - Pro subscription is inactive. Please upgrade your subscription to continue using the API. | - |
| `INVALID_API_KEY` | 401 | Invalid API key provided | - |
| `UNAUTHORIZED` | 403 | Unauthorized access | - |
| `INVALID_REQUEST` | 400 | Invalid request parameters | - |
| `INVALID_MODEL` | 400 | Invalid model specified | - |
| `CHARACTER_NOT_FOUND` | 404 | No character could be found from the provided character_slug | - |
| `INVALID_CONTENT_TYPE` | 415 | Invalid content type | - |
| `INVALID_FILE_SIZE` | 413 | File size exceeds maximum limit | - |
| `INVALID_IMAGE_FORMAT` | 400 | Invalid image format | - |
| `CORRUPTED_IMAGE` | 400 | The image file is corrupted or unreadable | - |
| `RATE_LIMIT_EXCEEDED` | 429 | Rate limit exceeded | - |
| `MODEL_NOT_FOUND` | 404 | Specified model not found | - |
| `INFERENCE_FAILED` | 500 | Inference processing failed | error |
| `UPSCALE_FAILED` | 500 | Image upscaling failed | error |
| `UNKNOWN_ERROR` | 500 | An unknown error occurred | error |

### Error Response Format

```json
{
  "error": {
    "message": "Detailed error description",
    "type": "invalid_request_error",
    "param": "parameter_name",
    "code": "INVALID_API_KEY"
  }
}
```

## Best Practices (Official Venice Recommendations)

1. **Error Handling**: Implement robust error handling for API responses
2. **Rate Limiting**: Be mindful of rate limits during the beta period
3. **System Prompts**: Test both with and without Venice's system prompts to determine the best fit for your use case
4. **API Keys**: Keep your API keys secure and rotate them regularly

## Differences from OpenAI's API (Official)

While Venice maintains high compatibility with the OpenAI API specification, there are some Venice-specific features and parameters:

1. **venice_parameters**: Venice offers additional configurations not available via OpenAI
2. **System Prompts**: Different default behavior for system prompt handling
3. **Model Names**: Venice provides transformation for some common OpenAI model selection to comparable Venice support models, although it is recommended to review the models available on Venice directly

### Venice-Specific Features

- **Web Search Integration**: Built-in web search with citations
- **Character Interactions**: AI personalities and conversation styles
- **Reasoning Models**: Advanced thinking and reasoning capabilities
- **Venice Parameters**: Unique configurations not available in OpenAI
- **Uncensored Responses**: Default prompts designed for natural responses

## Integration Examples

### Python with OpenAI Library

```python
import openai
import base64
import requests

client = openai.OpenAI(
    api_key="your-venice-api-key",
    base_url="https://api.venice.ai/api/v1"
)

# Chat completion
response = client.chat.completions.create(
    model="venice-uncensored",
    messages=[
        {"role": "user", "content": "Hello!"}
    ],
    venice_parameters={
        "include_venice_system_prompt": False,
        "enable_web_search": "auto"
    }
)

# Image generation
image_response = client.images.generate(
    prompt="A beautiful landscape",
    model="venice-sd35",
    size="1024x1024",
    response_format="url"
)

# Generate embeddings
embeddings_response = client.embeddings.create(
    model="text-embedding-bge-m3",
    input=["Hello world", "Venice AI is great"],
    encoding_format="float"
)

# Image upscale
with open("image.jpg", "rb") as f:
    image_data = base64.b64encode(f.read()).decode()

upscale_response = requests.post(
    "https://api.venice.ai/api/v1/image/upscale",
    headers={"Authorization": f"Bearer {client.api_key}"},
    json={
        "image": image_data,
        "scale": 2,
        "enhance": True,
        "enhancePrompt": "high quality, detailed"
    }
)

# Text-to-speech
tts_response = requests.post(
    "https://api.venice.ai/api/v1/audio/speech",
    headers={"Authorization": f"Bearer {client.api_key}"},
    json={
        "input": "Hello, this is Venice AI speaking!",
        "model": "tts-kokoro", 
        "voice": "af_alloy",
        "response_format": "mp3",
        "speed": 1.0
    }
)

# Chat with character
character_response = client.chat.completions.create(
    model="venice-uncensored",
    messages=[
        {"role": "user", "content": "What is the nature of consciousness?"}
    ],
    venice_parameters={
        "character_slug": "alan-watts"
    }
)
```

### JavaScript/Node.js

```javascript
import OpenAI from 'openai';

const client = new OpenAI({
  apiKey: 'your-venice-api-key',
  baseURL: 'https://api.venice.ai/api/v1',
});

// Chat completion
const response = await client.chat.completions.create({
  model: 'venice-uncensored',
  messages: [
    { role: 'user', content: 'Hello!' }
  ],
  venice_parameters: {
    include_venice_system_prompt: false,
    enable_web_search: 'auto'
  }
});

// Streaming
const stream = await client.chat.completions.create({
  model: 'venice-uncensored',
  messages: [{ role: 'user', content: 'Tell me a story' }],
  stream: true,
});

for await (const chunk of stream) {
  process.stdout.write(chunk.choices[0]?.delta?.content || '');
}

// Image generation
const imageResponse = await client.images.generate({
  prompt: "A beautiful sunset over mountains",
  model: "venice-sd35",
  size: "1024x1024",
  response_format: "url"
});

// Embeddings
const embeddingsResponse = await client.embeddings.create({
  model: "text-embedding-bge-m3",
  input: ["Hello world", "Venice AI is great"],
  encoding_format: "float"
});
```

### cURL Examples

```bash
# Chat completion
curl -X POST https://api.venice.ai/api/v1/chat/completions \
  -H "Authorization: Bearer $VENICE_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "venice-uncensored",
    "messages": [
      {"role": "user", "content": "Hello!"}
    ],
    "venice_parameters": {
      "include_venice_system_prompt": false
    }
  }'

# Image generation
curl -X POST https://api.venice.ai/api/v1/images/generations \
  -H "Authorization: Bearer $VENICE_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "A beautiful sunset over mountains",
    "model": "venice-sd35",
    "size": "1024x1024",
    "response_format": "url"
  }'

# Image upscale
curl -X POST https://api.venice.ai/api/v1/image/upscale \
  -H "Authorization: Bearer $VENICE_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{
    "image": "base64-encoded-image-data",
    "scale": 2,
    "enhance": true,
    "enhancePrompt": "high quality"
  }'

# Text-to-speech
curl -X POST https://api.venice.ai/api/v1/audio/speech \
  -H "Authorization: Bearer $VENICE_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{
    "input": "Hello, this is Venice AI!",
    "model": "tts-kokoro",
    "voice": "af_alloy",
    "response_format": "mp3"
  }' \
  --output speech.mp3

# Embeddings
curl -X POST https://api.venice.ai/api/v1/embeddings \
  -H "Authorization: Bearer $VENICE_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{
    "input": "Hello world",
    "model": "text-embedding-bge-m3",
    "encoding_format": "float"
  }'

# List models
curl -X GET https://api.venice.ai/api/v1/models \
  -H "Authorization: Bearer $VENICE_API_KEY"

# List characters
curl -X GET https://api.venice.ai/api/v1/characters \
  -H "Authorization: Bearer $VENICE_API_KEY"

# List image styles
curl -X GET https://api.venice.ai/api/v1/image/styles \
  -H "Authorization: Bearer $VENICE_API_KEY"
```

## Resources and Support

- **Documentation**: https://docs.venice.ai
- **API Specification**: https://api.venice.ai/doc/api/swagger.yaml
- **Terms of Service**: https://venice.ai/legal/tos
- **Discord Community**: Join for support and updates
- **Status Page**: Monitor API availability
- **GitHub**: Open-source documentation contributions
- **Support Email**: support@venice.ai

## API Version Information

- **Current API Version**: 20250604.155856
- **OpenAPI Specification**: 3.0.0
- **Bearer Token Format**: JWT

This comprehensive knowledge base has been verified against the official Venice.ai Swagger specification and documentation, ensuring accuracy and completeness for successful integration and development.

