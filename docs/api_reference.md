## 📄 `docs/api_reference.md`

```markdown
# 📡 API Reference

## Base URL
http://localhost:8000


## Endpoints

### Health Check
GET /health

Returns system status and readiness.

### Generate Image
POST /generate
Content-Type: application/json

{
"prompt": "A beautiful sunset over mountains",
"style": "photorealistic",
"quality": "high",
"num_inference_steps": 30,
"guidance_scale": 7.5,
"width": 512,
"height": 512,
"seed": 42,
"num_images": 1,
"enhance_prompt": true,
"output_format": "png"
}

**Response:**
```json
{
    "success": true,
    "images": ["base64_encoded_image_string"],
    "prompt_used": "enhanced prompt...",
    "negative_prompt_used": "negative prompt...",
    "settings": {...},
    "elapsed_time": 3.45,
    "timestamp": "2024-01-01T12:00:00"
}

POST /generate/stream
Returns image bytes directly instead of base64.

Analyze Prompt
POST /analyze
{"prompt": "your prompt here"}

Enhance Prompt
POST /enhance
{"prompt": "basic prompt", "style": "digital_art", "quality": "high"}

List Styles
GET /styles

List Schedulers
GET /schedulers

List Models
GET /models

Get History
GET /history?limit=10

Get Stats
GET /stats



