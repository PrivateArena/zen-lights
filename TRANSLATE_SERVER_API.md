# Zenlights Translate Server HTTP API Documentation

The `zenlights` translation server provides a flexible, dual-engine API for translating text between languages. It leverages both online services (Google Translate) and local offline translation models (ONNX Helsinki OPUS-MT transformer networks).

It supports three execution modes:
1. **Online (`online`)**: Queries the Google Translate API.
2. **Offline (`offline`)**: Performs fast local neural machine translation via ONNX Runtime on the CPU.
3. **Auto (`auto`)**: Attempts offline translation first, automatically falling back to online if the model is missing or fails.

---

## ⚙️ Configuration (`config.json`)

The translate server behavior is controlled under the `"translation"` block in `config.json`:

```json
  "translation": {
    "mode": "auto",
    "max_tokens": 128,
    "shared_lib_path": "/media/jang/home/Deve/zen-tts/piper/libonnxruntime.so.1.24.2",
    "online": {
      "api_url": "https://translate.googleapis.com/translate_a/single?client=gtx&sl=%s&tl=%s&dt=t&q=%s",
      "timeout_ms": 10000
    },
    "profiles": [
      {
        "id": "ja-en",
        "encoder_path": "/media/jang/home/Deve/zen-lights/models/ja-en/encoder_model.onnx",
        "decoder_path": "/media/jang/home/Deve/zen-lights/models/ja-en/decoder_model.onnx",
        "source_spm": "/media/jang/home/Deve/zen-lights/models/ja-en/source.spm",
        "target_spm": "/media/jang/home/Deve/zen-lights/models/ja-en/target.spm",
        "vocab_path": "/media/jang/home/Deve/zen-lights/models/ja-en/vocab.json",
        "eos_token_id": 0,
        "pad_token_id": 65000
      }
    ]
  }
```

### Config Options
* `mode` (String, optional): Translation routing mode. Can be `"online"`, `"offline"`, or `"auto"`. Defaults to `"auto"`.
* `max_tokens` (Integer, optional): The maximum number of tokens generated in offline mode. Defaults to `128`.
* `shared_lib_path` (String, optional): Absolute path to the ONNX Runtime shared library (`libonnxruntime.so`).
* `online.api_url` (String, optional): The API query template URL. You can specify a custom translation proxy or format.
* `online.timeout_ms` (Integer, optional): HTTP client timeout for online requests in milliseconds. Defaults to `10000`.
* `profiles` (Array, optional): A registry of registered local offline models. Each profile specifies the language-pair ID and local file paths to model assets.

---

## 📡 API Endpoints

### 1. Health Status
Verifies translation service and general server health.

* **Endpoint:** `GET /status`
* **Response:** `200 OK`
* **Example:**
  ```bash
  curl -i http://localhost:8765/status
  ```

---

### 2. Translate Text
Performs translation using the configured engine/mode routing.

* **Endpoint:** `POST /translate` (also accepts `GET`)
* **Request Parameters:**
  * `text` (String, required): The raw text to translate. Can be sent in a JSON body or as a query parameter.
  * `source` / `from` / `src` (String, optional): Source language code. Defaults to `"auto"`.
  * `target` / `to` / `tgt` (String, optional): Target language code. Defaults to `"en"`.

#### GET Request Example (Query Parameters)
```bash
curl -s "http://localhost:8765/translate?text=こんにちは&source=ja&target=en"
```

#### POST Request Example (JSON Body)
```bash
curl -s -X POST http://localhost:8765/translate \
  -H "Content-Type: application/json" \
  -d '{
    "text": "こんにちは",
    "source": "ja",
    "target": "en"
  }'
```

#### Sample Success Response
```json
{
  "translated": "Hello"
}
```

#### Sample Error Response (Missing Text Parameter)
```json
{
  "error": "Parameter 'text' is required"
}
```

#### Sample Error Response (Internal Failure / Missing Profile)
```json
{
  "error": "Translation failed: translation profile \"ja-en\" not found: unsupported language pair"
}
```

---

## ❓ Interactive API Help Guide
The server supports built-in interactive help guides directly from the terminal!

### 1. Manual Help Request
Pass `help=true` as a query parameter or `{"help": true}` in a JSON request body to retrieve this complete markdown guide formatted beautifully in your terminal:
```bash
curl -s "http://localhost:8765/translate?help=true"
```

### 2. Error Redirection
If any request has invalid parameters or wrong usage (causing a non-200 HTTP status code), the server automatically includes the full contents of this API guide under the `"help"` key of the JSON error response:
```json
{
  "error": "Parameter 'text' is required",
  "help": "# Zenlights Translate Server HTTP API Documentation..."
}
```
