# Zenlights OCR Server HTTP API Documentation

The `zenlights` persistent OCR server provides a high-performance HTTP interface for multi-language OCR, powered by PaddleOCR PP-OCRv4 models running on ONNX Runtime. 

It supports high-frequency text recognition (e.g. game scoreboard scraping) and full-frame layout parsing, automatically managing models in memory and allowing runtime dynamic model selection.

---

## 🚀 Starting the Server

The server can be started using either positional ports or traditional flags. Mixed formats are robustly resolved.

### Command Syntax
```bash
./zenlights ocr-server [address/port] [options]
```

### Options
* `[address/port]` (Positional, optional): Port number (e.g. `8765`) or address (e.g. `127.0.0.1:8765`). Defaults to `localhost:8080`.
* `-default-model` (Flag, optional): Specifies the language/model to run by default if not requested explicitly in API calls. Defaults to `"ch"`.
* `-config` (Flag, optional): Path to the JSON configuration mapping language IDs to model assets. Defaults to `config.json`.

### Starting Examples

**Using Positional Port & Custom Default Model:**
```bash
./zenlights ocr-server 8765 -default-model en
```

**Using Traditional Flags:**
```bash
./zenlights ocr-server -addr localhost:8765 -default-model ko -config my_custom_config.json
```

**Startup Log Output:**
```text
zenlights: Loaded language profiles from config.json
zenlights: 🤖 OCR Server listening on http://127.0.0.1:8765
zenlights: ⚙️ OCR: Pre-loading default model "en"...
zenlights: ⚙️ OCR: Loading model for language/profile "en"...
zenlights: ❇️ OCR: Successfully loaded model for language/profile "en"
```

---

## 📡 API Endpoints

### 1. Health Check
Checks the status of the server.

* **Endpoint:** `GET /status`
* **Response:** `200 OK`
* **Example:**
  ```bash
  curl -i http://localhost:8765/status
  ```

---

### 2. Unified OCR & Model Config
Unifies OCR recognition, retrieving active model configuration, and setting/updating the default model dynamically.

* **Endpoint:** `/ocr`
* **Supported Methods:** `GET`, `POST`, `PUT`

#### A. Get Active OCR Model Configuration
Retrieve the currently configured default OCR model.

* **Method:** `GET`
* **Response Format:** `application/json`
* **Example:**
  ```bash
  curl -s http://localhost:8765/ocr
  ```
* **Sample Response:**
  ```json
  {
    "default_model": "en"
  }
  ```

#### B. Dynamic Model Selection (Change Active Model)
Dynamically changes the default model for all subsequent API requests. The model must exist in the server config profiles.

* **Method:** `POST` or `PUT` (without image body)
* **Request Options:** Specify `model` or `lang` via query parameter or JSON body.
* **Example (Query Parameter):**
  ```bash
  curl -s -X POST "http://localhost:8765/ocr?model=ko"
  ```
* **Example (JSON Body):**
  ```bash
  curl -s -X POST http://localhost:8765/ocr \
    -H "Content-Type: application/json" \
    -d '{"model": "ja"}'
  ```
* **Sample Success Response:**
  ```json
  {
    "status": "success",
    "default_model": "ja"
  }
  ```

#### C. Perform OCR Recognition
Uploads an image file to run layout segmentation and text recognition. If a `model` or `lang` parameter is specified, it also updates the server's default/active model.

* **Method:** `POST`
* **Request Options:**
  * `image` (Multipart Form File, required): The image to process. Alternatively, you can send the raw image bytes as the request body.
  * `model` or `lang` (Query Parameter, optional): Explicitly selects which model to use for this specific request, and sets it as the active server default. If omitted, falls back to the server's currently active model.
* **Example (Using Active Model):**
  ```bash
  curl -s -X POST http://localhost:8765/ocr \
    -F "image=@/path/to/screenshot.png"
  ```
* **Example (Explicitly Requesting Korean Model & Updating Default):**
  ```bash
  curl -s -X POST "http://localhost:8765/ocr?model=ko" \
    -F "image=@/path/to/screenshot.png"
  ```
* **Sample Success Response:**
  ```json
  {
    "results": [
      {
        "Text": "SCORE",
        "Confidence": 0.985,
        "Bounds": {
          "Min": {"X": 45, "Y": 12},
          "Max": {"X": 110, "Y": 34}
        }
      }
    ]
  }
  ```

---

### 3. Translate (Dual-Engine Translation)
Translates text between languages using either online (Google Translate) or local offline models (Helsinki OPUS-MT).

* **Endpoint:** `POST /translate` (also supports `GET`)
* **Request Options:**
  * `text` (String, required): The text to translate. Can be sent in a JSON body or as a query parameter.
  * `source` / `from` / `src` (String, optional): The source language code (e.g. `ja`, `ko`, `zh`). Defaults to `auto`.
  * `target` / `to` / `tgt` (String, optional): The target language code (e.g. `en`). Defaults to `en`.
* **Example (GET Request):**
  ```bash
  curl -s "http://localhost:8765/translate?text=こんにちは&source=ja&target=en"
  ```
* **Example (POST Request with JSON Body):**
  ```bash
  curl -s -X POST http://localhost:8765/translate \
    -H "Content-Type: application/json" \
    -d '{"text": "こんにちは", "source": "ja", "target": "en"}'
  ```
* **Sample Success Response:**
  ```json
  {
    "translated": "Hello"
  }
  ```
* **Sample Error Response:**
  ```json
  {
    "error": "Parameter 'text' is required"
  }
  ```

---

## ❓ Interactive API Help Guide
The server supports built-in interactive help guides directly from the terminal!

### 1. Manual Help Request
Pass `help=true` as a query parameter or `{"help": true}` in a JSON request body to retrieve this complete markdown guide formatted beautifully in your terminal:
```bash
curl -s "http://localhost:8765/ocr?help=true"
```

### 2. Error Redirection
If any request has invalid parameters or wrong usage (causing a non-200 HTTP status code), the server automatically includes the full contents of this API guide under the `"help"` key of the JSON error response:
```json
{
  "error": "Failed to decode image: ...",
  "help": "# Zenlights OCR Server HTTP API Documentation..."
}
```
