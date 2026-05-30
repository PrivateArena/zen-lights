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

### 2. Get Default Model
Retrieves the currently configured default OCR model.

* **Endpoint:** `GET /default-model`
* **Response Format:** `application/json`
* **Example:**
  ```bash
  curl -s http://localhost:8765/default-model
  ```
* **Sample Response:**
  ```json
  {
    "default_model": "en"
  }
  ```

---

### 3. Set Default Model (Dynamic Selection)
Dynamically changes the default model for all subsequent API requests. The model must exist in the server config profiles.

* **Endpoint:** `POST /default-model` (also accepts `PUT`)
* **Request Options:** Specify `model` or `lang` via query parameter or JSON body.
* **Example (Query Parameter):**
  ```bash
  curl -s -X POST "http://localhost:8765/default-model?model=ko"
  ```
* **Example (JSON Body):**
  ```bash
  curl -s -X POST http://localhost:8765/default-model \
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
* **Sample Error Response (Invalid Model):**
  ```json
  {
    "error": "Model/Language \"fr\" not available in config"
  }
  ```

---

### 4. Recognize (Run OCR)
Uploads an image file to run layout segmentation and text recognition. 

* **Endpoint:** `POST /recognize`
* **Request Options:**
  * `image` (Multipart Form File, required): The image to process. Alternatively, you can send the raw image bytes as the request body.
  * `model` or `lang` (Query Parameter, optional): Explicitly selects which model to use for this specific request. If omitted, falls back to the server's configured `default_model`.
* **Example (Using Default Model):**
  ```bash
  curl -s -X POST http://localhost:8765/recognize \
    -F "image=@/path/to/screenshot.png"
  ```
* **Example (Explicitly Requesting Korean Model):**
  ```bash
  curl -s -X POST "http://localhost:8765/recognize?model=ko" \
    -F "image=@/path/to/screenshot.png"
  ```
* **Example (Using Raw Image Body):**
  ```bash
  curl -s -X POST "http://localhost:8765/recognize?model=en" \
    --data-binary "@/path/to/screenshot.png"
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
      },
      {
        "Text": "12 : 5",
        "Confidence": 0.992,
        "Bounds": {
          "Min": {"X": 120, "Y": 10},
          "Max": {"X": 210, "Y": 36}
        }
      }
    ]
  }
  ```
