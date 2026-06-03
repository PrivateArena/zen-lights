# Zenlights Summarize Server HTTP API Documentation

The `zenlights` summarization server provides an HTTP API for summarizing long documents. It supports both classical extractive algorithms (which rank and select key sentences from the source text) and an advanced local abstractive engine (which uses the Gemma-3 LLM to generate a concise summary from scratch via ONNX Runtime).

The following summarization algorithms are supported:
1. **`textrank` (Default)**: Builds a word-overlap sentence similarity graph and applies the PageRank algorithm.
2. **`lexrank`**: Ranks sentences using a TF-IDF cosine similarity graph and PageRank.
3. **`luhn`**: Ranks sentences based on Luhn significance-percentage and word chunk density.
4. **`lsa`**: Ranks sentences using Latent Semantic Analysis via Singular Value Decomposition (SVD).
5. **`sumbasic`**: Ranks sentences using word probability distributions, repeatedly updating counts to minimize redundancy.
6. **`llm`**: Generates a high-quality abstractive summary using a local Gemma-3 ONNX model.

---

## ⚙️ Configuration (`config.json`)

The summarization server behavior is controlled under the `"summarize"` block in `config.json`:

```json
  "summarize": {
    "algorithm": "llm",
    "shared_lib_path": "/media/jang/home/Deve/zen-tts/piper/libonnxruntime.so.1.24.2",
    "model_path": "/media/jang/home/Deve/zen-news/models/model.onnx",
    "model_dir": "/media/jang/home/Deve/zen-news/models/",
    "num_layers": 18,
    "vocab_size": 262144,
    "max_tokens_per_sentence": 80
  }
```

### Config Options
* `algorithm` (String, optional): The default engine algorithm to use. Can be `"textrank"`, `"lexrank"`, `"luhn"`, `"lsa"`, `"sumbasic"`, or `"llm"`. Defaults to `"textrank"`.
* `shared_lib_path` (String, optional): Absolute path to the ONNX Runtime shared library (`libonnxruntime.so`) needed for the `llm` engine.
* `model_path` (String, optional): Absolute path to the Gemma-3 ONNX model file.
* `model_dir` (String, optional): Absolute path to the model folder containing `tokenizer.json`.
* `num_layers` (Integer, optional): Number of model transformer layers (e.g. `18` for Gemma-3-270m).
* `vocab_size` (Integer, optional): Vocabulary dimension size (e.g. `262144` for Gemma-3).
* `max_tokens_per_sentence` (Integer, optional): The generation limit of summary tokens for the LLM. Defaults to `80`.

---

## 📡 API Endpoints

### 1. Health Status
Verifies general server health.

* **Endpoint:** `GET /status`
* **Response:** `200 OK`
* **Example:**
  ```bash
  curl -i http://localhost:8765/status
  ```

---

### 2. Summarize Text
Performs summarization on the provided text using the configured algorithm.

* **Endpoint:** `POST /summarize` (also accepts `GET`)
* **Request Parameters / Fields:**
  * `text` (String, required): The raw document text to summarize. Can be sent in a JSON body or as a query parameter.
  * `count` (Integer, optional): The number of summary sentences to return. Defaults to `3`.
  * `language` / `lang` (String, optional): Language code used for stop words filtering (e.g., `"en"`, `"cjk"`, `"vi"`). Defaults to `"en"`.

#### GET Request Example (Query Parameters)
```bash
curl -s "http://localhost:8765/summarize?text=Gemma-3+is+a+lightweight+open-source+model.+It+is+highly+optimized+for+local+inference.+ONNX+runtime+helps+to+execute+it+efficiently+on+CPU.&count=2&lang=en"
```

#### POST Request Example (JSON Body)
```bash
curl -s -X POST http://localhost:8765/summarize \
  -H "Content-Type: application/json" \
  -d '{
    "text": "Gemma-3 is a state-of-the-art open large language model developed by Google. It is designed to be highly efficient, especially for lightweight inference tasks on local consumer-grade hardware. ONNX runtime enables high-performance inference across various platforms.",
    "count": 1,
    "language": "en"
  }'
```

#### Sample Success Response
```json
{
  "summary": [
    "Gemma-3 is a state-of-the-art open-source open-science model designed for lightweight inference on local consumer-grade hardware."
  ]
}
```

#### Sample Error Response (e.g. missing text)
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
curl -s "http://localhost:8765/summarize?help=true"
```

### 2. Error Redirection
If any request has invalid parameters or wrong usage (causing a non-200 HTTP status code), the server automatically includes the full contents of this API guide under the `"help"` key of the JSON error response:
```json
{
  "error": "Parameter 'text' is required",
  "help": "# Zenlights Summarize Server HTTP API Documentation..."
}
```
