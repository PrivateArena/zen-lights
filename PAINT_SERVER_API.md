# Zenlights Paint Server HTTP API Documentation

The `zenlights` unified server integrates a production-grade, local image generation engine (ported from `zen-paint`). It supports ONNX-based Text-to-Image models using dual architectures:
1. **SDXL / SDXL-Turbo / LCM**: Fast, high-quality latent diffusion pipelines.
2. **FLUX.2 / Bonsai**: Multi-modal Diffusion Transformer (MMDiT) flow-matching pipelines.
3. **Pixel Studio (pixel)**: Production-ready pixelated asset generation with transparency flood-fill, downsampling grids, color quantization (presets + custom K-Means), outline generation, and diagonal line cleanup.

All inference is performed locally using **ONNX Runtime** and can utilize CPU multi-threading or hardware acceleration (CUDA, ROCm, DirectML, OpenVINO).

---

## ⚙️ Configuration (`config.json`)

The paint engine properties are configured under the `"paint"` block in `config.json`:

```json
  "paint": {
    "default_model": "sdxl-turbo",
    "execution_provider": "cpu",
    "num_threads": 0,
    "output_dir": "/tmp/zen-paint",
    "max_concurrency": 1,
    "models_dir": "models_paint"
  }
```

### Config Options
* `default_model` (String, required): Name of the model directory to load automatically at server startup (e.g., `"sdxl-turbo"`).
* `execution_provider` (String, optional): Hardware acceleration backend. Options: `"cpu"`, `"cuda"` (NVIDIA), `"rocm"` (AMD), `"directml"` (Windows), `"openvino"` (Intel). Defaults to `"cpu"`.
* `num_threads` (Integer, optional): Number of CPU execution threads. `0` defaults to standard auto-detection (capped at `16`).
* `output_dir` (String, required): Directory path on disk where generated PNG images are written.
* `max_concurrency` (Integer, optional): Max concurrent generation requests processed simultaneously. Excess requests receive a `429 Too Many Requests` response. Defaults to `1`.
* `models_dir` (String, required): Absolute folder path containing model subdirectories. This lets you reference weights without copying large gigabyte files.

---

## 📡 API Endpoints

All endpoints are hosted on the unified server port (default: `8765`) and are prefixed with `/paint`.

### 1. Engine Status
Retrieves current engine status, loaded model name, and backend execution metadata.

* **Endpoint:** `GET /paint/status`
* **Response:** `200 OK`
* **Sample Response:**
  ```json
  {
    "info": "SDXL pipeline | dir=/media/jang/home/Deve/zen-paint/models/sdxl-turbo threads=12 provider=cpu",
    "model": "sdxl-turbo",
    "status": "ok"
  }
  ```
* **Example:**
  ```bash
  curl -i http://localhost:8765/paint/status
  ```

---

### 2. List Models
Lists all valid model directories under the configured `models_dir` containing a `model.json` descriptor.

* **Endpoint:** `GET /paint/models`
* **Response:** `200 OK`
* **Sample Response:**
  ```json
  {
    "models": [
      "sdxl-turbo",
      "lcm-dreamshaper",
      "bonsai"
    ]
  }
  ```
* **Example:**
  ```bash
  curl -i http://localhost:8765/paint/models
  ```

---

### 3. Load Model
Triggers dynamic, on-demand loading of a different model from the available list.

* **Endpoint:** `POST /paint/load`
* **Request Fields:**
  * `model` (String, required): Name of the model subdirectory to load.
* **Sample Request:**
  ```json
  {
    "model": "bonsai"
  }
  ```
* **Sample Response:**
  ```json
  {
    "model": "bonsai",
    "status": "loaded"
  }
  ```
* **Example:**
  ```bash
  curl -s -X POST http://localhost:8765/paint/load \
    -H "Content-Type: application/json" \
    -d '{"model": "bonsai"}'
  ```

---

### 4. Generate Image
Generates a new PNG image from a text prompt.

* **Endpoint:** `POST /paint/generate`
* **Request Fields:**
  * `prompt` (String, required): Text description of the image to generate.
  * `negative_prompt` (String, optional): Elements to exclude (only utilized by SDXL backends).
  * `width` (Integer, optional): Width of the image. Defaults to `512`.
  * `height` (Integer, optional): Height of the image. Defaults to `512`.
  * `steps` (Integer, optional): Number of denoising inference steps. Defaults to `4`.
  * `seed` (Integer, optional): Random seed. Defaults to current timestamp.
  * `pixel_size` (Integer, optional): Target grid size (e.g. `16`, `32`, `64`, `128`) for pixel art. Defaults to `64`. (Only used by `pixel` architecture).
  * `palette` (String, optional): Palette preset (`"pico8"`, `"gameboy"`, `"nes"`, `"c64"`, `"custom"`). Defaults to `"custom"`. (Only used by `pixel` architecture).
  * `palette_size` (Integer, optional): Max colors for custom palette extraction. Defaults to `16`.
  * `outline` (Boolean, optional): Generates an outline around transparent shapes. Defaults to `true`.
  * `outline_color` (String, optional): Hex color (e.g. `"#ff0000"` or `"#000000"`) for custom outline colors.
  * `transparent` (Boolean, optional): Automatically key out/flood-fill solid backgrounds. Defaults to `true`.
  * `clean_doubles` (Boolean, optional): Cleans L-shaped double pixels on diagonals. Defaults to `true`.
  * `dither` (Boolean, optional): Apply Floyd-Steinberg error diffusion dithering (only with preset palettes). Defaults to `false`.
* **Sample Request:**
  ```json
  {
    "prompt": "A beautiful cinematic digital painting of antigravity physics in a vibrant cyberpunk city",
    "width": 512,
    "height": 512,
    "steps": 4,
    "seed": 42
  }
  ```
* **Sample Response:**
  ```json
  {
    "path": "/tmp/zen-paint/zp_1715456729000_42.png",
    "duration_ms": 3280,
    "width": 512,
    "height": 512,
    "seed": 42
  }
  ```
* **Example:**
  ```bash
  curl -s -X POST http://localhost:8765/paint/generate \
    -H "Content-Type: application/json" \
    -d '{
      "prompt": "A vibrant digital drawing of a flying cat",
      "steps": 4,
      "width": 512,
      "height": 512
    }'
  ```

---

### 5. Serve Output Images
Downloads or displays generated PNG files directly from the server.

* **Endpoint:** `GET /paint/outputs/:filename`
* **Response:** `200 OK` (binary image data)
* **Example:**
  To retrieve an image returned in the generation result path (e.g., `zp_1715456729000_42.png`):
  ```bash
  curl -o output.png http://localhost:8765/paint/outputs/zp_1715456729000_42.png
  ```

---

## ❓ Interactive API Help Guide
The server supports built-in interactive help guides directly from the terminal!

### 1. Manual Help Request
Pass `help=true` as a query parameter or `{"help": true}` in a JSON request body to retrieve this complete markdown guide formatted beautifully in your terminal:
```bash
curl -s "http://localhost:8765/paint/generate?help=true"
```

### 2. Error Redirection
If any request has invalid parameters or wrong usage (causing a non-200 HTTP status code), the server automatically includes the full contents of this API guide under the `"help"` key of the JSON error response:
```json
{
  "error": "invalid JSON body",
  "help": "# Zenlights Paint Server HTTP API Documentation..."
}
```
