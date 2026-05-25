You are spot on about hardware budget engineering: **every layer is a context switch, memory allocation, and kernel launch.** If your goal is a zero-latency CPU pipeline, your instinct to avoid adding a heavy object detection network (like YOLO or Mask R-CNN) just to find speech bubbles is absolutely correct.

On a CPU, a dedicated bubble detection model is not just overrated; it is a massive performance bottleneck.

However, the problem isn't that bubble detection as a *concept* is useless—it’s that executing it via a separate deep learning model is inefficient. You can achieve the exact same architectural benefits using zero-cost algorithmic heuristics directly on top of your existing `det_infer.onnx` output.

---

## Why a Bubble Model Sucks on CPU

If you add a standard layout detection layer (like `comic-text-bubble-detector`), your pipeline execution graph looks like this:

```
[Manga Page] ──► [Bubble Model (Heavy CNN)] ──► [Crop Images] ──► [Paddle Det] ──► [Paddle Rec]

```

* **The Math Doesn't Clear:** Running a second heavy structural CNN or ViT on an image size large enough to resolve fine manga text scales poorly on standard CPU instruction sets.
* **Memory Thrashing:** You end up copying, slicing, and cropping sub-tensors out of your main canvas image in memory before passing them to the next model stage, destroying cache locality.

---

## The Lean Architecture: Pure Math Instead of a Second Model

Since PaddleOCR's DBNet (`det_infer.onnx`) is exceptional at localizing text lines, you already have the precise geometric spatial coordinates of every piece of text on the page.

Instead of adding another model layer, you can use **spatial spatial-clustering algorithms** on the CPU to group those lines into semantic text blocks (bubbles). This introduces virtually zero CPU overhead (sub-millisecond execution times).

```
                      ┌───────────────────────────┐
                      │    Input Manga Page       │
                      └─────────────┬─────────────┘
                                    │
                                    ▼
                      ┌───────────────────────────┐
                      │    Paddle DBNet (Det)     │
                      └─────────────┬─────────────┘
                                    │ Outputs an array of individual line coordinates
                                    ▼
                      ┌───────────────────────────┐
                      │  DBSCAN / Graph Distance  │  <-- Pure algorithmic math.
                      │    (Line Clustering)      │      Groups lines into bubbles.
                      └─────────────┬─────────────┘
                                    │
                                    ▼
                      ┌───────────────────────────┐
                      │   Paddle SVTR/CTC (Rec)   │
                      └───────────────────────────┘

```

### How to Implement 0-Cost Clustering

Two lines of vertical Japanese text inside the same speech bubble will always have two properties:

1. They are physically close to each other on the X-axis.
2. They overlap significantly on the Y-axis.

By running a simple **DBSCAN (Density-Based Spatial Clustering)** or a custom bounding-box distance check in your code, you can group individual lines into virtual "bubbles" before passing them to translation.

```python
# Conceptual heuristic loop for line grouping
def group_lines_into_bubbles(detected_lines, x_threshold=40):
    # Sort lines from right to left (Manga reading order)
    detected_lines.sort(key=lambda box: box.center_x, reverse=True)
    
    bubbles = []
    for line in detected_lines:
        matched = False
        for bubble in bubbles:
            # If line is close horizontally and overlaps vertically, it belongs to this bubble
            if abs(line.center_x - bubble.center_x) < x_threshold and line.overlaps_y(bubble):
                bubble.add(line)
                matched = True
                break
        if not matched:
            bubbles.append(Bubble(line))
    return bubbles

```

---

## CPU-SOTA Quantization and Runtime Tactics

To make this execution graph blazing fast on a CPU, optimize the runtime configuration for the models you already have:

### 1. Model Quantization (INT8)

Do not run standard FP32 or FP16 ONNX exports on a CPU. Convert your PP-OCRv4 models to **INT8** using ONNX Runtime's quantization tool or OpenVINO.

* Modern CPUs have dedicated hardware vector extensions (**AVX-512 VNNI** or **AVX-VNNI**) explicitly designed to compute INT8 matrix multiplications at a fraction of the clock cycles.

### 2. Bypass PyTorch Completely

Ensure your execution layer uses native ONNX Runtime bindings (`onnxruntime` or `onnxruntime-openvino`) without any Python-heavy machine learning frameworks wrapped around them. A torchless execution context strips out gigabytes of overhead and memory footprint.

### 3. Concurrency Tactic: Batching vs. Threading

PaddleOCR's recognition model (`rec_infer.onnx`) handles variable-width shapes.

* **Avoid dynamic batch sizes on CPU:** Changing batch dimensions on the fly causes ONNX Runtime to constantly reallocate dynamic internal memory buffers.
* **The Performance Play:** Run your detection engine once on the full page image. Then, use a parallel processing pool (like Go channels/goroutines or Python's `concurrent.futures`) to send individual text line crops to a fixed pool of single-batch ONNX inference sessions running across your available CPU threads.