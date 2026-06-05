package paint

import (
	"bytes"
	"encoding/json"
	"fmt"
	"io"
	"log"
	"net/http"
	"os"
	"strings"
	"time"

	"github.com/zen-lights/zen-lights/internal/paint/engine"
)

// RegisterHandlers maps the /paint/* API endpoints to the router ServeMux.
func RegisterHandlers(mux *http.ServeMux, mgr *Manager) {
	mux.HandleFunc("/paint/status", mgr.HandleStatus)
	mux.HandleFunc("/paint/models", mgr.HandleModels)
	mux.HandleFunc("/paint/load", mgr.HandleLoad)
	mux.HandleFunc("/paint/generate", mgr.HandleGenerate)
	
	// Serve output images dynamically from output_dir
	mux.Handle("/paint/outputs/", http.StripPrefix("/paint/outputs/", http.FileServer(http.Dir(mgr.Config.OutputDir))))
}

// checkHelp checks if a manual help request is made via query param (?help=true) or json body ({"help": true})
func checkHelp(r *http.Request) (bool, []byte) {
	if r.URL.Query().Get("help") == "true" {
		return true, nil
	}
	if (r.Method == http.MethodPost || r.Method == http.MethodPut) && r.Body != nil {
		bodyBytes, err := io.ReadAll(r.Body)
		if err == nil {
			r.Body = io.NopCloser(bytes.NewReader(bodyBytes))
			var tmp struct {
				Help any `json:"help"`
			}
			if err := json.Unmarshal(bodyBytes, &tmp); err == nil {
				if b, ok := tmp.Help.(bool); ok && b {
					return true, bodyBytes
				}
				if s, ok := tmp.Help.(string); ok && (strings.ToLower(s) == "true" || s == "1" || strings.ToLower(s) == "yes") {
					return true, bodyBytes
				}
			}
		}
	}
	return false, nil
}

// sendHelp reads a markdown guide and writes it back as markdown (for 200) or wraps it in JSON error (for non-200)
func sendHelp(w http.ResponseWriter, filename string, errStr string, statusCode int) {
	var helpContent string
	data, err := os.ReadFile(filename)
	if err == nil {
		helpContent = string(data)
	} else {
		// Try traversing parent directories (up to 3 levels)
		for i := 1; i <= 3; i++ {
			prefix := ""
			for j := 0; j < i; j++ {
				prefix += "../"
			}
			data, err = os.ReadFile(prefix + filename)
			if err == nil {
				helpContent = string(data)
				break
			}
		}
	}

	if helpContent == "" {
		helpContent = fmt.Sprintf("Help guide %s not found on the server.", filename)
	}

	if errStr != "" || statusCode != http.StatusOK {
		w.Header().Set("Content-Type", "application/json")
		w.WriteHeader(statusCode)
		json.NewEncoder(w).Encode(map[string]any{
			"error": errStr,
			"help":  helpContent,
		})
		return
	}

	// Manual request: print raw markdown for beautiful terminal viewing
	w.Header().Set("Content-Type", "text/markdown; charset=utf-8")
	w.WriteHeader(http.StatusOK)
	w.Write([]byte(helpContent))
}

// HandleStatus reports active model and backend info.
func (mgr *Manager) HandleStatus(w http.ResponseWriter, r *http.Request) {
	isHelp, _ := checkHelp(r)
	if isHelp {
		sendHelp(w, "PAINT_SERVER_API.md", "", http.StatusOK)
		return
	}

	eng, model := mgr.GetActive()
	info := ""
	if eng != nil {
		info = eng.Info()
	}
	writeJSON(w, http.StatusOK, map[string]any{
		"status": "ok",
		"model":  model,
		"info":   info,
	})
}

// HandleModels lists all models available under the configured models_dir.
func (mgr *Manager) HandleModels(w http.ResponseWriter, r *http.Request) {
	isHelp, _ := checkHelp(r)
	if isHelp {
		sendHelp(w, "PAINT_SERVER_API.md", "", http.StatusOK)
		return
	}

	names, err := listModels(mgr.Config.ModelsDir)
	if err != nil {
		writeJSON(w, http.StatusInternalServerError, map[string]any{"error": err.Error()})
		return
	}
	writeJSON(w, http.StatusOK, map[string]any{"models": names})
}

// HandleLoad triggers on-demand loading of a specific model directory.
func (mgr *Manager) HandleLoad(w http.ResponseWriter, r *http.Request) {
	isHelp, bodyBytes := checkHelp(r)
	if isHelp {
		sendHelp(w, "PAINT_SERVER_API.md", "", http.StatusOK)
		return
	}

	if r.Method != http.MethodPost {
		sendHelp(w, "PAINT_SERVER_API.md", "Method not allowed", http.StatusMethodNotAllowed)
		return
	}
	var req struct {
		Model string `json:"model"`
	}
	
	if len(bodyBytes) == 0 && r.Body != nil {
		var err error
		bodyBytes, err = io.ReadAll(r.Body)
		if err == nil {
			r.Body = io.NopCloser(bytes.NewReader(bodyBytes))
		}
	}
	
	if len(bodyBytes) > 0 {
		if err := json.Unmarshal(bodyBytes, &req); err != nil || req.Model == "" {
			writeJSON(w, http.StatusBadRequest, map[string]any{"error": "missing or invalid 'model' parameter"})
			return
		}
	} else {
		writeJSON(w, http.StatusBadRequest, map[string]any{"error": "missing or invalid 'model' parameter"})
		return
	}

	if err := mgr.LoadEngine(req.Model); err != nil {
		writeJSON(w, http.StatusInternalServerError, map[string]any{"error": err.Error()})
		return
	}
	writeJSON(w, http.StatusOK, map[string]any{"status": "loaded", "model": req.Model})
}

// HandleGenerate queues a generation call and handles execution flow.
func (mgr *Manager) HandleGenerate(w http.ResponseWriter, r *http.Request) {
	isHelp, bodyBytes := checkHelp(r)
	if isHelp {
		sendHelp(w, "PAINT_SERVER_API.md", "", http.StatusOK)
		return
	}

	if r.Method != http.MethodPost {
		sendHelp(w, "PAINT_SERVER_API.md", "Method not allowed", http.StatusMethodNotAllowed)
		return
	}
	var req engine.GenerateRequest
	
	if len(bodyBytes) == 0 && r.Body != nil {
		var err error
		bodyBytes, err = io.ReadAll(r.Body)
		if err == nil {
			r.Body = io.NopCloser(bytes.NewReader(bodyBytes))
		}
	}

	if len(bodyBytes) > 0 {
		if err := json.Unmarshal(bodyBytes, &req); err != nil {
			writeJSON(w, http.StatusBadRequest, map[string]any{"error": "invalid JSON body"})
			return
		}
	} else {
		writeJSON(w, http.StatusBadRequest, map[string]any{"error": "invalid JSON body"})
		return
	}

	if !mgr.AcquireSemaphore() {
		writeJSON(w, http.StatusTooManyRequests, map[string]any{"error": "server busy: max generation concurrency reached"})
		return
	}
	defer mgr.ReleaseSemaphore()

	// Apply robust defaults
	if req.Width <= 0 {
		req.Width = 512
	}
	if req.Height <= 0 {
		req.Height = 512
	}
	if req.Steps <= 0 {
		req.Steps = 4
	}
	if req.Seed == 0 {
		req.Seed = time.Now().UnixNano()
	}
	if req.CFGScale <= 0 {
		req.CFGScale = 1.0
	}

	eng, _ := mgr.GetActive()
	if eng == nil {
		writeJSON(w, http.StatusInternalServerError, map[string]any{"error": "no image generation engine is loaded"})
		return
	}

	log.Printf("[paint] Generating %dx%d | steps=%d seed=%d | %q", req.Width, req.Height, req.Steps, req.Seed, req.Prompt)
	t0 := time.Now()
	result, err := eng.Generate(req)
	elapsed := time.Since(t0)

	if err != nil {
		log.Printf("[paint] Generate error: %v", err)
		writeJSON(w, http.StatusInternalServerError, map[string]any{"error": err.Error()})
		return
	}
	result.DurationMs = elapsed.Milliseconds()
	log.Printf("[paint] Done in %v → %s", elapsed, result.ImagePath)
	writeJSON(w, http.StatusOK, result)
}

// --- Internal Helper ---

func writeJSON(w http.ResponseWriter, status int, v any) {
	if status != http.StatusOK && status != http.StatusCreated && status != http.StatusAccepted {
		errStr := ""
		if m, ok := v.(map[string]any); ok {
			if e, exists := m["error"]; exists {
				errStr = fmt.Sprintf("%v", e)
			}
		} else if m, ok := v.(map[string]string); ok {
			if e, exists := m["error"]; exists {
				errStr = e
			}
		}
		if errStr != "" {
			sendHelp(w, "PAINT_SERVER_API.md", errStr, status)
			return
		}
	}

	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(status)
	_ = json.NewEncoder(w).Encode(v)
}
