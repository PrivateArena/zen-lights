package paint

import (
	"encoding/json"
	"log"
	"net/http"
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

// HandleStatus reports active model and backend info.
func (mgr *Manager) HandleStatus(w http.ResponseWriter, r *http.Request) {
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
	names, err := listModels(mgr.Config.ModelsDir)
	if err != nil {
		writeJSON(w, http.StatusInternalServerError, map[string]any{"error": err.Error()})
		return
	}
	writeJSON(w, http.StatusOK, map[string]any{"models": names})
}

// HandleLoad triggers on-demand loading of a specific model directory.
func (mgr *Manager) HandleLoad(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodPost {
		http.Error(w, "Method not allowed", http.StatusMethodNotAllowed)
		return
	}
	var req struct {
		Model string `json:"model"`
	}
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil || req.Model == "" {
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
	if r.Method != http.MethodPost {
		http.Error(w, "Method not allowed", http.StatusMethodNotAllowed)
		return
	}
	var req engine.GenerateRequest
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
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
	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(status)
	_ = json.NewEncoder(w).Encode(v)
}
