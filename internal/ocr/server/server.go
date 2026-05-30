package server

import (
	"encoding/json"
	"fmt"
	"image"
	"io"
	"log"
	"net"
	"net/http"

	_ "image/jpeg"
	_ "image/png"

	"github.com/zen-lights/zen-lights/internal/ocr"
	"github.com/zen-lights/zen-lights/internal/translate"
)

// Server provides a persistent HTTP API for multi-language OCR.
type Server struct {
	manager          *ocr.Manager
	translateManager *translate.Manager
	addr             string
	defaultModel     string
}

// New creates a new OCR server.
func New(addr string, manager *ocr.Manager, defaultModel string, translateManager *translate.Manager) *Server {
	if defaultModel == "" {
		defaultModel = "ch"
	}
	return &Server{
		addr:             addr,
		manager:          manager,
		translateManager: translateManager,
		defaultModel:     defaultModel,
	}
}

// Start runs the HTTP server.
func (s *Server) Start() error {
	mux := http.NewServeMux()
	mux.HandleFunc("/recognize", s.handleRecognize)
	mux.HandleFunc("/status", s.handleStatus)
	mux.HandleFunc("/default-model", s.handleDefaultModel)
	mux.HandleFunc("/translate", s.handleTranslate)

	ln, err := net.Listen("tcp", s.addr)
	if err != nil {
		return fmt.Errorf("listen %s: %w", s.addr, err)
	}

	host, port, err := net.SplitHostPort(ln.Addr().String())
	if err == nil {
		if host == "" || host == "::" || host == "0.0.0.0" {
			host = "127.0.0.1"
		}
		log.Printf("🤖 OCR Server listening on http://%s:%s", host, port)
	} else {
		log.Printf("🤖 OCR Server listening on http://%s", ln.Addr())
	}

	// Pre-load/verify the default model at startup
	log.Printf("⚙️ OCR: Pre-loading default model %q...", s.defaultModel)
	_, err = s.manager.GetClient(s.defaultModel)
	if err != nil {
		log.Printf("Warning: failed to pre-load default model %q: %v", s.defaultModel, err)
	}

	return http.Serve(ln, mux)
}

func (s *Server) handleStatus(w http.ResponseWriter, r *http.Request) {
	w.WriteHeader(http.StatusOK)
	w.Write([]byte("OK"))
}

type recognizeResponse struct {
	Results []ocr.TextResult `json:"results"`
	Error   string           `json:"error,omitempty"`
}

func (s *Server) handleRecognize(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodPost {
		http.Error(w, "Method not allowed", http.StatusMethodNotAllowed)
		return
	}

	// 1. Get language/model ID
	langID := r.URL.Query().Get("model")
	if langID == "" {
		langID = r.URL.Query().Get("lang")
	}
	if langID == "" {
		langID = s.defaultModel
	}

	// 2. Parse image from request body (multipart or raw)
	var img image.Image
	file, _, err := r.FormFile("image")
	if err == nil {
		defer file.Close()
		img, _, err = image.Decode(file)
	} else {
		// Try reading raw body if multipart fails
		img, _, err = image.Decode(r.Body)
	}

	if err != nil {
		s.jsonError(w, fmt.Sprintf("Failed to decode image: %v", err), http.StatusBadRequest)
		return
	}

	// 3. Get OCR client for the language
	client, err := s.manager.GetClient(langID)
	if err != nil {
		s.jsonError(w, fmt.Sprintf("Language %q not available: %v", langID, err), http.StatusNotFound)
		return
	}

	// 4. Run OCR
	// Convert image to RGB24 frameData for Client.ReadFullFrame
	bounds := img.Bounds()
	wImg, hImg := bounds.Dx(), bounds.Dy()
	frameData := make([]byte, wImg*hImg*3)
	for y := 0; y < hImg; y++ {
		for x := 0; x < wImg; x++ {
			r, g, b, _ := img.At(bounds.Min.X+x, bounds.Min.Y+y).RGBA()
			idx := (y*wImg + x) * 3
			frameData[idx] = uint8(r >> 8)
			frameData[idx+1] = uint8(g >> 8)
			frameData[idx+2] = uint8(b >> 8)
		}
	}

	results, err := client.ReadFullFrame(frameData, wImg, hImg)
	if err != nil {
		s.jsonError(w, fmt.Sprintf("OCR processing failed: %v", err), http.StatusInternalServerError)
		return
	}

	// 5. Respond
	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(recognizeResponse{Results: results})
}

func (s *Server) jsonError(w http.ResponseWriter, msg string, code int) {
	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(code)
	json.NewEncoder(w).Encode(recognizeResponse{Error: msg})
}

func (s *Server) handleDefaultModel(w http.ResponseWriter, r *http.Request) {
	if r.Method == http.MethodGet {
		w.Header().Set("Content-Type", "application/json")
		json.NewEncoder(w).Encode(map[string]string{
			"default_model": s.defaultModel,
		})
		return
	}

	if r.Method != http.MethodPost && r.Method != http.MethodPut {
		http.Error(w, "Method not allowed", http.StatusMethodNotAllowed)
		return
	}

	model := r.URL.Query().Get("model")
	if model == "" {
		model = r.URL.Query().Get("lang")
	}
	if model == "" {
		var body struct {
			Model string `json:"model"`
			Lang  string `json:"lang"`
		}
		if err := json.NewDecoder(r.Body).Decode(&body); err == nil {
			model = body.Model
			if model == "" {
				model = body.Lang
			}
		}
	}

	if model == "" {
		s.jsonError(w, "Model parameter is required", http.StatusBadRequest)
		return
	}

	// Verify the model exists in the profiles
	if !s.manager.HasLanguage(model) {
		s.jsonError(w, fmt.Sprintf("Model/Language %q not available in config", model), http.StatusNotFound)
		return
	}

	s.defaultModel = model
	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(map[string]string{
		"status":        "success",
		"default_model": s.defaultModel,
	})
}

type translateRequest struct {
	Text   string `json:"text"`
	Source string `json:"source"`
	Target string `json:"target"`
}

type translateResponse struct {
	Translated string `json:"translated,omitempty"`
	Error      string `json:"error,omitempty"`
}

func (s *Server) handleTranslate(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodPost && r.Method != http.MethodGet {
		http.Error(w, "Method not allowed", http.StatusMethodNotAllowed)
		return
	}

	if s.translateManager == nil {
		s.jsonTranslateError(w, "Translation engine not initialized", http.StatusServiceUnavailable)
		return
	}

	var req translateRequest

	if r.Method == http.MethodPost {
		if err := json.NewDecoder(r.Body).Decode(&req); err != nil && err != io.EOF {
			// fallback to parameters if body decoding fails
		}
	}

	if req.Text == "" {
		req.Text = r.URL.Query().Get("text")
	}
	if req.Source == "" {
		req.Source = r.URL.Query().Get("source")
		if req.Source == "" {
			req.Source = r.URL.Query().Get("from")
		}
		if req.Source == "" {
			req.Source = r.URL.Query().Get("src")
		}
	}
	if req.Target == "" {
		req.Target = r.URL.Query().Get("target")
		if req.Target == "" {
			req.Target = r.URL.Query().Get("to")
		}
		if req.Target == "" {
			req.Target = r.URL.Query().Get("tgt")
		}
	}

	if req.Text == "" {
		s.jsonTranslateError(w, "Parameter 'text' is required", http.StatusBadRequest)
		return
	}
	if req.Source == "" {
		req.Source = "auto"
	}
	if req.Target == "" {
		req.Target = "en"
	}

	translated, err := s.translateManager.Translate(r.Context(), req.Text, req.Source, req.Target)
	if err != nil {
		s.jsonTranslateError(w, fmt.Sprintf("Translation failed: %v", err), http.StatusInternalServerError)
		return
	}

	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(translateResponse{Translated: translated})
}

func (s *Server) jsonTranslateError(w http.ResponseWriter, msg string, code int) {
	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(code)
	json.NewEncoder(w).Encode(translateResponse{Error: msg})
}
