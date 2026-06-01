package server

import (
	"bytes"
	"encoding/json"
	"fmt"
	"image"
	"io"
	"log"
	"net"
	"net/http"
	"os"
	"strings"

	_ "image/jpeg"
	_ "image/png"

	"github.com/zen-lights/zen-lights/internal/ocr"
	"github.com/zen-lights/zen-lights/internal/paint"
	"github.com/zen-lights/zen-lights/internal/summarize"
	"github.com/zen-lights/zen-lights/internal/translate"
)

// Server provides a persistent HTTP API for OCR, translation, summarization, and image generation.
type Server struct {
	manager          *ocr.Manager
	translateManager *translate.Manager
	summarizeManager *summarize.Manager
	paintManager     *paint.Manager
	addr             string
	defaultModel     string
}

// New creates a new unified server.
func New(
	addr string,
	manager *ocr.Manager,
	defaultModel string,
	translateManager *translate.Manager,
	summarizeManager *summarize.Manager,
	paintManager *paint.Manager,
) *Server {
	if defaultModel == "" {
		defaultModel = "ch"
	}
	return &Server{
		addr:             addr,
		manager:          manager,
		translateManager: translateManager,
		summarizeManager: summarizeManager,
		paintManager:     paintManager,
		defaultModel:     defaultModel,
	}
}

// Start runs the HTTP server.
func (s *Server) Start() error {
	mux := http.NewServeMux()
	mux.HandleFunc("/ocr", s.handleOCR)
	mux.HandleFunc("/status", s.handleStatus)
	mux.HandleFunc("/translate", s.handleTranslate)
	mux.HandleFunc("/summarize", s.handleSummarize)

	// Register paint endpoints if the manager is initialized
	if s.paintManager != nil {
		paint.RegisterHandlers(mux, s.paintManager)
	}

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
func (s *Server) sendHelp(w http.ResponseWriter, filename string, errStr string, statusCode int) {
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

// handleOCR unifies OCR recognition and model settings selection.
func (s *Server) handleOCR(w http.ResponseWriter, r *http.Request) {
	isHelp, _ := checkHelp(r)
	if isHelp {
		s.sendHelp(w, "OCR_SERVER_API.md", "", http.StatusOK)
		return
	}

	if r.Method == http.MethodGet {
		w.Header().Set("Content-Type", "application/json")
		json.NewEncoder(w).Encode(map[string]string{
			"default_model": s.defaultModel,
		})
		return
	}

	if r.Method != http.MethodPost && r.Method != http.MethodPut {
		s.sendHelp(w, "OCR_SERVER_API.md", "Method not allowed", http.StatusMethodNotAllowed)
		return
	}

	// Read body bytes to safely inspect and reuse
	var bodyBytes []byte
	if r.Body != nil {
		var err error
		bodyBytes, err = io.ReadAll(r.Body)
		if err == nil {
			r.Body = io.NopCloser(bytes.NewReader(bodyBytes))
		}
	}

	contentType := r.Header.Get("Content-Type")
	var hasImage bool

	if strings.HasPrefix(contentType, "multipart/form-data") {
		r.ParseMultipartForm(32 << 20)
		_, _, errForm := r.FormFile("image")
		if errForm == nil {
			hasImage = true
		}
	} else if len(bodyBytes) > 0 {
		if !strings.Contains(contentType, "application/json") {
			hasImage = true
		}
	}

	// 1. If it's a dynamic model selection (no image uploaded, but has model/lang parameter)
	if !hasImage {
		model := r.URL.Query().Get("model")
		if model == "" {
			model = r.URL.Query().Get("lang")
		}
		if model == "" && strings.Contains(contentType, "application/json") {
			var body struct {
				Model string `json:"model"`
				Lang  string `json:"lang"`
			}
			if err := json.Unmarshal(bodyBytes, &body); err == nil {
				model = body.Model
				if model == "" {
					model = body.Lang
				}
			}
		}

		if model == "" {
			s.sendHelp(w, "OCR_SERVER_API.md", "Image or model parameter is required", http.StatusBadRequest)
			return
		}

		// Verify the model exists in the profiles
		if !s.manager.HasLanguage(model) {
			s.sendHelp(w, "OCR_SERVER_API.md", fmt.Sprintf("Model/Language %q not available in config", model), http.StatusNotFound)
			return
		}

		s.defaultModel = model
		w.Header().Set("Content-Type", "application/json")
		json.NewEncoder(w).Encode(map[string]string{
			"status":        "success",
			"default_model": s.defaultModel,
		})
		return
	}

	// 2. Perform OCR
	langID := r.URL.Query().Get("model")
	if langID == "" {
		langID = r.URL.Query().Get("lang")
	}
	if langID == "" && strings.Contains(contentType, "application/json") {
		var body struct {
			Model string `json:"model"`
			Lang  string `json:"lang"`
		}
		if err := json.Unmarshal(bodyBytes, &body); err == nil {
			langID = body.Model
			if langID == "" {
				langID = body.Lang
			}
		}
	}

	// If a model is explicitly requested, also update the active default model of the server
	if langID != "" {
		if !s.manager.HasLanguage(langID) {
			s.sendHelp(w, "OCR_SERVER_API.md", fmt.Sprintf("Language %q not available", langID), http.StatusNotFound)
			return
		}
		s.defaultModel = langID
	} else {
		langID = s.defaultModel
	}

	var img image.Image
	var imgErr error
	file, _, formErr := r.FormFile("image")
	if formErr == nil {
		defer file.Close()
		img, _, imgErr = image.Decode(file)
	} else {
		img, _, imgErr = image.Decode(r.Body)
	}

	if imgErr != nil {
		s.sendHelp(w, "OCR_SERVER_API.md", fmt.Sprintf("Failed to decode image: %v", imgErr), http.StatusBadRequest)
		return
	}

	client, err := s.manager.GetClient(langID)
	if err != nil {
		s.sendHelp(w, "OCR_SERVER_API.md", fmt.Sprintf("Language %q not available: %v", langID, err), http.StatusNotFound)
		return
	}

	bounds := img.Bounds()
	wImg, hImg := bounds.Dx(), bounds.Dy()
	frameData := make([]byte, wImg*hImg*3)
	for y := 0; y < hImg; y++ {
		for x := 0; x < wImg; x++ {
			rCol, g, b, _ := img.At(bounds.Min.X+x, bounds.Min.Y+y).RGBA()
			idx := (y*wImg + x) * 3
			frameData[idx] = uint8(rCol >> 8)
			frameData[idx+1] = uint8(g >> 8)
			frameData[idx+2] = uint8(b >> 8)
		}
	}

	results, err := client.ReadFullFrame(frameData, wImg, hImg)
	if err != nil {
		s.sendHelp(w, "OCR_SERVER_API.md", fmt.Sprintf("OCR processing failed: %v", err), http.StatusInternalServerError)
		return
	}

	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(recognizeResponse{Results: results})
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
	isHelp, bodyBytes := checkHelp(r)
	if isHelp {
		s.sendHelp(w, "TRANSLATE_SERVER_API.md", "", http.StatusOK)
		return
	}

	if r.Method != http.MethodPost && r.Method != http.MethodGet {
		s.sendHelp(w, "TRANSLATE_SERVER_API.md", "Method not allowed", http.StatusMethodNotAllowed)
		return
	}

	if s.translateManager == nil {
		s.sendHelp(w, "TRANSLATE_SERVER_API.md", "Translation engine not initialized", http.StatusServiceUnavailable)
		return
	}

	var req translateRequest

	if r.Method == http.MethodPost {
		if len(bodyBytes) == 0 {
			var err error
			bodyBytes, err = io.ReadAll(r.Body)
			if err == nil {
				r.Body = io.NopCloser(bytes.NewReader(bodyBytes))
			}
		}
		if len(bodyBytes) > 0 {
			if err := json.Unmarshal(bodyBytes, &req); err != nil {
				// fallback to parameters if body decoding fails
			}
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
		s.sendHelp(w, "TRANSLATE_SERVER_API.md", "Parameter 'text' is required", http.StatusBadRequest)
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
		s.sendHelp(w, "TRANSLATE_SERVER_API.md", fmt.Sprintf("Translation failed: %v", err), http.StatusInternalServerError)
		return
	}

	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(translateResponse{Translated: translated})
}

type summarizeRequest struct {
	Text     string `json:"text"`
	Count    int    `json:"count"`
	Language string `json:"language"`
}

type summarizeResponse struct {
	Summary []string `json:"summary,omitempty"`
	Error   string   `json:"error,omitempty"`
}

func (s *Server) handleSummarize(w http.ResponseWriter, r *http.Request) {
	isHelp, bodyBytes := checkHelp(r)
	if isHelp {
		s.sendHelp(w, "SUMMARIZE_SERVER_API.md", "", http.StatusOK)
		return
	}

	if r.Method != http.MethodPost && r.Method != http.MethodGet {
		s.sendHelp(w, "SUMMARIZE_SERVER_API.md", "Method not allowed", http.StatusMethodNotAllowed)
		return
	}

	if s.summarizeManager == nil {
		s.sendHelp(w, "SUMMARIZE_SERVER_API.md", "Summarization engine not initialized", http.StatusServiceUnavailable)
		return
	}

	var req summarizeRequest

	if r.Method == http.MethodPost {
		if len(bodyBytes) == 0 {
			var err error
			bodyBytes, err = io.ReadAll(r.Body)
			if err == nil {
				r.Body = io.NopCloser(bytes.NewReader(bodyBytes))
			}
		}
		if len(bodyBytes) > 0 {
			if err := json.Unmarshal(bodyBytes, &req); err != nil {
				// fallback to parameters if body decoding fails
			}
		}
	}

	if req.Text == "" {
		req.Text = r.URL.Query().Get("text")
	}
	if req.Count <= 0 {
		var countVal int
		_, err := fmt.Sscanf(r.URL.Query().Get("count"), "%d", &countVal)
		if err == nil && countVal > 0 {
			req.Count = countVal
		} else {
			req.Count = 3 // Default
		}
	}
	if req.Language == "" {
		req.Language = r.URL.Query().Get("language")
		if req.Language == "" {
			req.Language = r.URL.Query().Get("lang")
		}
		if req.Language == "" {
			req.Language = "en"
		}
	}

	if req.Text == "" {
		s.sendHelp(w, "SUMMARIZE_SERVER_API.md", "Parameter 'text' is required", http.StatusBadRequest)
		return
	}

	summary, err := s.summarizeManager.Summarize(req.Text, req.Count, req.Language)
	if err != nil {
		s.sendHelp(w, "SUMMARIZE_SERVER_API.md", fmt.Sprintf("Summarization failed: %v", err), http.StatusInternalServerError)
		return
	}

	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(summarizeResponse{Summary: summary})
}
