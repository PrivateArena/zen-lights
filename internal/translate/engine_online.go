package translate

import (
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"net/url"
	"strings"
	"time"
)

// OnlineEngine implements Engine using the free Google Translate API.
type OnlineEngine struct {
	client *http.Client
	apiURL string
}

// NewOnlineEngine creates a new online translation engine.
func NewOnlineEngine(cfg OnlineConfig) *OnlineEngine {
	timeout := 10 * time.Second
	if cfg.TimeoutMS > 0 {
		timeout = time.Duration(cfg.TimeoutMS) * time.Millisecond
	}

	apiURL := cfg.APIURL
	if apiURL == "" {
		apiURL = "https://translate.googleapis.com/translate_a/single?client=gtx&sl=%s&tl=%s&dt=t&q=%s"
	}

	return &OnlineEngine{
		client: &http.Client{
			Timeout: timeout,
		},
		apiURL: apiURL,
	}
}

// Translate translates the given text using the free Google Translate API.
func (e *OnlineEngine) Translate(ctx context.Context, text, srcLang, tgtLang string) (string, error) {
	if text == "" {
		return "", nil
	}

	// Format languages
	if srcLang == "" {
		srcLang = "auto"
	}
	if tgtLang == "" {
		tgtLang = "en"
	}

	apiURL := fmt.Sprintf(
		e.apiURL,
		url.QueryEscape(srcLang),
		url.QueryEscape(tgtLang),
		url.QueryEscape(text),
	)

	req, err := http.NewRequestWithContext(ctx, http.MethodGet, apiURL, nil)
	if err != nil {
		return "", fmt.Errorf("create request: %w", err)
	}

	resp, err := e.client.Do(req)
	if err != nil {
		return "", fmt.Errorf("http request: %w", err)
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		body, _ := io.ReadAll(resp.Body)
		return "", fmt.Errorf("bad status code: %d, response: %s", resp.StatusCode, string(body))
	}

	body, err := io.ReadAll(resp.Body)
	if err != nil {
		return "", fmt.Errorf("read response body: %w", err)
	}

	// Parse Google Translate nested array JSON response
	var raw []interface{}
	if err := json.Unmarshal(body, &raw); err != nil {
		return "", fmt.Errorf("parse json: %w", err)
	}

	if len(raw) == 0 {
		return "", fmt.Errorf("empty response")
	}

	firstElement, ok := raw[0].([]interface{})
	if !ok {
		return "", fmt.Errorf("invalid response structure (first element is not array)")
	}

	var builder strings.Builder
	for _, segment := range firstElement {
		parts, ok := segment.([]interface{})
		if !ok || len(parts) == 0 {
			continue
		}
		translatedPart, ok := parts[0].(string)
		if ok {
			builder.WriteString(translatedPart)
		}
	}

	return builder.String(), nil
}

// Close closes the engine.
func (e *OnlineEngine) Close() error {
	return nil
}
