package translate

import (
	"context"
	"net/http"
	"net/http/httptest"
	"testing"
)

func TestOnlineEngine_Translate(t *testing.T) {
	// Create mock server
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		// Mock Google Translate response shape: [[["seg1 trans", "seg1 src", ...]], ...]
		resp := `[[["Hello world","こんにちは世界",null,null,3]],null,"ja"]`
		w.Header().Set("Content-Type", "application/json")
		w.WriteHeader(http.StatusOK)
		w.Write([]byte(resp))
	}))
	defer server.Close()

	// Initialize engine and overwrite client transport to use mock server
	engine := NewOnlineEngine()
	engine.client.Transport = &mockTransport{mockServerURL: server.URL}

	translated, err := engine.Translate(context.Background(), "こんにちは世界", "ja", "en")
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	expected := "Hello world"
	if translated != expected {
		t.Errorf("expected %q, got %q", expected, translated)
	}
}

type mockTransport struct {
	mockServerURL string
}

func (m *mockTransport) RoundTrip(req *http.Request) (*http.Response, error) {
	// Rewrite request URL to point to mock server
	mockURL, _ := http.NewRequest(req.Method, m.mockServerURL, nil)
	req.URL.Scheme = mockURL.URL.Scheme
	req.URL.Host = mockURL.URL.Host
	return http.DefaultTransport.RoundTrip(req)
}
