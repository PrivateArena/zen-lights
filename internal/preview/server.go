// Package preview serves a lightweight HTTP UI for reviewing detected highlights.
// It streams the output video via a standard HTTP file server, so any browser
// with H.264 support can seek and play without additional software.
package preview

import (
	"encoding/json"
	"fmt"
	"html/template"
	"log"
	"net"
	"net/http"
	"time"

	"github.com/zen-lights/zen-lights/pkg/highlight"
)

// segmentView is the JSON/template-friendly view of a Segment.
type segmentView struct {
	Index    int     `json:"index"`
	StartSec float64 `json:"start_sec"`
	EndSec   float64 `json:"end_sec"`
	StartStr string  `json:"start"`
	EndStr   string  `json:"end"`
	Kills    int     `json:"kills"`
	Events   int     `json:"events"`
}

var pageTmpl = template.Must(template.New("page").Funcs(template.FuncMap{
	"add1": func(i int) int { return i + 1 },
}).Parse(`<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <title>zen-lights preview</title>
  <style>
    * { box-sizing: border-box; margin: 0; padding: 0; }
    body { background: #0d0d0d; color: #e0e0e0; font-family: "Courier New", monospace; padding: 1.5rem; }
    h1 { color: #f5a623; margin-bottom: 1rem; font-size: 1.4rem; letter-spacing: 2px; }
    video { width: 100%; max-width: 960px; display: block; border: 1px solid #333; margin-bottom: 1.5rem; border-radius: 4px; }
    h2 { font-size: 0.9rem; text-transform: uppercase; letter-spacing: 1px; color: #888; margin-bottom: 0.5rem; }
    .segments { max-width: 960px; }
    .seg { display: flex; align-items: center; gap: 1rem; padding: 0.6rem 1rem;
           margin-bottom: 0.35rem; background: #1a1a1a; border-radius: 4px;
           cursor: pointer; border: 1px solid transparent; transition: border-color 0.15s; }
    .seg:hover { border-color: #f5a623; }
    .seg.active { border-color: #f5a623; background: #231f13; }
    .num  { color: #f5a623; font-weight: bold; min-width: 2rem; }
    .time { color: #aaa; min-width: 14rem; }
    .kills { color: #fff; font-weight: bold; }
    .events { color: #666; font-size: 0.85rem; }
    .footer { margin-top: 1.5rem; color: #444; font-size: 0.8rem; }
  </style>
</head>
<body>
  <h1>⚡ zen-lights</h1>
  <video id="v" controls preload="metadata"></video>
  <h2>{{len .Segments}} highlight segment(s) — click to seek</h2>
  <div class="segments">
    {{range .Segments}}
    <div class="seg" id="seg-{{.Index}}" onclick="seekTo({{.StartSec}}, {{.Index}})">
      <span class="num">#{{add1 .Index}}</span>
      <span class="time">{{.StartStr}} → {{.EndStr}}</span>
      <span class="kills">{{.Kills}} kills</span>
      <span class="events">({{.Events}} event{{if gt .Events 1}}s{{end}})</span>
    </div>
    {{end}}
  </div>
  <div class="footer">output: {{.OutputPath}}</div>
  <script>
    const vid = document.getElementById('v');
    // Load source after DOM is ready to allow metadata preload
    vid.src = '/video/output';

    function seekTo(t, idx) {
      document.querySelectorAll('.seg').forEach(el => el.classList.remove('active'));
      document.getElementById('seg-' + idx).classList.add('active');
      vid.currentTime = t;
      vid.play();
    }

    // Highlight the segment matching current playback position
    const segs = {{.SegmentsJSON}};
    vid.addEventListener('timeupdate', () => {
      const t = vid.currentTime;
      segs.forEach((s, i) => {
        const el = document.getElementById('seg-' + i);
        if (el) el.classList.toggle('active', t >= s.start_sec && t <= s.end_sec);
      });
    });
  </script>
</body>
</html>`))

type pageData struct {
	Segments     []segmentView
	SegmentsJSON template.JS
	OutputPath   string
}

// Serve launches an HTTP server on addr that previews the highlight output.
//
//	addr       — listen address, e.g. "localhost:8080" or ":8080"
//	outputPath — path to the merged highlight .mp4 to serve
//	segments   — the highlight segments detected by the pipeline
//
// Blocks until the server is shut down. Open http://addr in any browser.
func Serve(addr, outputPath string, segments []highlight.Segment) error {
	views := make([]segmentView, len(segments))
	for i, s := range segments {
		views[i] = segmentView{
			Index:    i,
			StartSec: s.Start.Seconds(),
			EndSec:   s.End.Seconds(),
			StartStr: fmtDur(s.Start),
			EndStr:   fmtDur(s.End),
			Kills:    s.TotalKills(),
			Events:   len(s.Events),
		}
	}

	jsonBytes, _ := json.Marshal(views)
	data := pageData{
		Segments:     views,
		SegmentsJSON: template.JS(jsonBytes),
		OutputPath:   outputPath,
	}

	mux := http.NewServeMux()

	mux.HandleFunc("/", func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "text/html; charset=utf-8")
		if err := pageTmpl.Execute(w, data); err != nil {
			log.Printf("preview template: %v", err)
		}
	})

	mux.HandleFunc("/video/output", func(w http.ResponseWriter, r *http.Request) {
		http.ServeFile(w, r, outputPath)
	})

	mux.HandleFunc("/segments", func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		_, _ = w.Write(jsonBytes)
	})

	ln, err := net.Listen("tcp", addr)
	if err != nil {
		return fmt.Errorf("listen %s: %w", addr, err)
	}
	log.Printf("🎬  preview → http://%s", ln.Addr())
	return http.Serve(ln, mux)
}

func fmtDur(d time.Duration) string {
	if d < 0 {
		d = 0
	}
	h := int(d.Hours())
	m := int(d.Minutes()) % 60
	s := d.Seconds() - float64(h*3600) - float64(m*60)
	if h > 0 {
		return fmt.Sprintf("%d:%02d:%05.2f", h, m, s)
	}
	return fmt.Sprintf("%d:%05.2f", m, s)
}
