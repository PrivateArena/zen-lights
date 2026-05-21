// Package ffmpeg wraps FFmpeg/ffprobe CLI tools for video probing,
// frame streaming, and lossless clip extraction.
package ffmpeg

import (
	"encoding/json"
	"fmt"
	"os/exec"
	"strconv"
	"strings"
	"time"
)

// VideoInfo contains key metadata for a video file.
type VideoInfo struct {
	Path     string
	Width    int
	Height   int
	FPS      float64
	Duration time.Duration
}

// ffprobeFormat mirrors relevant fields from ffprobe's JSON output.
type ffprobeOutput struct {
	Streams []struct {
		CodecType  string `json:"codec_type"`
		Width      int    `json:"width"`
		Height     int    `json:"height"`
		RFrameRate string `json:"r_frame_rate"`
	} `json:"streams"`
	Format struct {
		Duration string `json:"duration"`
	} `json:"format"`
}

// Probe runs ffprobe on path and returns video metadata.
// Requires ffprobe to be available on PATH.
func Probe(path string) (*VideoInfo, error) {
	cmd := exec.Command("ffprobe",
		"-v", "quiet",
		"-select_streams", "v:0",
		"-show_streams",
		"-show_format",
		"-of", "json",
		path,
	)
	raw, err := cmd.Output()
	if err != nil {
		return nil, fmt.Errorf("ffprobe %q: %w", path, err)
	}

	var probe ffprobeOutput
	if err := json.Unmarshal(raw, &probe); err != nil {
		return nil, fmt.Errorf("ffprobe json: %w", err)
	}

	info := &VideoInfo{Path: path}

	for _, s := range probe.Streams {
		if s.Width > 0 {
			info.Width = s.Width
			info.Height = s.Height
			info.FPS = parseFrameRate(s.RFrameRate)
			break
		}
	}

	if info.Width == 0 {
		return nil, fmt.Errorf("no video stream found in %q", path)
	}

	if probe.Format.Duration != "" {
		secs, _ := strconv.ParseFloat(strings.TrimSpace(probe.Format.Duration), 64)
		info.Duration = time.Duration(secs * float64(time.Second))
	}

	return info, nil
}

// parseFrameRate parses FFmpeg's rational frame rate string (e.g. "30000/1001").
func parseFrameRate(r string) float64 {
	parts := strings.SplitN(r, "/", 2)
	if len(parts) != 2 {
		return 30
	}
	num, _ := strconv.ParseFloat(parts[0], 64)
	den, _ := strconv.ParseFloat(parts[1], 64)
	if den == 0 {
		return 30
	}
	return num / den
}
