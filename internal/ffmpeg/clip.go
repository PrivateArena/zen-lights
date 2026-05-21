package ffmpeg

import (
	"fmt"
	"os"
	"os/exec"
	"path/filepath"
	"strings"
	"time"
)

// Segment is a time range to extract from a video.
type Segment struct {
	Start time.Duration
	End   time.Duration
}

// CutAndMerge extracts segments from src and concatenates them into dst.
//
// Uses stream copy (-c copy) throughout — no re-encode, near-instant for any
// length of video. All intermediate part files are written to a temp directory
// that is cleaned up on return, so only the final dst file persists.
func CutAndMerge(src string, segments []Segment, dst string) error {
	if len(segments) == 0 {
		return fmt.Errorf("no segments provided")
	}

	tmp, err := os.MkdirTemp("", "zenlights-*")
	if err != nil {
		return fmt.Errorf("create tmp dir: %w", err)
	}
	defer os.RemoveAll(tmp)

	// --- 1. Cut individual segments ---
	parts := make([]string, 0, len(segments))
	for i, seg := range segments {
		out := filepath.Join(tmp, fmt.Sprintf("part%04d.mp4", i))
		if err := cutSegment(src, seg, out); err != nil {
			return fmt.Errorf("cut segment %d (%s–%s): %w",
				i, seg.Start.Round(time.Millisecond), seg.End.Round(time.Millisecond), err)
		}
		parts = append(parts, out)
	}

	// --- 2. Single segment: skip the concat pass ---
	if len(parts) == 1 {
		return moveFile(parts[0], dst)
	}

	// --- 3. Multi-segment: concat demuxer (stream copy, no decode) ---
	concatList := filepath.Join(tmp, "concat.txt")
	var sb strings.Builder
	for _, p := range parts {
		// Single-quoted paths in concat list; escape embedded single quotes
		safe := strings.ReplaceAll(p, "'", `'\''`)
		sb.WriteString(fmt.Sprintf("file '%s'\n", safe))
	}
	if err := os.WriteFile(concatList, []byte(sb.String()), 0o600); err != nil {
		return fmt.Errorf("write concat list: %w", err)
	}

	cmd := exec.Command("ffmpeg",
		"-y",
		"-f", "concat",
		"-safe", "0",
		"-i", concatList,
		"-c", "copy",
		dst,
	)
	if out, err := cmd.CombinedOutput(); err != nil {
		return fmt.Errorf("ffmpeg concat: %w\n%s", err, out)
	}
	return nil
}

// cutSegment extracts [seg.Start, seg.End] from src into dst using stream copy.
// -ss before -i = fast seek (keyframe-accurate input seek).
func cutSegment(src string, seg Segment, dst string) error {
	cmd := exec.Command("ffmpeg",
		"-y",
		"-ss", fmtDuration(seg.Start),
		"-to", fmtDuration(seg.End),
		"-i", src,
		"-c", "copy",
		"-avoid_negative_ts", "make_zero",
		"-movflags", "+faststart",
		dst,
	)
	if out, err := cmd.CombinedOutput(); err != nil {
		return fmt.Errorf("ffmpeg cut: %w\n%s", err, out)
	}
	return nil
}

// fmtDuration formats a duration as HH:MM:SS.mmm for FFmpeg's -ss / -to flags.
func fmtDuration(d time.Duration) string {
	if d < 0 {
		d = 0
	}
	h := int(d.Hours())
	m := int(d.Minutes()) % 60
	s := d.Seconds() - float64(h*3600) - float64(m*60)
	return fmt.Sprintf("%02d:%02d:%06.3f", h, m, s)
}

func moveFile(src, dst string) error {
	// Try a cheap rename first (same filesystem)
	if err := os.Rename(src, dst); err == nil {
		return nil
	}
	// Fall back to copy+delete across filesystems
	data, err := os.ReadFile(src)
	if err != nil {
		return err
	}
	return os.WriteFile(dst, data, 0o644)
}
