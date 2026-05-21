package ffmpeg

import (
	"fmt"
	"io"
	"os/exec"
	"time"
)

// Frame holds a single sampled video frame as raw RGB24 bytes.
// Data is row-major: pixel (x,y) starts at index (y*Width + x) * 3.
type Frame struct {
	Index     int           // sequential sample index (0-based)
	Timestamp time.Duration // position in the original video
	Data      []byte        // RGB24, len = Width * Height * 3
	Width     int
	Height    int
}

// StreamFrames starts an FFmpeg subprocess that decodes the video at sampleFPS
// frames per second and sends each frame to the returned channel.
//
// No intermediate files are written — frames travel through the pipe directly
// into the returned channel. The goroutine sends a single error (or nil on
// success) to errCh and then closes frameCh.
//
// The caller MUST drain frameCh fully or the FFmpeg process will block.
// To cancel early, use a separate goroutine and discard remaining frames.
func StreamFrames(info *VideoInfo, sampleFPS float64) (<-chan Frame, <-chan error) {
	frameCh := make(chan Frame, 16) // buffer 16 frames to absorb pipe jitter
	errCh := make(chan error, 1)

	go func() {
		var runErr error
		defer func() {
			close(frameCh)
			errCh <- runErr
		}()

		frameBytes := info.Width * info.Height * 3

		cmd := exec.Command("ffmpeg",
			"-i", info.Path,
			// Resample to exactly sampleFPS — FFmpeg handles the frame selection
			"-vf", fmt.Sprintf("fps=%g", sampleFPS),
			"-f", "rawvideo",
			"-pix_fmt", "rgb24",
			"-an",    // drop audio — we only need video
			"pipe:1", // write raw frames to stdout
		)

		stdout, err := cmd.StdoutPipe()
		if err != nil {
			runErr = fmt.Errorf("stdout pipe: %w", err)
			return
		}
		// Suppress FFmpeg's verbose stderr; set to os.Stderr for debugging
		cmd.Stderr = io.Discard

		if err := cmd.Start(); err != nil {
			runErr = fmt.Errorf("ffmpeg start: %w", err)
			return
		}

		// Reuse a single read buffer; copy into per-frame allocation only when sending.
		readBuf := make([]byte, frameBytes)
		idx := 0
		// Time step between sampled frames in the original video timeline
		dt := float64(time.Second) / sampleFPS

		for {
			_, err := io.ReadFull(stdout, readBuf)
			if err == io.EOF || err == io.ErrUnexpectedEOF {
				break // normal end-of-stream
			}
			if err != nil {
				runErr = fmt.Errorf("read frame %d: %w", idx, err)
				_ = cmd.Process.Kill()
				return
			}

			// Allocate per-frame copy so each Frame is independently addressable
			data := make([]byte, frameBytes)
			copy(data, readBuf)

			frameCh <- Frame{
				Index:     idx,
				Timestamp: time.Duration(float64(idx) * dt),
				Data:      data,
				Width:     info.Width,
				Height:    info.Height,
			}
			idx++
		}

		_ = cmd.Wait()
	}()

	return frameCh, errCh
}
