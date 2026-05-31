package summarize

import (
	"math"
)

// Transpose transposes a square or rectangular 2D float64 slice.
func Transpose(matrix [][]float64) [][]float64 {
	r := len(matrix)
	if r == 0 {
		return nil
	}
	c := len(matrix[0])
	t := make([][]float64, c)
	for i := range t {
		t[i] = make([]float64, r)
		for j := 0; j < r; j++ {
			t[i][j] = matrix[j][i]
		}
	}
	return t
}

// L2Norm calculates the L2 norm of a vector.
func L2Norm(v []float64) float64 {
	var sum float64
	for _, val := range v {
		sum += val * val
	}
	return math.Sqrt(sum)
}

// DotProduct calculates the dot product of a vector slice and another vector slice of the same size.
func DotProduct(row []float64, vec []float64) float64 {
	var sum float64
	limit := len(row)
	if len(vec) < limit {
		limit = len(vec)
	}
	for i := 0; i < limit; i++ {
		sum += row[i] * vec[i]
	}
	return sum
}

// PowerMethodTextRank runs the PageRank power method for TextRank.
func PowerMethodTextRank(matrix [][]float64, epsilon float64) []float64 {
	n := len(matrix)
	if n == 0 {
		return nil
	}
	transposed := Transpose(matrix)

	p := make([]float64, n)
	for i := range p {
		p[i] = 1.0 / float64(n)
	}

	lambdaVal := 1.0
	for iter := 0; iter < 1000 && lambdaVal > epsilon; iter++ {
		nextP := make([]float64, n)
		for i := 0; i < n; i++ {
			nextP[i] = DotProduct(transposed[i], p)
		}

		// Calculate L2 norm of (nextP - p)
		var diffSum float64
		for i := 0; i < n; i++ {
			diff := nextP[i] - p[i]
			diffSum += diff * diff
		}
		lambdaVal = math.Sqrt(diffSum)
		p = nextP
	}

	return p
}

// PowerMethodLexRank runs the PageRank power method for LexRank with L2 normalization.
func PowerMethodLexRank(matrix [][]float64, epsilon float64) []float64 {
	n := len(matrix)
	if n == 0 {
		return nil
	}
	transposed := Transpose(matrix)

	p := make([]float64, n)
	for i := range p {
		p[i] = 1.0 / float64(n)
	}

	lambdaVal := 1.0
	for iter := 0; iter < 1000 && lambdaVal > epsilon; iter++ {
		nextP := make([]float64, n)
		for i := 0; i < n; i++ {
			nextP[i] = DotProduct(transposed[i], p)
		}

		// Normalize nextP by its L2 norm
		norm := L2Norm(nextP)
		if norm > 1e-9 {
			for i := 0; i < n; i++ {
				nextP[i] /= norm
			}
		}

		// Calculate L2 norm of (nextP - p)
		var diffSum float64
		for i := 0; i < n; i++ {
			diff := nextP[i] - p[i]
			diffSum += diff * diff
		}
		lambdaVal = math.Sqrt(diffSum)
		p = nextP
	}

	return p
}
