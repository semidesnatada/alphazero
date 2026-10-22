package games

import "math/rand"

func randomKey(m map[int]bool) int {
	// expects a non-empty map

	var result int
	n := 0

	for k := range m {
		n++
		if rand.Intn(n) == 0 {
			result = k
		}
	}

	return result
}