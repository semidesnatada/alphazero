package neuralnets

import "math/rand"

func FloatSliceSum(nums []float64) float64 {
    // sums a slice of floats without requiring an external library
    var sum float64
    for _, n := range nums {
        sum += n
    }
    return sum
}

func RandomIndices(populationSize, sampleSize int) []int {
    // Conducts a fisher-yates randomisation of a slice of indices
    // to produce a random set of indices. Allows selecting a sample from a larger slice
    // without needing to mutate / shuffle the slice to be sampled from
    // use of a virtual slice of indices (using the map) saves memory and operations swapping slice entries

    if sampleSize <= 0 || populationSize <= 0 {
        return []int{}
    }
    if sampleSize > populationSize {
        sampleSize = populationSize
    }

    // swaps accounts for where entries that have been moved in our virtual slice
    swaps := make(map[int]int, sampleSize)
    // result acts as the head of the virtual slice and the output
    result := make([]int, 0, sampleSize)

    for i := 0; i < sampleSize; i++ {
        j := i + rand.Intn(populationSize - i)

        // Get the virtual value at position i
        valueI := i
        if v, ok := swaps[i]; ok {
            valueI = v
        }

        // Get the virtual value at position j
        valueJ := j
        if v, ok := swaps[j]; ok {
            valueJ = v
        }

        // Select the value at j
        result = append(result, valueJ)

        // Simulate swapping positions i and j
        swaps[j] = valueI
        delete(swaps, i)
    }

    return result
}