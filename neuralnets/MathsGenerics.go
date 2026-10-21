package neuralnets

func FloatSliceSum(nums []float64) float64 {
    var sum float64
    for _, n := range nums {
        sum += n
    }
    return sum
}