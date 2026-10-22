package neuralnets

import (
	"fmt"
	"math"
	"math/rand"
	"os"
)

type ReplayBuffer struct {
	Histories []BufferItem
	Capacity int
	Content int
	Next int
}

type BufferItem struct {
	BoardState []int
	MCTSPolicy []float64
	AssessedGameValue float64
}

func CreateReplayBuffer(capacity int) *ReplayBuffer {
	if capacity <= 0 {
        panic("replay buffer capacity must be positive")
    }
	return &ReplayBuffer{
		Histories: make([]BufferItem, capacity),
		Capacity: capacity,
		Content: 0,
		Next: 0,
	}
}

func (r *ReplayBuffer) AddItem(boardState []int, policy []float64, gameValue float64) {

	r.Histories[r.Next] = BufferItem{
		// hard copy the state so that we can modify the others objects as we wish
		BoardState: append([]int{}, boardState...),
		MCTSPolicy: append([]float64{}, policy...),
		AssessedGameValue: gameValue,
	}

	r.Next = (r.Next + 1) % r.Capacity
	
	if r.Content < r.Capacity {
		r.Content ++
	}

}

func (r ReplayBuffer) selectBatch(batchSize int) ([][]*Value, [][]*Value) {

	batchInputs := make([][]*Value, batchSize)
	batchOutputs := make([][]*Value, batchSize)

	if batchSize > r.Content {
		fmt.Printf("not enough training data in buffer for batch size. Closing\n")
		os.Exit(1)
	}
	// select random distinct indices to identify the batch
	batchIndices := RandomIndices(r.Content, batchSize)

	for i, index := range batchIndices {

		// reshape inputs into correct type
		boardInputs := make([]*Value, len(r.Histories[index].BoardState))
		for i, square := range r.Histories[index].BoardState {
			boardInputs[i] = NewValue(float64(square), fmt.Sprintf("board-square-%d", i))
		}
		batchInputs[i] = boardInputs

		// reshape outputs into correct type
		outs := []*Value{}
		for _, policyVal := range r.Histories[index].MCTSPolicy {
			outs = append(outs, NewValue(policyVal, ""))
		}
		outs = append(outs, NewValue(r.Histories[index].AssessedGameValue, ""))
		batchOutputs[i] = outs

	}

	return batchInputs, batchOutputs
}

func (r *ReplayBuffer) PopulateWithTest(boardSize, policySize int) {
	// creates fake training data for testing the other parts of the module
	//  not for use with the real algorithm / training
	//  assumes an empty buffer at the start

	testCount := r.Capacity / 3

	for i := 0; i < testCount; i ++ {

		fakeB := make([]int, boardSize)
		fakeP := make([]float64, policySize)

		for j := 0; j < boardSize; j ++ {
			fakeB[j] = rand.Intn(3) - 1
		}
		sumP := 0.0
		for k := 0; k < policySize; k ++ {
			newV := math.Exp(rand.Float64())
			fakeP[k] = newV
			sumP += newV
		}
		for k := 0; k < policySize; k ++ {
			fakeP[k] /= sumP
		}

		tot := 0.0
		for k := 0; k < policySize; k ++ {
			tot += fakeP[k]
		}
		// fmt.Println(tot)

		r.AddItem(fakeB, fakeP, rand.Float64() * 2.0 - 1.0)

	}


}