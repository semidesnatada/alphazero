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
	CurrentPlayer int
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

func (r *ReplayBuffer) AddItem(boardState []int, policy []float64, currentPlayer int, gameValue float64, sub bool) {

	savedB := make([]int, len(boardState))
	copy(savedB, boardState)

	savedP := make([]float64, len(policy))
	copy(savedP, policy)

	r.Histories[r.Next] = BufferItem{
		// hard copy the state so that we can modify the others objects as we wish
		BoardState: append([]int{}, boardState...),
		MCTSPolicy: append([]float64{}, policy...),
		CurrentPlayer: currentPlayer,
		AssessedGameValue: gameValue,
	}

	r.Next = (r.Next + 1) % r.Capacity
	
	if r.Content < r.Capacity {
		r.Content ++
	}

	if !sub {
		// add symmetries of the grid too
		// vertical reflection
		newBoard := make([]int, len(boardState))
		copy(newBoard, savedB)
		newPolicy := make([]float64, len(policy))
		copy(newPolicy, savedP)
		// fmt.Println(newBoard, gameValue, currentPlayer)
		// os.Exit(1)
		newBoard[0], newBoard[2] = newBoard[2], newBoard[0]
		newBoard[3], newBoard[5] = newBoard[5], newBoard[3]
		newBoard[6], newBoard[8] = newBoard[8], newBoard[6]
		newPolicy[0], newPolicy[2] = newPolicy[2], newPolicy[0]
		newPolicy[3], newPolicy[5] = newPolicy[5], newPolicy[3]
		newPolicy[6], newPolicy[8] = newPolicy[8], newPolicy[6]
		r.AddItem(newBoard, newPolicy, currentPlayer, gameValue, true)

		// horizontal reflection
		// newBoard = make([]int, len(boardState))
		copy(newBoard, savedB)
		copy(newPolicy, savedP)
		newBoard[0], newBoard[6] = newBoard[6], newBoard[0]
		newBoard[1], newBoard[7] = newBoard[7], newBoard[1]
		newBoard[2], newBoard[8] = newBoard[8], newBoard[2]
		newPolicy[0], newPolicy[6] = newPolicy[6], newPolicy[0]
		newPolicy[1], newPolicy[7] = newPolicy[7], newPolicy[1]
		newPolicy[2], newPolicy[8] = newPolicy[8], newPolicy[2]
		r.AddItem(newBoard, newPolicy, currentPlayer, gameValue, true)

		// primary diagonal reflection
		// newBoard = make([]int, len(boardState))
		copy(newBoard, savedB)
		copy(newPolicy, savedP)
		newBoard[1], newBoard[3] = newBoard[3], newBoard[1]
		newBoard[2], newBoard[6] = newBoard[6], newBoard[2]
		newBoard[5], newBoard[7] = newBoard[7], newBoard[5]
		newPolicy[1], newPolicy[3] = newPolicy[3], newPolicy[1]
		newPolicy[2], newPolicy[6] = newPolicy[6], newPolicy[2]
		newPolicy[5], newPolicy[7] = newPolicy[7], newPolicy[5]
		r.AddItem(newBoard, newPolicy, currentPlayer, gameValue, true)

		// secondary diagonal reflection
		// newBoard = make([]int, len(boardState))
		copy(newBoard, savedB)
		copy(newPolicy, savedP)
		newBoard[0], newBoard[8] = newBoard[8], newBoard[0]
		newBoard[1], newBoard[5] = newBoard[5], newBoard[1]
		newBoard[3], newBoard[7] = newBoard[7], newBoard[3]
		newPolicy[0], newPolicy[8] = newPolicy[8], newPolicy[0]
		newPolicy[1], newPolicy[5] = newPolicy[5], newPolicy[1]
		newPolicy[3], newPolicy[7] = newPolicy[7], newPolicy[3]
		r.AddItem(newBoard, newPolicy, currentPlayer, gameValue, true)

		// rotation 90 degrees clockwise
		// have not included rotations for now.

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

		r.AddItem(fakeB, fakeP, 1, rand.Float64() * 2.0 - 1.0, false)

	}


}