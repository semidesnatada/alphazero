package neuralnets

import (
	"fmt"
	"math"
)


func newAZNetwork(spec DenseNetworkSpecification) *DenseNetwork {
	if len(spec.LayerSizes) != len(spec.LayerActivations) {
		panic("AZ network specification must have slices of same length")
	}

	ls := []*DenseLayer{}

	numLayers := len(spec.LayerSizes)

	for i := 0; i < numLayers - 1; i++ {
		ls = append(ls, 
			newDenseLayer(
				spec.LayerSizes[i],
				spec.LayerSizes[i+1],
				fmt.Sprintf("%d",i+1),
				spec.LayerActivations[i+1],
			),
		)
	}

	return &DenseNetwork{
		layers: ls,
	}

}


func (m *DenseNetwork) CallAZFinal(inputs []*Value) []*Value {
	// returns the AZ model outputs but with the policy values
	//  softmaxed, for use in training and later model use.
	intermediate := m.CallTraining(inputs)

	numOutputs := len(intermediate)
	maxOutput := intermediate[0].data
    for i := 1; i < numOutputs; i++ {
        if intermediate[i].data > maxOutput {
            maxOutput = intermediate[i].data
        }
    }

	final := make([]*Value, numOutputs)
	final[numOutputs - 1] = intermediate[numOutputs - 1]

	expSum := 0.0
	for i := 0; i < numOutputs - 1; i++ {
		expSum += math.Exp(intermediate[i].data - maxOutput)
	}
	expSumVal := NewValue(expSum, "softmax-total-sum")
	for i := 0; i < numOutputs - 1; i++ {
		final[i] = intermediate[i].sub(
		NewValue(
			maxOutput,
			"softmax-sum",
		),
		"stability-difference",
		).exp(
			fmt.Sprintf("exped-output-%d",i),
		).divide(
			expSumVal,
			fmt.Sprintf("final-output-%d",i),
		)
	}
	return final

}


func (m *DenseNetwork) generateAZLoss(batch_inputs [][]*Value, batch_outputs [][]*Value) *Value {
	// requires an AZ Network with a a set of output policy nodes and one value node
	// loss function is:
	//  - total square error of value, 
	//  - cross-entropy of policy, and
	// note that regularisation is handled by the parameter weightDecay in the Gradient Descent method

	if len(batch_inputs) != len(batch_outputs) {
		panic("inputs to AZ loss function must be of equal length")
	}

	var lossInc *Value

	// this is code for the MLP
	for i, xset := range batch_inputs {

		outputLength := len(batch_outputs[i])
		model_outputs := m.CallTraining(xset)
		if outputLength != len(model_outputs) {
			panic("error in differing lengths between model and expected output")
		}

		// convert nth output to TSE with ys
		tseComponent := batch_outputs[i][outputLength - 1].sub(model_outputs[outputLength - 1], "intermediate-TSE").pow(2.0, "TSE")
		if lossInc == nil {
			lossInc = tseComponent
			lossInc.label = fmt.Sprintf("new-total-loss-withTSE-%d", i+1)
		} else {
			lossInc = lossInc.add(tseComponent, fmt.Sprintf("new-total-loss-withTSE-%d", i+1))
		}

		// outputs of calltraining on an AZ model are not softmaxed
		// convert first n-1 outputs to softmax
		// store the softmaxed values; 
		softMaxCrossEnt := smce(model_outputs[:outputLength - 1], batch_outputs[i][:outputLength - 1], "smce-total")
		lossInc = lossInc.add(softMaxCrossEnt, fmt.Sprintf("final-TSE-for-match-%d", i+1))

	}
	lossInc.label = "final-total-loss-for-all-matches"

	return lossInc

}

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

	if batchSize < r.Content {
		panic("not enough training data in buffer for batch size")
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

func (m DenseNetwork) AZStochasticGradientDescent(
		learningRate, weightDecay float64,
		numSteps int,
		buffer ReplayBuffer,
		batchSize int,
	) {

	// Randomly select a batch of histories from the buffer
	batch_inputs, batch_outputs := buffer.selectBatch(batchSize)

	fmt.Println()

	for i := 0; i < numSteps; i++ {

		loss := m.generateAZLoss(batch_inputs, batch_outputs)
		loss.backwardsPropagate()
		m.gradientDescent(learningRate, weightDecay)

		if i % 100 == 0 {
			fmt.Printf("After descent %d, loss is %f\n", i, loss.data)
		}
		m.ClearGrads()

	}
	
	fmt.Printf("After finishing, loss is %f\n", m.generateAZLoss(batch_inputs, batch_outputs).data)

}