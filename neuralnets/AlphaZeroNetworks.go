package neuralnets

import (
	"fmt"
	"math"
)


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


func (m DenseNetwork) AZStochasticGradientDescent(
		learningRate, weightDecay float64,
		buffer *ReplayBuffer,
		batchSize int,
		numBatches int,
		cyclesPerBatch int,
	) {

	// suggest starting with cyclesPerBatch = 1, as per 
	//  original AZ algorithm. Only increase if it proves more effective.

	for c := 0; c < numBatches; c++ {
		
		batch_inputs, batch_outputs := buffer.selectBatch(batchSize)
		
		for i := 0; i < cyclesPerBatch; i++ {
			
			loss := m.generateAZLoss(batch_inputs, batch_outputs)
			loss.backwardsPropagate()
			m.gradientDescent(learningRate, weightDecay)

			if descentNo := c * cyclesPerBatch + i + 1; descentNo % 100 == 0 {
				fmt.Printf("After descent %d, loss is %f\n", descentNo, loss.data)
			}

			m.ClearGrads()

		}
	}

}

func PrintAZModelOutputs(modelOutputs []*Value, boardShape [2]int) {

	fmt.Printf("Game Value is %f\n", modelOutputs[len(modelOutputs)-1].data)

	for row := 0; row < boardShape[0]; row ++ {
		fmt.Printf("move choice probabilities are: ")
		for col := 0; col < boardShape[1]; col ++ {
			fmt.Printf(" %f ", modelOutputs[row * boardShape[1] + col].data)
		}
		fmt.Printf("\n")

	}

}