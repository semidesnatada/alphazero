package neuralnets

import (
	"fmt"
	"math"
)


func newAZNetwork(spec DenseNetworkSpecification) *DenseNetwork {

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

	final := []*Value{}
	final[numOutputs - 1] = intermediate[numOutputs - 1]

	expSum := 0.0
	for i := 0; i < numOutputs - 1; i++ {
		expSum += math.Exp(intermediate[i].data)
	}
	expSumVal := NewValue(expSum, "softmax-total-sum")
	for i := 0; i < numOutputs - 1; i++ {
		final[i] = intermediate[i].exp(
			fmt.Sprintf("exped-output-%d",i)).divide(
				expSumVal, fmt.Sprintf("final-output-%d",i))
	}
	return final

}


func (m *DenseNetwork) generateAZLoss(xs [][]*Value, ys [][]*Value) *Value {
	// requires an AZ Network with a a set of output policy nodes and one value node
	// loss function is:
	//  - total square error of value, 
	//  - cross-entropy of policy, and
	// note that regularisation is handled by the parameter weightDecay in the Gradient Descent method

	var lossInc *Value

	// this is code for the MLP
	for i, xset := range xs {

		outputLength := len(ys[i])
		yModel := m.CallTraining(xset)

		// convert nth output to TSE with ys
		tseComponent := ys[i][outputLength - 1].sub(yModel[outputLength - 1], "intermediate-TSE").pow(2.0, "TSE")
		if lossInc == nil {
			lossInc = tseComponent
			lossInc.label = fmt.Sprintf("new-total-loss-withTSE-%d", i+1)
		} else {
			lossInc = lossInc.add(tseComponent, fmt.Sprintf("new-total-loss-withTSE-%d", i+1))
		}

		// outputs of calltraining on an AZ model are not softmaxed
		// convert first n-1 outputs to softmax
		// store the softmaxed values; 
		softMaxCrossEnt := smce(yModel[:outputLength - 1], ys[i][:outputLength - 1], "smce-total")
		lossInc = lossInc.add(softMaxCrossEnt, fmt.Sprintf("final-TSE-for-match-%d", i+1))

	}
	lossInc.label = "final-total-loss-for-all-matches"

	return lossInc

}