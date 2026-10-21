package neuralnets

import (
	"fmt"
)


func NewMLP(spec DenseNetworkSpecification) *DenseNetwork {

	ls := []*DenseLayer{}

	for i := 0; i < len(spec.LayerSizes)-1; i++ {
		ls = append(ls, 
			newDenseLayer(spec.LayerSizes[i], spec.LayerSizes[i+1], fmt.Sprintf("%d",i+1), spec.LayerActivations[i+1]),
		)
	}

	return &DenseNetwork{
		layers: ls,
	}

}

func (m *DenseNetwork) generateTSELoss(xs [][]*Value, ys []*Value) *Value {
	// requires an MLP with a one final output node
	// loss function is Total Square Error

	var out *Value

	for i, xset := range xs {
		yAct := m.Call(xset)[0]
		k := ys[i].sub(yAct, "diff")
		j := k.pow(2, "powd")
		if out == nil {
			j.label = "new-total-loss-1"
			out = j
		} else {
			out = out.add(j, fmt.Sprintf("new-total-loss-%d", i+1))
		}
	}
	out.label = "Final-total-loss"

	return out

}

func (m DenseNetwork) BinaryMLPStochasticGradientDescent(
		learningRate, weightDecay float64,
		numSteps int,
		xs [][]*Value,
		ys []*Value,
	) {

	fmt.Println()

	for i := 0; i <= numSteps; i++ {

		loss := m.generateTSELoss(xs, ys)
		loss.backwardsPropagate()
		m.gradientDescent(learningRate, weightDecay)

		if i % 100 == 0 {
			fmt.Printf("After descent %d, loss is %f\n", i, loss.data)
		}
		m.ClearGrads()

	}
	fmt.Printf("After finishing, loss is %f\n", m.generateTSELoss(xs, ys).data)

}

func (m DenseNetwork) PrintBinaryMLPLoss(xs [][]*Value, ys []*Value) {

	for i, xset := range xs {
		fmt.Printf("%d | Modelled: %s | Expected: %s\n", i+1, m.Call(xset)[0].printf(), ys[i].printf())
	}
	loss := m.generateTSELoss(xs, ys)
	fmt.Printf("Total Loss: %s \n",loss.printf())

}
