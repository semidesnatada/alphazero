package neuralnets

import "fmt"


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


func (m *DenseNetwork) generateAZLoss(xs [][]*Value, ys []*Value) *Value {
	// requires an AZ Network with a a set of output policy nodes and one value node
	// loss function is:
	//  - total square error of value, 
	//  - cross-entropy of policy, and
	// note that regularisation is handled by the parameter weightDecay in the Gradient Descent method

	var out *Value

	// this is code for the MLP
	// for i, xset := range xs {
	// 	yAct := m.Call(xset)[0]
	// 	k := ys[i].sub(yAct, "diff")
	// 	j := k.pow(2, "powd")
	// 	if out == nil {
	// 		j.label = "new-total-loss-1"
	// 		out = j
	// 	} else {
	// 		out = out.add(j, fmt.Sprintf("new-total-loss-%d", i+1))
	// 	}
	// }
	// out.label = "Final-total-loss"

	return out

}