package neuralnets

import (
	"fmt"
	"math"
	"slices"
)

type Value struct {
	data float64
	grad float64
	parents []*Value
	operation string
	label string
	back func()
}

func NewValue(inputData float64, inputLabel string) *Value {
	return &Value{
		data: inputData,
		grad: 0,
		label: inputLabel,
		parents: []*Value{},
		back: func() {},
	}
}

func (v *Value) printLong() {
	fmt.Printf("Value %s has data %f, gradient %f, parents %v, operation %s\n",
	v.label, v.data, v.grad, v.parents, v.operation)
}

func (v *Value) print() {
	fmt.Printf("Value[data: %f]\n",
	v.data)
}

func (v *Value) printf() string {
	return fmt.Sprintf("Value[data: %f]",
	v.data)
}

func (o *Value) printAll() {

	sortedTree := []*Value{}
	visited := make(map[*Value]bool)

	var buildSortedTree func(v *Value)
	buildSortedTree = func(v *Value) {
		if !visited[v] {
			visited[v] = true
			for _, child := range v.parents {
				buildSortedTree(child)
			}
			sortedTree = append(sortedTree, v)
		}
	}

	buildSortedTree(o)

	slices.Reverse(sortedTree)

	fmt.Println()
	for i, node := range sortedTree {
		fmt.Printf("%d, %s, value: %f, gradient: %f\n", i, node.label, node.data, node.grad)
	}

}

func (v *Value) relu(newLabel string) *Value {

	outDat := max(v.data, 0)

	newV := NewValue(
		outDat,
		newLabel,
	)

	backProp := func() {
		if outDat > 0.0 {
			v.grad += newV.grad
		}
	}

	newV.back = backProp
	newV.operation = "relu"
	newV.parents = append(newV.parents, v)

	return newV

}

func (v *Value) tanh(newLabel string) *Value {

	outDat := math.Tanh(v.data)

	newV := NewValue(
		outDat,
		newLabel,
	)

	backProp := func() {
		v.grad += (1 - math.Pow(outDat,2)) * newV.grad
	}

	newV.back = backProp
	newV.operation = "tanh"
	newV.parents = append(newV.parents, v)

	return newV

}

func (v *Value) add(o *Value, newLabel string) *Value {

	outDat := v.data + o.data

	newV := NewValue(
		outDat,
		newLabel,
	)

	backProp := func() {
		v.grad += newV.grad
		o.grad += newV.grad
	}

	newV.back = backProp
	newV.operation = "add"
	newV.parents = append(newV.parents, v, o)

	return newV

}

func (v *Value) sub(o *Value, newLabel string) *Value {

	outDat := v.data - o.data

	newV := NewValue(
		outDat,
		newLabel,
	)

	backProp := func() {
		v.grad += newV.grad
		o.grad -= newV.grad
	}

	newV.back = backProp
	newV.operation = "sub"
	newV.parents = append(newV.parents, v, o)

	return newV

}

func (v *Value) pow(n float64, newLabel string) *Value {
	// works only for integer powers, when v.data is negative, as NAN is returned.
	// if v.data is zero, negative powers return NAN.

	outDat := math.Pow(v.data, n)

	newV := NewValue(
		outDat,
		newLabel,
	)

	backProp := func() {
		v.grad += n * math.Pow(v.data, n-1) * newV.grad
	}

	newV.back = backProp
	newV.operation = fmt.Sprintf("pow-%f", n)
	newV.parents = append(newV.parents, v)

	return newV

}

func (v *Value) exp(newLabel string) *Value {
	// returns simple exp(v.data)

	outDat := math.Exp(v.data)

	newV := NewValue(
		outDat,
		newLabel,
	)

	backProp := func() {
		v.grad += outDat * newV.grad
	}

	newV.back = backProp
	newV.operation = "exp"
	newV.parents = append(newV.parents, v)

	return newV

}

func (v *Value) log(newLabel string) *Value {
	// returns simple natural log(v.data)
	// not appropriate for handling negative inputs - these cases are not handled

	outDat := math.Log(v.data)

	newV := NewValue(
		outDat,
		newLabel,
	)

	backProp := func() {
		v.grad += newV.grad / v.data
	}

	newV.back = backProp
	newV.operation = "log"
	newV.parents = append(newV.parents, v)

	return newV

}

func (v *Value) times(o *Value, newLabel string) *Value {

	outDat := v.data * o.data

	newV := NewValue(
		outDat,
		newLabel,
	)

	backProp := func() {
		v.grad += newV.grad * o.data
		o.grad += newV.grad * v.data
	}

	newV.back = backProp
	newV.operation = "times"
	newV.parents = append(newV.parents, v, o)

	return newV

}

func (v *Value) divide(o *Value, newLabel string) *Value {
	// v is dividend, o is divisor

	outDat := v.data / o.data

	newV := NewValue(
		outDat,
		newLabel,
	)

	backProp := func() {
		v.grad += newV.grad / o.data
		o.grad -= newV.grad * v.data / (o.data * o.data)
	}

	newV.back = backProp
	newV.operation = "divide"
	newV.parents = append(newV.parents, v, o)

	return newV

}

func smce(vs, ys []*Value, newLabel string) *Value {
	// calculates the softmaxcrossentropy loss of a series of values
	// note that input vs are not already softmaxed, they are raw model outputs
	// ys are the actuals we are using to calculate cross entropy
	//  loss. these must sum to 1 / be a valid probability distribution.

	numOutputs := len(vs)

	// find maximum of the output nodes
	maxOutput := vs[0].data
    for i := 1; i < numOutputs; i++ {
        if vs[i].data > maxOutput {
            maxOutput = vs[i].data
        }
    }

	ps := make([]float64, numOutputs)

	for i := 0; i < numOutputs; i++ {
		ps[i] = math.Exp(vs[i].data - maxOutput)
	}
	expSum := FloatSliceSum(ps)
	for i := 0; i < numOutputs; i++ {
		ps[i] /= expSum
	}

	// calculate the final cross-entropy loss
	cFactor := math.Log(expSum) + maxOutput
	outDat := 0.0
	for i := 0; i < numOutputs; i++ {
		outDat -= ys[i].data * (vs[i].data - cFactor)
	}

	newV := NewValue(
		outDat,
		newLabel,
	)

	// each output node gradient only depends on its softmax p and the actual, y.
	backProp := func() {
		for i := 0; i < numOutputs; i++ {
			vs[i].grad += newV.grad * (ps[i] - ys[i].data)
		}
	}

	newV.back = backProp
	newV.operation = "smce"
	newV.parents = append(newV.parents, vs...)

	return newV

}

func (o *Value) backwardsPropagate() {

	sortedTree := []*Value{}
	visited := make(map[*Value]bool)

	var buildSortedTree func(v *Value)
	buildSortedTree = func(v *Value) {
		if !visited[v] {
			visited[v] = true
			for _, child := range v.parents {
				buildSortedTree(child)
			}
			sortedTree = append(sortedTree, v)
		}
	}

	buildSortedTree(o)

	slices.Reverse(sortedTree)

	o.grad = 1.0

	for _, node := range sortedTree {
		node.back()
	}

}