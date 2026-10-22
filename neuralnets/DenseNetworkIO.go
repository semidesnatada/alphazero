package neuralnets

import (
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
)


type DenseNetworkFloat struct {
	Layers []DenseLayerFloat	`json:"layers"`
}


type DenseLayerFloat struct {
	Neurons []NeuronFloat	`json:"neurons"`
	Label string			`json:"label"`
	ActivationType string	`json:"activation-type"`
}

type NeuronFloat struct {
	Weights []float64 `json:"weights"`
	Bias float64	  `json:"bias"`
	Label string	  `json:"label"`
}

func (m DenseNetwork) convertNetworkToFloat() DenseNetworkFloat {

	outLayers := make([]DenseLayerFloat, len(m.layers))

	for i, layer := range m.layers {

		floatNeurons := make([]NeuronFloat, len(layer.neurons))
		for j, neu := range layer.neurons {

			floatWeights := make([]float64, len(neu.weights))

			for k, wei := range neu.weights {
				floatWeights[k] = wei.data
			}

			floatNeurons[j] = NeuronFloat{
				Weights: floatWeights,
				Bias: neu.bias.data,
				Label: neu.label,
			}
		}

		outLayers[i] = DenseLayerFloat{
			Neurons: floatNeurons,
			Label: layer.label,
			ActivationType: layer.activationType,
		}

	}

	return DenseNetworkFloat{
		Layers: outLayers,
	}
}

func (mf DenseNetworkFloat) convertFloatToNetwork() DenseNetwork {

	outLayers := make([]*DenseLayer, len(mf.Layers))

	for i, layer := range mf.Layers {

		valueNeurons := make([]*Neuron, len(layer.Neurons))
		for j, neu := range layer.Neurons {

			valueWeights := make([]*Value, len(neu.Weights))

			for k, wei := range neu.Weights {
				valueWeights[k] = NewValue(wei, fmt.Sprintf("w%d-%d-%d", i, j, k))
			}

			valueNeurons[j] = &Neuron{
				weights: valueWeights,
				bias: NewValue(neu.Bias, fmt.Sprintf("b%d-%d", i, j)),
				label: neu.Label,
			}
		}

		outLayers[i] = &DenseLayer{
			neurons: valueNeurons,
			label: layer.Label,
			activationType: layer.ActivationType,
		}

	}

	return DenseNetwork{
		layers: outLayers,
	}
}


func (m DenseNetwork) DumpNetworkToFile(filename string) error {
	// converts the network to JSON and saves to file.

	path := filepath.Join("model_dumps/", filename)

	mFloat := m.convertNetworkToFloat()

	f, err := os.Create(path)
	if err != nil {
		return fmt.Errorf("Creating model dump error: %w", err)
	}
	defer f.Close()

	enc := json.NewEncoder(f)
	if err := enc.Encode(mFloat); err != nil {
		return fmt.Errorf("Encode model error: %w", err)
	}

	return nil

}

func CreateNetworkFromFile(filename string) (*DenseNetwork, error) {
	// converts the network to JSON and saves to file.

	path := filepath.Join("model_dumps/", filename)

	f, err := os.Open(path)
	if err != nil {
		return nil, fmt.Errorf("Opening model dump error: %w", err)
	}
	defer f.Close()

	var mf DenseNetworkFloat

	dec := json.NewDecoder(f)
	if err := dec.Decode(&mf); err != nil {
		return nil, fmt.Errorf("Decoding file error: %w", err)
	}

	out := mf.convertFloatToNetwork()

	return &out, nil

}