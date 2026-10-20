package main

import "fmt"

func main() {

	SimpleBinaryClassifierMLP()

	g := newGame()
	g.printBoard()

	g.playMove(0)
	g.playMove(1)
	g.playMove(1)
	g.playMove(1)
	g.playMove(1)
	g.playMove(1)
	g.playMove(1)
	g.playMove(1)
	g.playMove(1)
	g.playMove(1)
	g.playMove(1)
	g.printBoard()

	g.playMove(2)
	g.playMove(3)
	g.playMove(2)
	g.playMove(1)
	g.playMove(6)
	g.playMove(6)
	g.playMove(5)
	g.playMove(6)
	g.playMove(5)
	g.playMove(6)
	g.playMove(5)
	g.playMove(6)
	g.playMove(6)
	g.playMove(5)

	g.printBoard()

	fmt.Printf("%49b\n", g.currentHead.boardState.playing)
	fmt.Printf("%49b\n", g.currentHead.boardState.allPlayed)

	// initialise neural network
	// 	 - MLP with 84 binary inputs. 42 for player pieces on board; 42 for all pieces on board.
	// 	 - several hidden layers (e.g. 2 layers, each of 128 neurons)
	// 	 - seven policy outputs - floats normalised to sum to one via softmax (valid p distribution)
	//   - one value output - float normalised to [-1, 1] via tanh (expected value of match given the current state + player)
	//   - custom alphazero loss function (takes an actual outcome and actual final option probabilities experienced at that state)
	// expose two methods to access the net:
	//   - one to feed-forward an input yielding the policy and value outputs
	//   - one to feed-forward an input and expected values, to allow optimisation of a loss function

	// initialise storage for game state + outcome histories (replay buffer, capacity of c. 30,000 positions, FIFO [discard oldest examples when full])
	
	// for each in a set number of training cycles.
	// 	 create training data by self play loop. 
	//   A. for each in a set number of matches (c. 25-50 minimum, to provide buffer of c. 1,000 for first training round)
	// 		1. initialise tree for use in MCTS
	//      2. initialise the game board (used binary bits; inversion switches player perspective)
	//      3. initialise empty list of game states
	//      4. for a loop until a match is over
	// 			- get list of valid moves from game board
	// 			- mask any invalid moves in probability distribution and prevent MCTS exploration to these invalid moves.
	// 			- use neural net policy head to assess a choice probability for each valid move. Add dirichlet noise to this.
	// 			
	// 			- run MCTS exploration, for a set number of steps (c. 100)
	// 				. use PUCT values to deterministically find the next leaf node (always choose highest PUCT)
	// 				. check if terminal state. if terminal, game is over so propagate result back up. ensure sign alternates.
	// 				. if not terminal, use neural net to evaluate the state, yielding move choice probabilities and expected game value at this leaf node. do not add dirichlet noise again. store at node
	// 				. if not terminal, propagate neural net assessed value back up the tree. ensure sign of value is properly alternated across each layer (to reflect different players)

	// 			- combine record of visits during MCTS to valid choices into final move choice distribution
	// 			- add board state and final move selection probabilities to game state list
	// 			- make move by selecting from the final move choice distribution (use temperature model so more likely to choice outside top option earlier in match).
	// 			- invert board state so next player can play
	// 			- rebase MCTS tree to next 
	// 			- check if game is over, if so, break and store outcome
	//      5. add / update each board state, final move choice probabilities (MCTS), and eventual match outcome data to storage (replay buffer), accounting for differing players on each move
	// 
	//   train the neural net
	//   B. Sample, uniformly, from the replay buffer
	// 		1. for each batch (size c. 128) in a number of batches of (game state, actual outcome, and MCTS distribution) histories.
	// 		2. feed batch into loss method of neural net, which does
	// 			- for each cycle in count of training cycles (c. 100, learning rate c. 0.01)
	// 				. evaluates inputs to get a predicted outcome and policy distribution
	// 				. combines (sums) losses across all inputs in batch, using MCTS distribution and actual outcomes to calculate losses
	// 				. backpropagates gradients
	// 				. use stochastic gradient descent to reduce loss with learning rate multiplier to optimise.
	// 				. wipe gradients


	// Immediate tasks
	//  1. Add ReLU, Log, Exp, Div, to Value module.
	//  2. Fix Neural Net module, to permit different shape net outputs, different normalisation (ReLU).
	//  3. Implement Softmax and policy cross-entropy in Neural Net module.
	//  4. Implement new Neural Net type which uses Connect Four structure, including dual policy and value output heads.
	//  5. Implement new loss method for new neural net type, which accounts for the new head and loss function. Add this as a bespoke value operation, given the gradient is trivial (p_i - pi_i)
	//  6. Implement new training method for new neural net type.
	//  7. Consider changing weight initialisation for new nets.
	//  8. Allow dumping of neural net weights to file and import of weights from file.

}
