package games

// couldn't get this to work with different board data types and attributes
// type Board interface {
//     PossibleMoves() map[int]bool
//     IsGameOver() bool
//     PrintBoard()
// }

type C4GameState struct {
    CurrentHead *C4GameStateNode
    PrevState   *C4GameStateNode
    GameOver    bool
}

type C4GameStateNode struct {
    PossibleMovesMap map[int]bool
    TimesVisited     int
    Wins             int
    Board            C4Board     
}

type C4Board struct {
    Playing   uint64
    AllPlayed uint64
}

type NACGameState struct {
    CurrentHead *NACGameStateNode
    PrevState   *NACGameStateNode
    GameOver    bool
    GameValue   float64
}

type NACGameStateNode struct {
    PossibleMovesMap map[int]bool
    TimesVisited     int
    Wins             int
    Board            NACBoard    
    CurrentPlayer    int 
}

type NACBoard []int