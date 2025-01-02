package src.pas.tetris.agents;


// SYSTEM IMPORTS
import java.util.Iterator;
import java.util.List;
import java.util.Random;
import java.util.Comparator;
import java.util.stream.Collectors;


// JAVA PROJECT IMPORTS
import edu.bu.tetris.agents.QAgent;
import edu.bu.tetris.agents.TrainerAgent.GameCounter;
import edu.bu.tetris.game.Board;
import edu.bu.tetris.game.Game.GameView;
import edu.bu.tetris.game.minos.Mino;
import edu.bu.tetris.linalg.Matrix;
import edu.bu.tetris.nn.Model;
import edu.bu.tetris.nn.LossFunction;
import edu.bu.tetris.nn.Optimizer;
import edu.bu.tetris.nn.models.Sequential;
import edu.bu.tetris.nn.layers.Dense; // fully connected layer
import edu.bu.tetris.nn.layers.ReLU;  // some activations (below too)
import edu.bu.tetris.nn.layers.Tanh;
import edu.bu.tetris.nn.layers.Sigmoid;
import edu.bu.tetris.training.data.Dataset;
import edu.bu.tetris.utils.Pair;
import edu.bu.tetris.linalg.Shape;


public class TetrisQAgent
    extends QAgent
{

    //public static final double EXPLORATION_PROB = 0.05;
    public static final double INITIAL_EXPLORATION_PROB = 1.0;
    public static final double FINAL_EXPLORATION_PROB = 0.05;
    public static final int EXPLORATION_DECAY_GAMES = 50000;

    private Random random;

    public TetrisQAgent(String name)
    {
        super(name);
        this.random = new Random(12345); // optional to have a seed
    }

    public Random getRandom() { return this.random; }

    @Override
public Model initQFunction() {
    final int inputDim = (Board.NUM_ROWS * Board.NUM_COLS) + 4;
    final int hiddenDim1 = 128;
    final int hiddenDim2 = 64;
    final int outDim = 1;

    Sequential qFunction = new Sequential();
    qFunction.add(new Dense(inputDim, hiddenDim1));
    qFunction.add(new ReLU());
    qFunction.add(new Dense(hiddenDim1, hiddenDim2));
    qFunction.add(new ReLU());
    qFunction.add(new Dense(hiddenDim2, outDim));
    return qFunction;
}


    /**
        This function is for you to figure out what your features
        are. This should end up being a single row-vector, and the
        dimensions should be what your qfunction is expecting.
        One thing we can do is get the grayscale image
        where squares in the image are 0.0 if unoccupied, 0.5 if
        there is a "background" square (i.e. that square is occupied
        but it is not the current piece being placed), and 1.0 for
        any squares that the current piece is being considered for.
        
        We can then flatten this image to get a row-vector, but we
        can do more than this! Try to be creative: how can you measure the
        "state" of the game without relying on the pixels? If you were given
        a tetris game midway through play, what properties would you look for?
     */
@Override
public Matrix getQFunctionInput(final GameView game, final Mino potentialAction) {
    try {
        Matrix boardState = game.getGrayscaleImage(potentialAction);
        Matrix flatBoard = boardState.flatten();
        
        // Create a larger feature matrix with more game state information
        Matrix featureMatrix = Matrix.zeros(1, 4);
        featureMatrix.set(0, 0, game.getTotalScore());
        featureMatrix.set(0, 1, game.getScoreThisTurn());
        featureMatrix.set(0, 2, calculateBoardHeight(boardState));
        featureMatrix.set(0, 3, calculateHoles(boardState));
        
        // Combine features with board state
        Matrix combined = Matrix.zeros(1, flatBoard.numel() + 4);
        for (int i = 0; i < flatBoard.numel(); i++) {
            combined.set(0, i, flatBoard.get(0, i));
        }
        for (int i = 0; i < 4; i++) {
            combined.set(0, flatBoard.numel() + i, featureMatrix.get(0, i));
        }
        return combined;
    } catch (Exception e) {
        return Matrix.zeros(1, 1);
    }
}

//helper functions for rewards :3
private double calculateBoardHeight(Matrix boardState) {
    try {
        for (int row = 0; row < boardState.getShape().getNumRows(); row++) {
            for (int col = 0; col < boardState.getShape().getNumCols(); col++) {
                if (boardState.get(row, col) > 0) {
                    return boardState.getShape().getNumRows() - row;
                }
            }
        }
        return 0;
    } catch (Exception e) {
        return 0;
    }
}

private double calculateHoles(Matrix boardState) {
    try {
        double holes = 0;
        for (int col = 0; col < boardState.getShape().getNumCols(); col++) {
            boolean foundBlock = false;
            for (int row = 0; row < boardState.getShape().getNumRows(); row++) {
                if (boardState.get(row, col) > 0) {
                    foundBlock = true;
                } else if (foundBlock && boardState.get(row, col) == 0) {
                    holes++;
                }
            }
        }
        return holes;
    } catch (Exception e) {
        return 0;
    }
}
    /**
     * This method is used to decide if we should follow our current policy
     * (i.e. our q-function), or if we should ignore it and take a random action
     * (i.e. explore).
     *
     * Remember, as the q-function learns, it will start to predict the same "good" actions
     * over and over again. This can prevent us from discovering new, potentially even
     * better states, which we want to do! So, sometimes we should ignore our policy
     * and explore to gain novel experiences.
     *
     * The current implementation chooses to ignore the current policy around 5% of the time.
     * While this strategy is easy to implement, it often doesn't perform well and is
     * really sensitive to the EXPLORATION_PROB. I would recommend devising your own
     * strategy here.
     */
//    @Override
//    public boolean shouldExplore(final GameView game,
//                                 final GameCounter gameCounter)
//    {
//        // System.out.println("phaseIdx=" + gameCounter.getCurrentPhaseIdx() + "\tgameIdx=" + gameCounter.getCurrentGameIdx());
//        //return this.getRandom().nextDouble() <= EXPLORATION_PROB;
//    }
        

    @Override
public boolean shouldExplore(final GameView game, final GameCounter gameCounter) {
    double baseProb = Math.max(FINAL_EXPLORATION_PROB, 
        INITIAL_EXPLORATION_PROB - (INITIAL_EXPLORATION_PROB - FINAL_EXPLORATION_PROB) * 
        (gameCounter.getCurrentPhaseIdx() / 100.0));
    
    // Increase exploration if score is low
    if (game.getTotalScore() < 10) {
        baseProb *= 1.5;
    }
    
    return this.getRandom().nextDouble() < baseProb;
}

    /**
     * This method is a counterpart to the "shouldExplore" method. Whenever we decide
     * that we should ignore our policy, we now have to actually choose an action.
     *
     * You should come up with a way of choosing an action so that the model gets
     * to experience something new. The current implemention just chooses a random
     * option, which in practice doesn't work as well as a more guided strategy.
     * I would recommend devising your own strategy here.
     */
    
    @Override
    public Mino getExplorationMove(final GameView game) {
        List<Mino> possibleMoves = game.getFinalMinoPositions();
        return possibleMoves.get(this.getRandom().nextInt(possibleMoves.size()));
    }

    /**
     * This method is called by the TrainerAgent after we have played enough training games.
     * In between the training section and the evaluation section of a phase, we need to use
     * the exprience we've collected (from the training games) to improve the q-function.
     *
     * You don't really need to change this method unless you want to. All that happens
     * is that we will use the experiences currently stored in the replay buffer to update
     * our model. Updates (i.e. gradient descent updates) will be applied per minibatch
     * (i.e. a subset of the entire dataset) rather than in a vanilla gradient descent manner
     * (i.e. all at once)...this often works better and is an active area of research.
     *
     * Each pass through the data is called an epoch, and we will perform "numUpdates" amount
     * of epochs in between the training and eval sections of each phase.
     */
    @Override
    public void trainQFunction(Dataset dataset,
                               LossFunction lossFunction,
                               Optimizer optimizer,
                               long numUpdates)
    {
        for(int epochIdx = 0; epochIdx < numUpdates; ++epochIdx)
        {
            dataset.shuffle();
            Iterator<Pair<Matrix, Matrix> > batchIterator = dataset.iterator();

            while(batchIterator.hasNext())
            {
                Pair<Matrix, Matrix> batch = batchIterator.next();

                try
                {
                    Matrix YHat = this.getQFunction().forward(batch.getFirst());

                    optimizer.reset();
                    this.getQFunction().backwards(batch.getFirst(),
                                                  lossFunction.backwards(YHat, batch.getSecond()));
                    optimizer.step();
                } catch(Exception e)
                {
                    e.printStackTrace();
                    System.exit(-1);
                }
            }
        }
    }

    /**
     * This method is where you will devise your own reward signal. Remember, the larger
     * the number, the more "pleasurable" it is to the model, and the smaller the number,
     * the more "painful" to the model.
     *
     * This is where you get to tell the model how "good" or "bad" the game is.
     * Since you earn points in this game, the reward should probably be influenced by the
     * points, however this is not all. In fact, just using the points earned this turn
     * is a **terrible** reward function, because earning points is hard!!
     *
     * I would recommend you to consider other ways of measuring "good"ness and "bad"ness
     * of the game. For instance, the higher the stack of minos gets....generally the worse
     * (unless you have a long hole waiting for an I-block). When you design a reward
     * signal that is less sparse, you should see your model optimize this reward over time.
     */
@Override
public double getReward(final GameView game) {
    double score = game.getScoreThisTurn();
    double reward = 0;
    
    // More aggressive progressive reward scaling
    if (score >= 6) {      // Perfect clear/T-spin double
        reward = score * 60;
    } else if (score >= 4) { // Tetris/T-spin single
        reward = score * 40;
    } else if (score >= 2) { // Triple/double line clear
        reward = score * 30;
    } else if (score >= 1) { // Single line clear
        reward = score * 20;
    }
    
    try {
        Matrix boardState = game.getGrayscaleImage(null);
        double height = calculateBoardHeight(boardState);
        
        // Only penalize when truly dangerous
        if (height > Board.NUM_ROWS * 0.8) {
            reward -= height * 2;
        }
        
        // Reduced game over penalty
        if (game.didAgentLose() && height >= Board.NUM_ROWS) {
            reward -= 100;
        }
    } catch (Exception e) {
        return reward;
    }
    
    return reward;
}


}
