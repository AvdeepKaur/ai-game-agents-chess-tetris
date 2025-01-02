package src.pas.chess.heuristics;


// SYSTEM IMPORTS
import edu.bu.chess.search.DFSTreeNode;
import edu.cwru.sepia.util.Direction;
import edu.bu.chess.game.piece.Piece;
import edu.bu.chess.game.piece.PieceType;
import edu.bu.chess.game.Board;
import edu.bu.chess.game.player.Player;
import edu.bu.chess.utils.Coordinate;

import java.util.Set;

// JAVA PROJECT IMPORTS
import src.pas.chess.heuristics.DefaultHeuristics;


public class CustomHeuristics extends Object
{

	/**
	 * TODO: implement me! The heuristics that I wrote are useful, but not very good for a good chessbot.
	 * Please use this class to add your heuristics here! I recommend taking a look at the ones I provided for you
	 * in DefaultHeuristics.java (which is in the same directory as this file)
	 */
	public static double getMaxPlayerHeuristicValue(DFSTreeNode node) {
        Board board = node.getGame().getBoard();
        Player maxPlayer = node.getMaxPlayer();

        // Calculate different components of the heuristic
        double kingSafetyValue = calculateKingSafetyValue(node, board, maxPlayer);
        double threatPenalty = calculateThreatPenalty(board, node);
        double pieceCountValue = calculateAlivePieceValue(board, maxPlayer);

        // weights
        double kingSafetyWeight = 1.8;
        double threatPenaltyWeight = -2.0;
        double pieceCountWeight = 1.5;

        // combine the components
        double heuristicValue = (kingSafetyWeight * kingSafetyValue) +
                                (threatPenaltyWeight * threatPenalty) +
                                (pieceCountWeight * pieceCountValue);

        // use the max and min to limit the value
        heuristicValue = Math.max(heuristicValue, -Double.MAX_VALUE);
        heuristicValue = Math.min(heuristicValue, Double.MAX_VALUE);

        return heuristicValue;
    }

    private static double calculateKingSafetyValue(DFSTreeNode node, Board board, Player maxPlayer) {
        Piece king = board.getPieces(maxPlayer, PieceType.KING).iterator().next(); // MAX player's king
        Coordinate kingPosition = node.getGame().getCurrentPosition(king);

        double safetyValue = 0;

        // Check neighboring squares 
        for (Direction direction : Direction.values()) {
    		// get the neighbor position
    		Coordinate neighborPosition = kingPosition.getNeighbor(direction);
    
   			// is in bounds and has a piece on it?
    		if (board.isInbounds(neighborPosition) && board.isPositionOccupied(neighborPosition)) {
        		Piece piece = board.getPieceAtPosition(neighborPosition);
  
        		if (piece != null) {
            		if (king.isEnemyPiece(piece)) {
            			// if it's enemy it's dangerous boys
                		safetyValue -= Piece.getPointValue(piece.getType()); 
            		} else {
                		safetyValue += Piece.getPointValue(piece.getType()); 
            		}
        		}
    		}
		}

        return Math.max(safetyValue, 0);
    }
    
    private static double calculateThreatPenalty(Board board, DFSTreeNode node) {
        Player minPlayer = node.getGame().getOtherPlayer(node.getMaxPlayer());
        int threatCount = 0;

        // count how many captures are possible by enemy
        Set<Piece> enemyPieces = board.getPieces(minPlayer);
        for (Piece piece : enemyPieces) {
            threatCount += piece.getAllCaptureMoves(node.getGame()).size();
        }

        return (double) threatCount; 
    }
    
    private static double calculateAlivePieceValue(Board board, Player maxPlayer) {
        double pieceValue = 0;

        //point values for our alive pieces
        for (PieceType type : PieceType.values()) {
            int count = board.getNumberOfAlivePieces(maxPlayer, type);
            pieceValue += count * Piece.getPointValue(type);
        }

        return pieceValue;
    }

}
