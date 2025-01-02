package src.pas.chess.moveorder;


// SYSTEM IMPORTS
import edu.bu.chess.search.DFSTreeNode;

//import edu.bu.chess.move;
import java.util.List;
import java.util.Collections;
import java.util.Comparator;
import src.pas.chess.heuristics.CustomHeuristics;


// JAVA PROJECT IMPORTS
import src.pas.chess.moveorder.DefaultMoveOrderer;

public class CustomMoveOrderer
    extends Object
{

	/**
	 * TODO: implement me!
	 * This method should perform move ordering. Remember, move ordering is how alpha-beta pruning gets part of its power from.
	 * You want to see nodes which are beneficial FIRST so you can prune as much as possible during the search (i.e. be faster)
	 * @param nodes. The nodes to order (these are children of a DFSTreeNode) that we are about to consider in the search.
	 * @return The ordered nodes.
	 */
	public static List<DFSTreeNode> order(List<DFSTreeNode> nodes) {
        
        Collections.sort(nodes, new Comparator<DFSTreeNode>() {
            @Override
            public int compare(DFSTreeNode node1, DFSTreeNode node2) {
                double value1 = CustomHeuristics.getMaxPlayerHeuristicValue(node1);
                double value2 = CustomHeuristics.getMaxPlayerHeuristicValue(node2);

                // Sort in descending order
                return Double.compare(value2, value1);
            }
        });

        return nodes;
    }

}
