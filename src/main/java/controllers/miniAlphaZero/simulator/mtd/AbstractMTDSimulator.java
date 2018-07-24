package controllers.miniAlphaZero.simulator.mtd;

import controllers.Heuristics.StateHeuristic;
import controllers.miniAlphaZero.pretraining.QPolicy;
import controllers.miniAlphaZero.pretraining.RLDataExtractor;
import controllers.miniAlphaZero.simulator.Simulator;
import core.game.StateObservation;
import ontology.Types;

import java.util.*;

public abstract class AbstractMTDSimulator implements Simulator {

    public void setClassifierPrediction(boolean classifierPrediction) {
        isClassifierPrediction = classifierPrediction;
    }

	// These are the different types of values which could be stored in our
	// transposition table
	public enum EntryType {
		EXACT_VALUE, LOWERBOUND, UPPERBOUND;
	}

	// Thrown if we are out of time
	private class OutOfTimeException extends Exception {
	}

	// Helper class for storing several items in the transposition table
	private class SearchNode {
		EntryType type;
		int value;
		int depth;

	}
	
	// Helper class for storing several items in the transposition table
	private class SearchStatistics {
		int searchDepth;
		int timeSpent;
		int nodesEvaluated;

	}

	// These are easier to see than maxint and minint
	public static final int LOSE = -100000;
	public static final int WIN = 100000;

	private static final boolean DEBUG = false;

	// Time we have to compute a move in milliseconds
	private int searchTime;

	// Time we have left to search
	private long startTimeMillis;

	// Are we maximizing the heuristic?
	private boolean maximizer;

	// The maximum search depth to use
	private int SEARCH_DEPTH;

	// A transposition table for caching repeated states. Critical since
	// iterative deepening hits states over and over
	private Map<StateObservation, SearchNode> transpositionTable;

	// Counter to see how many nodes we touch
	private int checkedNodes;
	
	private List<SearchStatistics> statsList;

	private int cacheHits;
	private int leafNodes;

	protected StateHeuristic heuristic;
	protected double m_gamma;
	protected Map<Types.ACTIONS, Integer> inverted_action_mapping = new HashMap<>();
	protected Map<Integer, Types.ACTIONS> action_mapping = new HashMap<>();
	protected QPolicy policy;
	protected boolean isClassifierPrediction = false;
    protected int SIMULATION_DEPTH;

	/**
	 * Creates a new MTD(f)-based game-player
	 * 
	 * @param maximizer
	 *            Is this player maximizing the heuristic score?
	 * @param searchTimeMSec
	 *            How much time per move can we get
	 * @param SEARCH_DEPTH
	 *            What is the maximum depth we should ever search to?
	 */
	public AbstractMTDSimulator(HashMap<Integer, Types.ACTIONS> action_mapping, boolean maximizer, int searchTimeMSec, int SIMULATION_DEPTH, double m_gamma, int SEARCH_DEPTH, StateHeuristic heuristic) {
        searchTime = searchTimeMSec;
        this.maximizer = maximizer;
        this.SIMULATION_DEPTH = SIMULATION_DEPTH;
        this.SEARCH_DEPTH = SEARCH_DEPTH;
        statsList = new ArrayList<>();
        this.heuristic = heuristic;
        this.m_gamma = m_gamma;

        for (int i = 0; i < action_mapping.size(); i++) {
            this.inverted_action_mapping.put(action_mapping.get(i), i);
        }
        this.action_mapping = action_mapping;
	}

	/** {@inheritDoc} */
	public Types.ACTIONS decide(StateObservation state) {
		startTimeMillis = System.currentTimeMillis();
		transpositionTable = new HashMap<>(10000);
		return iterative_deepening(state);
	}

	/**
	 * Helper to iteratively recurse down the search tree, getting deeper each
	 * time
	 * 
	 * @param root
	 *            The root of the tree to search from
	 * @return The best action to take from the root
	 */
	private Types.ACTIONS iterative_deepening(StateObservation root) {
		// Create ActionValuePairs so that we can order Actions
		List<ActionValuePair> actions = buildAVPList(root.getAvailableActions());
		checkedNodes = 0; cacheHits=0; leafNodes = 0;
		
		int d;
		for (d = 1; d < SEARCH_DEPTH; d++) {
			int alpha = LOSE; int beta = WIN; int actionsExplored = 0;
			for (ActionValuePair a : actions) {
				try {

					StateObservation n = root.copy();
					n.advance(a.action);
					
					int value;
					value = MTDF(n, a.value, d);
					actionsExplored++;
					// Store the computed value for move ordering
					a.value = value;
					/*
					if (maximizer) {
						alpha = Math.max(alpha, value);
					} else*/
					//	beta = Math.min(beta, value);
				} catch (OutOfTimeException e) {
					// System.out.println("Out of time");
					// revert to the previously computed values. 
					//HOWEVER, if our best value is found to be catastrophic, keep its value.
					// TODO: this should keep all found catastrophic values, not just the first!
					boolean resetBest = true;
					if (actionsExplored > 1) {// If we have looked at more than one possible action
						ActionValuePair bestAction = actions.get(0);
						// check to see if the best action is worse than another action
						for (int i=0; i < actionsExplored; i++) {
							if (bestAction.value < actions.get(i).value) {
								// don't reset the first choice
								resetBest = false;
								break;
							}
						}	
					}
					
					if (resetBest) {
						for (ActionValuePair ac: actions) {
							ac.value = ac.previousValue;
						}
					} else {
						for (int i=1; i < actionsExplored; i++) {
							actions.get(i).value = actions.get(i).previousValue;
						}
					}
					break;
				} catch (Exception e){
				    e.printStackTrace();
                }
			}
			// Sort the actions for move ordering on the next iteration
			Collections.sort(actions, Collections.reverseOrder());
			
			// And update the previous value field
			for (ActionValuePair a: actions) {
				a.previousValue = a.value;
			}
			
			// System.out.printf("%2.2f",0.001*(System.currentTimeMillis() - startTimeMillis));
			// System.out.println(": " + d + ": "+actions.get(0));
			
			if (times_up()) {
				break;
			}
		}

		SearchStatistics s = new SearchStatistics();
		s.nodesEvaluated = leafNodes;
		s.timeSpent = (int) (System.currentTimeMillis() - startTimeMillis);
		s.searchDepth = d;
		statsList.add(s);
		
		double nodesPerSec = (1000.0*s.nodesEvaluated) / s.timeSpent;
		double EBF = Math.log(s.nodesEvaluated)/Math.log(s.searchDepth);
		double searchEfficiency = (1.0 * leafNodes) / checkedNodes;
			
		// System.out.printf("NPS:%.2f EBF:%.2f eff:%.2f\n", nodesPerSec, EBF, searchEfficiency);
		// System.out.println("Cache hits:"+cacheHits);
			
		// System.out.println("Available actions:"+actions);
		return getRandomBestAction(actions);
	}

	/**
	 * Helper to check if we are out of time for our search
	 * 
	 * @return true if we are out of time, false otherwise
	 */
	private boolean times_up() {
		return (System.currentTimeMillis() - startTimeMillis) > searchTime;
	}

	/**
	 * Main MTD(f) search algorithm which recursively uses alpha-beta with
	 * varying bounds to compute the minimax value of the root
	 * 
	 * @param root
	 *            The root of the search tree
	 * @param firstGuess
	 *            An initial guess as to what the minimax value could be
	 * @param depth
	 *            The maximum depth to search
	 * @return The best guess for the minimax value given the available search
	 *         depth
	 * @throws OutOfTimeException
	 *             If we ran out of search time during the search.
	 */
	private int MTDF(StateObservation root, int firstGuess, int depth)
			throws OutOfTimeException {
		int g = firstGuess;
		int beta;
		int upperbound = WIN;
		int lowerbound = LOSE;

		int flag = maximizer ? 1 : -1;

		while (lowerbound < upperbound) {
			if (g == lowerbound) {
				beta = g + 1;
			} else {
				beta = g;
			}
			// Traditional NegaMax call, just with different bounds
			g = -AlphaBetaWithMemory(root, beta - 1, beta, depth, -flag);
			if (g < beta) {
				upperbound = g;
			} else {
				lowerbound = g;
			}
			// System.out.println("g: "+g+", beta: "+beta+", ["+lowerbound+","+upperbound+"]");
		}

		return g;
	}
	
	/**
	 * Implementation of NegaMax with Alpha-Beta pruning and transposition-table
	 * lookup
	 * 
	 * @param state
	 *            The State we are currently parsing.
	 * @param alpha
	 *            The alpha bound for alpha-beta pruning.
	 * @param beta
	 *            The beta bound for alpha-beta pruning.
	 * @param depth
	 *            The current depth we are at.
	 * @return The best point count we can get on this branch of the state space
	 *         to the specified depth.
	 * @throws OutOfTimeException
	 *             If we ran out of time during the search
	 */
	@SuppressWarnings( { "rawtypes", "unchecked" })
	private int AlphaBetaWithMemory(StateObservation state, int alpha, int beta,
			int depth, int color) throws OutOfTimeException {

		/**
		 * If we are not at a low depth (have at least more recursive calls
		 * below us) then we are called infrequently enough that we can afford
		 * to check if we are out of time
		 */
		if (depth > 4) {
			if (times_up())
				throw new OutOfTimeException();
		}

		// Note that we checked a new node
		checkedNodes++;
		// Specify us
		// Has this state already been computed?
		SearchNode node = transpositionTable.get(state);
		// TODO: shoot myself. This code wasn't working because I had node.depth
		// >= depth rather than >
		if (node != null && node.depth >= depth) {
			cacheHits++;
			switch (node.type) {
			case EXACT_VALUE:
				return node.value;
			/*case UPPERBOUND:
				if (node.value > alpha)
					alpha = node.value;
				break;
			case LOWERBOUND:
				if (node.value < beta)
					beta = node.value;
				break;*/
			}
		}
		// Is this state/our search done?
		if (depth == 0 || state.isGameOver()) {
			double h = heuristic.evaluateState(state);
			if (policy != null && isClassifierPrediction){
                try {
                    double[] Q = policy.getQArray(RLDataExtractor.featureExtract(state));
                    h = Double.NEGATIVE_INFINITY;
                    for (int i = 0; i < Q.length; i++) {
                        if (h < Q[i])
                            h = Q[i];
                    }
                } catch (Exception e) {
                    e.printStackTrace();
                }
            }
			leafNodes++;
			int value = ((int) (color * h));
			return saveAndReturnState(state, alpha, beta, depth, value, color);
		}

		int bestValue = LOSE;

		// Partial move ordering. Check value up to depth D-3 and order by that
		int[] depthsToSearch;
		if (depth > 4) {
			depthsToSearch = new int[2];
			depthsToSearch[0] = depth - 2; // TODO: this should be easily adjustable
			depthsToSearch[1] = depth;
		} else {
			depthsToSearch = new int[1];
			depthsToSearch[0] = depth;
		}

		List<ActionValuePair> actions = buildAVPList(state.getAvailableActions());
		// Do our shorter depth search first to order moves on the longer search
		for (int i = 0; i < depthsToSearch.length; i++) {
			for (ActionValuePair a : actions) {
				int newValue;
				StateObservation childState = state.copy();
				childState.advance(a.action);
				// Traditional NegaMax call
				newValue = -AlphaBetaWithMemory(childState, -beta, -alpha,
						depthsToSearch[i] - 1, -color);
				// Store the value in the ActionValuePair for action ordering
				a.value = newValue;
				if (newValue > bestValue)
					bestValue = newValue;
				if (bestValue > alpha)
					alpha = bestValue;
				if (bestValue >= beta)
					break;
			}
			// Sort the actions to order moves on the deeper search
			Collections.sort(actions, Collections.reverseOrder());

		}
		return saveAndReturnState(state, alpha, beta, depth, bestValue, color);
	}

	/**
	 * Helper to save a given search state to the transposition table
	 * 
	 * @param state
	 *            The node to act as a key to this search node
	 * @param alpha
	 *            The current alpha bound
	 * @param beta
	 *            The current beta bound
	 * @param depth
	 *            The current search depth
	 * @param value
	 *            The computed value of the state
	 * @param color
	 *            The current "color" i.e. whether we are maximizing the
	 *            heuristic or the negative heuristic
	 * @return The value that we stored
	 */
	private int saveAndReturnState(StateObservation state, int alpha, int beta, int depth,
			int value, int color) {
		// Store so we don't have to compute it again.
		SearchNode saveNode = new SearchNode();
		if (value <= alpha) {
			saveNode.type = EntryType.LOWERBOUND;
		} else if (value >= beta) {
			saveNode.type = EntryType.UPPERBOUND;
		} else {
			saveNode.type = EntryType.EXACT_VALUE;
		}

		saveNode.depth = depth;
		saveNode.value = value;
		transpositionTable.put(state, saveNode);

		return value;
	}

	/**
	 * Helper to create a list of ActionValuePairs with value of 0 from a list
	 * of actions
	 * 
	 * @param actions
	 *            The actions to convert
	 * @return A list of actionvaluepairs
	 */
	private List<ActionValuePair> buildAVPList(List<Types.ACTIONS> actions) {
		List<ActionValuePair> res = new ArrayList<>();

		for (Types.ACTIONS a : actions) {
			ActionValuePair p = new ActionValuePair(a, 0);
			res.add(p);
		}

		return res;
	}

	/**
	 * Returns a random action from among the best actions in the given list
	 * NOTE: this assumes the list is already sorted with the best move first,
	 * and that the list is nonempty!
	 * 
	 * @param actions
	 *            The actions to examine
	 * @return The selected action
	 */
	private Types.ACTIONS getRandomBestAction(List<ActionValuePair> actions) {
		List<Types.ACTIONS> bestActions = new LinkedList<>();

		int bestV = actions.get(0).value;
		for (ActionValuePair avp : actions) {
			if (avp.value != bestV)
				break;

			bestActions.add(avp.action);
		}

		//Collections.shuffle(bestActions);
		return bestActions.get(0);
	}
	
	public void printSearchStatistics() {
		double avgNodesPerSec = 0; double avgEBF = 0;
		for (SearchStatistics s: statsList) {
			double nodesPerSec = (1000.0*s.nodesEvaluated) / s.timeSpent;
			avgNodesPerSec += nodesPerSec;
			double EBF = Math.log(s.nodesEvaluated)/Math.log(s.searchDepth);
			avgEBF += EBF;
		}
		
		avgNodesPerSec /= statsList.size();
		avgEBF /= statsList.size();
		
		// System.out.printf("Average Nodes Per Second:%.2f\n", avgNodesPerSec);
		// System.out.printf("Average EBF:%.2f\n", avgEBF);
	}
}
