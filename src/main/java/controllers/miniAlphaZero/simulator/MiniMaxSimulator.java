package controllers.miniAlphaZero.simulator;

import controllers.Heuristics.StateHeuristic;
import controllers.miniAlphaZero.pretraining.QPolicy;
import controllers.miniAlphaZero.pretraining.RLDataExtractor;
import core.game.StateObservation;
import ontology.Types;
import weka.core.Instance;
import weka.core.Instances;

import java.util.*;

/**
 * This class represents an AI Decider that uses a MiniMax algorithm.
 * We use alpha-beta pruning, but besides that we're pretty vanilla.
 * @author Ashoat Tevosyan
 * @author Peter Brook 
 * @since Mon April 28 2011
 * @version CSE 473
 */
public class MiniMaxSimulator implements Simulator {
	
	// Are we maximizing or minimizing?
	private boolean maximize;
	// The depth to which we should analyze the search space
	// HashMap to avoid recalculating States
	private Map<StateObservation, Double> computedStates;
	// Used to generate a graph of the search space for each turn in SVG format
	private int SIMULATION_DEPTH;
	private int SEARCH_DEPTH;
    private final HashMap<Types.ACTIONS, Integer> action_mapping = new HashMap<>();
    private double m_gamma;
	
	/**
	 * Initialize this MiniMaxSimulator.
	 * @param SIMULATION_DEPTH    The depth of each simulation.
	 * @param SEARCH_DEPTH    The depth to which we should analyze the search space at each decision step.
	 */
	public MiniMaxSimulator(HashMap<Integer, Types.ACTIONS> action_mapping, int SIMULATION_DEPTH, double m_gamma, int SEARCH_DEPTH) {
		this.maximize = true;
		this.SIMULATION_DEPTH = SIMULATION_DEPTH;
		this.SEARCH_DEPTH = SEARCH_DEPTH;
		computedStates = new HashMap<>();
		this.m_gamma = m_gamma;
        for (int i = 0; i < action_mapping.size(); i++) {
            this.action_mapping.put(action_mapping.get(i), i);
        }
    }
	
	/**
	 * Decide which state to go into.
	 * We manually MiniMax the first layer so we can figure out which heuristic is from which Action.
	 * Also, we want to be able to choose randomly between equally good options.
	 * "I'm the decider, and I decide what is best." - George W. Bush
	 * @param state The start State for our search.
	 * @return The Action we are deciding to take.
	 */
	@SuppressWarnings({ "unchecked", "rawtypes" })
	public Types.ACTIONS decide(StateObservation state, StateHeuristic heuristic) {
		// Choose randomly between equally good options
		double value = maximize ? Double.NEGATIVE_INFINITY : Double.POSITIVE_INFINITY;
		List<Types.ACTIONS> bestActions = new ArrayList<>();
		// Iterate!
		int flag = maximize ? 1 : -1;
		double alpha = Double.NEGATIVE_INFINITY, beta = Double.POSITIVE_INFINITY;
		for (Types.ACTIONS action : state.getAvailableActions()) {
			try {
				// Algorithm!
				StateObservation newState = state.copy();
				newState.advance(action);
				double newValue = this.miniMaxRecursor(newState, alpha, beta, 1, !this.maximize, heuristic);
				// Better candidates?
				if (flag * newValue > flag * value) {
					value = newValue;
					bestActions.clear();
				}
				// Add it to the list of candidates?
				if (flag * newValue >= flag * value) bestActions.add(action);
                if (maximize){
                    if (value >= beta)
                        break;
                    alpha = alpha > value ? alpha : value;
                }
                else{
                    if (value <= alpha)
                        break;
                    beta = beta < value ? beta : value;
                }

			} catch (Exception e) {
				throw new RuntimeException("Invalid action!");
			}
		}
		// If there are more than one best actions, pick one of the best randomly
		Collections.shuffle(bestActions);
		return bestActions.get(0);
	}
	
	/**
	 * The true implementation of the MiniMax algorithm!
	 * Thoroughly commented for your convenience.
	 * @param state    The State we are currently parsing.
	 * @param alpha    The alpha bound for alpha-beta pruning.
	 * @param beta     The beta bound for alpha-beta pruning.
	 * @param depth    The current depth we are at.
	 * @param maximize Are we maximizing? If not, we are minimizing.
	 * @return The best point count we can get on this branch of the state space to the specified depth.
	 */
	@SuppressWarnings({ "rawtypes", "unchecked" })
	public double miniMaxRecursor(StateObservation state,
								  double alpha, double beta,
								  int depth, boolean maximize, StateHeuristic heuristic) {
		// Has this state already been computed?
		if (computedStates.containsKey(state)) 
                    // Return the stored result
                    return computedStates.get(state);
		// Is this state done?
		if (state.isGameOver())
                    // Store and return
                    return heuristic.evaluateState(state);
		// Have we reached the end of the line?
		if (depth == SEARCH_DEPTH)
                    //Return the heuristic value
                    return heuristic.evaluateState(state);
                
		// If not, recurse further. Identify the best actions to take.
		double value = maximize ? Double.NEGATIVE_INFINITY : Double.POSITIVE_INFINITY;
		int flag = maximize ? 1 : -1;
		List<Types.ACTIONS> test = state.getAvailableActions();
		for (Types.ACTIONS action : test) {
			// Check it. Is it better? If so, keep it.
			try {
				StateObservation childState = state.copy();
				childState.advance(action);
				double newValue = this.miniMaxRecursor(childState, alpha, beta, depth + 1, !maximize, heuristic);
				//Record the best value
                if (flag * newValue > flag * value)
                    value = newValue;
                // update alpha and beta
                if (maximize){
                    if (value >= beta)
                        return value;
                    alpha = alpha > value ? alpha : value;
                }
                else{
                    if (value <= alpha)
                        return value;
                    beta = beta < value ? beta : value;
                }
			} catch (Exception e) {
                                //Should not go here
				throw new RuntimeException("Invalid action!");
			}
		}
		// Store so we don't have to compute it again.
		return value;
	}

	@Override
	public void setClassifierPrediction(boolean classifierPrediction) {

	}

	@Override
	public Instances simulate(StateObservation stateObs, StateHeuristic heuristic, QPolicy policy) {
        Instances data = new Instances(RLDataExtractor.datasetHeader(), 0);
        stateObs = stateObs.copy();

        Instance sequence[] = new Instance[SIMULATION_DEPTH];
        int depth = 0;
        double factor = 1;
        for (; depth < SIMULATION_DEPTH; depth++) {
            try {
                double score_before = heuristic.evaluateState(stateObs);

                // simulate
                Types.ACTIONS action = decide(stateObs, heuristic);
                stateObs.advance(action);

                double score_after = heuristic.evaluateState(stateObs);

                double delta_score = factor * (score_after - score_before);
                factor = factor * m_gamma;
                // collect data
                double[] features = RLDataExtractor.featureExtract(stateObs);
                sequence[depth] = RLDataExtractor.makeInstance(features, action_mapping.get(action), delta_score);

            } catch (Exception exc) {
                exc.printStackTrace();
                break;
            }
            if (stateObs.isGameOver()) {
                depth++;
                break;
            }
        }

        // get the predicted Q from the last state
        double accQ = 0;
        if (!stateObs.isGameOver()) {
            try {
                accQ = factor * policy.getMaxQ(RLDataExtractor.featureExtract(stateObs));
            } catch (Exception exc) {
                exc.printStackTrace();
            }
        }

        // calculate the accumulated Q
        for (depth = depth - 1; depth >= 0; depth--) {
            accQ += sequence[depth].classValue();
            sequence[depth].setClassValue(accQ);
            data.add(sequence[depth]);
        }
        return data;
    }
}