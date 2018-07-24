package controllers.miniAlphaZero.simulator.mtd;

import controllers.Heuristics.StateHeuristic;
import controllers.miniAlphaZero.pretraining.QPolicy;
import controllers.miniAlphaZero.pretraining.RLDataExtractor;
import core.game.StateObservation;
import ontology.Types;
import weka.core.Instance;
import weka.core.Instances;

import java.util.ArrayList;
import java.util.HashMap;

/**
 * Created with IntelliJ IDEA.
 * User: Swimiltylers
 * Date: 2018/6/15
 * Time: 12:07
 */
public class lastMTDSimulator extends AbstractMTDSimulator {
    /**
     * Creates a new MTD(f)-based game-player
     *
     * @param action_mapping
     * @param maximizer        Is this player maximizing the heuristic score?
     * @param searchTimeMSec   How much time per move can we get
     * @param SIMULATION_DEPTH
     * @param m_gamma
     * @param SEARCH_DEPTH
     * @param heuristic
     */
    public lastMTDSimulator(HashMap<Integer, Types.ACTIONS> action_mapping, boolean maximizer, int searchTimeMSec, int SIMULATION_DEPTH, double m_gamma, int SEARCH_DEPTH, StateHeuristic heuristic) {
        super(action_mapping, maximizer, searchTimeMSec, SIMULATION_DEPTH, m_gamma, SEARCH_DEPTH, heuristic);
    }

    @Override
    public Instances simulate(StateObservation stateObs, StateHeuristic heuristic, QPolicy policy) {
        Instances data = new Instances(RLDataExtractor.datasetHeader(), 0);
        stateObs = stateObs.copy();
        this.policy = policy;

        Instance sequence[] = new Instance[SIMULATION_DEPTH];
        int depth = 0;
        double factor = 1;
        int[] last_action = null;
        for (; depth < SIMULATION_DEPTH; depth++) {
            try {
                double score_before = heuristic.evaluateState(stateObs);

                ArrayList<Types.ACTIONS> availableActions = stateObs.getAvailableActions();
                int[] toNum_availableActions = new int[availableActions.size()];
                for (int i=0; i<toNum_availableActions.length; ++i) {
                    StateObservation simObs = stateObs.copy();
                    simObs.advance(availableActions.get(i));
                    if (!simObs.getAvatarPosition().equals(stateObs.getAvatarPosition()))
                        toNum_availableActions[i] = inverted_action_mapping.get(availableActions.get(i));
                }
                // simulate
                int action_num = policy.getAction(RLDataExtractor.featureExtract(stateObs), toNum_availableActions, last_action, m_gamma*m_gamma);
                Types.ACTIONS action = action_mapping.get(action_num);
                stateObs.advance(action);

                double score_after = heuristic.evaluateState(stateObs);

                double delta_score = factor * (score_after - score_before);
                factor = factor * m_gamma;
                // collect data
                double[] features = RLDataExtractor.featureExtract(stateObs);
                sequence[depth] = RLDataExtractor.makeInstance(features, action_num, delta_score);

                last_action = new int[]{inverted_action_mapping.get(stateObs.getAvatarLastAction())};

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
                // accQ = factor * policy.getMaxQ(RLDataExtractor.featureExtract(stateObs));
                stateObs.advance(decide(stateObs));
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

        this.policy = null;

        return data;
    }
}
