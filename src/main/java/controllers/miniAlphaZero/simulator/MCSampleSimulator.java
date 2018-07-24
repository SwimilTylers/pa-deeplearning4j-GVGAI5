package controllers.miniAlphaZero.simulator;

import controllers.Heuristics.StateHeuristic;
import controllers.miniAlphaZero.pretraining.QPolicy;
import controllers.miniAlphaZero.pretraining.RLDataExtractor;
import core.game.StateObservation;
import ontology.Types;
import weka.core.Instance;
import weka.core.Instances;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.Map;

/**
 * Created with IntelliJ IDEA.
 * User: Swimiltylers
 * Date: 2018/6/7
 * Time: 20:26
 */
public class MCSampleSimulator implements Simulator {
    private int SIMULATION_DEPTH = 20;
    protected double m_gamma = 0.99;
    private final HashMap<Integer, Types.ACTIONS> action_mapping;
    private final HashMap<Types.ACTIONS, Integer> inverted_action_mapping;
    private final int[] counter;

    public MCSampleSimulator(HashMap<Integer, Types.ACTIONS> action_mapping, int SIMULATION_DEPTH, double m_gamma){
        this.action_mapping = action_mapping;
        this.SIMULATION_DEPTH = SIMULATION_DEPTH;
        this.m_gamma = m_gamma;
        inverted_action_mapping = new HashMap<>();
        action_mapping.forEach((Integer key, Types.ACTIONS value)->inverted_action_mapping.put(value, key));

        counter = new int[action_mapping.size()];
        action_mapping.forEach((Integer key, Types.ACTIONS value)->{
            int counter_num = -1;
            switch (value){
                case ACTION_UP: counter_num = inverted_action_mapping.get(Types.ACTIONS.ACTION_DOWN);   break;
                case ACTION_DOWN: counter_num = inverted_action_mapping.get(Types.ACTIONS.ACTION_UP);   break;
                case ACTION_LEFT: counter_num = inverted_action_mapping.get(Types.ACTIONS.ACTION_RIGHT);    break;
                case ACTION_RIGHT: counter_num = inverted_action_mapping.get(Types.ACTIONS.ACTION_LEFT);    break;
            }
            counter[key] = counter_num;
        });
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
        int[] last_action = null;
        for (; depth < SIMULATION_DEPTH; depth++) {
            try {

                ArrayList<Types.ACTIONS> availableActions = stateObs.getAvailableActions();
                int[] toNum_availableActions = new int[availableActions.size()];
                StateObservation simObs;
                for (int i=0; i<toNum_availableActions.length; ++i) {
                    simObs = stateObs.copy();
                    simObs.advance(availableActions.get(i));
                    if (!simObs.getAvatarPosition().equals(stateObs.getAvatarPosition()))
                        toNum_availableActions[i] = inverted_action_mapping.get(availableActions.get(i));
                }

                double[] features = RLDataExtractor.featureExtract(stateObs);

                int action_num = policy.getAction(features, toNum_availableActions, last_action, m_gamma*m_gamma);

                double score_before = heuristic.evaluateState(stateObs);

                // simulate
                Types.ACTIONS action = action_mapping.get(action_num);
                stateObs.advance(action);

                double score_after = heuristic.evaluateState(stateObs);

                double delta_score = factor * (score_after - score_before);
                factor = factor * m_gamma;
                // collect data
                sequence[depth] = RLDataExtractor.makeInstance(features, action_num, delta_score);
                last_action = new int[]{counter[inverted_action_mapping.get(stateObs.getAvatarLastAction())]};

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
                accQ = factor*policy.getMaxQ(RLDataExtractor.featureExtract(stateObs));
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
