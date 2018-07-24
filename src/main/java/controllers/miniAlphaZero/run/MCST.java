package controllers.miniAlphaZero.run;

import controllers.Heuristics.StateHeuristic;
import controllers.miniAlphaZero.pretraining.*;
import controllers.miniAlphaZero.simulator.Simulator;
import core.game.StateObservation;
import ontology.Types;
import weka.classifiers.Classifier;
import weka.core.Instance;
import weka.core.Instances;

import java.io.FileInputStream;
import java.io.IOException;
import java.io.ObjectInputStream;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.Map;

/**
 * Created with IntelliJ IDEA.
 * User: Swimiltylers
 * Date: 2018/6/25
 * Time: 21:54
 */
public class MCST implements Simulator{
    private Classifier policy_net;
    private Classifier value_net;
    private Instances pHeader = InstancesStore.pHeader;
    private Instances vHeader = InstancesStore.vHeader;
    private int SIMULATION_DEPTH = 20;
    protected double m_gamma = 0.99;
    private final HashMap<Integer, Types.ACTIONS> action_mapping;
    private final HashMap<Types.ACTIONS, Integer> inverted_action_mapping;
    private final int[] counter;

    public MCST(HashMap<Integer, Types.ACTIONS> action_mapping, int SIMULATION_DEPTH, double m_gamma,
                Classifier policy_net, Classifier value_net){
        this.action_mapping = action_mapping;
        this.SIMULATION_DEPTH = SIMULATION_DEPTH;
        this.m_gamma = m_gamma;

        this.policy_net = policy_net;
        this.value_net = value_net;

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

        Instance sequence[] = new Instance[SIMULATION_DEPTH];
        stateObs = stateObs.copy();
        int depth = 0;
        double factor = 1;
        int[] last_action = null;
        for (; depth < SIMULATION_DEPTH; depth++) {
            try {

                Instance pins = new Instance(1, pDataExtractor.featureExtract(stateObs, -1));
                pins.setDataset(pHeader);
                int action_num = (int)policy_net.classifyInstance(pins);


                Instance vins = new Instance(1, vDataExtractor.featureExtract(stateObs));
                vins.setDataset(vHeader);
                double score_before = value_net.classifyInstance(vins);

                // double score_before = heuristic.evaluateState(stateObs);

                // simulate
                Types.ACTIONS action = action_mapping.get(action_num);
                StateObservation simObs = stateObs.copy();
                simObs.advance(action);

                if (simObs.getAvatarPosition().equals(stateObs.getAvatarPosition())){
                    ArrayList<Types.ACTIONS> availableActions = stateObs.getAvailableActions();
                    int[] toNum_availableActions = new int[availableActions.size()];
                    for (int i=0; i<toNum_availableActions.length; ++i) {
                        simObs = stateObs.copy();
                        simObs.advance(availableActions.get(i));
                        if (!simObs.getAvatarPosition().equals(stateObs.getAvatarPosition()))
                            toNum_availableActions[i] = inverted_action_mapping.get(availableActions.get(i));
                    }
                    // simulate
                    action_num = policy.getAction(RLDataExtractor.featureExtract(stateObs), toNum_availableActions, last_action, m_gamma*m_gamma);
                    action = action_mapping.get(action_num);
                }

                stateObs.advance(action);

                vins = new Instance(1, vDataExtractor.featureExtract(stateObs));
                vins.setDataset(vHeader);
                double score_after = value_net.classifyInstance(vins);

                double delta_score = factor * (score_after - score_before);
                factor = factor * m_gamma;

                double[] features = RLDataExtractor.featureExtract(stateObs);
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
