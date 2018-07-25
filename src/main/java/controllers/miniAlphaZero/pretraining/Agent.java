package controllers.miniAlphaZero.pretraining;

import controllers.Heuristics.StateHeuristic;
import controllers.Heuristics.WinScoreHeuristic;
import controllers.miniAlphaZero.simulator.MCSampleSimulator;
import controllers.miniAlphaZero.simulator.Simulator;
import core.game.StateObservation;
import core.player.AbstractPlayer;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.Random;

import ontology.Types;
import tools.ElapsedCpuTimer;
import tools.Vector2d;
import weka.classifiers.Classifier;
import weka.core.Instance;
import weka.core.Instances;

public class Agent extends AbstractPlayer {

    protected Classifier m_model;
    protected Random m_rnd;
    private static int SIMULATION_DEPTH = 20;
    private final HashMap<Integer, Types.ACTIONS> action_mapping;
    private final HashMap<Types.ACTIONS, Integer> inverted_action_mapping;
    private final int[] counter;
    protected QPolicy m_policy;
    protected int N_ACTIONS;
    protected static Instances m_dataset;
    protected int m_maxPoolSize = 1000;
    protected double m_gamma = 0.9;
    protected Simulator simulator = null;
    protected int[] last_Action = null;
    protected Instances pHeader = InstancesStore.pHeader;
    protected Instances vHeader = InstancesStore.vHeader;
    protected double beta = 0.3;

    protected HashMap<String, Double> location_score = new HashMap<>();

    public Agent(StateObservation stateObs, ElapsedCpuTimer elapsedTimer) {
        m_rnd = new Random();

        // convert numbers to actions
        action_mapping = new HashMap<>();
        inverted_action_mapping = new HashMap<>();
        int i = 0;
        for (Types.ACTIONS action : stateObs.getAvailableActions()) {
            action_mapping.put(i, action);
            inverted_action_mapping.put(action, i);
            i++;
        }

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

        N_ACTIONS = stateObs.getAvailableActions().size();
        m_policy = new QPolicy(N_ACTIONS);
        m_dataset = new Instances(RLDataExtractor.s_datasetHeader);

        simulator = new MCSampleSimulator(action_mapping, SIMULATION_DEPTH, m_gamma);
        // simulator = new MiniMaxSimulator(action_mapping, SIMULATION_DEPTH, m_gamma, 4);
        // simulator = new lastMTDSimulator(action_mapping, true, 20, SIMULATION_DEPTH, m_gamma, 5, new SimpleStateHeuristic(null));
        // simulator = new restMTDSimulator(action_mapping, true, 20, SIMULATION_DEPTH, m_gamma, 5, new SimpleStateHeuristic(null));
    }

    /**
     *
     * Learning based agent.
     *
     * @param stateObs Observation of the current state.
     * @param elapsedTimer Timer when the action returned is due.
     * @return An action for the current state
     */
    public Types.ACTIONS act(StateObservation stateObs, ElapsedCpuTimer elapsedTimer) {

        //m_timer = elapsedTimer;
        learnPolicy(stateObs, SIMULATION_DEPTH, new WinScoreHeuristic(stateObs));
        int action_num = 0;
        Types.ACTIONS bestAction = null;
        try {
            ArrayList<Types.ACTIONS> availableActions = stateObs.getAvailableActions();
            int[] toNum_availableActions = new int[availableActions.size()];
            for (int i=0; i<toNum_availableActions.length; ++i) {
                StateObservation simObs = stateObs.copy();
                simObs.advance(availableActions.get(i));
                if (!simObs.getAvatarPosition().equals(stateObs.getAvatarPosition()))
                    toNum_availableActions[i] = inverted_action_mapping.get(availableActions.get(i));
            }

            double[] features = RLDataExtractor.featureExtract(stateObs);
            action_num = m_policy.getActionNoExplore(features, toNum_availableActions, last_Action, m_gamma*m_gamma); // no exploration
            bestAction = action_mapping.get(action_num);
            last_Action = new int[]{counter[inverted_action_mapping.get(bestAction)]};

            String location_signature = "["+stateObs.getAvatarPosition().x+","+stateObs.getAvatarPosition().y+"]";
            // System.out.print(location_signature+" "+stateObs.getGameScore());


            if (!location_score.containsKey(location_signature)){
                location_score.put(location_signature, stateObs.getGameScore());
                // System.out.println();

                double[] pfeatures = pDataExtractor.featureExtract(stateObs, action_num);
                Instance pins = new Instance(1, pfeatures);
                pins.setDataset(pHeader);
                pHeader.add(pins);

                double[] vfeatures = vDataExtractor.featureExtract(stateObs);
                Instance vins = new Instance(1, vfeatures);
                vins.setDataset(vHeader);
                vHeader.add(vins);
            }
            else {
                // System.out.println(" TRUE ["+location_score.get(location_signature)+"]");
                if (stateObs.getGameScore() > location_score.get(location_signature) + 1){
                    location_score.replace(location_signature, stateObs.getGameScore());

                    double[] pfeatures = pDataExtractor.featureExtract(stateObs, action_num);
                    Instance pins = new Instance(1, pfeatures);
                    pins.setDataset(pHeader);
                    pHeader.add(pins);

                    double[] vfeatures = vDataExtractor.featureExtract(stateObs);
                    Instance vins = new Instance(1, vfeatures);
                    vins.setDataset(vHeader);
                    vHeader.add(vins);
                }
                else if (stateObs.getGameScore() == location_score.get(location_signature) + 1){
                    location_score.replace(location_signature, stateObs.getGameScore());
                    if (new Random().nextDouble() < beta){
                        double[] pfeatures = pDataExtractor.featureExtract(stateObs, action_num);
                        Instance pins = new Instance(1, pfeatures);
                        pins.setDataset(pHeader);
                        pHeader.add(pins);

                        double[] vfeatures = vDataExtractor.featureExtract(stateObs);
                        Instance vins = new Instance(1, vfeatures);
                        vins.setDataset(vHeader);
                        vHeader.add(vins);
                    }
                    else;
                        // System.out.println("redundant training data, dropped [UNLUCKY]");

                }
                else;
                    // System.out.println("redundant training data, dropped ["+stateObs.getGameScore()+":"+location_score.get(location_signature)+"]");
            }
        } catch (Exception exc) {
            exc.printStackTrace();
        }

        // System.out.println("====================");


        return bestAction;
    }

    private Instances simulate(StateObservation stateObs, StateHeuristic heuristic, QPolicy policy) {
        return simulator.simulate(stateObs, heuristic, policy);
    }

    private void learnPolicy(StateObservation stateObs, int maxdepth, StateHeuristic heuristic) {

        // assume we need SIMULATION_DEPTH*10 milliseconds for one iteration
        int iter = 0;
        while (iter++ <= 10 //truem_timer.remainingTimeMillis() > SIMULATION_DEPTH*10
                ) {

            // get NetTuning data of the MC sampling
            Instances dataset = simulate(stateObs, heuristic, m_policy);

            // update dataset
            m_dataset.randomize(m_rnd);
            for (int i = 0; i < dataset.numInstances(); i++) {
                m_dataset.add(dataset.instance(i)); // add to the last
            }
            while (m_dataset.numInstances() > m_maxPoolSize) {
                m_dataset.delete(0);
            }
        }
        // train policy
        try {
            m_policy.fitQ(m_dataset);
            simulator.setClassifierPrediction(true);
        } catch (Exception exc) {
            exc.printStackTrace();
        }
    }
}
