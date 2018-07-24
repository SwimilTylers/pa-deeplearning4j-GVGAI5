package controllers.miniAlphaZero.simulator;

import controllers.Heuristics.StateHeuristic;
import controllers.miniAlphaZero.pretraining.QPolicy;
import core.game.StateObservation;
import weka.core.Instances;

/**
 * Created with IntelliJ IDEA.
 * User: Swimiltylers
 * Date: 2018/6/7
 * Time: 19:34
 */
public interface Simulator {
    /* dummy func */
    void setClassifierPrediction(boolean classifierPrediction);

    Instances simulate(StateObservation stateObs, StateHeuristic heuristic, QPolicy policy);
}
