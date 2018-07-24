/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package controllers.miniAlphaZero.pretraining;

import java.util.Arrays;
import java.util.Random;

import weka.classifiers.Classifier;
import weka.classifiers.trees.REPTree;
import weka.core.Instance;
import weka.core.Instances;
/**
 *
 * @author yuy
 */
public class QPolicy {
    protected double m_epsilon=0.3;
    protected Classifier m_c;
    protected Instances m_dataset;
    protected Random m_rnd;
    protected int m_numActions;
    
    
    public QPolicy(int N_ACTIONS){
        m_numActions = N_ACTIONS;
        m_dataset = RLDataExtractor.datasetHeader();
        m_rnd = new Random();
        m_c = null;
    }
    
    public void setEpsilon(double epsilon){
        m_epsilon = epsilon;
    }
    
    // max Q action without epsilon-greedy 
    public int getActionNoExplore(double[] feature) throws Exception{
        double[] Q = getQArray(feature);
        
        // find best action according to Q value
        int bestaction = 0;
        for(int action=1; action<m_numActions; action++){
            if( Q[bestaction] < Q[action] ){
                bestaction = action;
            }
        }
        // among the same best actions, choose a random one
        int sameactions =0;
        for(int action=bestaction+1; action<m_numActions; action++){
            if(Q[bestaction] == Q[action]){
                sameactions++;
                if( m_rnd.nextDouble() < 1.0/(double)sameactions )
                    bestaction = action;
            }
        }
        
        return bestaction;
    }

    public int getActionNoExplore(double[] feature, int[] available,  int[] penalty, double penalty_ratio) throws Exception{
        double[] Q = getQArray(feature);

        boolean[] chk = new boolean[m_numActions];
        Arrays.fill(chk, false);

        for (int i=0; i<available.length; ++i)
            chk[available[i]] = true;
        if (penalty != null)
            for (int i=0; i<penalty.length; ++i)
                Q[penalty[i]] = Q[penalty[i]] >= 0
                        ? (penalty_ratio * Q[penalty[i]])
                        : ((2-penalty_ratio) * Q[penalty[i]]);

        // find best action according to Q value
        int best_action = 0;
        int action = 1;
        for (; action<m_numActions&&!chk[action]; action++);
        if (action == m_numActions) return m_rnd.nextInt(m_numActions);
        for(; action<m_numActions; action++){
            if( Q[best_action] < Q[action] && chk[action]){
                best_action = action;
            }
        }
        // among the same best actions, choose a random one
        int same_actions =0;
        for(action=best_action+1; action<m_numActions; action++){
            if(Q[best_action] == Q[action] && chk[action]){
                same_actions++;
                if( m_rnd.nextDouble() < 1.0/(double)same_actions )
                    best_action = action;
            }
        }

        return best_action;
    }
    
    // max Q action with epsilon-greedy 
    public int getAction(double[] feature) throws Exception{
        double[] Q = getQArray(feature);
        
        // find best action according to Q value
        int bestaction = 0;
        for(int action=1; action<m_numActions; action++){
            if( Q[bestaction] < Q[action] ){
                bestaction = action;
            }
        }
        // among the same best actions, choose a random one
        int sameactions =0;
        for(int action=bestaction+1; action<m_numActions; action++){
            if(Q[bestaction] == Q[action]){
                sameactions++;
                if( m_rnd.nextDouble() < 1.0/(double)sameactions )
                    bestaction = action;
            }
        }
        
        // epsilon greedy
        if( m_rnd.nextDouble() < m_epsilon ){
            bestaction = m_rnd.nextInt(m_numActions);
        }
        
        return bestaction;
    }

    // max Q action with epsilon-greedy
    public int getAction(double[] feature, int[] available,  int[] penalty, double penalty_ratio) throws Exception{
        double[] Q = getQArray(feature);

        boolean[] chk = new boolean[m_numActions];
        Arrays.fill(chk, false);

        for (int i=0; i<available.length; ++i)
            chk[available[i]] = true;
        if (penalty != null)
            for (int i=0; i<penalty.length; ++i)
                Q[penalty[i]] = Q[penalty[i]] >= 0
                        ? (penalty_ratio * Q[penalty[i]])
                        : ((2-penalty_ratio) * Q[penalty[i]]);

        // find best action according to Q value
        int best_action = 0;
        int action = 1;
        for (; action<m_numActions&&!chk[action]; action++);
        if (action == m_numActions) return m_rnd.nextInt(m_numActions);
        for(; action<m_numActions; action++){
            if( Q[best_action] < Q[action] && chk[action]){
                best_action = action;
            }
        }
        // among the same best actions, choose a random one
        int same_actions =0;
        for(action=best_action+1; action<m_numActions; action++){
            if(Q[best_action] == Q[action] && chk[action]){
                same_actions++;
                if( m_rnd.nextDouble() < 1.0/(double)same_actions )
                    best_action = action;
            }
        }

        // epsilon greedy
        if( m_rnd.nextDouble() < m_epsilon ){
            best_action = m_rnd.nextInt(m_numActions);
        }

        return best_action;
    }
    
    public double getMaxQ(double[] feature) throws Exception{
        double[] Q = getQArray(feature);
        
        // find best action according to Q value
        int bestaction = 0;
        for(int action=1; action<m_numActions; action++){
            if( Q[bestaction] < Q[action] )
                bestaction = action;
        }
        
        return Q[bestaction];
    }
    
    public double[] getQArray(double[] feature) throws Exception{
        
        double[] Q = new double[m_numActions];
        
        // get Q value from the prediction model
        for(int action = 0; action<m_numActions; action++){
            feature[feature.length-2] = action;
            feature[feature.length-1] = Double.NaN;
            Q[action] = m_c == null ? 0 : m_c.classifyInstance(makeInstance(feature));
        }
        
        return Q;
    }
    
    public void fitQ(Instances data) throws Exception{
        if( m_c == null ){
            m_c = new weka.classifiers.trees.REPTree();
            ((REPTree)m_c).setMinNum(1);
            ((REPTree)m_c).setNoPruning(true);
        }
        m_c.buildClassifier(data);   
    }
        
    protected Instance makeInstance(double[] vector){
        Instance ins = new Instance(1,vector);
        ins.setDataset(m_dataset);
        return ins;
    }
}
