package controllers.miniAlphaZero.pretraining;

import core.game.Observation;
import core.game.StateObservation;
import tools.Vector2d;
import weka.core.Attribute;
import weka.core.FastVector;
import weka.core.Instances;

import java.util.ArrayList;
import java.util.LinkedList;

/**
 * Created with IntelliJ IDEA.
 * User: Swimiltylers
 * Date: 2018/6/25
 * Time: 17:45
 */
public class pDataExtractor {
    public static double[] featureExtract(StateObservation obs, int action_to_num){

        double[] feature = new double[873];  // 868 + 4 + 1(action)

        // 448 locations
        int[][] map = new int[28][31];
        // Extract features
        LinkedList<Observation> allobj = new LinkedList<>();
        if( obs.getImmovablePositions()!=null )
            for(ArrayList<Observation> l : obs.getImmovablePositions()) allobj.addAll(l);
        if( obs.getMovablePositions()!=null )
            for(ArrayList<Observation> l : obs.getMovablePositions()) allobj.addAll(l);
        if( obs.getNPCPositions()!=null )
            for(ArrayList<Observation> l : obs.getNPCPositions()) allobj.addAll(l);

        for(Observation o : allobj){
            Vector2d p = o.position;
            int x = (int)(p.x/20); //squre size is 20 for pacman
            int y= (int)(p.y/20);
            map[x][y] = o.itype;
        }
        for(int y=0; y<31; y++)
            for(int x=0; x<28; x++)
                feature[y*28+x] = map[x][y];

        // 4 states
        feature[868] = obs.getGameTick();
        feature[869] = obs.getAvatarSpeed();
        try {
            feature[870] = obs.getAvatarHealthPoints();
        }catch (Exception e){
            feature[870] = 0;
        }
        feature[871] = obs.getAvatarType();
        feature[872] = action_to_num;

        return feature;
    }

    public static Instances datasetHeader(){

        FastVector attInfo = new FastVector();
        // 448 locations
        for(int y=0; y<28; y++){
            for(int x=0; x<31; x++){
                Attribute att = new Attribute("object_at_position_x=" + x + "_y=" + y);
                attInfo.addElement(att);
            }
        }
        Attribute att = new Attribute("GameTick" ); attInfo.addElement(att);
        att = new Attribute("AvatarSpeed" ); attInfo.addElement(att);
        att = new Attribute("AvatarHealthPoints" ); attInfo.addElement(att);
        att = new Attribute("AvatarType" ); attInfo.addElement(att);
        //action
        FastVector actions = new FastVector();
        actions.addElement("0");
        actions.addElement("1");
        actions.addElement("2");
        actions.addElement("3");
        att = new Attribute("actions", actions);
        attInfo.addElement(att);

        Instances instances = new Instances("Pacman_pnet_data", attInfo, 0);
        instances.setClassIndex( instances.numAttributes() - 1);

        return instances;
    }
}
