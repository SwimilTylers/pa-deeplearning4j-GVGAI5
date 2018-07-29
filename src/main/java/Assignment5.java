import java.lang.annotation.Repeatable;
import java.util.Random;

import controllers.miniAlphaZero.deeplearning4j.Dl4jLocalUtils;
import controllers.miniAlphaZero.deeplearning4j.NetTuning;
import controllers.miniAlphaZero.pretraining.InstancesStore;
import core.competition.CompetitionParameters;

import core.ArcadeMachine;
import weka.core.Instances;

/**
 * Created with IntelliJ IDEA.
 * User: Diego
 * Date: 04/10/13
 * Time: 16:29
 * This is a Java port from Tom Schaul's VGDL - https://github.com/schaul/py-vgdl
 */
public class Assignment5
{
    //Only pacman game
    private static String game = "/examples/gridphysics/pacman.txt";
    private static String level = "/examples/gridphysics/pacman_lvl";

    private static boolean[] gui_switch = new boolean[]{false, true};

    public static void pretraining(int epoch){
        //Reinforcement learning controllers:
        String rlController = "controllers.miniAlphaZero.pretraining.Agent";

        //Other settings
        int seed = new Random().nextInt();

        //Game and level to play

        // Monte-Carlo RL NetTuning
        CompetitionParameters.ACTION_TIME = 1000000;
        //ArcadeMachine.runOneGame(game, level, visuals, rlController, null, seed, false);
        //String level2 = gamesPath + games[gameIdx] + "_lvl" + 1 +".txt";
        for(int i=0; i<epoch; i++){
            String levelfile = level + "0.txt";
            System.out.println("Pre-Training at ["+i+","+epoch+"]");
            ArcadeMachine.runOneGame(makeResourcePath(game), makeResourcePath(levelfile), gui_switch[0], rlController, null, seed, false);
            InstancesStore.save();
        }
    }

    public static void training(boolean CV){
        if (!InstancesStore.isStored)
            InstancesStore.load();

        Instances ptrain = InstancesStore.pHeader;
        Instances vtrain = InstancesStore.vHeader;
        ;
        try {
            NetTuning.train(Dl4jLocalUtils.getDataSetFromInstances(ptrain,4), Dl4jLocalUtils.getDataSetFromInstances(vtrain));
        } catch (Exception e) {
            e.printStackTrace();
        }
    }

    public static void game(int epoch, int fine_tuning_interval){
        //Reinforcement learning controllers:
        String rlController = "controllers.miniAlphaZero.run.Agent";

        //Other settings
        int seed = new Random().nextInt();

        //Game and level to play

        // Monte-Carlo RL NetTuning
        CompetitionParameters.ACTION_TIME = 1000000;
        //ArcadeMachine.runOneGame(game, level, visuals, rlController, null, seed, false);
        //String level2 = gamesPath + games[gameIdx] + "_lvl" + 1 +".txt";
        if (!InstancesStore.isStored)
            InstancesStore.load();
        for(int i=0; i<epoch; i++){
            String levelfile = level + "0.txt";
            ArcadeMachine.runOneGame(makeResourcePath(game), makeResourcePath(levelfile), gui_switch[1], rlController, null, seed, false);
            if ((i+1)%fine_tuning_interval == 0)
                try {
                    NetTuning.fine_tuning(Dl4jLocalUtils.getDataSetFromInstances(InstancesStore.pHeader,4), Dl4jLocalUtils.getDataSetFromInstances(InstancesStore.vHeader));
                } catch (Exception e) {
                    e.printStackTrace();
                }
        }
    }

    private static String makeResourcePath(String template) {
        return Assignment5.class.getResource(template).getPath();
    }

    public static void main(String[] args)
    {
        //pretraining(5);
        //training(false);
        game(4, 2);
    }
}
