package edu.iu.kmeans.common;

import edu.iu.fileformat.MultiFileInputFormat;
import edu.iu.kmeans.allreduce.MiniBatchKmeansMapper;
import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.conf.Configured;
import org.apache.hadoop.fs.FileSystem;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.mapred.JobConf;
import org.apache.hadoop.mapreduce.Job;
import org.apache.hadoop.mapreduce.lib.input.FileInputFormat;
import org.apache.hadoop.mapreduce.lib.output.FileOutputFormat;
import org.apache.hadoop.util.Tool;
import org.apache.hadoop.util.ToolRunner;

import java.io.IOException;
import java.net.URISyntaxException;
import java.text.SimpleDateFormat;
import java.util.Calendar;
import java.util.concurrent.ExecutionException;

/**
 * Finding centroids using Mini-batch K-Means algorithm
 */

public class MiniBatchKmeansMapCollective extends Configured implements Tool {

    public static void main(String [] args) throws Exception {
        int res = ToolRunner.run(new Configuration(), new MiniBatchKmeansMapCollective(), args);
        System.exit(res);
    }

    @Override
    public int run(String[] args) throws Exception {
        if (args.length < 8) {
            System.err.println("Usage: Mini-BatchKmeansMapCollective " +
                    "<dataPointsCount> " +
                    "<centroidsCount> " +
                    "<vectorSize> " +
                    "<batchSize> " +
                    "<mapTaskCount> " +
                    "<iterationCount> " +
                    "<workPath> " +
                    "<localPath> ") ;
            ToolRunner.printGenericCommandUsage(System.err);
            return -1;
        }

        int dataPointsCount = Integer.parseInt(args[0]);
        System.out.println("dataPointsCount: " + dataPointsCount) ;

        int centroidsCount = Integer.parseInt(args[1]);
        System.out.println("centroidsCount: " + centroidsCount) ;

        int vectorSize = Integer.parseInt(args[2]);
        System.out.println("vectorSize: " + vectorSize) ;

        int batchSize = Integer.parseInt(args[3]);
        System.out.println("batchSize: " + batchSize) ;

        int mapTaskCount = Integer.parseInt(args[4]);
        System.out.println("mapTaskCount: " + mapTaskCount) ;

        int iterationCount = Integer.parseInt(args[5]);
        System.out.println("iterationCount: " + iterationCount) ;

        String workPath = args[6];
        System.out.println("workPath: " + workPath) ;

        String localPath = args[7];
        System.out.println("localPath: " + localPath) ;

        launch(dataPointsCount, centroidsCount, vectorSize, batchSize
                , mapTaskCount, iterationCount, workPath, localPath);
        System.out.println("Success :) :)");
        return 0;
    }

    void launch(int dataPointsCount, int centroidsCount, int vectorSize,
                int batchSize, int mapTaskCount, int iterationCount,
                String workDPath, String localPath)
            throws ClassNotFoundException, InterruptedException, IOException,
            URISyntaxException, ExecutionException {

        Configuration configuration = getConf();
        Path workPath = new Path(workDPath);
        FileSystem fs = FileSystem.get(configuration);
        Path dataPath = new Path(workPath, "data");
        Path cenDir = new Path(workPath, "centroids");
        Path outDir = new Path(workPath, "out");

        if (fs.exists(outDir)) {
            fs.delete(outDir, true);
        }
        fs.mkdirs(outDir);

        System.out.println("Generate data.");
        Utils.generateData(dataPointsCount, vectorSize, mapTaskCount, fs, localPath, dataPath);

        int jobId = 0;
        Utils.generateInitialCentroids(centroidsCount, vectorSize, configuration, cenDir, fs, jobId);

        long startTime = System.currentTimeMillis();

        runMiniBatchKmeans(dataPointsCount,centroidsCount, vectorSize, batchSize,
                iterationCount, jobId,  mapTaskCount, configuration,
                workPath, dataPath, cenDir, outDir);
        long endTime = System.currentTimeMillis();
        System.out.println("Total Mini-Batch K-means Execution Time: "+ (endTime - startTime));
    }

    private void runMiniBatchKmeans(int dataPointsCount, int centroidsCount,
                                    int vectorSize, int batchSize,
                                    int numIterations, int jobId,
                                    int mapTaskCount, Configuration configuration,
                                    Path workPath, Path dataPath,
                                    Path cDir, Path outDir)
            throws ClassNotFoundException, IOException,URISyntaxException, InterruptedException {

        System.out.println("Starting Job");
        long jobSubmitTime;
        boolean jobSuccess = true;
        int jobRetryCount = 0;

        do {
            // ----------------------------------------------------------------------
            jobSubmitTime = System.currentTimeMillis();
            System.out.println("Start Job#" + jobId + " "+ new SimpleDateFormat("HH:mm:ss.SSS").format(Calendar.getInstance().getTime()));

            Job kmeansJob = configureMiniBatchKmeansJob(dataPointsCount, centroidsCount,
                    vectorSize, batchSize, mapTaskCount, configuration,
                    workPath, dataPath,cDir, outDir, jobId, numIterations);

            System.out.println("| Job#"+ jobId+ " configure in "+ (System.currentTimeMillis() - jobSubmitTime)+ " miliseconds |");

            // ----------------------------------------------------------
            jobSuccess =kmeansJob.waitForCompletion(true);

            System.out.println("end Jod#" + jobId + " "
                    + new SimpleDateFormat("HH:mm:ss.SSS")
                    .format(Calendar.getInstance().getTime()));
            System.out.println("| Job#"+ jobId + " Finished in "
                    + (System.currentTimeMillis() - jobSubmitTime)
                    + " miliseconds |");

            // ---------------------------------------------------------
            if (!jobSuccess) {
                System.out.println("Mini-Batch KMeans Job failed. Job ID:"+ jobId);
                jobRetryCount++;
                if (jobRetryCount == 3) {
                    break;
                }
            }else{
                break;
            }
        } while (true);
    }

    private Job configureMiniBatchKmeansJob(int numOfDataPoints, int numCentroids,
                                            int vectorSize, int batchSize,
                                            int numMapTasks, Configuration configuration,
                                            Path workDirPath, Path dataDir, Path cDir,
                                            Path outDir, int jobID, int numIterations)
            throws IOException, URISyntaxException {

        Job job = Job.getInstance(configuration, "mbkmeans_job_"+ jobID);
        Configuration jobConfig = job.getConfiguration();
        Path jobOutDir = new Path(outDir, "mbkmeans_out_" + jobID);
        FileSystem fs = FileSystem.get(configuration);
        if (fs.exists(jobOutDir)) {
            fs.delete(jobOutDir, true);
        }
        FileInputFormat.setInputPaths(job, dataDir);
        FileOutputFormat.setOutputPath(job, jobOutDir);

        Path cFile = new Path(cDir, MiniBatchKmeansConstants.CENTROID_FILE_PREFIX + jobID);
        System.out.println("Centroid File Path: "+ cFile.toString());
        jobConfig.set(MiniBatchKmeansConstants.CFILE,cFile.toString());
        jobConfig.setInt(MiniBatchKmeansConstants.JOB_ID, jobID);
        jobConfig.setInt(MiniBatchKmeansConstants.NUM_ITERATONS, numIterations);
        job.setInputFormatClass(MultiFileInputFormat.class);
        job.setJarByClass(MiniBatchKmeansConstants.class);
        job.setMapperClass(MiniBatchKmeansMapper.class);
        JobConf jobConf = (JobConf) job.getConfiguration();
        jobConf.set("mapreduce.framework.name", "map-collective");
        jobConf.setNumMapTasks(numMapTasks);
        jobConf.setInt("mapreduce.job.max.split.locations", 10000);
        job.setNumReduceTasks(0);
        jobConfig.setInt(MiniBatchKmeansConstants.VECTOR_SIZE,vectorSize);
        jobConfig.setInt(MiniBatchKmeansConstants.NUM_CENTROIDS, numCentroids);
        jobConfig.set(MiniBatchKmeansConstants.WORK_DIR,workDirPath.toString());
        jobConfig.setInt(MiniBatchKmeansConstants.NUM_MAPPERS, numMapTasks);
        return job;
    }

}