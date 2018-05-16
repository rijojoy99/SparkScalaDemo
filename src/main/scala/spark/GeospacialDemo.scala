package spark

import java.sql.Timestamp
import com.typesafe.config.ConfigFactory
import org.apache.spark.sql.SparkSession
import org.apache.spark.sql.catalyst.expressions.Literal

object GeospacialDemo {

  def run_prog = {

    val spark = SparkSession.builder.appName("My Spark GeoSpatial Application").master("local[8]").getOrCreate

    System.setProperty("hadoop.home.dir", "E:\\Hadoop\\HADOOP_HOME\\bin")

    import spark.implicits._
    val geo_locs = spark.read.option("header", "true").option("delimiter", ",").csv("src/main/resources/single_trip.csv")
      .select($"trip_id".cast("String"), $"timestamp".cast("Timestamp"), $"lat".cast("double"), $"lon".cast("double"))
      .as[geoLocs]
    //      .toDF("trip_id", "timestamp", "lat", "lon")

    val key_pnts = geo_locs.filter('trip_id === Literal("00001")).rdd.groupBy(x => x.trip_id).keyBy(x => (x._1, x._2))


    key_pnts.take(20).foreach(println)
    println("Count=> " + geo_locs.count())

    val conf = ConfigFactory.load()
    val value = conf.getString("my.demoVal.demo")
    val value1 = conf.getString("my.demoVal.demo1")
    val value2 = conf.getString("my.demoVal.demo2")
    val value3 = conf.getInt("my.demoVal.demo3")
    val value4 = conf.getDouble("my.demoVal.demo4")
    println(s"******************************************************************My secret value is $value"
      + "\n" + value1 + "\n" + value2 + "\n" + value3 + "\n" + value4 + "********")

    type ScoreCollector = (Int, Double)
    type PersonScores = (String, (Int, Double))

    val initialScores = Array(("Fred", 88.0), ("Fred", 95.0), ("Fred", 91.0), ("Wilma", 93.0), ("Wilma", 95.0), ("Wilma", 98.0))

    val wilmaAndFredScores = spark.sparkContext.parallelize(initialScores).cache()

    val createScoreCombiner = (score: Double) => (1, score)

    val scoreCombiner = (collector: ScoreCollector, score: Double) => {
      val (numberScores, totalScore) = collector
      (numberScores + 1, totalScore + score)
    }

    val scoreMerger = (collector1: ScoreCollector, collector2: ScoreCollector) => {
      val (numScores1, totalScore1) = collector1
      val (numScores2, totalScore2) = collector2
      (numScores1 + numScores2, totalScore1 + totalScore2)
    }
    val scores = wilmaAndFredScores.combineByKey(createScoreCombiner, scoreCombiner, scoreMerger)

    val averagingFunction = (personScore: PersonScores) => {
      val (name, (numberScores, totalScore)) = personScore
      (name, totalScore / numberScores)
    }

    val averageScores = scores.collectAsMap().map(averagingFunction)

    println("Average Scores using CombingByKey")
    averageScores.foreach((ps) => {
      val(name,average) = ps
      println(name+ "'s average score : " + average)
    })
  }

  case class geoLocs(trip_id: String, timestamp: Timestamp, lat: Double, lon: Double)
}
