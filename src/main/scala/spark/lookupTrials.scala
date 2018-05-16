package spark

import org.apache.spark.sql.SparkSession

object lookupTrials {

  def lookUptest ={

    val spark = SparkSession.builder.appName("My Spark Lookup Application").master("local[8]").getOrCreate

    System.setProperty("hadoop.home.dir", "E:\\Hadoop\\HADOOP_HOME\\bin")

    import spark.implicits._
    val rdd1 = Seq((1,2),(2,4),(3,6),(6,7)).toDF("id","num_val").rdd
    val brdCmrdd1 = spark.sparkContext.broadcast(rdd1.map( x=> ((x.getAs("id"),x.getAs("num_val")))).collect())

    val rdd2 = Seq( ("rijo",2,"Hyderabad"), ("Sowji",3, "Warangal"), ("Kiwi",10,"Singapore")).toDF("name","id","place").rdd



   /* val outputRDD = rdd2.map{ row => (
      row.getAs("name"), brdCmrdd1.value.getOrElse(row.getAs("id").else("None"))
    )}*/
  }
}

