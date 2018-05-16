package spark

import org.apache.spark.sql.{DataFrame, SparkSession}
import org.apache.spark.sql.functions._
import org.apache.spark.ml.Pipeline
import org.apache.spark.ml.classification.{RandomForestClassificationModel, RandomForestClassifier}
import org.apache.spark.ml.evaluation.BinaryClassificationEvaluator
import org.apache.spark.ml.feature._
import org.apache.spark.ml.tuning.{CrossValidator, ParamGridBuilder}




object ModelSparkloreBestMovie {
  case class rfParams() {
    val algoNumTrees: Seq[Int] = Seq(40)
    val algoMaxDepth: Seq[Int] = Seq(6)
    val algoMaxBins: Seq[Int] = Seq(32)
  }

  val params = new rfParams

  println("Param=>" + params.algoNumTrees)

  def getDataPrep = {
    println("Dataprep for Best Movie")

    val spark = SparkSession.builder().appName("BestMovieModel").master("local[8]").getOrCreate()

    import spark.implicits._
    val oscarData = spark.read.option("header", true).option("inferSchema", true).csv("src/main/resources/oscar-data.csv")
      .select(col("Year").cast("Int"),
        col("Movie"),
        col("Rated"),
        to_date(unix_timestamp(col("Release_Date"), "MM/dd/yyyy").cast("Timestamp")).as("Release_Date"),
        regexp_replace(col("Runtime"), "[^0-9]+", "").as("Runtime"),
        col("Genres"),
        col("Language"),
        col("Country"),
        col("Awards").as("String").as("Awards"),
        col("Metascore"),
        col("Rating"),
        col("Votes"),
        when(col("Oscar_Winner").equalTo("Yes"), 1).otherwise(0).as("Oscar_Winner"))

    /*  oscarData.schema.foreach(println)
      oscarData.describe().collect().foreach(println)
      oscarData.show(10)*/

    println()

    val PatternGG = "Won ([0-9]+) Golden Globes".r
    val PatternGGNomi = "Nominated for ([0-9]+) Golden Globes".r
    val PatternAnotherWins = "Another ([0-9]+) wins".r
    val PatternNomis = "([0-9]+) nominations".r
    val PatternOscar = "Won ([0-9]+) Oscars".r
    val PatternOscarNomi = "Nominated for ([0-9]+) Oscars".r

    // Check different prizes
    val awardsDistinct = oscarData.select(col("Awards"), regexp_replace(col("Awards"), "[.&]", "|").as("repCol")).rdd
      .map(y => {
        val origVal = y.getAs[String]("Awards")
        val arr = y.getAs[String]("repCol").split('|').toList
        arr.foreach(println)
        var oscarWin = 0
        var oscarNomi = 0
        var nomi = 0
        var anotherWins = 0
        var goldenGlobesWon = 0
        var goldenGlobesNomis = 0

        for (awardVal <- arr) {
          //              if ( 0 < i ) {
          println("awardVal=>" + awardVal)
          awardVal.trim match {
            case PatternGG(num) => goldenGlobesWon = num.toInt
            case PatternGGNomi(num) => goldenGlobesNomis = num.toInt
            case PatternAnotherWins(num) => anotherWins = num.toInt
            case PatternNomis(num) => nomi = num.toInt
            case PatternOscar(num) => oscarWin = num.toInt
            case PatternOscarNomi(num) => oscarNomi = num.toInt
            case _ => {
              oscarWin = 0
              oscarNomi = 0
              nomi = 0
              anotherWins = 0
              goldenGlobesWon = 0
              goldenGlobesNomis = 0
            }
          }
          //              }
        }
        (origVal.toString(), oscarWin.toString(), oscarNomi.toString(), goldenGlobesWon.toString(), goldenGlobesNomis.toString(), anotherWins.toString(), nomi.toString())
      }).toDF("Award_n", "OscarWin", "OscarNomi", "GoldenGlobesWon", "GoldenGlobesNomis", "AnotherWins", "Nomi")

    awardsDistinct.take(10).foreach(println)

    val finalData = oscarData.join(awardsDistinct, col("Awards") === col("Award_n"), "inner")
      .select("Year", "Movie", "Rated", "Release_Date", "Runtime", "Genres",
        "Language", "Country", "OscarWin", "OscarNomi", "GoldenGlobesWon", "GoldenGlobesNomis",
        "AnotherWins", "Nomi", "Metascore", "Rating", "Votes", "Oscar_Winner")

    finalData.show()
    ModelTrain(finalData)
  }


  def ModelTrain(mainData: DataFrame) = {
    println("Train the model")

    RandomForestModel(mainData.select(col("Year"),
      col("Movie").cast("String"),
      col("Rated").cast("String"),
      col("Release_Date"),
      col("Runtime"),
      col("Genres").cast("String"),
      col("Language").cast("String"),
      col("Country").cast("String"),
      col("OscarWin"),
      col("OscarNomi"),
      col("GoldenGlobesWon"),
      col("GoldenGlobesNomis"),
      col("AnotherWins"),
      col("Nomi"),
      col("Metascore"),
      col("Rating"),
      col("Votes"),
      col("Oscar_Winner")
    ))

  }

  def RandomForestModel(data: DataFrame) = {

    // StringIndexer for categorical columns (OneHotEncoder should be evaluated as well)
    def isCateg(c: String): Boolean = c.startsWith("cat")

    def categNewCol(c: String): String = if (isCateg(c)) s"idx_${c}" else c

    def hotEncodeNewCol(c: String): String = if (isCateg(c)) s"hotencd_${c}" else c

    // Function to select only feature columns (omit id and label)
    def onlyFeatureCols(c: String): Boolean = !(c matches "id|label")

    val mainData = data.withColumnRenamed("Oscar_Winner", "label")

    val split = mainData.randomSplit(Array(0.7, 0.3), seed = 12345)

    val (trainData, testData) = (split(0), split(1))

    /*val featureCols = Array("Year", "Movie", "Rated", "Release_Date", "Runtime", "Genres",
      "Language", "Country", "OscarWin", "OscarNomi", "GoldenGlobesWon", "GoldenGlobesNomis",
      "AnotherWins", "Nomi", "Metascore", "Rating", "Votes")
*/
    val stringIndexerStages = trainData.columns.filter(isCateg)
      .map(c => new StringIndexer()
        .setInputCol(c)
        .setOutputCol(categNewCol(c))
        .fit(mainData.select(c)))

//    val indexed = stringIndexerStages.ransform()

    val oneHotEncodeStages = trainData.columns.filter(isCateg)
      .map(c => new OneHotEncoder()
        .setInputCol(c)
        .setOutputCol(hotEncodeNewCol(c))
        .transform(mainData.select(c)))

    // Definitive set of feature columns
    val featureCols = trainData.columns
      .filter(onlyFeatureCols)
      .map(categNewCol)

    //set the input and output column names**
    val assembler = new VectorAssembler().setInputCols(featureCols).setOutputCol("features")

    // Estimator algorithm
    val algo = new RandomForestClassifier().setFeaturesCol("features")

    // Building the Pipeline for transformations and predictor
    val pipeline = new Pipeline().setStages((stringIndexerStages :+ assembler ) :+ algo )


    // ***********************************************************
    //    log.info("Preparing K-fold Cross Validation and Grid Search")
    // ***********************************************************

    val paramGrid = new ParamGridBuilder()
      .addGrid(algo.numTrees, params.algoNumTrees)
      .addGrid(algo.maxDepth, params.algoMaxDepth)
      .addGrid(algo.maxBins, params.algoMaxBins)
      //        .addGrid(algo.)
      .build()

    val cv = new CrossValidator()
      .setEstimator(pipeline)
      .setEvaluator(new BinaryClassificationEvaluator())
      .setEstimatorParamMaps(paramGrid)

    //      .setNumFolds(params.numFolds)
    // ************************************************************
    //    log.info("Training model with RandomForest algorithm")
    // ************************************************************

    val cvModel = cv.fit(trainData)


    // **********************************************************************
    //    log.info("Evaluating model on train and test data and calculating RMSE")
    // **********************************************************************

    val trainPredictionsAndLabels = cvModel.transform(testData).select("label", "prediction")
    trainPredictionsAndLabels.show()
    // create an Evaluator for binary classification, which expects two input columns: rawPrediction and label.**
    val evaluator = new BinaryClassificationEvaluator().setMetricName("areaUnderROC").setLabelCol("label")
    // Evaluates predictions and returns a scalar metric areaUnderROC(larger is better).**
    println("Eval==>" + evaluator.evaluate(testData))

  }

}
