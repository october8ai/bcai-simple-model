import org.apache.spark.ml.classification.{LogisticRegression, LogisticRegressionModel}
import org.apache.spark.ml.feature.LabeledPoint
import org.apache.spark.ml.linalg.Vectors
import org.apache.spark.ml.{Pipeline, PipelineStage}
import org.apache.spark.sql.functions._
import org.apache.spark.sql.{DataFrame, Dataset, SparkSession}

import scala.collection.mutable

object SimpleModel {
  def main(args: Array[String]) {

    val spark = SparkSession.builder()
      .appName("Spark Interview")
      .master("local[4]")
      .getOrCreate()

    spark.sparkContext.setLogLevel("ERROR")


    import spark.implicits._

    val eventsFile = "src/main/resources/events.csv"
    val conversionsFile = "src/main/resources/conversions.csv"
    val events = spark
      .read
      .option("header", "true")
      .option("inferSchema", "true")
      .csv(eventsFile)
      .cache()

    val conversions = spark
      .read
      .option("header", "true")
      .option("inferSchema", "true")
      .csv(conversionsFile)
      .withColumn("CONVERTED", lit(1.0))
      .cache()



    val labeledEvents: DataFrame =  events.join(conversions, Seq("ID"), "left")

    labeledEvents.printSchema()
    labeledEvents.show()

    val labeledFeatures: Dataset[LabeledPoint] =
      labeledEvents
        .map(r => LabeledPoint(
          r.getAs[Double]("CONVERTED"),
          Vectors.dense(Array(Option(r.getAs[Int]("HOTEL_CITY_ID")).map(_.toDouble).getOrElse(0.0),
            Option(r.getAs[Int]("TIME_TO_ARRIVAL")).map(_.toDouble).getOrElse(0.0),
            Option(r.getAs[Int]("TIME_SPENT_ON_SITE")).map(_.toDouble).getOrElse(0.0)))))

    val Array(trainingDataSet, testData) = labeledFeatures.randomSplit(Array(0.7, 0.3), 100)
    val lr = new LogisticRegression().setProbabilityCol("probability")

    val pipelineStages = Array(lr)


    val pipeline = new Pipeline().setStages(pipelineStages)

    println("Training Model...")

    val pipelineModel = pipeline.fit(trainingDataSet)
    val model = pipelineModel.stages.last.asInstanceOf[LogisticRegressionModel]
    val trainingAuc = model.binarySummary.areaUnderROC
    val test = model.transform(testData)
    test.select("features", "label", "probability").show()
    println(s"Training AUC $trainingAuc")
    spark.stop()
  }
}
