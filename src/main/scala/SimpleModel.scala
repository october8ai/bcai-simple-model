import org.apache.spark.ml.classification.{LogisticRegression, LogisticRegressionModel}
import org.apache.spark.ml.feature.LabeledPoint
import org.apache.spark.ml.linalg.Vectors
import org.apache.spark.ml.{Pipeline, PipelineStage}
import org.apache.spark.sql.functions._
import org.apache.spark.sql.{Dataset, SparkSession}

import scala.collection.mutable

object SimpleModel {
  def main(args: Array[String]) {

    val spark = SparkSession.builder()
      .appName("Spark Interview")
      .master("local[4]")
      .getOrCreate()

    import spark.implicits._

    val eventsFile = "src/main/resources/events.csv"
    val conversionsFile = "src/main/resources/conversions.csv"
    val events = spark
      .read
      .option("header", "true")
      .csv(eventsFile)
      .cache()

    val conversions = spark
      .read
      .option("header", "true")
      .csv(conversionsFile)
      .withColumn("converted", lit(1.0))
      .cache()

    val stages = new mutable.ArrayBuffer[PipelineStage]()

    val dataset: Dataset[LabeledPoint] =
      events.join(conversions, Seq("ID"), "left")
        .map(r => LabeledPoint(
          r.getAs[Double]("converted"),
          Vectors.dense(Array(Option(r.getAs[String]("HOTEL_CITY_ID")).map(_.toDouble).getOrElse(0.0),
            Option(r.getAs[String]("TIME_TO_ARRIVAL")).map(_.toDouble).getOrElse(0.0),
            Option(r.getAs[String]("TIME_SPENT_ON_SITE")).map(_.toDouble).getOrElse(0.0)))))

    val Array(trainingDataSet, testData) = dataset.randomSplit(Array(0.7, 0.3), 100)
    val lr = new LogisticRegression().setProbabilityCol("probability")

    stages += lr

    val pipeline = new Pipeline().setStages(stages.toArray)
    val pipelineModel = pipeline.fit(trainingDataSet)
    val model = pipelineModel.stages.last.asInstanceOf[LogisticRegressionModel]
    val trainingAuc = model.binarySummary.areaUnderROC
    val test = model.transform(testData)
    test.select("features", "label", "probability").show()
    println(s"Training AUC $trainingAuc")
    spark.stop()
  }
}
