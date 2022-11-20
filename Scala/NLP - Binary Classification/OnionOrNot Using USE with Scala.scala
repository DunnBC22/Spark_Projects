// Databricks notebook source
// MAGIC %md
// MAGIC ## OnionOrNot Using USE with Scala
// MAGIC 
// MAGIC Project Objective: to correctly classify if a title is for an Onion news article or not.  
// MAGIC 
// MAGIC Dataset Source: https://www.kaggle.com/datasets/chrisfilo/onion-or-not

// COMMAND ----------

// MAGIC %md
// MAGIC ##### Import Necessary Libraries

// COMMAND ----------

import spark.implicits._
import com.johnsnowlabs.nlp.annotators.Tokenizer
import org.apache.spark.ml.Pipeline

import org.apache.spark.sql.{DataFrame, Column}
import org.apache.spark.sql.functions.{col, element_at, explode, when, split, trim, concat_ws, size, count, sum, filter}
import org.apache.spark.util.random
import com.johnsnowlabs.nlp.base.DocumentAssembler
import com.johnsnowlabs.nlp.annotators.classifier.dl.ClassifierDLApproach
import com.johnsnowlabs.nlp.embeddings.UniversalSentenceEncoder
import com.johnsnowlabs.nlp.annotator.SentenceDetector

import org.apache.spark.mllib.util.MLUtils

import org.apache.spark.ml.evaluation.{BinaryClassificationEvaluator, MulticlassClassificationEvaluator}
import org.apache.spark.mllib.evaluation.BinaryClassificationMetrics
import org.apache.spark.mllib.util.MLUtils

// COMMAND ----------

// MAGIC %md
// MAGIC ##### Ingest & Preprocess Data

// COMMAND ----------

// File location and type
val file_location = "/FileStore/tables/OnionOrNot.csv"
val file_type = "csv"

// CSV options
val infer_schema = "false"
val first_row_is_header = "true"
val delimiter = ","

// The applied options are for CSV files. For other file types, these will be ignored.
var df = spark.read.format(file_type)
  .option("inferSchema", infer_schema)
  .option("header", first_row_is_header)
  .option("sep", delimiter)
  .load(file_location)

df = df.where("label==0 or label==1")

df = df.na.drop() 

display(df)

// COMMAND ----------

// MAGIC %md
// MAGIC ##### Check for Unique Values in Label Feature

// COMMAND ----------

var dropDisDF = df.dropDuplicates(Array("label")).select("label")
dropDisDF.show()

// COMMAND ----------

// MAGIC %md
// MAGIC ##### Calculate & Display Histogram of Length of Text (Feature)

// COMMAND ----------

df = df.withColumn("text_length", size(split(col("text"), " ")))
display(df)

// COMMAND ----------

// MAGIC %md 
// MAGIC ##### Remove Class Imabalance

// COMMAND ----------

def oversample(dataset : DataFrame, label_col: String) : DataFrame = {
  val minor_df : DataFrame = df.filter(col(label_col) === 0)
  val major_df : DataFrame = df.filter(col(label_col) === 1)
  
  var minor_count : Double = (minor_df.count().asInstanceOf[Double])
  var major_count : Double = (major_df.count().asInstanceOf[Double])
  
  val ratio : Double = (minor_count/major_count)
  println("ratio: " + ratio)
  
  val major_resampled = major_df.sample(withReplacement=true, fraction=ratio, seed=42)
  val bal_df = minor_df.unionAll(major_resampled)
  
  return bal_df;
}

var balanced_ds = oversample(df, "label")
balanced_ds.groupBy("label").count().show()

// COMMAND ----------

// MAGIC %md
// MAGIC ##### Trim Overly Lengthy Samples from Dataset & Final Preprocessing/Cleanup

// COMMAND ----------

df = balanced_ds.filter(col("text_length") < 36)
df = df.drop(col("text_length"))
print(df.count())

// COMMAND ----------

// MAGIC %md
// MAGIC ##### Print Number of Samples in Dataset & Dataset Schema

// COMMAND ----------

df.printSchema()
df.count()

// COMMAND ----------

// MAGIC %md
// MAGIC ##### Split Dataset into Training & Testing Dataset

// COMMAND ----------

val Array(train_ds, test_ds) = df.randomSplit(weights=Array(0.80, 0.20), seed=42)

print(train_ds.count())
print(test_ds.count())

// COMMAND ----------

// MAGIC %md
// MAGIC ##### Basic Values/Constants

// COMMAND ----------

val NUM_OF_EPOCHS : Int = 15
val BATCH_SIZE : Int = 128
val MAX_INPUT_LEN : Int = 45
val LR : Float = 5e-3f
val VERBOSITY_LEVEL : Int = 1

// COMMAND ----------

// MAGIC %md
// MAGIC ##### Define Piepline Stages

// COMMAND ----------

// DocumentAssembler
val doc = new DocumentAssembler()
    .setInputCol("text")
    .setOutputCol("document")

// Universal Sentence Encoder
val use = UniversalSentenceEncoder.pretrained("tfhub_use", "en")
    .setInputCols("document")
    .setOutputCol("sent_embeds")

// clf_model
val clf = new ClassifierDLApproach()
    .setInputCols(Array("sent_embeds"))
    .setOutputCol("class")
    .setLabelColumn("label")
    .setBatchSize(BATCH_SIZE)
    .setLr(LR)
    .setMaxEpochs(NUM_OF_EPOCHS)
    .setVerbose(VERBOSITY_LEVEL)

// COMMAND ----------

// MAGIC %md
// MAGIC ##### Build Pipeline for Training

// COMMAND ----------

val nlp_clf_pipeline = new Pipeline().setStages(Array(
    doc, use, clf
))

// COMMAND ----------

// MAGIC %md
// MAGIC ##### Fit/Train Model

// COMMAND ----------

val nlp_clf_model = nlp_clf_pipeline.fit(train_ds)

// COMMAND ----------

// MAGIC %md
// MAGIC ##### Inference: Predictions Using Testing Dataset

// COMMAND ----------

val predictions = nlp_clf_model.transform(test_ds)

var predictions_in_pandas = (predictions.select(col("text").as("text"), col("label").as("ground_truth"), col("class.result").as("prediction"))).toDF()

predictions_in_pandas = predictions_in_pandas.withColumn("prediction", element_at($"prediction", 1))

predictions_in_pandas = predictions_in_pandas.withColumn("prediction", col("prediction").cast("double"))
                    .withColumn("ground_truth", col("ground_truth").cast("double"))

display(predictions_in_pandas)

// COMMAND ----------

// MAGIC %md
// MAGIC ##### Calculate Metrics

// COMMAND ----------

val roc_metric =  new BinaryClassificationEvaluator()
  .setLabelCol("ground_truth")
  .setRawPredictionCol("prediction")
  .setMetricName("areaUnderROC")

val pr_metric =  new BinaryClassificationEvaluator()
  .setLabelCol("ground_truth")
  .setRawPredictionCol("prediction")
  .setMetricName("areaUnderPR")

val areaUnderROC = roc_metric.evaluate(predictions_in_pandas)

val PR = pr_metric.evaluate(predictions_in_pandas)

// COMMAND ----------

// MAGIC %md
// MAGIC ##### Calculate More Metrics

// COMMAND ----------

val accuracy_eval = new MulticlassClassificationEvaluator()
  .setLabelCol("ground_truth")
  .setPredictionCol("prediction")
  .setMetricName("accuracy")

val precision_eval = new MulticlassClassificationEvaluator()
  .setLabelCol("ground_truth")
  .setPredictionCol("prediction")
  .setMetricName("weightedPrecision")

val recall_eval = new MulticlassClassificationEvaluator()
  .setLabelCol("ground_truth")
  .setPredictionCol("prediction")
  .setMetricName("weightedRecall")

val fl_eval = new MulticlassClassificationEvaluator()
  .setLabelCol("ground_truth")
  .setPredictionCol("prediction")
  .setMetricName("f1")
  .setBeta(0.5)

val accuracy = accuracy_eval.evaluate(predictions_in_pandas)
val precision = precision_eval.evaluate(predictions_in_pandas)

val recall = recall_eval.evaluate(predictions_in_pandas)
val f1 = fl_eval.evaluate(predictions_in_pandas)

// COMMAND ----------

// MAGIC %md
// MAGIC ##### Display Metrics

// COMMAND ----------

val metrics = Map(
  "accuracy" -> accuracy,
  "precision" -> precision, 
  "recall" -> recall,
  "f1 Score" -> f1,
  "ROC Score" -> areaUnderROC,
  "PR Curve" -> PR
)

println("+---------------------------------------+")
println("| Metric \t| Value \t\t|")
println("+---------------------------------------+")

for ((metric_name, metric_value) <- metrics) {
  println(s"| $metric_name\t| $metric_value\t|")
  println("+---------------------------------------+")
}

println()

// COMMAND ----------

// MAGIC %md
// MAGIC ### Notes & Other Takeaways
// MAGIC ****
// MAGIC - Normally, I would only post the HTML version of the project on GitHub, but the file size is too large for it to display in GitHub. I have posted both versions. If the HTML version does not load and you would like to see the HTML Version, feel free to ask me. I would be elated to share that copy with you!
// MAGIC ****
// MAGIC - Unfortunately, this model did not turn out as well as the HuggingFace Trainer API-trained model that used this same dataset; however, it did train much faster! 
// MAGIC ****
