// Databricks notebook source
// MAGIC %md
// MAGIC 
// MAGIC ## Spam Filter
// MAGIC 
// MAGIC Dataset Source: https://www.kaggle.com/datasets/uciml/sms-spam-collection-dataset

// COMMAND ----------

// MAGIC %md
// MAGIC ##### Import Necessary Libraries

// COMMAND ----------

import spark.implicits._
import com.johnsnowlabs.nlp.annotators._ 
import org.apache.spark.ml.Pipeline

import com.johnsnowlabs.nlp.annotators.classifier.dl

import org.apache.spark.sql.functions.{col, element_at, when}
import com.johnsnowlabs.nlp.base.DocumentAssembler
import com.johnsnowlabs.nlp.annotators.classifier.dl.ClassifierDLModel
import com.johnsnowlabs.nlp.embeddings.UniversalSentenceEncoder

import org.apache.spark.ml.evaluation.{BinaryClassificationEvaluator, MulticlassClassificationEvaluator}

// COMMAND ----------

// MAGIC %md
// MAGIC ##### Data Ingestion & Data Preprocessing

// COMMAND ----------

// File location and type
val file_location = "/FileStore/tables/spam.csv"
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

df = df.withColumnRenamed("v1", "label").withColumnRenamed("v2", "text")

df = df.where("label in ('ham', 'spam')")

df = df.drop("_c2").drop("_c3").drop("_c4")

df = df.withColumn("label", when(col("label") === "ham", 0)
                   .when(col("label") === "spam", 1))

display(df)

// COMMAND ----------

// MAGIC %md
// MAGIC ##### Display Unique Label Values

// COMMAND ----------

val unique_label_values = df.dropDuplicates("label").select("label")
unique_label_values.show()

// COMMAND ----------

// MAGIC %md
// MAGIC ##### Split Dataset into Training & Testing Dataset

// COMMAND ----------

val Array(train_ds, test_ds) = (df.randomSplit(Array(0.65,0.35), seed=42))

// COMMAND ----------

// MAGIC %md
// MAGIC ##### Instantiate Pipeline Stages

// COMMAND ----------

val doc = new DocumentAssembler()
  .setInputCol("text")
  .setOutputCol("document")

val use = UniversalSentenceEncoder.pretrained("tfhub_use", lang="en")
  .setInputCols(Array("document"))
  .setOutputCol("sent_embeds")

val document_classifier = ClassifierDLModel.pretrained("classifierdl_use_spam", "en")
  .setInputCols(Array("sent_embeds"))
  .setOutputCol("class")

// COMMAND ----------

// MAGIC %md
// MAGIC ##### Build Pipeline

// COMMAND ----------

val spam_nlp_pipeline = new Pipeline().setStages(Array(doc, use, document_classifier))

// COMMAND ----------

// MAGIC %md
// MAGIC ##### Fit/Train Model

// COMMAND ----------

val spam_nlp_model = spam_nlp_pipeline.fit(train_ds)

// COMMAND ----------

// MAGIC %md
// MAGIC ##### Predict Values Based on Testing Dataset

// COMMAND ----------

val predictions = spam_nlp_model.transform(test_ds)

// COMMAND ----------

// MAGIC %md
// MAGIC ##### Display Predictions

// COMMAND ----------

display(predictions)

// COMMAND ----------

// MAGIC %md
// MAGIC ##### Convert Necessary Features to Pandas DataFrame

// COMMAND ----------

var predictions_in_pandas = predictions.select(col("label").as("ground_truth"), col(raw"class.result").as("prediction")).toDF()

display(predictions_in_pandas)

// COMMAND ----------

// MAGIC %md
// MAGIC ##### Instantiate & Display Classification Report

// COMMAND ----------

predictions_in_pandas = predictions_in_pandas.withColumn("prediction", element_at($"prediction", 1))

predictions_in_pandas = predictions_in_pandas.withColumn("prediction", when(col("prediction") === "ham", 0)
                   .when(col("prediction") === "spam", 1))

predictions_in_pandas = predictions_in_pandas.withColumn("prediction", col("prediction").cast("Double"))

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

println("+------------------------------------+")
println("| Metric \t| Value \t     |")
println("+------------------------------------+")

for ((metric_name, metric_value) <- metrics) {
  println(s"| $metric_name\t| $metric_value |")
  println("+------------------------------------+")
}

println()

// COMMAND ----------

// MAGIC %md
// MAGIC ### Notes & Other Takeaways
// MAGIC ****
// MAGIC - Normally, I would only post the HTML version of the project on GitHub, but the file size is too large for it to display in GitHub. I have posted both versions. If the HTML version does not load and you would like to see the HTML Version, feel free to ask me. I would be elated to share that copy with you!
// MAGIC ****
// MAGIC - Overall, the outcome of this project was excellent. The metrics were similar to the SpamFilter model that I built using the HuggingFace Trainer API; however, this project took considerably less time to train and evaluate.
// MAGIC ****
