// Databricks notebook source
// MAGIC %md
// MAGIC ## Machine Translation - Wikipedia Biographies - English to Spanish (Scala)
// MAGIC 
// MAGIC Project Objective: To tune a model for translation from English to Spanish
// MAGIC 
// MAGIC Dataset Source: https://www.kaggle.com/datasets/paultimothymooney/translated-wikipedia-biographies?select=Translated+Wikipedia+Biographies+-+EN_ES.csv

// COMMAND ----------

// MAGIC %md
// MAGIC ##### Import Necessary Libraries

// COMMAND ----------

import com.johnsnowlabs.nlp.SparkNLP
import com.johnsnowlabs.nlp.annotator._
import com.johnsnowlabs.nlp.base._

import org.apache.spark.sql.functions.col
import org.apache.spark.ml.Pipeline
import org.apache.spark.sql.SparkSession

// COMMAND ----------

// MAGIC %md
// MAGIC ##### Ingest & Start Preprocessing Data

// COMMAND ----------

// File location and type
val file_location = "/FileStore/tables/Translated_Wikipedia_Biographies___EN_DE.tsv"
val file_type = "csv"

// CSV options
val infer_schema = "false"
val first_row_is_header = "true"
val delimiter = "\t"

// The applied options are for CSV files. For other file types, these will be ignored.
var df = spark.read.format(file_type)
  .option("inferSchema", infer_schema)
  .option("header", first_row_is_header)
  .option("sep", delimiter)
  .load(file_location)

df = df.drop("sourceLanguage", "targetLanguage", "documentID", "stringID", "entityName", "sourceURL")

display(df)

// COMMAND ----------

// MAGIC %md
// MAGIC ##### Return Number of Samples in Dataset

// COMMAND ----------

df.count()

// COMMAND ----------

// MAGIC %md
// MAGIC ##### Return Unique Values in 'perceivedGender' Features (& Number of Unique Values)

// COMMAND ----------

val unique_label_vals = df.select("perceivedGender").distinct().count()

println(unique_label_vals)
println(df.select("perceivedGender").distinct().show())

// COMMAND ----------

// MAGIC %md
// MAGIC ##### Filter Columns to Remove Incorrect Samples

// COMMAND ----------

val genders = Array("Female", "Male")

df = df.filter(df("perceivedGender").isin(genders:_*))

var unique_label_vals = df.select("perceivedGender").distinct().count()

println(unique_label_vals)
println(df.select("perceivedGender").distinct().show())

// COMMAND ----------

// MAGIC %md
// MAGIC ##### Return Total Number of Samples in Processed Dataset & Drop Unnecessary Feature

// COMMAND ----------

df = df.drop("perceivedGender")
df.count()

// COMMAND ----------

// MAGIC %md
// MAGIC ##### Split Dataset into Training & Testing Datasets

// COMMAND ----------

val Array(train_ds, test_ds) = df.randomSplit(weights = Array(0.80, 0.20), seed=42)

// COMMAND ----------

// MAGIC %md
// MAGIC ##### Define Pipeline Stages & Pipeline

// COMMAND ----------

val doc = new DocumentAssembler()
    .setInputCol("sourceText")
    .setOutputCol("document")

val sentence = SentenceDetectorDLModel.pretrained("sentence_detector_dl", "xx")
    .setInputCols(Array("document"))
    .setOutputCol("sentence")

val translator = MarianTransformer.pretrained("opus_mt_en_es", "xx")
    .setInputCols(Array("sentence"))
    .setOutputCol("translation")

val en_es_translation_pipeline = new Pipeline().setStages(Array(doc, sentence, translator))

// COMMAND ----------

// MAGIC %md
// MAGIC ##### Fit/Train Model

// COMMAND ----------

val en_es_translation_model = en_es_translation_pipeline.fit(train_ds)

// COMMAND ----------

// MAGIC %md
// MAGIC ##### Inference: Predictions Using Test Dataset

// COMMAND ----------

val preds = en_es_translation_model.transform(test_ds)

// COMMAND ----------

// MAGIC %md
// MAGIC ##### Return Only Necessary Features & Convert to Pandas DataFrame

// COMMAND ----------

val preds_in_pandas = preds.select(col("sourceText").as("source"), col("translatedText").as("ground_truth"), col("translation.result").as("predictions")).toDF()

// COMMAND ----------

// MAGIC %md
// MAGIC ##### Display Condensed Predictions Output

// COMMAND ----------

display(preds_in_pandas)

// COMMAND ----------

// MAGIC %md
// MAGIC ##### Save Model

// COMMAND ----------

en_es_translation_model.save("/FileStore/tables/NLP-Machine-Translation/English_to_Spanish-Translator-Model-in-Scala")

// COMMAND ----------

// MAGIC %md
// MAGIC ### Notes & Other Takeaways
// MAGIC ****
// MAGIC - Normally, I would only post the HTML version of the project on GitHub, but the file size is too large for it to display in GitHub. I have posted both versions. If the HTML version does not load and you would like to see the HTML Version, feel free to ask me. I would be elated to share that copy with you!
// MAGIC ****
// MAGIC - I tried to find a way to evaluate the results of this model, but I was unable to find a way to do so. I tried many different options, but to no avail. If you have any ideas, feel free to reach out and let me know!
// MAGIC ****
