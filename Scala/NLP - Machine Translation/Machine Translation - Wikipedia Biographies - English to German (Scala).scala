// Databricks notebook source
// MAGIC %md
// MAGIC ## Machine Translation - Wikipedia Biographies - English to German (Scala)
// MAGIC 
// MAGIC Project Objective: To tune a model for translation from English to German
// MAGIC 
// MAGIC Dataset Source: https://www.kaggle.com/datasets/paultimothymooney/translated-wikipedia-biographies?select=Translated+Wikipedia+Biographies+-+EN_DE.csv

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
// MAGIC ##### Ingest & Start to Preprocess Data

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

df = df.drop("sourceLanguage", "targetLanguage", "documentID", "entityName", "sourceURL")

display(df)

// COMMAND ----------

// MAGIC %md
// MAGIC ##### Return Number of Samples in Dataset

// COMMAND ----------

df.count()

// COMMAND ----------

// MAGIC %md
// MAGIC ##### Identify Unique Values for 'perceivedGender' Feature (& Total Number of Unqiue Values)

// COMMAND ----------

val unique_label_vals = df.select("perceivedGender").distinct().count()
println(unique_label_vals)
println(df.select("perceivedGender").distinct().show())

// COMMAND ----------

// MAGIC %md
// MAGIC ##### Filter Columns to Remove Incorrect Samples

// COMMAND ----------

val genders: Seq[String] = Seq("Female", "Male")

df = df.filter(df("perceivedGender").isin(genders:_*))

val unique_label_vals = df.select("perceivedGender").distinct().count()

println(unique_label_vals)
println(df.select("perceivedGender").distinct().show())

// COMMAND ----------

// MAGIC %md
// MAGIC ##### Final Data Preprocessing

// COMMAND ----------

df = df.drop("perceivedGender")
df.count()

// COMMAND ----------

// MAGIC %md
// MAGIC ##### Split Dataset into Training & Testing Dataset (75/25)

// COMMAND ----------

val Array(train_ds, test_ds) = df.randomSplit(weights=Array(0.75, 0.25), seed=42)

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

val translator = MarianTransformer.pretrained("opus_mt_en_de", "xx")
    .setInputCols(Array("sentence"))
    .setOutputCol("translation")

val en_de_translation_pipeline = new Pipeline().setStages(Array(doc, sentence, translator))

// COMMAND ----------

// MAGIC %md
// MAGIC ##### Fit/Train Model

// COMMAND ----------

val en_de_translation_model = en_de_translation_pipeline.fit(train_ds)

// COMMAND ----------

// MAGIC %md
// MAGIC ##### Inference: Predictions Using Test Dataset

// COMMAND ----------

val preds = en_de_translation_model.transform(test_ds)

// COMMAND ----------

// MAGIC %md
// MAGIC ##### Return Only Necessary Features & Convert to Pandas DataFrame

// COMMAND ----------

val preds_in_pandas = preds.select(col("sourceText").as("source"), col("translatedText").as("ground_truth"), col("translation.result").as("predictions")).toDF()

// COMMAND ----------

// MAGIC %md
// MAGIC ##### Display Condensed Predictions Results

// COMMAND ----------

display(preds_in_pandas)

// COMMAND ----------

// MAGIC %md
// MAGIC ##### Save Model

// COMMAND ----------

val path = "/FileStore/tables/NLP-Machine-Translation/English_to_German-Translator-Model-in-Scala"
en_de_translation_model.write().overwrite().save(path)

// COMMAND ----------

// MAGIC %md
// MAGIC ### Notes & Other Takeaways
// MAGIC ****
// MAGIC - Normally, I would only post the HTML version of the project on GitHub, but the file size is too large for it to display in GitHub. I have posted both versions. If the HTML version does not load and you would like to see the HTML Version, feel free to ask me. I would be elated to share that copy with you!
// MAGIC ****
// MAGIC - For some reason, this project kept on stopping somwheres near displaying the condensed predictions. Since the code is very similar to the English to Spanish version and that version worked seemlessly, I am confident that the problem is not due to a coding error.
// MAGIC ****
// MAGIC - I tried to find a way to evaluate the results of this model, but I was unable to find a way to do so. I tried many different options, but to no avail. If you have any ideas, feel free to reach out and let me know!
// MAGIC ****
