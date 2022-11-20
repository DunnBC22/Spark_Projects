// Databricks notebook source
// MAGIC %md
// MAGIC ## Fake Jobs Posting
// MAGIC 
// MAGIC Dataset Source: https://www.kaggle.com/datasets/shivamb/real-or-fake-fake-jobposting-prediction

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
import com.johnsnowlabs.nlp.embeddings.{DistilBertEmbeddings, SentenceEmbeddings}
import com.johnsnowlabs.nlp.base.DocumentAssembler
import com.johnsnowlabs.nlp.annotators.classifier.dl.ClassifierDLApproach

import org.apache.spark.mllib.util.MLUtils

import org.apache.spark.ml.evaluation.{BinaryClassificationEvaluator, MulticlassClassificationEvaluator}

// COMMAND ----------

// MAGIC %md
// MAGIC ##### Import Dataset & Display DataFrame Before Preprocessing

// COMMAND ----------

val file_location = "/FileStore/tables/fake_job_postings.tsv"
val file_type = "csv"

val infer_schema = "false"
val first_row_is_header = "true"
val delimiter = "\t"

var df = spark.read.format(file_type)
  .option("inferSchema", infer_schema)
  .option("header", first_row_is_header)
  .option("sep", delimiter)
  .load(file_location)

df = df.drop(col("job_id"))

print("The number of samples in this dataset is: ", df.count())

display(df)

// COMMAND ----------

// MAGIC %md
// MAGIC ##### Print Schema of DataFrame

// COMMAND ----------

df.printSchema()

// COMMAND ----------

// MAGIC %md
// MAGIC ##### Data Preprocessing

// COMMAND ----------

// location
df = df.na.fill("location not listed. ", Seq("location"))

// department
df = df.na.fill("department not listed. ", Seq("department"))

// salary_range
df = df.na.fill("salary range not listed. ", Seq("salary_range"))

// company_profile
df = df.na.fill("company profile not listed. ", Seq("company_profile"))

// description
df = df.na.fill("job description not listed. ", Seq("description"))

// requirements
df = df.na.fill("job requirements not listed. ", Seq("requirements"))

// benefits
df = df.na.fill("benefits not listed. ", Seq("benefits"))

// telecommuting
df = df.withColumn("telecommuting", when(col("telecommuting")==="0", "telecommuting not available. ")
                   .when(col("has_company_logo")==="1", "telecommuting available. "))

df = df.na.fill("no telecommuting decision listed. ", Seq("telecommuting"))

// has_company_logo
df = df.withColumn("has_company_logo", when(col("has_company_logo")==="0", "company logo not available. ")
                   .when(col("has_company_logo")==="1", "company logo available. "))

df = df.na.fill("no company logo is shown. ", Seq("has_company_logo"))

// has_questions
df = df.na.fill("no questions listed. ", Seq("has_questions"))
df = df.withColumn("has_questions", when(col("has_questions")==="0", "no questions asked. ")
                   .when(col("has_questions")==="1", "questions asked. "))

// employment_type
df = df.na.fill("employment type not listed. ", Seq("employment_type"))

// required_experience
df = df.na.fill("experience requirement not listed. ", Seq("required_experience"))

// required_education
df = df.na.fill("education requirement not listed. ", Seq("required_education"))

// industry
df = df.na.fill("industry not listed. ", Seq("industry"))

// function
df = df.na.fill("job function not listed. ", Seq("function"))

// fraudulent
val fraud_labels = List(0,1)
df = df.filter(($"fraudulent").isin(fraud_labels:_*))
//df = df.filter($"genre".isin(cols_to_keep:_*))
df = df.withColumnRenamed("fraudulent", "label")

// title
df = df.na.drop(Seq("title"))

// Trim Leading & Trailing Whitespace for each feature
for (col_name <- df.columns){
    df = df.withColumn(col_name, trim(col(col_name)))
}
display(df)

// COMMAND ----------

// MAGIC %md
// MAGIC ##### Concatenate the Outputs of Features into One Feature, Named "Fraudulent"

// COMMAND ----------

df = df.select(col("label").cast("int").alias("label"),
              concat_ws(" ", col("title"), col("location"), col("department"), col("salary_range"), col("company_profile"), col("description"), col("requirements"), col("benefits"), col("telecommuting"), col("has_company_logo"), col("has_questions"), col("employment_type"), col("required_experience"), col("required_education"), col("industry"), col("function")).as("text"))

// COMMAND ----------

// MAGIC %md
// MAGIC ##### Display DataFrame After Preprocessing Steps

// COMMAND ----------

df = df.withColumn("text_length", size(split(col("text"), " ")))
display(df)

// COMMAND ----------

// MAGIC %md
// MAGIC ##### Check DataFrame Schema

// COMMAND ----------

println(df.printSchema())

// COMMAND ----------

// MAGIC %md
// MAGIC ##### Balance Dataset Classes (Outputs)

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
// MAGIC ##### Check Impact of Balancing Dataset

// COMMAND ----------

println(balanced_ds.printSchema())

val duplicates = balanced_ds.groupBy("text").count.filter("count > 1").sort(col("count").desc).show(50)

// COMMAND ----------

// MAGIC %md
// MAGIC ##### Reduce Number of Samples in Dataset

// COMMAND ----------

df = balanced_ds.filter(col("text_length") < 401)
df = df.drop(col("text_length"))

df = df.sample(0.30)

println(df.count())

// COMMAND ----------

// MAGIC %md
// MAGIC ##### Check Changes So Far

// COMMAND ----------

display(df)

// COMMAND ----------

// MAGIC %md
// MAGIC ##### Split Dataset into Training & Testing Datasets (80/20)

// COMMAND ----------

val Array(train_ds, test_ds) = df.randomSplit(weights=Array(0.80, 0.20), seed=42)

println("Training Dataset:" + train_ds.count())
println("Training Dataset:" + test_ds.count())

// COMMAND ----------

// MAGIC %md
// MAGIC ##### Basic Values/Constants

// COMMAND ----------

val NUM_OF_EPOCHS : Int = 2
val BATCH_SIZE : Int = 64
val MAX_INPUT_LEN : Int = 511
val LR : Float = 5e-3f
val VERBOSITY_LEVEL : Int = 2

// COMMAND ----------

// MAGIC %md
// MAGIC ##### Create Pipeline Stages

// COMMAND ----------

// DocumentAssembler
val doc = new DocumentAssembler()
    .setInputCol("text")
    .setOutputCol("document")

// Tokenizer
val tokenizer = new Tokenizer()
    .setInputCols(Array("document"))
    .setOutputCol("token")

// Bert Embeddings
val bert_embeds = DistilBertEmbeddings.pretrained()
    .setInputCols(Array("document", "token"))
    .setOutputCol("embeds")
    .setMaxSentenceLength(MAX_INPUT_LEN)

// Sentence Embeddings
val sent_embeds = new SentenceEmbeddings()
    .setInputCols(Array("document", "embeds"))
    .setOutputCol("sent_embeds")
    .setPoolingStrategy("AVERAGE")

// Classifier
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
// MAGIC ##### Create Pipeline

// COMMAND ----------

val lf_clf_pipeline = new Pipeline().setStages(Array(
    doc, 
    tokenizer, 
    bert_embeds, 
    sent_embeds,
    clf
))

// COMMAND ----------

// MAGIC %md
// MAGIC ##### Fit Model from Pipeline

// COMMAND ----------

val lf_clf_model = lf_clf_pipeline.fit(train_ds)

// COMMAND ----------

// MAGIC %md
// MAGIC ##### Inference: Predict (& Display) Using Test Dataset

// COMMAND ----------

var preds = lf_clf_model.transform(test_ds)

display(preds)

// COMMAND ----------

// MAGIC %md
// MAGIC ##### Convert Predictions to Pandas DataFrame & Condense to Only Necessary Features/Classes

// COMMAND ----------

var predictions_in_pandas = (preds.select(col("text").as("text"), col("label").as("ground_truth"), col("class.result").as("prediction"))).toDF()

predictions_in_pandas["prediction"] = predictions_in_pandas["prediction"].apply(lambda x : x[0])

display(predictions_in_pandas)

// COMMAND ----------

// MAGIC %md
// MAGIC ##### Evaluation of Metrics

// COMMAND ----------

acc_evaluator = BinaryClassificationEvaluator(
    labelCol="label", predictionCol="prediction", metricName="accuracy")
accuracy = acc_evaluator.evaluate(predictions_in_pandas)
print("Accuracy = %g" % (accuracy))

prec_evaluator = MulticlassClassificationEvaluator(
    labelCol="label", predictionCol="prediction", metricName="weightedPrecision")
wprecision = prec_evaluator.evaluate(predictions_in_pandas)
print("Weighted Precision = %g" % (wprecision))

rec_evaluator = MulticlassClassificationEvaluator(
    labelCol="label", predictionCol="prediction", metricName="weightedRecall")
wrecall = rec_evaluator.evaluate(predictions_in_pandas)
print("Weighted Recall = %g" % (wrecall))

// COMMAND ----------

// MAGIC %md
// MAGIC #### Notes & Other Takeaways From This Project
// MAGIC ****
// MAGIC - I admit that the variable naming conventions used are not the typical Java/Scala naming conventions. I used the Python scripts as a starting point for many of these Scala projects. I just completely goofed and forgot to change from snake casing to camel-casing. Since these projects can take some time to train and I want to move on to new topics, I am noting it here and will make sure to use the proper naming conventions in the future.
// MAGIC ****
// MAGIC - I had to reduce the size of the dataset to conform with the limitations of the Community Edition of Databricks. I would rather have something trained to demonstrate my abilities than nothing at all. I am aware of this. Some of the ways I reduced the dataset size were to eliminate overly length inputs and random sample the remaining samples.
// MAGIC ****
// MAGIC - My philosophy when learning a new technology is to start with a 'no-frills' project. With each project that I complete using that same technology, I work to improve the quality and difficulty. 
// MAGIC ****
// MAGIC - One way that I improved projects with Apache Spark and Databricks is that I have converted projects to use Scala (in addition to projects that use Python [PySpark]).
// MAGIC ****
// MAGIC - Some items that I am currently working on including are: type checking and wrapping code in functions.
// MAGIC ****
// MAGIC - While the accuracy of the model is not what I hoped it would be, I am confident that if I could run the full dataset on more than 2 epochs, the results would improve immensely.
// MAGIC ****
