<h1>Spark Projects</h1>

<p>
    This repository houses all of my Apache Spark projects. They were all completed using Databricks.
</p>

<details open>

<summary>Natural Language Processing</summary>

<br />

<caption>
    All NLP projects were completed using John Snow's open source NLP Algorithms. You can find more information here: <a href="https://www.johnsnowlabs.com/">https://www.johnsnowlabs.com/</a>. 
</caption>

<h4>
    Binary Text Classification
</h4>

| Project Name | Sentence Embedder/ Encoder | Transformer Used | Accuracy | Macro Precision | Macro Recall | Macro F1-Score |
|  :----------: | :----------: | :----------: | :----------: | :----------: | :----------: | :----------: |
| Clickbait Classification (Part 1) | `Universal Sentence Encoder` | `Classifier DL Approach` | 0.97 | 0.97 | 0.97 | 0.97 |
| Clickbait Classification (Part 2) | Regular Built-In Tokenizer | `BERT Sequence Classifier` | 1.0 | 1.0 | 1.0 | 1.0 |
| Clickbait Classification (Part 3) | `BERT Sequence Classifier` | `Classifier DL Approach` | 0.98 | 0.98 | 0.98 | 0.98 |
| Is There Depression in This Reddit Post? | `Universal Sentence Encoder` | `Classifier DL Approach` | 0.97 | 0.97 | 0.97 | 0.97 |
| Onion Or Not | `Universal Sentence Encoder` | `Classifier DL Approach` | 0.87 | 0.87 | 0.87 | 0.87 |
| Onion Or Not with Extra Stages[^1] | `Universal Sentence Encoder` | `Classifier DL Approach` | 0.86 | 0.86 | 0.86 | 0.86 |
| Real vs Fake News (Pretrained Model)[^3] |  `Universal Sentence Encoder` | `Classifier DL Model`[^2] | 0.608 | 0.605 | 0.610 | 0.608 |
| Real vs Fake News (Deep Learning Approach)[^3] |  `Universal Sentence Encoder` | `Classifier DL Approach` | 0.975 | 0.975 | 0.975 | 0.975 |
| Sarcasm Detection | `Universal Sentence Encoder` | `Classifier DL Approach` | 0.89 | 0.89 | 0.89 | 0.89 |
| Spam Filter | `Universal Sentence Encoder` | `Classifier DL Approach` | 0.98 | 0.97 | 0.96 | 0.96 |

<h4>
    Multiclass Text Classification
</h4>

| Project Name | Sentence Embedder/ Encoder | Transformer Used | Accuracy | Macro Precision | Macro Recall | Macro F1-Score |
| :----------: | :----------: | :----------: | :----------: | :----------: | :----------: |:----------: |
| CNN News Articles | `Sentence Embeddings`[^4] | `Classifier Deep Learning Approach` | 0.72 | 0.87 | 0.47 | 0.55 |
| CNN News Articles v2 | `Sentence Embeddings`[^4] | `Classifier Deep Learning Approach` | 0.75 | 0.83 | 0.47 | 0.55 |
| Cancer Classification (After Removing Class Imbalance)[^3] | `Universal Sentence Encoder` | `Classifier Deep Learning Approach` | 0.854 | 0.853 | 0.862 | 0.854 |
| Cancer Classification (Without Removing Class Imbalance)[^3] | `Universal Sentence Encoder` | `Classifier Deep Learning Approach` | 0.863 | 0.862 | 0.862 | 0.863 |
| Cyberbullying Classification | `Universal Sentence Encoder` | `Classifier Deep Learning Approach` | 0.82 | 0.81 | 0.82 | 0.81 |
| Ford Sentence Classification | `Universal Sentence Encoder` | `Classifier Deep Learning Approach` | 0.74 | 0.73 | 0.73 | 0.73 |
| IMDb Genres | `Sentence Embeddings`[^4] | `Classifier Deep Learning Approach` | 0.66 | 0.66 | 0.66 | 0.65 |


<h4>
    Multilabel Text Classification
</h4>

| Project Name | Sentence Embedder/ Encoder | Transformer Used | Accuracy | Micro Precision | Micro Recall | Micro F1 Score | Subset Accuracy | Hamming Loss |
| :----------: | :----------: | :----------: |  :----------: | :----------: | :----------: | :----------: | :----------: | :----------: |
| GoEmotions | `Universal Sentence Encoder` | `Multi Classifier DL Approach` | 0.934 | 0.965 | 0.965 | 0.965 | 0.125 | 0.973 |
| Research Articles | `Universal Sentence Encoder` | `Multi Classifier DL Approach` | 0.941 | 0.964 | 0.964 | 0.964 | 0.792 | 0.179 |
| uHack Reviews | `Universal Sentence Encoder` | `Multi Classifier DL Approach` | 0.913 | 0.951 | 0.952 | 0.952 | 0.389 | 0.581 |

<h4>
    Language Detection
</h4>

| Project Name | Transformer Used | Accuracy | F1 | Weighted Precision | Weighted Recall |
| :----------: | :----------: | :----------: | :----------: | :----------: | :----------: |
| All Languages | `Language Detector DL`[^5] | 0.980 | 0.986 | 0.991 | 0.980 |
| Top 5 Languages | `Language Detector DL` | 0.990 | 0.992 | 0.995 | 0.990 |


<h4>
    Machine Translation
</h4>

<p>
    At the time that I completed these projects, I was unable to find the Rouge metric code for Apache Spark. I have since used the Rouge metric with text summarization projects. I encourage you to view those projects.
</p>

<h4>
    Sentiment Analysis
</h4>

| Project Name | Sentence Encoder/Embeddings | Transformer Used | Accuracy | Macro Precision | Macro Recall | Macro F1 |
| :----------: | :----------: | :----------: | :----------: | :----------: | :----------: | :----------: |
| Sentiment Analysis of Reviews | `Universal Sentence Encoder` | `Sentiment DL Model`[^6] | 0.81 | 0.55 | 0.87 | 0.55 |
| Sentiment Analysis of Nearly 600,000 Tweets[^3] | `Universal Sentence Encoder` | `Classifier DL Approach` | 0.749 | 0.748 | 0.749 | 0.748 |
| Twitter Sentiment Analysis | `Universal Sentence Encoder` | `Sentiment DL Model`[^6] | 0.50 | 0.45 | 0.50 | 0.42 |

<h4>
    Text Summarization
</h4>

| Project Name | Transformer Used | Rouge1 | Rouge | RougeL | RougeLsum |
| :----------: | :----------: | :----------: | :----------: | :----------: | :----------: |
| CNN News Articles | `T5 Transformer` (`t5_small`) | 36.4 | 22.2 | 30.7 | 30.7 |

</details>

<br />

<details>

<summary>Structured Data</summary>

<h4>
    Binary Classification
</h4>

| Project Name | Classifier Used | Accuracy |  Macro F1 | Macro Precision | Macro Recall | Best Algorithm |
| :----------: | :----------: | :----------: | :----------: | :----------: | :----------: | :----------: |
| Banking Campaign[^3] | `GBTClassifier` (Gradient Boosted Tree) | 0.885 | 0.885 | 0.887 | 0.885 | - |
| Car Insurance Claim Predictor[^3] | `MultilayerPerceptronClassifier` | 0.502 | 0.335 | 0.252 | 0.502 | - |
| Car Insurance Claim Predictor (Class Imbalance Removed)[^3] | `RandomForestClassifier` | 0.627 | 0.626 | 0.628 | 0.627 | - |
| Car Insurance Claim Predictor[^3] | `RandomForestClassifier` | 0.935 | 0.904 | 0.875 | 0.935 | - |
| Diabetes Health Indicators (v1) | - | 0.90 | 0.90 | 0.90 | 0.90 | `GBTClassifier` |
| Diabetes Health Indicators (v2) | - | 0.90 | 0.90 | 0.90 | 0.90 | `GBTClassifier` |

<h4>
    Multiclass Classification
</h4>

| Project Name | Accuracy | Macro F1 | Macro Precision | Macro Recall | Best Algorithm |
| :----------: | :----------: | :----------: | :----------: | :----------: | :----------: |
| Mobile Phone Price Classification (v1)[^7] | 0.88 | 0.88 | 0.89 | 0.89 | `DecisionTreeClassifier` |
| Mobile Phone Price Classification (v2)[^7] | 0.88 | 0.88 | 0.89 | 0.89 | `DecisionTreeClassifier` |

<h4>
    Regression
</h4>

| Project Name | Algorithm Used | Root Mean Squared Error (RMSE) |
| :----------: | :----------: | :----------: |
| Predict AI & ML Salaries Predictor | `GBTRegressor` | $58932.71 |
| Absenteeism at Work | `GBTRegressor` | 0.789 hours |
| Email Click Through Rate Predictor | `GBTRegressor` | 0.044 |
| Data-Related Salaries Predictor | `GBTRegressor` | $57073 |

<h4>
    Word Cloud
</h4>

<caption>
    This is a word cloud of the most popular words used in the script from 'The Office.'
</caption>

<img src="https://github.com/DunnBC22/Spark_Projects/raw/main/Python%20(PySpark)/Structured%20Data/WordCloud/The%20Office%20Script%20WordCloud.png">

</details>

<br />

<details>

<summary>Computer Vision/Image Classification</summary>

<br />

<h4>
    Image Classification
</h4>

| Project Name | Pretrained Model or Untuned Checkpoint | Accuracy | F1 Score | Weighted Precision | Weighted Recall |
| :---: | :---: | :---: | :---: | :---: | :---: |
| Is It a Cat or a Dog? | Untuned Checkpoint | 0.975 | 0.982229 | 0.990196 | 0.975 |
| Is It a Cat or a Dog? | Pretrained Model | 0.995 | 0.995 | 0.99505 | 0.995 |
| Planes, Cars, & Boats[^3] | Untuned Checkpoint | 0.89 | 0.935331 | 0.994898 | 0.89 |

</details>

<br />

<details>

<summary>All Projects in Scala</summary>

<h4>
    Binary Text Classification
</h4>

| Project Name | Transformer Used | Accuracy | F1 Score | Precision | Recall | PR Score | ROC Score |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| Fake Job Postings | `ClassifierDLApproach` | - | - | - | - | - | - |
| Onion or Not | `ClassifierDLApproach` | 0.854 | 0.854 | 0.855 | 0.854 | 0.823 | 0.854 |
| Spam Filter | `ClassifierDLModel`[^8] | 0.986 | 0.986 | 0.986 | 0.987 | 0.936 | 0.965 |

<h4>
    Machine Translation
</h4>

<p>
    The Scala version of the Machine Translation projects were completed at about the same time as the Python version, so I was unable to find the Rouge metric code for these projects using Apache Spark. I have since used the Rouge metric with text summarization projects. I encourage you to view those projects.
</p>

</details>

<br />

Footnotes:

[^1]: The extra stages included in this project are: Sentence Detector (Deep Learning Model), Tokenizer (regular built in Tokenizer), Stop Words Cleaner, Spell Checker, Lemmatizer, and Token Assembler. Whereas most of my projects had three (3) stages, this project had nine (9).

[^2]: The pretrained model used was: classifierdl_use_fakenews.

[^3]: Regrettably, I did not include Macro Averaged versions of the metrics in these projects.

[^4]: Almost every instance of the SentenceEmbeddings sentence embedding is preceeded by a DistilBert Embeddings Stage (DistilBertEmbeddings).

[^5]: The pretrained model used was: ld_wiki_tatoeba_cnn_95.

[^6]: The pretrained model was: sentimentdl_use_twitter.

[^7]: Even though the Decision Tree Classifier performed best (I ran the project a couple times to check), I understand that it is likely due to lucky sampling. There is likelihood for bias in the outcome of the Decision Tree Classifier.

[^8]: The pretrained model was: classifierdl_use_spam.

<hr />

<h4>
    Additional Notes About This Repository
</h4>

<ul>
    <li>
        I noticed that some of the HTML file versions of projects were 'too large to load', so I included the Python Notebook (ipynb) versions as well.
    </li>
    <li>
        Unfortunately, I forgot to evaluate the training datasets to compare with the testing datasets to make sure that I did not overtrain projects. Due to the large number of projects, it is impractical to go back and retrain all of them just to include the evaluation of the training datasets. I will make sure to include that information going forward in new Spark projects.
    </li>
    <li>
        If there is a topic that you are interesting in seeing if I have completed any work with, feel free to reach out to me and ask.
    </li>