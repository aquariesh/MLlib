package project

import org.apache.spark.ml.classification.NaiveBayes
import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator
import org.apache.spark.ml.feature.{HashingTF, IDF}
import org.apache.spark.sql.SparkSession

/**
  * 文本情感分类demo
  */
object SentimentAnalysis {
  def main(args: Array[String]): Unit = {
    val spark = SparkSession.builder()
      .appName("SentimentAnalysis")
      .master("local")
      .getOrCreate()
    spark.sparkContext.setLogLevel("WARN")

    import spark.implicits._
    val rand = new util.Random()
    val neg = spark.read.textFile("/Users/wangjx/data/neg.txt").map(line =>{
      (line.split(" ").filter(!_.equals(" ")),0,rand.nextDouble())
    }).toDF("words","value","random")
    val pos = spark.read.textFile("/Users/wangjx/data/pos.txt").map(line =>{
      (line.split(" ").filter(!_.equals(" ")),1,rand.nextDouble())
    }).toDF("words","value","random")
    neg.show()
    pos.show()
    val data = neg.union(pos).sort("random")

    //文本特征提取
    val hashingTf = new HashingTF()
      .setInputCol("words")
      .setOutputCol("hashing")
      .transform(data)
    val idfModel = new IDF()
      .setInputCol("hashing")
      .setOutputCol("tfidf")
      .fit(hashingTf)
    val transformedData = idfModel
      .transform(hashingTf)
    val Array(training,test) =transformedData.randomSplit(Array(0.7,0.3))
    //根据抽取到的文本特征，使用分类器进行分类，这是一个二分类问题
    //分类器是可替换的
    val bayes = new NaiveBayes()
      .setFeaturesCol("tfidf") //X
      .setLabelCol("value")    //y
      .fit(training)
    val result = bayes.transform(test) //交叉验证
    result.show(false)

    //对模型的准确率进行评估
    val evaluator = new MulticlassClassificationEvaluator()
      .setLabelCol("value")
      .setPredictionCol("prediction")
      .setMetricName("accuracy")
    val accuracy = evaluator.evaluate(result)
    println(s"""accuracy is $accuracy""")
  }
}
