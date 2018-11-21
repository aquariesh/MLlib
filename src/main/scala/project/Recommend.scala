package project

import org.apache.spark.SparkConf
import org.apache.spark.ml.evaluation.RegressionEvaluator
import org.apache.spark.ml.recommendation.{ALS, ALSModel}
import org.apache.spark.ml.recommendation.ALS.Rating
import org.apache.spark.sql.SparkSession

object Recommend {
  def main(args: Array[String]): Unit = {
    val conf = new SparkConf().setMaster("local").setAppName("Recommend")
    val spark = SparkSession.builder().config(conf).getOrCreate()
    spark.sparkContext.setLogLevel("WARN")

    val parseRating = (string: String) =>{
      val stringArray = string.split("\t")
      Rating(stringArray(0).toInt,stringArray(1).toInt,stringArray(2).toFloat)
    }
    import spark.implicits._
    val data = spark.read.textFile("/Users/wangjx/data/u.data")
      .map(parseRating)
      .toDF("userID","itemID","rating")
    //data.show(false)
    val Array(traing,test) = data.randomSplit(Array(0.8,0.2))
    /**
      * ALS用于矩阵分解 分解协同过滤中复杂矩阵
      */
    val als = new ALS()
      .setMaxIter(20)
      .setUserCol("userID")
      .setItemCol("itemID")
      .setRatingCol("rating")
      .setRegParam(0.01)//正则化参数

    val model = als.fit(traing)
    model.setColdStartStrategy("drop")//冷启动策略，这是推荐系统的一个重点内容哦～

    val predictions = model.transform(test)
    predictions.show(false)//根据(userID,itemID)预测rating


    //MovieLens
    val users = spark.createDataset(Array(196)).toDF("userID")
    users.show(false)
    model.recommendForUserSubset(users,10).show(false)//想一想工业实践该怎么结合这段代码？

    //模型评估
    val evaluator = new RegressionEvaluator()
      .setMetricName("rmse")
      .setLabelCol("rating")
      .setPredictionCol("prediction")
    val rmse = evaluator.evaluate(predictions)
    println(s"Root-mean-square error is $rmse \n")

    //Spark机器学习模型的持久化
    //模型保存
    //model.save("./xxx")
    //模型加载
    //val model = ALS.load("xxxx")


  }
}

