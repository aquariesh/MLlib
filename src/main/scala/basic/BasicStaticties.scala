package basic

import org.apache.spark.mllib._
import org.apache.spark.mllib.linalg.Vectors
import org.apache.spark.mllib.stat.MultivariateStatisticalSummary
import org.apache.spark.sql.SparkSession

/**
  * 基本统计demo 转化成向量 求个数 最大值 最小值等
  */
object BasicStaticties {
  def main(args: Array[String]): Unit = {
    val spark = SparkSession.builder()
      .master("local")
      .appName("BasicStaticties")
      .getOrCreate()
    val txt = spark.sparkContext.textFile("/Users/wangjx/data/beijing.txt")
    val data = txt.flatMap(_.split(","))
      .map(valie => Vectors.dense(valie.toDouble))
    // colStats（）返回一个MultivariateStatisticalSummary的实例，它包含列的最大值，最小值，平均值，方差和非零序数，以及总计数
    val result:MultivariateStatisticalSummary = stat.Statistics.colStats(data)
    println(result.count)
    println(result.max)
    println(result.mean)
    println(result.min)
  }
}
