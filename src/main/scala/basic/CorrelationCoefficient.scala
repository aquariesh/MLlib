package basic

import org.apache.spark.sql.SparkSession
import org.apache.spark.mllib.stat._
/**
  * 相关系数demo
  */
object CorrelationCoefficient {
  def main(args: Array[String]): Unit = {
    val spark = SparkSession.builder()
      .master("local")
      .appName("CorrelationCoefficient")
      .getOrCreate()
    val txt = spark.sparkContext.textFile("/Users/wangjx/data/beijing2.txt")
    val data = txt.flatMap(_.split(",")).map(m=>m.toDouble)
    val year = data.filter(_ > 1000)
    val rain = data.filter(_ <= 1000)
    // corr相关系数 参数1为x轴 参数2为y轴 参数3为类型 默认为皮尔逊相关系数
    val result: Double = Statistics.corr(year, rain)
    println(result)
    //res：Double = -0.4385****    由于年份是倒序排列 所以是负相关 数值不大 说明关联并不大
  }
}
