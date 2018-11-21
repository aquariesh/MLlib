package basic

import org.apache.spark.mllib.stat.test.ChiSqTestResult
import org.apache.spark.mllib.{linalg, stat}
import org.apache.spark.sql.SparkSession
/**
  * 假设检验demo
  */
object HypothesisTest {
  def main(args: Array[String]): Unit = {
    val spark = SparkSession.builder()
      .master("local")
      .appName("HypothesisTest")
      .getOrCreate()
    val data = linalg.Matrices.dense(2,2,Array(127,19,147,10))
    //根据pValue的值（0.04）做判断 0.05 大于是支持 小于是反对 是否相对独立 即是有关系的
    val result: ChiSqTestResult = stat.Statistics.chiSqTest(data)
    println(result)
  }
}
