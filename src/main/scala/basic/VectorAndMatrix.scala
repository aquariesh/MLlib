package basic

import breeze.linalg._
import org.apache.spark.mllib.linalg.{Matrices, Vectors}
/**
  * 基于rdd的mllib的向量与矩阵的demo测试
  */
object VectorAndMatrix {
  def main(args: Array[String]): Unit = {
    /**
     *  向量
     */
    val v1 = Vectors.dense(1,2,3,4)
    println(v1)
    val v2 = DenseVector(1,2,3,4)
    println(v2+v2)
    println(v2.t)
    println(v2*v2.t)

    /**
      * 矩阵
      */
    val m1 = Matrices.dense(2,3,Array(1,4,2,5,3,6))
    println(m1)
    val m2 = DenseMatrix(Array(1,2,3),Array(4,5,6))
    println(m2)
    println(m2.t)
  }
}
