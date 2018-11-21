package classification

import org.apache.spark.ml.classification.NaiveBayes
import org.apache.spark.ml.feature.VectorAssembler
import org.apache.spark.sql.SparkSession

/**
  * 朴素贝叶斯算法分类鸢尾花数据集
  */
object NaiveBayes {
  def main(args: Array[String]): Unit = {
    val spark = SparkSession.builder()
      .master("local")
      .appName("NaiveBayes")
      .getOrCreate()
    spark.sparkContext.setLogLevel("WARN")
    val file = spark.read.format("csv").load("/Users/wangjx/data/iris.data")
    val random = new util.Random()
    import spark.implicits._
    val data = file.map(row=>{
     val label = row.getString(4) match {
       case "Iris-setosa" => 0
       case "Iris-versicolor" => 1
       case "Iris-virginica" => 2
     }
      (row.getString(0).toDouble,
        row.getString(1).toDouble,
        row.getString(2).toDouble,
        row.getString(3).toDouble,
        label,
        random.nextDouble())
    }
    ).toDF("_c0","_c1","_c2","_c3","label","random").sort("random")

    val assembler = new VectorAssembler()
      .setInputCols(Array("_c0","_c1","_c2","_c3"))
      .setOutputCol("features")

    val dataset = assembler.transform(data)
    val Array(train,test) = dataset.randomSplit(Array(0.8,0.2))

    val bayes = new NaiveBayes().setLabelCol("label").setFeaturesCol("features")
    val model = bayes.fit(train)
    val result = model.transform(test)
    result.show()
  }
}
