package other

import org.apache.spark.ml.classification.DecisionTreeClassifier
import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator
import org.apache.spark.ml.feature.{PCA, VectorAssembler}
import org.apache.spark.sql.SparkSession

import scala.util.Random

/**
  * pac线性降维 实现鸢尾花数据集决策树分类
  */
object PCA {
  def main(args: Array[String]): Unit = {
    val spark = SparkSession
      .builder()
      .appName("PCA")
      .master("local")
      .getOrCreate()
    spark.sparkContext.setLogLevel("WARN")

    val file = spark.read.format("csv").load("/Users/wangjx/data/iris.data")
    import spark.implicits._
    val random = new Random()
    val data = file.map(row =>{
      val label =  row.getString(4) match {
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
    }).toDF("_c0","_c1","_c2","_c3","label","rand").sort("rand")

    val assembler = new VectorAssembler().setInputCols(Array("_c0","_c1","_c2","_c3"))
      .setOutputCol("features")

    val pca = new PCA()
      .setInputCol("features").setOutputCol("features2").setK(3)
    val dataset = assembler.transform(data)
    val pcaModel = pca.fit(dataset)
    val dataset2 = pcaModel.transform(dataset)
    dataset2.show()
    val Array(train,test) = dataset2.randomSplit(Array(0.8,0.2))
    val dt = new DecisionTreeClassifier().setFeaturesCol("features2").setLabelCol("label")
    val model = dt.fit(train)
    val result = model.transform(test)
    result.show(false)
    val evaluator = new MulticlassClassificationEvaluator()
      .setLabelCol("label")
      .setPredictionCol("prediction")
      .setMetricName("accuracy")
    val accuracy = evaluator.evaluate(result)
    println(s"""accuracy is $accuracy""")
  }
}
