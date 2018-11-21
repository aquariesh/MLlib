package classification

import org.apache.spark.ml.classification.{DecisionTreeClassifier, LinearSVC, NaiveBayes}
import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator
import org.apache.spark.ml.feature.VectorAssembler
import org.apache.spark.sql.SparkSession

/**
  * 决策树算法分类鸢尾花数据集
  */
object Tree_1 {
  def main(args: Array[String]): Unit = {
    val spark = SparkSession.builder()
      .master("local")
      .appName("tree_1")
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

    val dt = new DecisionTreeClassifier().setFeaturesCol("features").setLabelCol("label")
    val model = dt.fit(train)
    val result = model.transform(test)
    result.show()

    val evaluator = new MulticlassClassificationEvaluator()
      .setLabelCol("label")
      .setPredictionCol("prediction")
      .setMetricName("accuracy")
    val accuracy = evaluator.evaluate(result)
    println(s"""accuracy is $accuracy""")
  }
}
