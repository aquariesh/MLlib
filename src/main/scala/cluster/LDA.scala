package classification

import org.apache.spark.ml.classification.{LinearSVC, NaiveBayes}
import org.apache.spark.ml.clustering.{KMeans, LDA}
import org.apache.spark.ml.feature.VectorAssembler
import org.apache.spark.sql.SparkSession

/**
  * LDA聚类鸢尾花数据集
  */
object LDA {
  def main(args: Array[String]): Unit = {
    val spark = SparkSession.builder()
      .master("local")
      .appName("LDA")
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

    val lda = new LDA().setFeaturesCol("features").setK(3).setMaxIter(40)
    val model = lda.fit(train)
    val prediction = model.transform(train)
    //prediction.show()
    val ll = model.logLikelihood(train) //最大似然估计 越大越好
    val lp = model.logPerplexity(train) //对数迷惑度 越小越好

    // Describe topics.
    val topics = model.describeTopics(3)
    prediction.select("label","topicDistribution").show(false)
    println("The topics described by their top-weighted terms:")
    topics.show(false)
    println(s"The lower bound on the log likelihood of the entire corpus: $ll")
    println(s"The upper bound on perplexity: $lp")

  }
}
