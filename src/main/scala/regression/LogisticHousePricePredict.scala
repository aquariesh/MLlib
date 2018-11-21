package regression

import org.apache.spark.ml.classification.LogisticRegression
import org.apache.spark.ml.feature.VectorAssembler
import org.apache.spark.ml.regression.LinearRegression
import org.apache.spark.sql.SparkSession

/**
  * 逻辑回归算法预测房价
  */
object LogisticHousePricePredict {
  def main(args: Array[String]): Unit = {
    val spark = SparkSession.builder()
      .master("local")
      .appName("LinearHousePricePredict")
      .getOrCreate()
    //为了转换成df 必须导入隐式转换包
    import spark.implicits._
    val file = spark.read.format("csv")
      .option("sep",";").option("header","true").load("/Users/wangjx/data/house.csv")
    //生成随机数 为了打乱已经有序的数据
    val random = new util.Random()
    //选出面积与房价两个字段 并生成随机数
    val data = file.select("square","price")
      .map(row=>(row.getAs[String](0).toDouble,row.getString(1).toDouble,random.nextDouble()))
      .toDF("square","price","random")
      .sort("random")
    //封装传递
    val assembler = new VectorAssembler()
      //设置自变量 特征
      .setInputCols(Array("square"))
      .setOutputCol("features")
    val dataset = assembler.transform(data)

    var Array(train,test) = dataset.randomSplit(Array(0.8,0.2),1234L)
    //创建逻辑回归实例
    val lr = new LogisticRegression().setLabelCol("price").setFeaturesCol("features")
      .setMaxIter(1).setRegParam(0.3).setElasticNetParam(0.8)
    val model = lr.fit(train)
    val result = model.transform(test)
    result.show()
    println(model.summary.totalIterations)
  }
}
