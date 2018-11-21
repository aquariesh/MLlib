package scala

object Constructor {
  def main(args: Array[String]): Unit = {
    val p1 = new person("wjx",28)
    val p2 = new person("wjx",28,"male")
    p1.test()

  }
}


class person(val name:String,val age:Int){
  /**
    * 附属构造器的第一行代码必须调用主构造器或者其他附属构造器
    * 对于访问权限 默认为public外部可以调用  private只能在定义成员的类或者方法内部调用
    * private[this] 不是类的成员
    * 如果遇到类名或者object名（）的形式 调用的是object里的apply方法
    * 如果是对象 则调用class里的apply方法
    * 在object的apply方法中去new class 不用单独new了
    */

  private[this] val school = "BUU"
  var gender:String = _
  private val heigth = 180

  def test(): Unit ={
    println(school)
    println(heigth)
  }

  def this(name:String,age:Int,gender:String){
    this(name,age)
    this.gender=gender
  }
}