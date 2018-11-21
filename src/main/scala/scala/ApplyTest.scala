package scala

object ApplyTest {
  def main(args: Array[String]): Unit = {
    val test = Test()
    println(test.name)
  }
}


class Test{
   val name = "wjx"
}

object Test{
  def apply(): Test = {
    new Test
  }
}