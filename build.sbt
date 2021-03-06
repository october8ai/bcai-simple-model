name := "SimpleModel"

version := "0.1"

scalaVersion := "2.12.8"
javacOptions ++= Seq("-source", "1.8", "-target", "1.8")

libraryDependencies += "org.apache.spark" %% "spark-core" % "2.4.3"
libraryDependencies += "org.apache.spark" %% "spark-sql" % "2.4.3"
libraryDependencies += "org.apache.spark" %% "spark-mllib" % "2.4.3"
