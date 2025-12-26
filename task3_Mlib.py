import os
import sys
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, when, udf, to_date, datediff, dayofweek, count, avg, round as sql_round
from pyspark.sql.types import DoubleType, IntegerType
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.classification import LogisticRegression, RandomForestClassifier, DecisionTreeClassifier
from pyspark.ml.classification import GBTClassifier
from pyspark.ml.evaluation import BinaryClassificationEvaluator
from pyspark.ml.tuning import CrossValidator, ParamGridBuilder

def main():
    spark = SparkSession.builder \
        .appName("CouponPredictionML") \
        .getOrCreate()
        
    spark.sparkContext.setLogLevel("ERROR")

    # 1. 加载数据 (离线训练集)
    raw_df = spark.read.csv("dataset/ccf_offline_stage1_train.csv", header=True, inferSchema=True)
    eval_df = spark.read.csv("dataset/ccf_offline_stage1_test_revised.csv", header=True, inferSchema=True)

    # 过滤掉没有领取优惠券的记录
    df = raw_df.filter(col("Coupon_id") != "null")

    # 2. 特征工程
    # 处理折扣率 (Discount_rate)
    def parse_discount_rate(rate):
        if rate == "null": return 1.0
        if ":" in rate:
            parts = rate.split(":")
            return 1.0 - float(parts[1]) / float(parts[0])
        return float(rate)

    def parse_discount_type(rate):
        return 1 if ":" in rate else 0
    
    def parse_discount_threshold(rate):
        if ":" in str(rate):
            return float(rate.split(":")[0])
        return 0.0

    def parse_discount_abs(rate):
        if ":" in str(rate):
            return float(rate.split(":")[1])
        return 0.0

    discount_rate_udf = udf(parse_discount_rate, DoubleType())
    discount_type_udf = udf(parse_discount_type, IntegerType())
    threshold_udf = udf(parse_discount_threshold, DoubleType())
    abs_udf = udf(parse_discount_abs, DoubleType())


    df = df.\
        withColumn("rate", discount_rate_udf(col("Discount_rate"))) \
        .withColumn("is_manjian", discount_type_udf(col("Discount_rate"))) \
        .withColumn("threshold", threshold_udf(col("Discount_rate"))) \
        .withColumn("discount_amt", abs_udf(col("Discount_rate")))

    # 处理距离 (Distance)
    df = df.withColumn("distance_val", when(col("Distance") == "null", -1).otherwise(col("Distance").cast(IntegerType())))

    # 处理日期特征 (Date_received)
    df = df.withColumn("received_date", to_date(col("Date_received").cast("string"), "yyyyMMdd"))
    df = df.withColumn("day_of_week", dayofweek(col("received_date")))
    df = df.withColumn("is_weekend", when(col("day_of_week").isin(1,7), 1).otherwise(0))

    # 标签：15天内使用优惠券记为1，否则为0
    df = df.withColumn("consume_date", to_date(col("Date").cast("string"), "yyyyMMdd"))
    df = df.withColumn("label", when(
        (col("Date") != "null") & (datediff(col("consume_date"), col("received_date")) <= 15), 1
    ).otherwise(0))

    # 划分数据集
    train_raw, test_raw = df.randomSplit([0.8, 0.2], seed=42)

    # 在训练集上计算统计特征
    user_stats = train_raw.groupBy("User_id").agg(
        count("Coupon_id").alias("user_receive_count"),
        avg("label").alias("user_verify_rate")
    )
    merchant_stats = train_raw.groupBy("Merchant_id").agg(
        count("Coupon_id").alias("merchant_issue_count"),
        avg("label").alias("merchant_verify_rate")
    )
    um_stats = train_raw.groupBy("User_id", "Merchant_id").agg(
        avg("label").alias("um_verify_rate")
    )

      
    # 将统计特征关联回各自的数据集
    train_proc = train_raw.join(user_stats, on="User_id", how="left") \
                          .join(merchant_stats, on="Merchant_id", how="left") \
                          .join(um_stats, on=["User_id", "Merchant_id"], how="left") \
                          .fillna(0, subset=["user_verify_rate", "merchant_verify_rate", "um_verify_rate"])
    
    # # 计算正负样本比例
    # dataset_size = train_proc.count()
    # positive_count = train_proc.filter(col("label") == 1).count()
    # balancing_ratio = (dataset_size - positive_count) / positive_count
    # # 增加权重列
    # train_proc = train_proc.withColumn("weight", when(col("label") == 1, balancing_ratio).otherwise(1.0))
  
    test_proc = test_raw.join(user_stats, on="User_id", how="left") \
                        .join(merchant_stats, on="Merchant_id", how="left") \
                        .join(um_stats, on=["User_id", "Merchant_id"], how="left") \
                        .fillna(0, subset=["user_verify_rate", "merchant_verify_rate", "um_verify_rate"])

    # 4. 准备训练数据
    feature_cols = ["rate", "is_manjian", "distance_val", "day_of_week", 
                    "user_verify_rate", "merchant_verify_rate", "is_weekend",
                    "threshold", "discount_amt", "um_verify_rate"]
    assembler = VectorAssembler(inputCols=feature_cols, outputCol="features", handleInvalid="keep")
    
    train_data = assembler.transform(train_proc).select("features", "label", "weight")
    test_data = assembler.transform(test_proc).select("features", "label")

    evaluator = BinaryClassificationEvaluator(rawPredictionCol="rawPrediction", labelCol="label")
    
    # 5. 训练模型
    # print("使用逻辑回归模型进行训练...")
    # lr = LogisticRegression(featuresCol="features", labelCol="label", maxIter=50)
    # model = lr.fit(train_data)
    print("使用随机森林模型进行训练...")
    # rf = RandomForestClassifier(featuresCol="features", labelCol="label", weightCol="weight", numTrees=150, maxDepth=5)    
    rf = RandomForestClassifier(featuresCol="features", labelCol="label", numTrees=150, maxDepth=5) 
    model = rf.fit(train_data)
    # print("使用决策树模型进行训练...")
    # dm = DecisionTreeClassifier(featuresCol="features", labelCol="label", maxDepth=10)
    # model = dm.fit(train_data)

    # 6. 预测与评估
    predictions = model.transform(test_data)
    auc = evaluator.evaluate(predictions)

    print(f"验证集上模型 AUC 评估结果: {auc:.4f}")
    
    # 展示部分预测结果
    print("预测结果示例 (1表示15天内使用，0表示未使用):")
    predictions.select(
        "features", 
        "label", 
        "prediction", 
        sql_round(udf(lambda v: float(v[1]), DoubleType())("probability"), 4).alias("prob_1")
    ).show(10)

    # 预测新数据 (eval_df)
    print("\n正在处理测试集并进行预测...")
    
    # 对测试集应用相同的特征工程
    eval_proc = eval_df \
        .withColumn("rate", discount_rate_udf(col("Discount_rate"))) \
        .withColumn("is_manjian", discount_type_udf(col("Discount_rate"))) \
        .withColumn("distance_val", when(col("Distance") == "null", -1).otherwise(col("Distance").cast(IntegerType()))) \
        .withColumn("received_date", to_date(col("Date_received").cast("string"), "yyyyMMdd")) \
        .withColumn("day_of_week", dayofweek(col("received_date"))) \
        .withColumn("is_weekend", when(col("day_of_week").isin(1,7), 1).otherwise(0)) \
        .withColumn("threshold", threshold_udf(col("Discount_rate"))) \
        .withColumn("discount_amt", abs_udf(col("Discount_rate")))

    # 关联训练集得到的统计特征
    eval_proc = eval_proc.join(user_stats, on="User_id", how="left") \
                         .join(merchant_stats, on="Merchant_id", how="left") \
                         .join(um_stats, on=["User_id", "Merchant_id"], how="left") \
                         .fillna(0, subset=["user_verify_rate", "merchant_verify_rate", "um_verify_rate"])

    # 3. 转换特征向量并预测
    eval_ml_data = assembler.transform(eval_proc)
    final_predictions = model.transform(eval_ml_data)

    # 将 probability 向量转换为 1 的概率值（float）
    prob1_udf = udf(lambda v: float(v[1]) if v is not None else 0.0, DoubleType())
    # 用标量概率替换原来的向量概率列，便于后续保存；同时保留一个四舍五入显示列
    final_predictions = final_predictions.withColumn("probability", prob1_udf(col("probability"))) \
                                         .withColumn("prob_1", sql_round(col("probability"), 4))

    # 4. 输出结果
    print("测试集预测结果 (前10行):")
    final_predictions.select("User_id", "Coupon_id", "Date_received", "prob_1").show(10)

    # 5. 保存结果到本地
    final_predictions.select("User_id", "Coupon_id", "Date_received", "probability") \
        .coalesce(1).write.mode("overwrite").csv("result/predicted_results", header=False)


    spark.stop()

if __name__ == "__main__":
    main()