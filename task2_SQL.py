import os
import sys
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, substring, when, sum, count, round

# 解决环境兼容性问题
os.environ['PYSPARK_PYTHON'] = sys.executable
os.environ['PYSPARK_DRIVER_PYTHON'] = sys.executable

def main():
    # 初始化 SparkSession
    spark = SparkSession.builder \
        .appName("CouponAnalysisSQL") \
        .getOrCreate()

    # --- 任务 1: 优惠券使用时间分布统计 ---
    # 读取离线数据
    offline_df = spark.read.csv("dataset/ccf_offline_stage1_train.csv", header=True, inferSchema=True)
    offline_df.createOrReplaceTempView("offline_table")

    # SQL 逻辑：
    # 1. 过滤 Date != 'null' 且 Coupon_id != 'null' (表示已使用的优惠券)
    # 2. 提取日期中的“天” (假设格式为 YYYYMMDD)
    # 3. 分类：1-10上旬，11-20中旬，21-31下旬
    # 4. 按 Coupon_id 分组计算概率
    task1_query = """
    WITH processed_data AS (
        SELECT 
            Coupon_id, 
            CAST(SUBSTRING(CAST(Date AS STRING), 7, 2) AS INT) as day
        FROM offline_table
        WHERE Date != 'null' AND Coupon_id != 'null'
    )
    SELECT 
        Coupon_id,
        ROUND(SUM(CASE WHEN day <= 10 THEN 1 ELSE 0 END) / COUNT(*), 4) as early_prob,
        ROUND(SUM(CASE WHEN day > 10 AND day <= 20 THEN 1 ELSE 0 END) / COUNT(*), 4) as mid_prob,
        ROUND(SUM(CASE WHEN day > 20 THEN 1 ELSE 0 END) / COUNT(*), 4) as late_prob
    FROM processed_data
    GROUP BY Coupon_id
    """
    
    coupon_dist = spark.sql(task1_query)
    rows = coupon_dist.limit(10).collect()
    print("任务 1：优惠券使用时间分布（前十行）")
    for row in rows:
        print(f"{row['Coupon_id']}  {row['early_prob']}  {row['mid_prob']}  {row['late_prob']}")

    # 存储结果 (可选)
    coupon_dist.write.csv("result/2_coupon_time_distribution", header=True)

    # --- 任务 2: 商家正样本比例统计 ---
    # 读取上一个任务生成的 online_consumption_table (假设为逗号分隔的 CSV)
    # 结构: Merchant_id, Neg_Count, Norm_Count, Pos_Count
    online_stats_df = spark.read.csv("result/1_online_consumption_table/part-00000", header=False, inferSchema=True) \
        .toDF("Merchant_id", "Neg_Count", "Norm_Count", "Pos_Count")
    
    online_stats_df.createOrReplaceTempView("online_table")

    # SQL 逻辑：
    # 1. 计算总样本数 = 负 + 普通 + 正
    # 2. 计算正样本比例 = 正 / 总
    # 3. 按比例降序排序，取前十
    task2_query = """
    SELECT 
        Merchant_id,
        ROUND(Pos_Count / (Neg_Count + Norm_Count + Pos_Count), 4) as pos_ratio,
        Pos_Count,
        (Neg_Count + Norm_Count + Pos_Count) as total_count
    FROM online_table
    WHERE (Neg_Count + Norm_Count + Pos_Count) > 0
    ORDER BY pos_ratio DESC, Merchant_id ASC
    """

    top_merchants = spark.sql(task2_query)
    rows = top_merchants.limit(10).collect()
    print("\n任务 2：正样本比例最高的前十个商家")
    for row in rows:
        print(f"{row['Merchant_id']}   {row['pos_ratio']}    {row['Pos_Count']}    {row['total_count']}")

    # 存储结果
    top_merchants.write.csv("result/2_top_merchants", header=True)

    spark.stop()

if __name__ == "__main__":
    main()