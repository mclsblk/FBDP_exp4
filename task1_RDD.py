from pyspark import SparkConf, SparkContext


# online_stru: User_id,Merchant_id,Action,Coupon_id,Discount_rate,Date_received,Date
# offline_stru: User_id,Merchant_id,Coupon_id,Discount_rate,Distance,Date_received,Date

def main():
    conf = SparkConf().setAppName("CouponAnalysis")
    sc = SparkContext(conf=conf)

    # 读取数据，假设文件在当前目录下，跳过表头
    raw_data = sc.textFile("dataset/ccf_online_stage1_train.csv")
    header = raw_data.first()
    data = raw_data.filter(lambda line: line != header).map(lambda line: line.split(','))

    # 数据列索引参考:
    # 0:User_id, 1:Merchant_id, 2:Action, 3:Coupon_id, 4:Discount_rate, 5:Date_received, 6:Date

    # --- 任务 1: 统计优惠券发放数量 (被使用次数) ---
    # 过滤条件: Coupon_id != 'null' 且 Date != 'null'
    used_coupons = data.filter(lambda x: x[3] != 'null' and x[3] != 'fixed' and x[6] != 'null') \
                       .map(lambda x: (x[3], 1)) \
                       .reduceByKey(lambda a, b: a + b) \
                       .sortBy(lambda x: x[1], ascending=False)

    print("任务 1 前十名结果：")
    for item in used_coupons.take(10):
        print(f"{item[0]}  {item[1]}")
    
    used_coupons.map(lambda x: f"{x[0]},{x[1]}") \
                .coalesce(1) \
                .saveAsTextFile("result/1_coupon_use_count")


    # --- 任务 2: 查询指定商家优惠券使用情况 ---
    # 逻辑判断函数
    def classify_consumption(fields):
        merchant_id = fields[1]
        coupon_id = fields[3]
        date = fields[6]
        
        neg, norm, pos = 0, 0, 0
        if date == 'null' and coupon_id != 'null':
            neg = 1  # 负样本
        elif date != 'null' and coupon_id == 'null':
            norm = 1 # 普通消费
        elif date != 'null' and coupon_id != 'null':
            pos = 1  # 正样本
            
        return (merchant_id, (neg, norm, pos))

    merchant_stats = data.map(classify_consumption) \
                         .reduceByKey(lambda a, b: (a[0] + b[0], a[1] + b[1], a[2] + b[2])) \
                         .sortByKey()


    # 格式化输出并存储结果
    formatted_stats = merchant_stats.map(lambda x: f"{x[0]}  {x[1][0]}  {x[1][1]}  {x[1][2]}")
    
    print("\n任务 2 前十行结果：")    
    for item in formatted_stats.take(10):
        print(item)

    # 将结果保存到目录（online_consumption_table）
    merchant_stats.map(lambda x: f"{x[0]},{x[1][0]},{x[1][1]},{x[1][2]}") \
        .coalesce(1) \
        .saveAsTextFile("result/1_online_consumption_table") \
    
    sc.stop()

if __name__ == "__main__":
    main()