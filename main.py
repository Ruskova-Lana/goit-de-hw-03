from pyspark.sql import SparkSession
from pyspark.sql import functions as F

spark = SparkSession.builder.appName("goit-de-hw-03").getOrCreate()

# 1. Завантаження CSV-файлів як окремих DataFrame
users_df = spark.read.csv("data/users.csv", header=True, inferSchema=True)
purchases_df = spark.read.csv("data/purchases.csv", header=True, inferSchema=True)
products_df = spark.read.csv("data/products.csv", header=True, inferSchema=True)

print("=== 1. Початкові дані ===")
for name, df in [("users", users_df), ("purchases", purchases_df), ("products", products_df)]:
    print(f"{name}: {df.count()} рядків")
    df.show(5)

# 2. Очищення даних: видалення рядків з пропущеними значеннями
users_df = users_df.dropna()
purchases_df = purchases_df.dropna()
products_df = products_df.dropna()

print("=== 2. Після очищення ===")
for name, df in [("users", users_df), ("purchases", purchases_df), ("products", products_df)]:
    print(f"{name}: {df.count()} рядків")

# Об'єднання покупок з продуктами та користувачами, розрахунок суми покупки
full_df = (
    purchases_df
    .join(products_df, on="product_id", how="inner")
    .join(users_df, on="user_id", how="inner")
    .withColumn("total_amount", F.col("quantity") * F.col("price"))
)

# 3. Загальна сума покупок за кожною категорією продуктів
print("=== 3. Загальна сума покупок за категоріями ===")
total_by_category = (
    full_df.groupBy("category")
    .agg(F.round(F.sum("total_amount"), 2).alias("total_amount"))
    .orderBy(F.desc("total_amount"))
)
total_by_category.show()

# 4. Сума покупок за категоріями для віку 18–25 включно
print("=== 4. Сума покупок за категоріями (вік 18–25) ===")
young_df = full_df.filter(F.col("age").between(18, 25))
young_by_category = (
    young_df.groupBy("category")
    .agg(F.sum("total_amount").alias("total_amount"))
)
young_by_category.withColumn("total_amount", F.round("total_amount", 2)) \
    .orderBy(F.desc("total_amount")).show()

# 5. Частка покупок за категоріями від сумарних витрат для віку 18–25
print("=== 5. Частка витрат за категоріями (вік 18–25), % ===")
young_total = young_df.agg(F.sum("total_amount")).collect()[0][0]
young_share = (
    young_by_category
    .withColumn("percentage", F.round(F.col("total_amount") / F.lit(young_total) * 100, 2))
    .withColumn("total_amount", F.round("total_amount", 2))
    .orderBy(F.desc("percentage"))
)
young_share.show()

# 6. Топ-3 категорії з найвищим відсотком витрат для віку 18–25
print("=== 6. Топ-3 категорії (вік 18–25) ===")
young_share.select("category", "percentage").limit(3).show()

spark.stop()
