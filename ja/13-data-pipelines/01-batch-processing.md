# バッチ処理

> **注:** この記事は英語版からの翻訳です。コードブロックおよびMermaidダイアグラムは原文のまま保持しています。

## TL;DR

バッチ処理は、スケジュールされたジョブで大量のデータを処理し、レイテンシよりもスループットを最適化します。MapReduceが分散バッチ処理の先駆けであり、Sparkなどの現代的なシステムはインメモリ計算で改善しています。データの完全性が鮮度よりも重要な場合にバッチ処理を選択してください。

---

## バッチ処理を使用するタイミング

```
Batch Processing:                    Stream Processing:
─────────────────                    ──────────────────
• Data completeness required         • Real-time insights needed
• High throughput priority           • Low latency priority
• Complex aggregations               • Simple transformations
• Historical analysis                • Event-driven actions
• Cost-effective compute             • Continuous processing

Examples:                            Examples:
• Daily reports                      • Fraud detection
• ETL pipelines                      • Live dashboards
• ML model training                  • Alerting
• Data warehouse loads               • Session tracking
```

---

## MapReduceの基礎

### プログラミングモデル

```
Input Data ──► Map ──► Shuffle ──► Reduce ──► Output

Map:     (k1, v1) → list[(k2, v2)]
Reduce:  (k2, list[v2]) → list[(k3, v3)]
```

### ワードカウントの例

```python
# Input: Collection of documents
# Output: Word frequency counts

def map(document_id, document_text):
    """Emit (word, 1) for each word"""
    for word in document_text.split():
        emit(word.lower(), 1)

def reduce(word, counts):
    """Sum all counts for a word"""
    emit(word, sum(counts))

# Execution flow:
#
# Document 1: "hello world"
# Document 2: "hello hadoop"
#
# Map output:
#   ("hello", 1), ("world", 1)
#   ("hello", 1), ("hadoop", 1)
#
# Shuffle (group by key):
#   "hello" → [1, 1]
#   "world" → [1]
#   "hadoop" → [1]
#
# Reduce output:
#   ("hello", 2), ("world", 1), ("hadoop", 1)
```

### MapReduceアーキテクチャ

```mermaid
graph TD
    JT["Job Tracker<br/><br/>Job scheduling<br/>Task assignment<br/>Progress monitoring"]
    JT --> TT1
    JT --> TT2
    JT --> TT3

    subgraph TT1["Task Tracker (Node 1)"]
        M1[Map Task]
        R1[Reduce Task]
        H1[("HDFS Block")]
    end

    subgraph TT2["Task Tracker (Node 2)"]
        M2[Map Task]
        R2[Reduce Task]
        H2[("HDFS Block")]
    end

    subgraph TT3["Task Tracker (Node 3)"]
        M3[Map Task]
        R3[Reduce Task]
        H3[("HDFS Block")]
    end
```

### データローカリティ

```mermaid
graph TD
    subgraph N1["Node 1"]
        BA1[Block A]
        BB1[Block B]
    end
    subgraph N2["Node 2"]
        BA2[Block A]
        BC2[Block C]
    end
    subgraph N3["Node 3"]
        BB3[Block B]
        BC3[Block C]
    end
```

```
原則: データを計算の場所に移動するのではなく、計算をデータの場所に移動する

タスクスケジューリングの優先順位:
1. データローカル:     ブロックがあるノードで実行（最適）
2. ラックローカル:     同じラック内で実行（ラック内ネットワークは高速）
3. ラック外:           任意の場所で実行（最終手段）
```

---

## Apache Spark

### MapReduceよりSparkが優れている理由

```
MapReduceの制限:
- ステージ間のディスクI/O（低速）
- MapとReduceプリミティブのみ
- 冗長なプログラミングモデル
- インタラクティブクエリ不可

Sparkの改善点:
- インメモリ計算（10〜100倍高速）
- 豊富な変換API
- 最適化付き遅延評価
- インタラクティブシェル
- バッチとストリーミングで同じコード

パフォーマンス比較（100TBソート）:
MapReduce: 72分、2100ノード
Spark:     23分、206ノード
```

### RDD（弾力的分散データセット）

```python
from pyspark import SparkContext

sc = SparkContext("local", "WordCount")

# Create RDD from file
text_file = sc.textFile("hdfs://path/to/file")

# Transformations (lazy - not executed yet)
words = text_file.flatMap(lambda line: line.split())
word_pairs = words.map(lambda word: (word, 1))
word_counts = word_pairs.reduceByKey(lambda a, b: a + b)

# Action (triggers execution)
results = word_counts.collect()

# Execution plan (DAG):
# textFile ──► flatMap ──► map ──► reduceByKey ──► collect
#                                                    │
#                              Narrow               Wide
#                              transforms           transform
#                              (no shuffle)         (shuffle)
```

### DataFrame API

```python
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, sum, avg, window

spark = SparkSession.builder.appName("Analytics").getOrCreate()

# Read data
orders = spark.read.parquet("s3://bucket/orders/")
products = spark.read.parquet("s3://bucket/products/")

# Transformations
daily_revenue = (
    orders
    .filter(col("status") == "completed")
    .join(products, orders.product_id == products.id)
    .groupBy(
        window(col("created_at"), "1 day"),
        col("category")
    )
    .agg(
        sum("total").alias("revenue"),
        avg("total").alias("avg_order_value"),
        count("*").alias("order_count")
    )
    .orderBy("window", "category")
)

# Write output
daily_revenue.write.mode("overwrite").parquet("s3://bucket/reports/daily_revenue/")
```

### Sparkアーキテクチャ

```mermaid
graph TD
    DP["Driver Program<br/><br/>SparkContext<br/>Create RDDs, Build DAG, Schedule tasks"]
    CM["Cluster Manager<br/>(YARN/Mesos/K8s)"]
    DP --> CM

    CM --> E1
    CM --> E2
    CM --> E3

    subgraph E1["Executor (Node 1)"]
        T1A[Task] --- T1B[Task] --- C1[Cache]
    end

    subgraph E2["Executor (Node 2)"]
        T2A[Task] --- T2B[Task] --- C2[Cache]
    end

    subgraph E3["Executor (Node 3)"]
        T3A[Task] --- T3B[Task] --- C3[Cache]
    end
```

---

## ETLパイプライン設計

### Extract, Transform, Load

```mermaid
graph TD
    E["EXTRACT<br/><br/>Databases<br/>APIs<br/>Files<br/>Streams"]
    T["TRANSFORM<br/><br/>Clean<br/>Validate<br/>Enrich<br/>Aggregate<br/>Join"]
    L["LOAD<br/><br/>Data Warehouse<br/>Data Lake<br/>Search Index<br/>Analytics DB"]

    E --> T --> L
```

### パイプライン実装

```python
from pyspark.sql import SparkSession
from pyspark.sql.functions import *
from pyspark.sql.types import *

class SalesETLPipeline:
    def __init__(self, spark):
        self.spark = spark

    def extract(self, date):
        """Extract data from various sources"""

        # From database
        orders = (
            self.spark.read
            .format("jdbc")
            .option("url", "jdbc:postgresql://db/sales")
            .option("dbtable", f"(SELECT * FROM orders WHERE date = '{date}') t")
            .load()
        )

        # From API (stored as JSON)
        exchange_rates = (
            self.spark.read
            .json(f"s3://bucket/exchange_rates/{date}/*.json")
        )

        # From data lake
        products = self.spark.read.parquet("s3://bucket/products/")
        customers = self.spark.read.parquet("s3://bucket/customers/")

        return orders, exchange_rates, products, customers

    def transform(self, orders, exchange_rates, products, customers):
        """Apply business transformations"""

        # Data cleaning
        cleaned_orders = (
            orders
            .filter(col("total") > 0)
            .filter(col("status").isin(["completed", "shipped"]))
            .dropDuplicates(["order_id"])
            .withColumn("total", col("total").cast("decimal(10,2)"))
        )

        # Data enrichment
        enriched_orders = (
            cleaned_orders
            .join(products, "product_id")
            .join(customers, "customer_id")
            .join(
                exchange_rates,
                cleaned_orders.currency == exchange_rates.currency_code
            )
            .withColumn(
                "total_usd",
                col("total") / col("exchange_rate")
            )
        )

        # Aggregations
        daily_metrics = (
            enriched_orders
            .groupBy("date", "category", "region")
            .agg(
                sum("total_usd").alias("revenue"),
                count("*").alias("order_count"),
                countDistinct("customer_id").alias("unique_customers"),
                avg("total_usd").alias("avg_order_value")
            )
        )

        return enriched_orders, daily_metrics

    def load(self, enriched_orders, daily_metrics, date):
        """Load to destination"""

        # Detailed data to data lake (partitioned)
        (
            enriched_orders
            .write
            .mode("overwrite")
            .partitionBy("date")
            .parquet(f"s3://bucket/enriched_orders/date={date}/")
        )

        # Aggregates to data warehouse
        (
            daily_metrics
            .write
            .format("jdbc")
            .option("url", "jdbc:redshift://cluster/warehouse")
            .option("dbtable", "daily_sales_metrics")
            .mode("append")
            .save()
        )

    def run(self, date):
        """Execute full pipeline"""
        orders, rates, products, customers = self.extract(date)
        enriched, metrics = self.transform(orders, rates, products, customers)
        self.load(enriched, metrics, date)
```

---

## オーケストレーション

### Apache Airflow

```python
from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.providers.apache.spark.operators.spark_submit import SparkSubmitOperator
from airflow.sensors.external_task import ExternalTaskSensor
from datetime import datetime, timedelta

default_args = {
    'owner': 'data-team',
    'depends_on_past': True,
    'retries': 3,
    'retry_delay': timedelta(minutes=5),
    'email_on_failure': True,
    'email': ['data-team@company.com']
}

with DAG(
    'daily_sales_etl',
    default_args=default_args,
    schedule_interval='0 2 * * *',  # 2 AM daily
    start_date=datetime(2024, 1, 1),
    catchup=True,
    max_active_runs=1
) as dag:

    # Wait for upstream data
    wait_for_orders = ExternalTaskSensor(
        task_id='wait_for_orders_export',
        external_dag_id='orders_export',
        external_task_id='export_complete',
        timeout=3600
    )

    # Extract from sources
    extract_task = SparkSubmitOperator(
        task_id='extract_data',
        application='s3://jobs/extract.py',
        conf={'spark.executor.memory': '4g'},
        application_args=['--date', '{{ ds }}']
    )

    # Transform
    transform_task = SparkSubmitOperator(
        task_id='transform_data',
        application='s3://jobs/transform.py',
        conf={'spark.executor.memory': '8g'},
        application_args=['--date', '{{ ds }}']
    )

    # Load to warehouse
    load_task = SparkSubmitOperator(
        task_id='load_warehouse',
        application='s3://jobs/load.py',
        application_args=['--date', '{{ ds }}']
    )

    # Data quality checks
    quality_check = PythonOperator(
        task_id='data_quality_check',
        python_callable=run_quality_checks,
        op_kwargs={'date': '{{ ds }}'}
    )

    # Dependencies
    wait_for_orders >> extract_task >> transform_task >> load_task >> quality_check
```

### DAGの可視化

```mermaid
graph TD
    W["wait_for_orders<br/>(ExternalSensor)"]
    E["extract_data<br/>(SparkSubmit)"]
    T["transform_data<br/>(SparkSubmit)"]
    L["load_warehouse<br/>(SparkSubmit)"]
    Q["quality_check<br/>(Python)"]

    W --> E --> T --> L --> Q
```

---

## データ品質

### バリデーションフレームワーク

```python
from great_expectations import DataContext
from pyspark.sql.functions import *

class DataQualityValidator:
    def __init__(self, spark):
        self.spark = spark

    def validate_orders(self, df, date):
        """Validate orders data quality"""

        results = {
            'date': date,
            'checks': [],
            'passed': True
        }

        # Check 1: No null primary keys
        null_ids = df.filter(col("order_id").isNull()).count()
        results['checks'].append({
            'name': 'no_null_primary_keys',
            'passed': null_ids == 0,
            'details': f'Found {null_ids} null order_ids'
        })

        # Check 2: Amounts are positive
        negative_amounts = df.filter(col("total") < 0).count()
        results['checks'].append({
            'name': 'positive_amounts',
            'passed': negative_amounts == 0,
            'details': f'Found {negative_amounts} negative amounts'
        })

        # Check 3: Dates are valid
        invalid_dates = df.filter(
            (col("created_at") > current_timestamp()) |
            (col("created_at") < "2020-01-01")
        ).count()
        results['checks'].append({
            'name': 'valid_dates',
            'passed': invalid_dates == 0,
            'details': f'Found {invalid_dates} invalid dates'
        })

        # Check 4: Referential integrity
        product_ids = df.select("product_id").distinct()
        products = self.spark.read.parquet("s3://bucket/products/")
        orphans = product_ids.subtract(
            products.select("id").withColumnRenamed("id", "product_id")
        ).count()
        results['checks'].append({
            'name': 'referential_integrity',
            'passed': orphans == 0,
            'details': f'Found {orphans} orphan product_ids'
        })

        # Check 5: Completeness - row count within expected range
        row_count = df.count()
        expected_min = 10000  # Based on historical data
        expected_max = 1000000
        results['checks'].append({
            'name': 'row_count_in_range',
            'passed': expected_min <= row_count <= expected_max,
            'details': f'Row count: {row_count}'
        })

        # Overall result
        results['passed'] = all(c['passed'] for c in results['checks'])

        return results
```

---

## パーティショニング戦略

### 時間ベースのパーティショニング

```
s3://bucket/orders/
├── year=2024/
│   ├── month=01/
│   │   ├── day=01/
│   │   │   ├── part-00000.parquet
│   │   │   └── part-00001.parquet
│   │   └── day=02/
│   │       └── ...
│   └── month=02/
│       └── ...
└── year=2023/
    └── ...

メリット:
- 関連するパーティションのみをクエリ
- ライフサイクル管理が容易（古いパーティションの削除）
- 自然なデータ整理
```

```python
# Writing partitioned data
(
    orders
    .withColumn("year", year("created_at"))
    .withColumn("month", month("created_at"))
    .withColumn("day", dayofmonth("created_at"))
    .write
    .partitionBy("year", "month", "day")
    .parquet("s3://bucket/orders/")
)

# Reading with partition pruning
spark.read.parquet("s3://bucket/orders/") \
    .filter(col("year") == 2024) \
    .filter(col("month") == 1)
# Only reads year=2024/month=01/ partitions
```

### ハッシュパーティショニング

```python
# Partition by hash of customer_id for even distribution
(
    orders
    .withColumn("partition", hash("customer_id") % 100)
    .write
    .partitionBy("partition")
    .parquet("s3://bucket/orders_by_customer/")
)
```

---

## ベストプラクティス

### パフォーマンス最適化

```python
# 1. Cache intermediate results used multiple times
enriched_orders = transform(raw_orders)
enriched_orders.cache()  # Keep in memory

daily_metrics = enriched_orders.groupBy("date").agg(...)
weekly_metrics = enriched_orders.groupBy(weekofyear("date")).agg(...)

# 2. Broadcast small tables in joins
from pyspark.sql.functions import broadcast

large_table.join(
    broadcast(small_lookup_table),  # < 10MB
    "key"
)

# 3. Avoid shuffles when possible
# BAD: Two shuffles
df.groupBy("key1").agg(...).groupBy("key2").agg(...)

# GOOD: Single shuffle
df.groupBy("key1", "key2").agg(...)

# 4. Use appropriate file formats
# Parquet: Column-oriented, great for analytics
# ORC: Similar to Parquet, better for Hive
# Avro: Row-oriented, good for write-heavy workloads
```

### べき等パイプライン

```python
# Pipeline should produce same output when run multiple times
def run_pipeline(date):
    output_path = f"s3://bucket/output/date={date}/"

    # Delete existing output (if any)
    delete_path(output_path)

    # Process
    result = process_data(date)

    # Write
    result.write.mode("overwrite").parquet(output_path)

# Now safe to re-run failed pipelines
```

---

## 参考文献

- [MapReduce Paper](https://research.google/pubs/pub62/)
- [Spark: Cluster Computing with Working Sets](https://www.usenix.org/legacy/event/hotcloud10/tech/full_papers/Zaharia.pdf)
- [Apache Spark Documentation](https://spark.apache.org/docs/latest/)
- [Apache Airflow Documentation](https://airflow.apache.org/docs/)
- [Designing Data-Intensive Applications - Chapter 10](https://dataintensive.net/)
