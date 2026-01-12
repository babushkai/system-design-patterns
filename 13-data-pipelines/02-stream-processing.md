# Stream Processing

## TL;DR

Stream processing handles unbounded data in real-time, trading perfect accuracy for low latency. Key challenges include handling late/out-of-order data, managing state, and achieving exactly-once semantics. Apache Kafka and Apache Flink are the dominant technologies.

---

## Batch vs. Stream Processing

```
Batch Processing:
─────────────────
Input:    [████████████████████]  Bounded dataset
Process:  Wait for all data, then process
Latency:  Minutes to hours
Accuracy: Complete, consistent
Example:  Daily reports, ETL

Stream Processing:
──────────────────
Input:    ──►──►──►──►──►──►──►  Unbounded events
Process:  Process each event as it arrives
Latency:  Milliseconds to seconds
Accuracy: Approximate, eventually consistent
Example:  Fraud detection, live dashboards
```

---

## Stream Processing Concepts

### Event Time vs. Processing Time

```
Event Time:    When the event actually occurred
Processing Time: When the event is processed by the system

Reality:
┌─────────────────────────────────────────────────────────────────┐
│ Event Time  │  00:01  │  00:02  │  00:03  │  00:04  │  00:05   │
│─────────────┼─────────┼─────────┼─────────┼─────────┼──────────│
│ Event A     │    ●    │         │         │         │          │
│ Event B     │         │    ●    │         │         │          │
│ Event C     │         │         │    ●    │         │          │
│─────────────┼─────────┼─────────┼─────────┼─────────┼──────────│
│ Processing  │         │         │         │  A,C    │    B     │
│ Time        │         │         │         │ (00:04) │  (00:05) │
└─────────────────────────────────────────────────────────────────┘

Notice: Event B arrived AFTER Event C (out of order)
        Events A and C arrived together (network batching)

Why it matters:
- If aggregating by event time (correct), must handle late arrivals
- If aggregating by processing time (simple), results may be wrong
```

### Windowing

```
Event Stream: ──A──B──C──D──E──F──G──H──I──J──►

Tumbling Windows (fixed, non-overlapping):
┌─────────┐ ┌─────────┐ ┌─────────┐
│ A  B  C │ │ D  E  F │ │ G  H  I │ ...
└─────────┘ └─────────┘ └─────────┘
   Window 1    Window 2    Window 3

Sliding Windows (overlapping):
┌─────────────────┐
│   A  B  C  D    │ Window 1
└─────────────────┘
      ┌─────────────────┐
      │   B  C  D  E    │ Window 2
      └─────────────────┘
            ┌─────────────────┐
            │   C  D  E  F    │ Window 3
            └─────────────────┘

Session Windows (activity-based):
┌───────────────────┐     ┌─────────────┐     ┌───────────┐
│  A  B  C  D       │     │  E  F       │     │  G  H  I  │
└───────────────────┘     └─────────────┘     └───────────┘
    Session 1                Session 2           Session 3
(events close together)   (gap > threshold)   (new session)
```

### Watermarks

```
Watermark = "No more events with timestamp < W will arrive"

Purpose: Tell the system when a window can be closed

Stream with watermark:
Time:    │00:01│00:02│00:03│00:04│00:05│00:06│00:07│
Events:  │  A  │  B  │     │  C  │  D  │  E  │     │
         │     │     │     │     │     │     │     │
Watermark:─────────────────────► W=00:03
                                 │
                                 └── "Safe to compute results 
                                      for windows ending at 00:03"

Handling late data (after watermark):
- Drop: Discard late events (simplest)
- Update: Recompute and emit updated result
- Side output: Send to separate late data stream
```

---

## Apache Kafka Streams

### Streams DSL

```java
StreamsBuilder builder = new StreamsBuilder();

// Source: Read from topic
KStream<String, Order> orders = builder.stream("orders");

// Stateless transformation
KStream<String, Order> validOrders = orders
    .filter((key, order) -> order.getAmount() > 0)
    .mapValues(order -> enrichOrder(order));

// Windowed aggregation
KTable<Windowed<String>, Long> ordersPerMinute = validOrders
    .groupBy((key, order) -> order.getCustomerId())
    .windowedBy(TimeWindows.of(Duration.ofMinutes(1)))
    .count();

// Join streams
KStream<String, EnrichedOrder> enriched = orders.join(
    customers,  // KTable
    (order, customer) -> new EnrichedOrder(order, customer),
    Joined.with(Serdes.String(), orderSerde, customerSerde)
);

// Sink: Write to topic
enriched.to("enriched-orders");

// Build and start
KafkaStreams streams = new KafkaStreams(builder.build(), config);
streams.start();
```

### Stateful Processing

```java
// State store for deduplication
StoreBuilder<KeyValueStore<String, Long>> storeBuilder = 
    Stores.keyValueStoreBuilder(
        Stores.persistentKeyValueStore("seen-events"),
        Serdes.String(),
        Serdes.Long()
    );

builder.addStateStore(storeBuilder);

// Processor with state access
KStream<String, Event> deduplicated = events.transform(
    () -> new Transformer<String, Event, KeyValue<String, Event>>() {
        private KeyValueStore<String, Long> store;
        
        @Override
        public void init(ProcessorContext context) {
            store = context.getStateStore("seen-events");
        }
        
        @Override
        public KeyValue<String, Event> transform(String key, Event event) {
            String eventId = event.getId();
            
            if (store.get(eventId) != null) {
                return null;  // Duplicate, skip
            }
            
            store.put(eventId, System.currentTimeMillis());
            return KeyValue.pair(key, event);
        }
        
        @Override
        public void close() {}
    },
    "seen-events"
);
```

---

## Apache Flink

### DataStream API

```java
StreamExecutionEnvironment env = StreamExecutionEnvironment.getExecutionEnvironment();

// Enable checkpointing for exactly-once
env.enableCheckpointing(60000);
env.getCheckpointConfig().setCheckpointingMode(CheckpointingMode.EXACTLY_ONCE);

// Source
DataStream<Event> events = env
    .addSource(new FlinkKafkaConsumer<>("events", new EventDeserializer(), properties))
    .assignTimestampsAndWatermarks(
        WatermarkStrategy
            .<Event>forBoundedOutOfOrderness(Duration.ofSeconds(10))
            .withTimestampAssigner((event, timestamp) -> event.getTimestamp())
    );

// Process with event time windows
DataStream<Result> results = events
    .keyBy(Event::getUserId)
    .window(TumblingEventTimeWindows.of(Time.minutes(5)))
    .aggregate(new CountAggregate());

// Handle late data
OutputTag<Event> lateDataTag = new OutputTag<Event>("late-data"){};

SingleOutputStreamOperator<Result> mainStream = events
    .keyBy(Event::getUserId)
    .window(TumblingEventTimeWindows.of(Time.minutes(5)))
    .allowedLateness(Time.minutes(1))
    .sideOutputLateData(lateDataTag)
    .aggregate(new CountAggregate());

DataStream<Event> lateData = mainStream.getSideOutput(lateDataTag);
lateData.addSink(new LateDataSink());

env.execute("Stream Processing Job");
```

### Flink Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        Job Manager                               │
│                                                                 │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────────────────┐ │
│  │ Checkpoint  │  │   Job       │  │    Resource             │ │
│  │ Coordinator │  │  Scheduler  │  │    Manager              │ │
│  └─────────────┘  └─────────────┘  └─────────────────────────┘ │
└────────────────────────────┬────────────────────────────────────┘
                             │
         ┌───────────────────┼───────────────────┐
         ▼                   ▼                   ▼
┌─────────────────┐ ┌─────────────────┐ ┌─────────────────┐
│   Task Manager  │ │   Task Manager  │ │   Task Manager  │
│                 │ │                 │ │                 │
│ ┌─────────────┐ │ │ ┌─────────────┐ │ │ ┌─────────────┐ │
│ │   Task      │ │ │ │   Task      │ │ │ │   Task      │ │
│ │   Slot      │ │ │ │   Slot      │ │ │ │   Slot      │ │
│ └─────────────┘ │ │ └─────────────┘ │ │ └─────────────┘ │
│ ┌─────────────┐ │ │ ┌─────────────┐ │ │ ┌─────────────┐ │
│ │   State     │ │ │ │   State     │ │ │ │   State     │ │
│ │   Backend   │ │ │ │   Backend   │ │ │ │   Backend   │ │
│ │  (RocksDB)  │ │ │ │  (RocksDB)  │ │ │ │  (RocksDB)  │ │
│ └─────────────┘ │ │ └─────────────┘ │ │ └─────────────┘ │
└─────────────────┘ └─────────────────┘ └─────────────────┘
```

---

## Exactly-Once Semantics

### The Challenge

```
At-most-once:
  Process each event 0 or 1 times
  May lose events on failure
  Simple but not for critical data

At-least-once:
  Process each event 1 or more times
  No data loss, but may duplicate
  Need idempotent processing or deduplication

Exactly-once:
  Process each event exactly 1 time
  No loss, no duplicates
  Hard to achieve in distributed systems
```

### Checkpointing (Flink)

```
Normal Processing:
Source ──► Map ──► Window ──► Sink
  │          │        │         │
  │          │        │         │
State: S1  State: S2  State: S3  Output

Checkpoint:
1. Inject barrier into stream
2. Operators save state when barrier passes
3. State saved to durable storage

Source ──[barrier]──► Map ──[barrier]──► Window ──[barrier]──► Sink
            │                  │                    │
            ▼                  ▼                    ▼
         ┌──────────────────────────────────────────────┐
         │              Checkpoint Storage              │
         │  S1: offset=1000                             │
         │  S2: count=500                               │
         │  S3: window_state={...}                      │
         └──────────────────────────────────────────────┘

Recovery:
1. Restore all operator state from checkpoint
2. Replay events from checkpoint offset
3. Continue processing
→ No events lost, no duplicates
```

### Transactional Sinks

```java
// Kafka transactional producer for exactly-once sink
FlinkKafkaProducer<String> producer = new FlinkKafkaProducer<>(
    "output-topic",
    new SimpleStringSchema(),
    properties,
    FlinkKafkaProducer.Semantic.EXACTLY_ONCE  // Enable transactions
);

// Two-phase commit:
// 1. Pre-commit: Write to Kafka (uncommitted)
// 2. Checkpoint: Save state
// 3. Commit: Mark Kafka writes as committed

// If failure after pre-commit but before commit:
// → Kafka transaction times out
// → Events not visible to consumers
// → Replay from checkpoint (events re-written)
```

---

## Common Patterns

### Event Deduplication

```python
# Flink example with keyed state
class DeduplicationFunction(KeyedProcessFunction):
    def __init__(self, ttl_seconds):
        self.ttl = ttl_seconds
        self.seen_ids = None
    
    def open(self, runtime_context):
        descriptor = ValueStateDescriptor(
            "seen-ids",
            Types.BOOLEAN()
        )
        # Set TTL to auto-cleanup old entries
        ttl_config = StateTtlConfig.builder(Time.seconds(self.ttl)) \
            .setUpdateType(StateTtlConfig.UpdateType.OnCreateAndWrite) \
            .build()
        descriptor.enable_time_to_live(ttl_config)
        self.seen_ids = self.get_runtime_context().get_state(descriptor)
    
    def process_element(self, event, ctx, out):
        if self.seen_ids.value() is None:
            self.seen_ids.update(True)
            out.collect(event)
        # else: duplicate, skip
```

### Sessionization

```python
# Group events into user sessions
class SessionWindowFunction(ProcessWindowFunction):
    def process(self, key, context, events, out):
        session = {
            'user_id': key,
            'start_time': min(e.timestamp for e in events),
            'end_time': max(e.timestamp for e in events),
            'event_count': len(events),
            'events': list(events)
        }
        out.collect(session)

# Apply session windows (gap-based)
events \
    .key_by(lambda e: e.user_id) \
    .window(EventTimeSessionWindows.with_gap(Time.minutes(30))) \
    .process(SessionWindowFunction())
```

### Real-Time Aggregation

```java
// Running count of events per category
DataStream<CategoryCount> counts = events
    .keyBy(Event::getCategory)
    .process(new KeyedProcessFunction<String, Event, CategoryCount>() {
        private ValueState<Long> countState;
        
        @Override
        public void open(Configuration parameters) {
            countState = getRuntimeContext().getState(
                new ValueStateDescriptor<>("count", Long.class)
            );
        }
        
        @Override
        public void processElement(Event event, Context ctx, Collector<CategoryCount> out) {
            Long currentCount = countState.value();
            if (currentCount == null) {
                currentCount = 0L;
            }
            currentCount++;
            countState.update(currentCount);
            
            out.collect(new CategoryCount(event.getCategory(), currentCount));
        }
    });
```

### Stream-Table Join

```java
// Enrich events with static reference data
DataStream<Event> events = ...;

// Broadcast small reference table to all workers
MapStateDescriptor<String, Product> descriptor = new MapStateDescriptor<>(
    "products",
    BasicTypeInfo.STRING_TYPE_INFO,
    TypeInformation.of(Product.class)
);

BroadcastStream<Product> productsBroadcast = products.broadcast(descriptor);

DataStream<EnrichedEvent> enriched = events
    .connect(productsBroadcast)
    .process(new BroadcastProcessFunction<Event, Product, EnrichedEvent>() {
        @Override
        public void processElement(Event event, ReadOnlyContext ctx, Collector<EnrichedEvent> out) {
            ReadOnlyBroadcastState<String, Product> state = 
                ctx.getBroadcastState(descriptor);
            Product product = state.get(event.getProductId());
            out.collect(new EnrichedEvent(event, product));
        }
        
        @Override
        public void processBroadcastElement(Product product, Context ctx, Collector<EnrichedEvent> out) {
            BroadcastState<String, Product> state = ctx.getBroadcastState(descriptor);
            state.put(product.getId(), product);
        }
    });
```

---

## Backpressure Handling

### The Problem

```
Producer (fast) ──────────► Consumer (slow)
    100 msg/s                   50 msg/s
         │
         └── Where do extra 50 msg/s go?

Options:
1. Buffer (memory exhaustion)
2. Drop (data loss)
3. Backpressure (slow down producer)
```

### Flink Credit-Based Flow Control

```
┌─────────────────┐          ┌─────────────────┐
│    Upstream     │          │   Downstream    │
│    Operator     │          │    Operator     │
│                 │          │                 │
│  Output Buffer  │ ◄─────── │  Input Buffer   │
│    [████░░░░]   │  Credits │    [████████]   │
│                 │    (2)   │                 │
└─────────────────┘          └─────────────────┘

Credits = number of buffers downstream can accept
- When downstream buffer has space → send credit
- When upstream receives credit → can send that many buffers
- No credit → upstream blocks (backpressure)

Effect propagates upstream:
Sink slow → Operator 3 blocked → Operator 2 blocked → Source slows
```

### Kafka Consumer Backpressure

```python
# Control consumption rate
consumer = KafkaConsumer(
    'topic',
    max_poll_records=100,  # Limit batch size
    max_poll_interval_ms=300000  # Time to process batch
)

# Manual flow control
def consume_with_backpressure():
    while True:
        records = consumer.poll(timeout_ms=1000)
        
        if len(processing_queue) > MAX_QUEUE_SIZE:
            consumer.pause(consumer.assignment())
            time.sleep(1)
            consumer.resume(consumer.assignment())
        else:
            for record in records:
                processing_queue.put(record)
```

---

## Deployment Patterns

### Stream Processing in Kubernetes

```yaml
apiVersion: flink.apache.org/v1beta1
kind: FlinkDeployment
metadata:
  name: stream-processor
spec:
  image: flink:1.17
  flinkVersion: v1_17
  flinkConfiguration:
    taskmanager.numberOfTaskSlots: "2"
    state.backend: rocksdb
    state.checkpoints.dir: s3://bucket/checkpoints
    execution.checkpointing.interval: "60000"
  serviceAccount: flink
  jobManager:
    resource:
      memory: "2048m"
      cpu: 1
  taskManager:
    replicas: 3
    resource:
      memory: "4096m"
      cpu: 2
  job:
    jarURI: s3://bucket/jobs/stream-processor.jar
    parallelism: 6
    upgradeMode: savepoint
```

### Scaling Strategies

```
Parallelism scaling:
┌─────────────────────────────────────────────────────────────────┐
│ Kafka Topic (6 partitions)                                       │
│   P0   P1   P2   P3   P4   P5                                   │
│    │    │    │    │    │    │                                   │
│    └────┼────┼────┴────┼────┘                                   │
│         │    │         │                                         │
│    ┌────┴────┴────┐ ┌──┴───────────┐                            │
│    │  Task (0-2)  │ │  Task (3-5)  │   Parallelism = 2          │
│    └──────────────┘ └──────────────┘                            │
│                                                                  │
│    Increase parallelism:                                         │
│    ┌────┐ ┌────┐ ┌────┐ ┌────┐ ┌────┐ ┌────┐                   │
│    │ T0 │ │ T1 │ │ T2 │ │ T3 │ │ T4 │ │ T5 │  Parallelism = 6  │
│    └────┘ └────┘ └────┘ └────┘ └────┘ └────┘                   │
└─────────────────────────────────────────────────────────────────┘

Note: Max parallelism = number of partitions
      To scale beyond, repartition the topic
```

---

## Best Practices

### Design Principles

```
1. Design for failure
   - Checkpointing enabled
   - Idempotent operations where possible
   - Graceful degradation

2. Handle out-of-order data
   - Use event time, not processing time
   - Set appropriate watermarks
   - Define strategy for late data

3. Manage state carefully
   - Use incremental checkpoints for large state
   - Set TTL for state that should expire
   - Monitor state size

4. Test thoroughly
   - Unit test transformations
   - Integration test with embedded Kafka/Flink
   - Chaos testing for failure recovery
```

### Monitoring

```yaml
# Key metrics to track
Stream Processing Metrics:
  - Records processed per second
  - Processing latency (event time - processing time)
  - Checkpoint duration
  - Checkpoint size
  - Backpressure indicators
  - Consumer lag (Kafka)
  
Alerts:
  - Consumer lag > threshold
  - Checkpoint failing
  - Backpressure sustained > 5 minutes
  - Processing latency > SLA
```

---

## References

- [Streaming Systems](https://www.oreilly.com/library/view/streaming-systems/9781491983867/)
- [Apache Flink Documentation](https://flink.apache.org/docs/stable/)
- [Kafka Streams Documentation](https://kafka.apache.org/documentation/streams/)
- [The Dataflow Model Paper](https://research.google/pubs/pub43864/)
- [Exactly-Once Semantics in Flink](https://flink.apache.org/features/2018/03/01/end-to-end-exactly-once-apache-flink.html)
