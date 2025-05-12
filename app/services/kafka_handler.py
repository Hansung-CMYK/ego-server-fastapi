import json
import logging
import asyncio
from aiokafka import AIOKafkaConsumer, AIOKafkaProducer

LOG = logging.getLogger("kafka-handler")

KAFKA_BOOTSTRAP      = "localhost:9092"
REQUEST_TOPIC        = "chat-requests"
RESPONSE_TOPIC       = "chat-responses"
GROUP_ID             = "fastapi-consumer-group"
SESSION_TIMEOUT_MS   = 300_000   # 5분
MAX_POLL_INTERVAL_MS = 300_000   # 5분

consumer: AIOKafkaConsumer
producer: AIOKafkaProducer

async def init_kafka():
    global consumer, producer
    consumer = AIOKafkaConsumer(
        REQUEST_TOPIC,
        bootstrap_servers=KAFKA_BOOTSTRAP,
        group_id=GROUP_ID,
        session_timeout_ms=SESSION_TIMEOUT_MS,
        max_poll_interval_ms=MAX_POLL_INTERVAL_MS,
        auto_offset_reset="earliest",
        enable_auto_commit=False,
        value_deserializer=lambda m: json.loads(m.decode("utf-8")),
    )
    producer = AIOKafkaProducer(
        bootstrap_servers=KAFKA_BOOTSTRAP,
        value_serializer=lambda v: json.dumps(v).encode("utf-8"),
        key_serializer=lambda k: k.encode("utf-8"),
    )
    await consumer.start()
    await producer.start()
    LOG.info("Kafka initialized")

async def shutdown_kafka():
    await consumer.stop()
    await producer.stop()
    LOG.info("Kafka shutdown complete")

async def consume_loop():
    LOG.info("Starting consume loop")
    async for msg in consumer:
        asyncio.create_task(handle_message(msg))

async def handle_message(msg):
    print(msg)
    try:
        data = msg.value
        if data.get("type") != "TEXT":
            await consumer.commit()
            return

        user = data["from"]

        loop   = asyncio.get_running_loop()
        result = await loop.run_in_executor(None, to_response_type, data)

        await producer.send_and_wait(
            RESPONSE_TOPIC,
            key=user,
            value=result
        )
        await consumer.commit()
    except Exception:
        LOG.exception("Error processing message")

def to_response_type(msg: dict) -> dict:
    return {
        "from":      msg.get('to'),
        "to":        msg.get('from'),
        "content":   '테스트',
        "type":      "TEXT",
        "mcpEnabled": False,
    }
