import json
import logging
import traceback
import uuid

from aiokafka import AIOKafkaConsumer, AIOKafkaProducer

from app.routers.chat.chat_service import chat_stream
from config.image.image_descriptor import ImageDescriptor
from config.kafka.chat_message import ChatMessage, ContentType
from config.session.session_config import SessionConfig

LOG = logging.getLogger("kafka-handler")

KAFKA_BOOTSTRAP         = "localhost:9092"
REQUEST_TOPIC           = "chat-requests"
RESPONSE_TOPIC          = "chat-responses"
RESPONSE_AI_TOPIC       = "chat-ai-responses"
RESPONSE_CLIENT_TOPIC   = "chat-client-responses"
GROUP_ID                = f"fastapi-consumer-group-{uuid.uuid4()}"
SESSION_TIMEOUT_MS      = 300_000
MAX_POLL_INTERVAL_MS    = 300_000

consumer: AIOKafkaConsumer = None
_producer: AIOKafkaProducer = None


import asyncio

_kafka_ready = asyncio.Event()

def get_producer() -> AIOKafkaProducer:
    if _producer is None:
        raise RuntimeError("Kafka producer is not initialized. Call init_kafka() first.")
    return _producer

async def init_kafka():
    global consumer, _producer
    consumer = AIOKafkaConsumer(
        REQUEST_TOPIC,
        bootstrap_servers=KAFKA_BOOTSTRAP,
        group_id=GROUP_ID,
        session_timeout_ms=SESSION_TIMEOUT_MS,
        max_poll_interval_ms=MAX_POLL_INTERVAL_MS,
        auto_offset_reset="latest",
        enable_auto_commit=False,
        value_deserializer=lambda m: json.loads(m.decode("utf-8")),
    )
    _producer = AIOKafkaProducer(
        bootstrap_servers=KAFKA_BOOTSTRAP,
        value_serializer=lambda v: json.dumps(v.dict(by_alias=True)).encode("utf-8"),
        key_serializer=lambda k: k.encode("utf-8"),
    )
    
    await consumer.start()

    while not consumer.assignment():
        await asyncio.sleep(0.1)

    for tp in consumer.assignment():
        await consumer.seek_to_end(tp)

    await _producer.start()
    _kafka_ready.set()
    LOG.info("Kafka initialized")


async def wait_until_kafka_ready():
    await _kafka_ready.wait()


async def shutdown_kafka():
    await consumer.stop()
    await _producer.stop()
    LOG.info("Kafka shutdown complete")

async def consume_loop():
    LOG.info("Starting consume loop")
    async for msg in consumer:
        asyncio.create_task(handle_message(msg))

async def handle_image(message: ChatMessage):
    b64_image = message.content
    image_description = ImageDescriptor.invoke(b64_image=b64_image)
    ImageDescriptor.store(image_description, SessionConfig(message.from_, message.to))

async def handle_message(msg):
    LOG.debug(msg)
    try:
        data = msg.value

        required_fields = ["chatRoomId", "from", "to", "content", "contentType"]
        missing = [field for field in required_fields if field not in data or data[field] is None]
        if missing:
            LOG.warning(f"Skipping message due to missing fields: {missing}")
            await consumer.commit()
            return

        message = ChatMessage.model_validate(data)

        if message.contentType != ContentType.TEXT:
            await consumer.commit()
            await handle_image(message)
            return

        loop = asyncio.get_running_loop()
        result = await loop.run_in_executor(None, to_response_message, message)

        await _producer.send_and_wait(
            RESPONSE_TOPIC,
            key=message.from_,
            value=result,
        )
        await consumer.commit()
    except Exception:
        LOG.exception("Error processing message")

def to_response_message(req_msg: ChatMessage) -> ChatMessage:
    content = ""
    try:
        session_config = SessionConfig(req_msg.from_, req_msg.to)
        prompt = str(req_msg.content).strip()

        if not prompt:
            raise Exception("Empty prompt")

        # NOTE 문장 단위로 Produce 로직 필요
        for chunk in chat_stream(prompt, session_config):
            content += chunk

    except Exception as e:
        LOG.error(
            f"메시지 처리간 오류: from:{req_msg.from_} to:{req_msg.to}\n{traceback.format_exc()}"
        )

    return ChatMessage(
        chatRoomId=req_msg.chatRoomId,
        from_=req_msg.to,
        to=req_msg.from_,
        content=content,
        contentType=ContentType.TEXT,
        mcpEnabled=False,
    )
